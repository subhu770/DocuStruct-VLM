"""OpenCV-based Document Preprocessing Pipeline.

Provides robust skew angle detection, affine deskewing, bounding-box coordinate
transformation, contrast normalization (CLAHE), adaptive binarization, and noise filtering.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from src.api.schemas import BoundingBox, OCRToken


class DocumentPreprocessor:
    """Production-grade document vision preprocessor for OCR & spatial extraction."""

    def __init__(
        self,
        max_skew_angle: float = 45.0,
        min_line_length: int = 30,
        max_line_gap: int = 12,
        hough_threshold: int = 40,
        border_fill_value: Union[int, Tuple[int, int, int]] = (255, 255, 255),
    ):
        self.max_skew_angle = max_skew_angle
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.hough_threshold = hough_threshold
        self.border_fill_value = border_fill_value

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Ensures image is single-channel 8-bit grayscale."""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        if len(image.shape) == 2:
            return image
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def detect_skew_angle(self, image: np.ndarray) -> float:
        """Detects document skew angle in degrees using Hough Line Transform and morphological edges.

        Returns:
            Detected skew angle in degrees in range [-max_skew_angle, max_skew_angle].
            A positive angle indicates the image was rotated counter-clockwise by that angle.
        """
        gray = self.to_grayscale(image)
        h, w = gray.shape[:2]

        # Invert so text is bright on dark background
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )

        # Detect line edges
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        angles: List[float] = []
        weights: List[float] = []

        if lines is not None:
            for line in lines:
                pts = line.reshape(-1)
                if len(pts) < 4:
                    continue
                x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length < self.min_line_length:
                    continue

                angle_deg = math.degrees(math.atan2(dy, dx))
                # Normalize angle to [-90, 90]
                if angle_deg > 90:
                    angle_deg -= 180
                elif angle_deg < -90:
                    angle_deg += 180

                # In image coordinates where Y points down, OpenCV counter-clockwise
                # rotation produces atan2(dy, dx) = -rotation_angle.
                recovered_angle = -angle_deg

                # Filter to near-horizontal text lines within allowable skew range
                if abs(recovered_angle) <= self.max_skew_angle:
                    angles.append(recovered_angle)
                    weights.append(length)

        if not angles:
            # Fallback to contour minAreaRect method
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    rect = cv2.minAreaRect(cnt)
                    (_, (rw, rh), r_angle) = rect
                    if rw < rh:
                        rw, rh = rh, rw
                        r_angle = r_angle + 90 if r_angle < 0 else r_angle - 90
                    if abs(r_angle) <= self.max_skew_angle:
                        angles.append(r_angle)
                        weights.append(float(area))

        if not angles:
            return 0.0

        # Weighted median of angles
        sorted_indices = np.argsort(angles)
        sorted_angles = np.array(angles)[sorted_indices]
        sorted_weights = np.array(weights)[sorted_indices]
        cum_weights = np.cumsum(sorted_weights)
        cutoff = cum_weights[-1] / 2.0
        median_idx = np.where(cum_weights >= cutoff)[0][0]
        detected_angle = float(sorted_angles[median_idx])

        return round(detected_angle, 2)

    def deskew(
        self,
        image: np.ndarray,
        angle: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Deskews an image by rotating around center.

        Args:
            image: Input image array
            angle: Skew angle in degrees. If None, angle is auto-detected.

        Returns:
            Tuple of (deskewed_image, 2x3 affine_matrix, applied_correction_angle)
        """
        if angle is None:
            angle = self.detect_skew_angle(image)

        # Correction angle is negative of detected skew
        correction_angle = -angle
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        # Compute affine rotation matrix
        rot_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)

        # Determine border color
        if len(image.shape) == 2:
            border_val = 255 if isinstance(self.border_fill_value, (int, float)) else 255
        else:
            border_val = self.border_fill_value if isinstance(self.border_fill_value, tuple) else (255, 255, 255)

        deskewed = cv2.warpAffine(
            image,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )

        return deskewed, rot_matrix, angle

    def transform_bboxes(
        self,
        bboxes: List[List[int]],
        affine_matrix: np.ndarray,
        img_width: int,
        img_height: int,
    ) -> List[List[int]]:
        """Applies 2x3 affine matrix to bounding boxes [x1, y1, x2, y2].

        Transforms the four rectangle corners and computes the new axis-aligned bounding box.
        """
        if not bboxes:
            return []

        transformed: List[List[int]] = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # 4 rectangle vertices
            corners = np.array([
                [x1, y1, 1],
                [x2, y1, 1],
                [x2, y2, 1],
                [x1, y2, 1]
            ], dtype=np.float32).T  # Shape: (3, 4)

            # Multiply by 2x3 affine matrix -> Shape: (2, 4)
            rotated_corners = affine_matrix.dot(corners)
            rx = rotated_corners[0, :]
            ry = rotated_corners[1, :]

            # New axis-aligned bounding box
            nx1 = max(0, min(img_width - 1, int(np.floor(np.min(rx)))))
            ny1 = max(0, min(img_height - 1, int(np.floor(np.min(ry)))))
            nx2 = max(0, min(img_width, int(np.ceil(np.max(rx)))))
            ny2 = max(0, min(img_height, int(np.ceil(np.max(ry)))))

            # Guard against degenerated boxes
            if nx2 <= nx1:
                nx2 = nx1 + 1
            if ny2 <= ny1:
                ny2 = ny1 + 1

            transformed.append([nx1, ny1, nx2, ny2])

        return transformed

    def transform_tokens(
        self,
        tokens: List[OCRToken],
        affine_matrix: np.ndarray,
        img_width: int,
        img_height: int,
    ) -> List[OCRToken]:
        """Transforms bounding boxes of a list of OCRToken instances."""
        if not tokens:
            return []
        bboxes = [t.bbox for t in tokens]
        new_bboxes = self.transform_bboxes(bboxes, affine_matrix, img_width, img_height)
        return [
            OCRToken(text=t.text, bbox=nb, confidence=t.confidence)
            for t, nb in zip(tokens, new_bboxes)
        ]

    def normalize_contrast(self, image: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        gray = self.to_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def adaptive_binarize(self, image: np.ndarray) -> np.ndarray:
        """Produces clean binary text image via Gaussian adaptive thresholding."""
        gray = self.to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 9
        )

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Reduces document grain and high-frequency noise while preserving stroke edges."""
        if len(image.shape) == 2:
            return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
        return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

    def preprocess(
        self,
        image: np.ndarray,
        tokens: Optional[List[OCRToken]] = None,
        deskew: bool = True,
    ) -> Tuple[np.ndarray, Optional[List[OCRToken]], float]:
        """Runs the complete vision preprocessing pipeline.

        Returns:
            Tuple of (preprocessed_image, transformed_tokens, detected_skew_angle)
        """
        img = image.copy()
        angle = 0.0
        h, w = img.shape[:2]

        if deskew:
            angle = self.detect_skew_angle(img)
            if abs(angle) > 0.5:
                img, matrix, _ = self.deskew(img, angle=angle)
                if tokens:
                    tokens = self.transform_tokens(tokens, matrix, w, h)

        img = self.denoise(img)
        img = self.normalize_contrast(img)

        return img, tokens, angle
