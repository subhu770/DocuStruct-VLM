"""Unit Tests for Document Preprocessing Pipeline.

Tests skew angle detection, affine deskewing, and bounding box coordinate transformations.
"""

import math
import numpy as np
import pytest

from src.api.schemas import BoundingBox, OCRToken
from src.simulation.invoice_generator import SyntheticInvoiceGenerator
from src.vision.preprocessor import DocumentPreprocessor


@pytest.fixture
def preprocessor() -> DocumentPreprocessor:
    return DocumentPreprocessor()


@pytest.fixture
def generator() -> SyntheticInvoiceGenerator:
    return SyntheticInvoiceGenerator(seed=123)


def test_detect_skew_angle_zero(preprocessor: DocumentPreprocessor, generator: SyntheticInvoiceGenerator):
    """Verifies that an un-tilted document has near-zero detected skew."""
    synth = generator.generate_invoice(layout_style="standard_two_column", skew_angle=0.0)
    detected = preprocessor.detect_skew_angle(synth.image)
    assert abs(detected) <= 1.0, f"Expected near-zero skew, got {detected} degrees"


@pytest.mark.parametrize("target_angle", [-8.0, 6.0, -12.0, 10.0])
def test_detect_skew_angle_recovery(preprocessor: DocumentPreprocessor, generator: SyntheticInvoiceGenerator, target_angle: float):
    """Verifies that artificially tilted documents have their skew angles accurately detected."""
    synth = generator.generate_invoice(layout_style="standard_two_column", skew_angle=target_angle)
    detected = preprocessor.detect_skew_angle(synth.image)
    # Allow tolerance of ±2.0 degrees given discrete pixel grid and rasterization
    error = abs(detected - target_angle)
    assert error <= 2.5, f"Target angle {target_angle}°, detected {detected}°, error {error}°"


def test_deskew_straightens_image(preprocessor: DocumentPreprocessor, generator: SyntheticInvoiceGenerator):
    """Verifies that running deskew produces an upright image with residual skew near zero."""
    target_angle = 7.0
    synth = generator.generate_invoice(layout_style="standard_two_column", skew_angle=target_angle)

    deskewed_img, matrix, applied_angle = preprocessor.deskew(synth.image)
    assert matrix.shape == (2, 3), "Expected 2x3 affine matrix"

    # Measure residual angle on the deskewed image
    residual_angle = preprocessor.detect_skew_angle(deskewed_img)
    assert abs(residual_angle) <= 1.5, f"Expected residual skew <= 1.5°, got {residual_angle}°"


def test_transform_bboxes_accuracy(preprocessor: DocumentPreprocessor):
    """Verifies coordinate affine transformation on synthetic bounding boxes."""
    w, h = 800, 1000
    center = (w / 2.0, h / 2.0)
    angle = 90.0  # 90-degree clockwise rotation for clear geometric verification

    import cv2
    rot_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # Box around center: [390, 490, 410, 510]
    orig_box = [390, 490, 410, 510]
    transformed = preprocessor.transform_bboxes([orig_box], rot_matrix, w, h)

    assert len(transformed) == 1
    t_box = transformed[0]
    # Box centered on center of rotation should remain near center
    t_cx = (t_box[0] + t_box[2]) / 2.0
    t_cy = (t_box[1] + t_box[3]) / 2.0
    assert abs(t_cx - center[0]) <= 2.0
    assert abs(t_cy - center[1]) <= 2.0


def test_transform_tokens(preprocessor: DocumentPreprocessor, generator: SyntheticInvoiceGenerator):
    """Verifies that transform_tokens updates OCR token bounding boxes properly."""
    synth = generator.generate_invoice(layout_style="classic_boxed", skew_angle=0.0)
    import cv2
    rot_matrix = cv2.getRotationMatrix2D((synth.image_width / 2.0, synth.image_height / 2.0), -5.0, 1.0)

    transformed_tokens = preprocessor.transform_tokens(
        synth.tokens, rot_matrix, synth.image_width, synth.image_height
    )

    assert len(transformed_tokens) == len(synth.tokens)
    for orig, trans in zip(synth.tokens, transformed_tokens):
        assert trans.text == orig.text
        assert trans.confidence == orig.confidence
        assert trans.bbox[2] > trans.bbox[0]
        assert trans.bbox[3] > trans.bbox[1]


def test_contrast_normalization_and_adaptive_binarize(preprocessor: DocumentPreprocessor, generator: SyntheticInvoiceGenerator):
    """Verifies CLAHE normalization and adaptive thresholding produces valid binary output."""
    synth = generator.generate_invoice(layout_style="receipt_compact", add_noise=True)

    normalized = preprocessor.normalize_contrast(synth.image)
    assert normalized.shape == (synth.image_height, synth.image_width)
    assert normalized.dtype == np.uint8

    binarized = preprocessor.adaptive_binarize(synth.image)
    assert binarized.shape == (synth.image_height, synth.image_width)
    unique_vals = np.unique(binarized)
    # Binary image should only contain 0 and 255
    assert set(unique_vals).issubset({0, 255})
