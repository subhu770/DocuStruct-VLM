"""DocuStruct-VLM Data Schemas and Validation Models.

Pydantic v2 strict schemas for OCR tokens, spatial layouts, invoice fields,
arithmetic reconciliation, extraction responses, and Human-in-the-Loop review queues.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field, field_validator, model_validator


class ExtractionStatus(str, Enum):
    """Extraction routing outcome status."""
    AUTO_APPROVED = "AUTO_APPROVED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"


class ReviewStatus(str, Enum):
    """Status of an item in the Human-in-the-Loop (HITL) review queue."""
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"


class BoundingBox(BaseModel):
    """2D Spatial Bounding Box in [x1, y1, x2, y2] format.

    Compatible with standard OCR and LayoutLM [0..1000] coordinate conventions.
    """
    x1: int = Field(..., description="Top-left x coordinate")
    y1: int = Field(..., description="Top-left y coordinate")
    x2: int = Field(..., description="Bottom-right x coordinate")
    y2: int = Field(..., description="Bottom-right y coordinate")

    @model_validator(mode="after")
    def validate_coordinates(self) -> BoundingBox:
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1
        return self

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union with another box."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def vertical_overlap(self, other: BoundingBox) -> float:
        """Calculates 1D vertical overlap ratio relative to min height."""
        inter_y1 = max(self.y1, other.y1)
        inter_y2 = min(self.y2, other.y2)
        if inter_y2 <= inter_y1:
            return 0.0
        min_h = min(self.height, other.height)
        return (inter_y2 - inter_y1) / min_h if min_h > 0 else 0.0

    def horizontal_overlap(self, other: BoundingBox) -> float:
        """Calculates 1D horizontal overlap ratio relative to min width."""
        inter_x1 = max(self.x1, other.x1)
        inter_x2 = min(self.x2, other.x2)
        if inter_x2 <= inter_x1:
            return 0.0
        min_w = min(self.width, other.width)
        return (inter_x2 - inter_x1) / min_w if min_w > 0 else 0.0


class OCRToken(BaseModel):
    """Raw or preprocessed OCR Token with text, spatial bbox, and confidence."""
    text: str = Field(..., description="Recognized text content of the token")
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] coordinates")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="OCR recognition confidence [0.0..1.0]")

    @field_validator("bbox")
    @classmethod
    def sanitize_bbox(cls, v: List[int]) -> List[int]:
        x1, y1, x2, y2 = v
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [int(x1), int(y1), int(x2), int(y2)]

    @property
    def box(self) -> BoundingBox:
        return BoundingBox(x1=self.bbox[0], y1=self.bbox[1], x2=self.bbox[2], y2=self.bbox[3])


T = TypeVar("T")


class ExtractedField(BaseModel, Generic[T]):
    """Extracted semantic field with value, calibrated confidence, and provenance."""
    value: Optional[T] = Field(default=None, description="Extracted and normalized typed value")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Calibrated field confidence [0.0..1.0]")
    bbox: Optional[List[int]] = Field(default=None, description="Enclosing [x1, y1, x2, y2] bbox")
    raw_text: Optional[str] = Field(default=None, description="Original unparsed text snippet")


class LineItem(BaseModel):
    """Invoice line-item row with itemized details and arithmetic reconciliation."""
    description: str = Field(..., description="Product or service line description")
    quantity: float = Field(default=1.0, ge=0.0, description="Quantity billed")
    unit_price: float = Field(default=0.0, ge=0.0, description="Price per unit")
    amount: float = Field(default=0.0, description="Total line amount")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Line extraction confidence")
    bbox: Optional[List[int]] = Field(default=None, description="Line item row bounding box")

    def is_arithmetically_valid(self, tolerance: float = 0.05) -> bool:
        """Validates that quantity * unit_price == amount within tolerance."""
        expected = self.quantity * self.unit_price
        return math.isclose(expected, self.amount, abs_tol=tolerance)


class InvoiceSchema(BaseModel):
    """Layout-aware unstructured invoice extraction schema."""
    vendor_name: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField(value=None, confidence=0.0),
        description="Seller / Vendor name"
    )
    invoice_number: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField(value=None, confidence=0.0),
        description="Invoice identification number / code"
    )
    invoice_date: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField(value=None, confidence=0.0),
        description="Date of invoice issuance (ISO-8601 or YYYY-MM-DD)"
    )
    subtotal: ExtractedField[float] = Field(
        default_factory=lambda: ExtractedField(value=None, confidence=0.0),
        description="Subtotal amount before tax"
    )
    tax: ExtractedField[float] = Field(
        default_factory=lambda: ExtractedField(value=None, confidence=0.0),
        description="Total tax amount"
    )
    total: ExtractedField[float] = Field(
        default_factory=lambda: ExtractedField(value=None, confidence=0.0),
        description="Final invoice total payable"
    )
    line_items: List[LineItem] = Field(
        default_factory=list,
        description="Extracted table line items"
    )
    currency: str = Field(default="USD", description="Currency ISO symbol or code")

    def to_clean_dict(self) -> Dict[str, Any]:
        """Flattens extracted fields to clean values for business systems."""
        return {
            "vendor_name": self.vendor_name.value,
            "invoice_number": self.invoice_number.value,
            "invoice_date": self.invoice_date.value,
            "subtotal": self.subtotal.value,
            "tax": self.tax.value,
            "total": self.total.value,
            "currency": self.currency,
            "line_items": [item.model_dump() for item in self.line_items],
        }


class ValidationResult(BaseModel):
    """Deterministic arithmetic cross-checking and confidence threshold assessment."""
    is_valid: bool = Field(..., description="True if both arithmetic and confidence requirements are met")
    arithmetic_passed: bool = Field(..., description="True if subtotal + tax == total within tolerance")
    confidence_passed: bool = Field(..., description="True if overall confidence >= threshold")
    subtotal: Optional[float] = Field(default=None, description="Extracted subtotal")
    tax: Optional[float] = Field(default=None, description="Extracted tax")
    total: Optional[float] = Field(default=None, description="Reported invoice total")
    calculated_total: Optional[float] = Field(default=None, description="Computed total (subtotal + tax)")
    arithmetic_discrepancy: float = Field(default=0.0, description="Absolute difference between reported and computed total")
    line_items_sum: Optional[float] = Field(default=None, description="Sum of line item amounts")
    line_items_arithmetic_passed: bool = Field(default=True, description="True if line items sum up to subtotal/total")
    issues: List[str] = Field(default_factory=list, description="Detailed list of detected discrepancies or low-confidence fields")


class ExtractionRequest(BaseModel):
    """Payload for POST /extract endpoint."""
    doc_id: Optional[str] = Field(default=None, description="Optional document tracking ID")
    tokens: Optional[List[OCRToken]] = Field(default=None, description="Pre-extracted OCR tokens with bboxes")
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded invoice image (PNG/JPEG)")
    image_width: Optional[int] = Field(default=None, description="Image pixel width for coordinate normalization")
    image_height: Optional[int] = Field(default=None, description="Image pixel height for coordinate normalization")
    deskew: bool = Field(default=True, description="Whether to run automatic skew detection and correction")
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Threshold for auto-approval")
    simulate: bool = Field(default=False, description="Generate a synthetic invoice on-the-fly if tokens/image omitted")
    simulation_vendor: Optional[str] = Field(default=None, description="Vendor to use for simulation")
    simulation_layout: Optional[str] = Field(default=None, description="Layout style for simulation")
    simulation_corruption: Optional[str] = Field(default=None, description="Optional corruption to inject in simulation")


class ExtractionResponse(BaseModel):
    """Response returned by POST /extract endpoint."""
    doc_id: str = Field(..., description="Document identifier")
    status: ExtractionStatus = Field(..., description="AUTO_APPROVED or NEEDS_REVIEW")
    data: InvoiceSchema = Field(..., description="Extracted structured invoice data")
    validation: ValidationResult = Field(..., description="Arithmetic reconciliation and confidence evaluation")
    overall_confidence: float = Field(..., description="Aggregated calibrated document confidence")
    processing_time_ms: float = Field(..., description="End-to-end extraction latency in milliseconds")
    review_queue_id: Optional[str] = Field(default=None, description="ID in HITL review queue if status is NEEDS_REVIEW")


class ReviewQueueItem(BaseModel):
    """Item record in Human-in-the-Loop review queue."""
    doc_id: str = Field(..., description="Document identifier")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Queue timestamp")
    status: ReviewStatus = Field(default=ReviewStatus.PENDING, description="Current review status")
    extracted_data: InvoiceSchema = Field(..., description="Original extracted data flagged for review")
    validation: ValidationResult = Field(..., description="Validation failure details")
    overall_confidence: float = Field(..., description="Overall confidence score")
    resolved_data: Optional[InvoiceSchema] = Field(default=None, description="Human-corrected data after resolution")
    reviewer_notes: Optional[str] = Field(default=None, description="Audit notes from human reviewer")


class ResolveReviewRequest(BaseModel):
    """Payload for POST /review-queue/{doc_id}/resolve endpoint."""
    corrected_data: InvoiceSchema = Field(..., description="Human-corrected invoice fields")
    reviewer_notes: Optional[str] = Field(default=None, description="Audit notes from human reviewer")
    status: ReviewStatus = Field(default=ReviewStatus.RESOLVED, description="Resolution status")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    service: str = "DocuStruct-VLM"
    version: str = "1.0.0"
    queue_size: int = 0
