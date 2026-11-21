"""Unit Tests for Confidence Routing and Human-in-the-Loop (HITL) Queue.

Verifies deterministic routing to AUTO_APPROVED vs NEEDS_REVIEW based on arithmetic
reconciliation and calibrated confidence scores, and verifies the resolution API.
"""

from fastapi.testclient import TestClient
import pytest

from src.api.main import app
from src.api.schemas import (
    ExtractionStatus,
    InvoiceSchema,
    ResolveReviewRequest,
    ReviewStatus,
)
from src.extraction.spatial_extractor import SpatialLayoutExtractor
from src.extraction.validator import ArithmeticValidator
from src.queue.hitl_store import hitl_queue_store
from src.simulation.invoice_generator import SyntheticInvoiceGenerator


@pytest.fixture(autouse=True)
def clean_queue():
    """Ensures a clean HITL queue for each test."""
    hitl_queue_store.clear()
    yield
    hitl_queue_store.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def generator() -> SyntheticInvoiceGenerator:
    return SyntheticInvoiceGenerator(seed=777)


@pytest.fixture
def extractor() -> SpatialLayoutExtractor:
    return SpatialLayoutExtractor()


@pytest.fixture
def validator() -> ArithmeticValidator:
    return ArithmeticValidator()


def test_clean_invoice_auto_approved(
    generator: SyntheticInvoiceGenerator,
    extractor: SpatialLayoutExtractor,
    validator: ArithmeticValidator,
):
    """Verifies that an uncorrupted invoice passes arithmetic and confidence checks to be AUTO_APPROVED."""
    synth = generator.generate_invoice(layout_style="standard_two_column")
    extracted = extractor.extract(synth.tokens, synth.image_width, synth.image_height)

    status, val_result, conf = validator.evaluate_routing(extracted, confidence_threshold=0.80)

    assert status == ExtractionStatus.AUTO_APPROVED
    assert val_result.is_valid is True
    assert val_result.arithmetic_passed is True
    assert val_result.confidence_passed is True
    assert conf >= 0.80


def test_arithmetic_mismatch_routes_to_hitl(
    generator: SyntheticInvoiceGenerator,
    extractor: SpatialLayoutExtractor,
    validator: ArithmeticValidator,
):
    """Verifies that an invoice with arithmetic mismatch is routed to NEEDS_REVIEW with explicit mismatch issues."""
    synth = generator.generate_invoice(
        layout_style="classic_boxed",
        corruption="arithmetic_mismatch",
    )
    extracted = extractor.extract(synth.tokens, synth.image_width, synth.image_height)

    status, val_result, conf = validator.evaluate_routing(extracted, confidence_threshold=0.80)

    assert status == ExtractionStatus.NEEDS_REVIEW
    assert val_result.is_valid is False
    assert val_result.arithmetic_passed is False
    assert val_result.arithmetic_discrepancy > 0.05
    assert any("ARITHMETIC_MISMATCH" in issue for issue in val_result.issues)


def test_low_confidence_routes_to_hitl(
    generator: SyntheticInvoiceGenerator,
    extractor: SpatialLayoutExtractor,
    validator: ArithmeticValidator,
):
    """Verifies that degraded OCR confidence routes document to NEEDS_REVIEW."""
    synth = generator.generate_invoice(
        layout_style="receipt_compact",
        corruption="low_confidence",
    )
    extracted = extractor.extract(synth.tokens, synth.image_width, synth.image_height)

    status, val_result, conf = validator.evaluate_routing(extracted, confidence_threshold=0.85)

    assert status == ExtractionStatus.NEEDS_REVIEW
    assert val_result.confidence_passed is False
    assert any("LOW_CONFIDENCE" in issue for issue in val_result.issues)


def test_fastapi_extract_and_resolve_flow(
    client: TestClient,
    generator: SyntheticInvoiceGenerator,
):
    """Verifies end-to-end API: /extract -> HITL enqueue -> /review-queue -> /resolve."""
    # 1. Send corrupted invoice to POST /extract
    synth = generator.generate_invoice(
        doc_id="doc_audit_1001",
        layout_style="modern_minimal",
        corruption="arithmetic_mismatch",
    )

    token_payload = [t.model_dump() for t in synth.tokens]
    response = client.post(
        "/extract",
        json={
            "doc_id": "doc_audit_1001",
            "tokens": token_payload,
            "image_width": synth.image_width,
            "image_height": synth.image_height,
            "confidence_threshold": 0.85,
        },
    )

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "NEEDS_REVIEW"
    assert res_json["review_queue_id"] == "doc_audit_1001"
    assert res_json["validation"]["arithmetic_passed"] is False

    # 2. Check GET /review-queue
    q_resp = client.get("/review-queue")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert q_data["stats"]["pending"] == 1
    assert len(q_data["items"]) == 1
    assert q_data["items"][0]["doc_id"] == "doc_audit_1001"

    # 3. Resolve the item via POST /review-queue/{doc_id}/resolve
    extracted_data = res_json["data"]
    # Human fixes the total so subtotal + tax == total
    fixed_total = round(extracted_data["subtotal"]["value"] + extracted_data["tax"]["value"], 2)
    extracted_data["total"]["value"] = fixed_total
    extracted_data["total"]["confidence"] = 1.0

    resolve_resp = client.post(
        "/review-queue/doc_audit_1001/resolve",
        json={
            "corrected_data": extracted_data,
            "reviewer_notes": "Corrected total mismatch in billing portal",
        },
    )

    assert resolve_resp.status_code == 200
    resolved_item = resolve_resp.json()
    assert resolved_item["status"] == "RESOLVED"
    assert resolved_item["resolved_data"]["total"]["value"] == fixed_total
    assert resolved_item["reviewer_notes"] == "Corrected total mismatch in billing portal"

    # 4. Verify updated stats
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["queue_size"] == 0  # 0 pending
