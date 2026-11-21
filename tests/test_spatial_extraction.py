"""Unit Tests for 2D Spatial Layout-Aware Extractor.

Verifies template-free key-value and line-item extraction across multiple layout paradigms:
Standard Two-Column, Classic Boxed, Compact Receipt, and Modern Minimal.
"""

import math
import pytest

from src.api.schemas import OCRToken
from src.extraction.spatial_extractor import SpatialLayoutExtractor
from src.simulation.invoice_generator import SyntheticInvoiceGenerator


@pytest.fixture
def extractor() -> SpatialLayoutExtractor:
    return SpatialLayoutExtractor()


@pytest.fixture
def generator() -> SyntheticInvoiceGenerator:
    return SyntheticInvoiceGenerator(seed=42)


@pytest.mark.parametrize(
    "layout_style",
    [
        "standard_two_column",
        "classic_boxed",
        "receipt_compact",
        "modern_minimal",
    ],
)
def test_template_free_extraction_across_layouts(
    extractor: SpatialLayoutExtractor,
    generator: SyntheticInvoiceGenerator,
    layout_style: str,
):
    """Verifies that key invoice fields and line items are accurately extracted across diverse layout templates."""
    synth = generator.generate_invoice(layout_style=layout_style)
    gt = synth.ground_truth

    extracted = extractor.extract(
        tokens=synth.tokens,
        image_width=synth.image_width,
        image_height=synth.image_height,
    )

    # 1. Total Extraction Check
    assert extracted.total.value is not None, f"Failed to extract total for {layout_style}"
    assert math.isclose(extracted.total.value, gt["total"], abs_tol=0.05), (
        f"Total mismatch for {layout_style}: extracted {extracted.total.value} vs GT {gt['total']}"
    )

    # 2. Subtotal Extraction Check
    assert extracted.subtotal.value is not None, f"Failed to extract subtotal for {layout_style}"
    assert math.isclose(extracted.subtotal.value, gt["subtotal"], abs_tol=0.05), (
        f"Subtotal mismatch for {layout_style}: extracted {extracted.subtotal.value} vs GT {gt['subtotal']}"
    )

    # 3. Tax Extraction Check
    assert extracted.tax.value is not None, f"Failed to extract tax for {layout_style}"
    assert math.isclose(extracted.tax.value, gt["tax"], abs_tol=0.05), (
        f"Tax mismatch for {layout_style}: extracted {extracted.tax.value} vs GT {gt['tax']}"
    )

    # 4. Invoice Number Extraction Check
    assert extracted.invoice_number.value is not None, f"Failed to extract invoice number for {layout_style}"
    assert gt["invoice_number"] in extracted.invoice_number.value, (
        f"Invoice number mismatch: extracted '{extracted.invoice_number.value}' vs GT '{gt['invoice_number']}'"
    )

    # 5. Invoice Date Extraction Check
    assert extracted.invoice_date.value is not None, f"Failed to extract invoice date for {layout_style}"
    assert extracted.invoice_date.value == gt["invoice_date"], (
        f"Date mismatch: extracted '{extracted.invoice_date.value}' vs GT '{gt['invoice_date']}'"
    )

    # 6. Vendor Name Extraction Check
    assert extracted.vendor_name.value is not None, f"Failed to extract vendor for {layout_style}"
    # Verify core vendor name appears in extracted string
    vendor_first_word = gt["vendor_name"].split()[0].lower()
    assert vendor_first_word in extracted.vendor_name.value.lower(), (
        f"Vendor mismatch: extracted '{extracted.vendor_name.value}' vs GT '{gt['vendor_name']}'"
    )

    # 7. Line Item Extraction Check
    assert len(extracted.line_items) > 0, f"No line items extracted for {layout_style}"
    for item in extracted.line_items:
        assert item.description, "Line item description should not be empty"
        assert item.quantity > 0, "Line item quantity must be positive"
        assert item.unit_price > 0, "Line item unit price must be positive"
        assert item.amount > 0, "Line item amount must be positive"
        # Line arithmetic check
        assert math.isclose(item.quantity * item.unit_price, item.amount, abs_tol=0.05), (
            f"Line item internal math error: {item.quantity} * {item.unit_price} != {item.amount}"
        )


def test_layoutlm_normalized_coordinates(
    extractor: SpatialLayoutExtractor,
    generator: SyntheticInvoiceGenerator,
):
    """Verifies compatibility with LayoutLM [0..1000] normalized bounding-box coordinates."""
    synth = generator.generate_invoice(layout_style="standard_two_column")
    gt = synth.ground_truth
    w, h = synth.image_width, synth.image_height

    # Normalize tokens to [0..1000] scale
    normalized_tokens = [
        OCRToken(
            text=t.text,
            bbox=[
                int(t.bbox[0] * 1000.0 / w),
                int(t.bbox[1] * 1000.0 / h),
                int(t.bbox[2] * 1000.0 / w),
                int(t.bbox[3] * 1000.0 / h),
            ],
            confidence=t.confidence,
        )
        for t in synth.tokens
    ]

    extracted = extractor.extract(normalized_tokens, image_width=1000, image_height=1000)

    assert extracted.total.value is not None
    assert math.isclose(extracted.total.value, gt["total"], abs_tol=0.05)
    assert extracted.subtotal.value is not None
    assert math.isclose(extracted.subtotal.value, gt["subtotal"], abs_tol=0.05)
