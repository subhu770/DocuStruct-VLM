"""Deterministic Arithmetic Cross-Checker & Calibrated Confidence Threshold Gate.

Reconciles document-level math (subtotal + tax == total), validates table line-item
equations (qty * unit_price == amount and sum(amounts) == subtotal), calculates
calibrated document confidence, and routes documents to AUTO_APPROVED or NEEDS_REVIEW.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from src.api.schemas import (
    ExtractionStatus,
    InvoiceSchema,
    ValidationResult,
)


class ArithmeticValidator:
    """Production validator for invoice mathematical consistency and confidence gating."""

    FIELD_WEIGHTS = {
        "total": 0.25,
        "subtotal": 0.20,
        "invoice_number": 0.15,
        "vendor_name": 0.15,
        "tax": 0.10,
        "invoice_date": 0.10,
        "line_items": 0.05,
    }

    def __init__(
        self,
        default_tolerance: float = 0.05,
        default_confidence_threshold: float = 0.80,
    ):
        self.default_tolerance = default_tolerance
        self.default_confidence_threshold = default_confidence_threshold

    def reconcile_arithmetic(
        self,
        invoice: InvoiceSchema,
        tolerance: Optional[float] = None,
    ) -> ValidationResult:
        """Cross-checks arithmetic across subtotal, tax, total, and line items."""
        tol = tolerance if tolerance is not None else self.default_tolerance
        issues: List[str] = []

        subtotal = invoice.subtotal.value
        tax = invoice.tax.value if invoice.tax.value is not None else 0.0
        total = invoice.total.value

        arithmetic_passed = True
        calc_total = None
        discrepancy = 0.0

        if subtotal is not None and total is not None:
            calc_total = round(subtotal + tax, 2)
            discrepancy = round(abs(calc_total - total), 2)
            if discrepancy > tol:
                arithmetic_passed = False
                issues.append(
                    f"ARITHMETIC_MISMATCH: subtotal ({subtotal:.2f}) + tax ({tax:.2f}) != total ({total:.2f}), "
                    f"discrepancy={discrepancy:.2f}"
                )
        elif total is None:
            arithmetic_passed = False
            issues.append("MISSING_FIELD: total amount could not be extracted")
        elif subtotal is None:
            arithmetic_passed = False
            issues.append("MISSING_FIELD: subtotal could not be extracted")

        # Validate line items internal math and sum
        line_items_arithmetic_passed = True
        line_items_sum = 0.0

        if invoice.line_items:
            for idx, item in enumerate(invoice.line_items):
                expected_amt = item.quantity * item.unit_price
                if not math.isclose(expected_amt, item.amount, abs_tol=tol):
                    line_items_arithmetic_passed = False
                    issues.append(
                        f"LINE_ITEM_ARITHMETIC_MISMATCH: item[{idx}] '{item.description}' "
                        f"({item.quantity} * {item.unit_price:.2f} = {expected_amt:.2f}) != amount ({item.amount:.2f})"
                    )
                line_items_sum += item.amount

            line_items_sum = round(line_items_sum, 2)

            if subtotal is not None:
                item_sum_discrepancy = round(abs(line_items_sum - subtotal), 2)
                max_sum_tol = max(tol, 0.02 * len(invoice.line_items))
                if item_sum_discrepancy > max_sum_tol:
                    issues.append(
                        f"LINE_ITEMS_SUM_MISMATCH: sum(items)={line_items_sum:.2f} != subtotal={subtotal:.2f}"
                    )
                    if item_sum_discrepancy > 1.0:
                        line_items_arithmetic_passed = False
        else:
            line_items_sum = None

        is_valid = arithmetic_passed and line_items_arithmetic_passed

        return ValidationResult(
            is_valid=is_valid,
            arithmetic_passed=arithmetic_passed,
            confidence_passed=False,
            subtotal=subtotal,
            tax=tax,
            total=total,
            calculated_total=calc_total,
            arithmetic_discrepancy=discrepancy,
            line_items_sum=line_items_sum,
            line_items_arithmetic_passed=line_items_arithmetic_passed,
            issues=issues,
        )

    def calibrate_confidence(self, invoice: InvoiceSchema, arithmetic_passed: bool = False) -> float:
        """Computes calibrated document-level confidence score [0.0..1.0]."""
        total_weight = 0.0
        weighted_sum = 0.0

        fields: Dict[str, float] = {
            "total": invoice.total.confidence if invoice.total.value is not None else 0.0,
            "subtotal": invoice.subtotal.confidence if invoice.subtotal.value is not None else 0.0,
            "invoice_number": invoice.invoice_number.confidence if invoice.invoice_number.value is not None else 0.0,
            "vendor_name": invoice.vendor_name.confidence if invoice.vendor_name.value is not None else 0.0,
            "tax": invoice.tax.confidence if invoice.tax.value is not None else 0.5,
            "invoice_date": invoice.invoice_date.confidence if invoice.invoice_date.value is not None else 0.0,
        }

        for f_name, weight in self.FIELD_WEIGHTS.items():
            if f_name == "line_items":
                continue
            f_score = fields.get(f_name, 0.0)
            weighted_sum += f_score * weight
            total_weight += weight

        if invoice.line_items:
            li_conf = sum(it.confidence for it in invoice.line_items) / len(invoice.line_items)
            li_weight = self.FIELD_WEIGHTS["line_items"]
            weighted_sum += li_conf * li_weight
            total_weight += li_weight

        score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Mathematical verification boost: when arithmetic passes formally, boost confidence
        if arithmetic_passed:
            score = min(1.0, score + 0.06)

        critical_fields = [invoice.total.value, invoice.subtotal.value, invoice.invoice_number.value]
        missing_count = sum(1 for v in critical_fields if v is None)
        if missing_count > 0:
            score = max(0.0, score - (0.25 * missing_count))

        return round(min(1.0, max(0.0, score)), 3)

    def evaluate_routing(
        self,
        invoice: InvoiceSchema,
        confidence_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> Tuple[ExtractionStatus, ValidationResult, float]:
        """Evaluates arithmetic and confidence, returning routing status (AUTO_APPROVED or NEEDS_REVIEW)."""
        threshold = confidence_threshold if confidence_threshold is not None else self.default_confidence_threshold
        val_result = self.reconcile_arithmetic(invoice, tolerance=tolerance)
        calibrated_conf = self.calibrate_confidence(invoice, arithmetic_passed=val_result.arithmetic_passed)

        # Flag individual degraded fields (low confidence < critical_thresh)
        has_critical_degradation = False
        critical_thresh = min(threshold, 0.70)
        if invoice.total.value is not None and invoice.total.confidence < critical_thresh:
            val_result.issues.append(f"LOW_CONFIDENCE: total confidence ({invoice.total.confidence:.2f}) < {critical_thresh}")
            has_critical_degradation = True
        if invoice.subtotal.value is not None and invoice.subtotal.confidence < critical_thresh:
            val_result.issues.append(f"LOW_CONFIDENCE: subtotal confidence ({invoice.subtotal.confidence:.2f}) < {critical_thresh}")
            has_critical_degradation = True
        if invoice.invoice_number.value is not None and invoice.invoice_number.confidence < critical_thresh:
            val_result.issues.append(f"LOW_CONFIDENCE: invoice_number confidence ({invoice.invoice_number.confidence:.2f}) < {critical_thresh}")
            has_critical_degradation = True

        confidence_passed = (calibrated_conf >= threshold) and not has_critical_degradation
        val_result.confidence_passed = confidence_passed
        val_result.is_valid = val_result.arithmetic_passed and confidence_passed

        if val_result.is_valid:
            status = ExtractionStatus.AUTO_APPROVED
        else:
            status = ExtractionStatus.NEEDS_REVIEW

        return status, val_result, calibrated_conf
