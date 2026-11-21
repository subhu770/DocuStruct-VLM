"""Evaluation Benchmark for DocuStruct-VLM.

Evaluates field-level Precision, Recall, F1 score, arithmetic reconciliation pass rate,
HITL routing accuracy, and P50/P90/P95/P99 latency across 250 synthetic unstructured invoices.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.schemas import ExtractionStatus
from src.extraction.spatial_extractor import SpatialLayoutExtractor
from src.extraction.validator import ArithmeticValidator
from src.simulation.invoice_generator import SyntheticInvoice, SyntheticInvoiceGenerator
from src.vision.preprocessor import DocumentPreprocessor


class BenchmarkEvaluator:
    """Rigorous evaluation suite for layout-aware document extraction engines."""

    def __init__(self, num_samples: int = 250, seed: int = 42, deskew: bool = True):
        self.num_samples = num_samples
        self.seed = seed
        self.deskew = deskew
        self.generator = SyntheticInvoiceGenerator(seed=seed)
        self.preprocessor = DocumentPreprocessor()
        self.extractor = SpatialLayoutExtractor()
        self.validator = ArithmeticValidator()

    def run(self) -> Dict[str, Any]:
        """Runs benchmark across the test cohort and computes metrics."""
        print(f"\n=======================================================================")
        print(f"  DocuStruct-VLM Extraction & Arithmetic Benchmark ({self.num_samples} Invoices)")
        print(f"=======================================================================\n")

        # Track field TP, FP, FN
        fields = ["vendor_name", "invoice_number", "invoice_date", "subtotal", "tax", "total", "line_items"]
        stats = {f: {"tp": 0, "fp": 0, "fn": 0} for f in fields}

        arithmetic_clean_total = 0
        arithmetic_clean_passed = 0

        hitl_corrupted_total = 0
        hitl_corrupted_flagged = 0
        hitl_clean_total = 0
        hitl_clean_flagged = 0

        latencies_ms: List[float] = []

        start_bench_time = time.perf_counter()

        for idx in range(self.num_samples):
            # Controlled distribution of layouts and edge cases
            layout = self.generator.LAYOUT_STYLES[idx % len(self.generator.LAYOUT_STYLES)]
            vendor_idx = idx % len(self.generator.VENDORS)

            # 10% corrupted arithmetic, 5% low confidence, 15% tilted
            corruption = None
            if idx % 10 == 0:
                corruption = "arithmetic_mismatch"
            elif idx % 20 == 5:
                corruption = "low_confidence"

            skew = 0.0
            if idx % 7 == 0:
                skew = float(((idx % 5) - 2) * 4.0)  # -8, -4, 0, 4, 8 deg

            synth = self.generator.generate_invoice(
                doc_id=f"bench_doc_{idx:04d}",
                layout_style=layout,
                vendor_idx=vendor_idx,
                skew_angle=skew,
                add_noise=(idx % 12 == 0),
                corruption=corruption,
            )
            gt = synth.ground_truth

            # Run extraction and time latency
            t0 = time.perf_counter()

            tokens = synth.tokens
            img = synth.image
            if self.deskew and abs(skew) > 0.5:
                img, tokens, _ = self.preprocessor.preprocess(img, tokens=tokens, deskew=True)

            extracted = self.extractor.extract(
                tokens=tokens,
                image_width=synth.image_width,
                image_height=synth.image_height,
            )

            route_status, val_result, conf = self.validator.evaluate_routing(extracted)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            # Evaluate Fields
            # 1. Total
            gt_total = gt["total"]
            ext_total = extracted.total.value
            if ext_total is not None and math.isclose(ext_total, gt_total, abs_tol=0.05):
                stats["total"]["tp"] += 1
            elif ext_total is not None:
                stats["total"]["fp"] += 1
            else:
                stats["total"]["fn"] += 1

            # 2. Subtotal
            gt_subtotal = gt["subtotal"]
            ext_subtotal = extracted.subtotal.value
            if ext_subtotal is not None and math.isclose(ext_subtotal, gt_subtotal, abs_tol=0.05):
                stats["subtotal"]["tp"] += 1
            elif ext_subtotal is not None:
                stats["subtotal"]["fp"] += 1
            else:
                stats["subtotal"]["fn"] += 1

            # 3. Tax
            gt_tax = gt["tax"]
            ext_tax = extracted.tax.value
            if ext_tax is not None and math.isclose(ext_tax, gt_tax, abs_tol=0.05):
                stats["tax"]["tp"] += 1
            elif ext_tax is not None:
                stats["tax"]["fp"] += 1
            else:
                stats["tax"]["fn"] += 1

            # 4. Invoice Number
            gt_inv_num = gt["invoice_number"]
            ext_inv_num = extracted.invoice_number.value
            if ext_inv_num and gt_inv_num in ext_inv_num:
                stats["invoice_number"]["tp"] += 1
            elif ext_inv_num:
                stats["invoice_number"]["fp"] += 1
            else:
                stats["invoice_number"]["fn"] += 1

            # 5. Invoice Date
            gt_date = gt["invoice_date"]
            ext_date = extracted.invoice_date.value
            if ext_date and ext_date == gt_date:
                stats["invoice_date"]["tp"] += 1
            elif ext_date:
                stats["invoice_date"]["fp"] += 1
            else:
                stats["invoice_date"]["fn"] += 1

            # 6. Vendor Name
            gt_vendor = gt["vendor_name"].split()[0].lower()
            ext_vendor = extracted.vendor_name.value
            if ext_vendor and gt_vendor in ext_vendor.lower():
                stats["vendor_name"]["tp"] += 1
            elif ext_vendor:
                stats["vendor_name"]["fp"] += 1
            else:
                stats["vendor_name"]["fn"] += 1

            # 7. Line Items (Item-level matching)
            gt_items = gt["line_items"]
            matched_items = 0
            for ext_item in extracted.line_items:
                for gt_item in gt_items:
                    if math.isclose(ext_item.amount, gt_item["amount"], abs_tol=0.05):
                        matched_items += 1
                        break

            stats["line_items"]["tp"] += matched_items
            stats["line_items"]["fp"] += max(0, len(extracted.line_items) - matched_items)
            stats["line_items"]["fn"] += max(0, len(gt_items) - matched_items)

            # Evaluate Arithmetic Reconciliation on Clean Invoices
            if not gt["is_corrupted"]:
                arithmetic_clean_total += 1
                if val_result.arithmetic_passed:
                    arithmetic_clean_passed += 1

            # Evaluate HITL Routing
            if gt["is_corrupted"] or corruption is not None:
                hitl_corrupted_total += 1
                if route_status == ExtractionStatus.NEEDS_REVIEW:
                    hitl_corrupted_flagged += 1
            else:
                hitl_clean_total += 1
                if route_status == ExtractionStatus.NEEDS_REVIEW:
                    hitl_clean_flagged += 1

            if (idx + 1) % 50 == 0 or idx == self.num_samples - 1:
                print(f"Processed [{idx + 1}/{self.num_samples}] documents...")

        total_bench_duration = time.perf_counter() - start_bench_time

        # Compute Metrics
        metrics = {}
        for f in fields:
            tp = stats[f]["tp"]
            fp = stats[f]["fp"]
            fn = stats[f]["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            metrics[f] = {
                "precision": round(prec * 100.0, 2),
                "recall": round(rec * 100.0, 2),
                "f1": round(f1 * 100.0, 2),
            }

        # Arithmetic Pass Rate
        arithmetic_pass_rate = (
            (arithmetic_clean_passed / arithmetic_clean_total) * 100.0 if arithmetic_clean_total > 0 else 0.0
        )

        # HITL Routing Metrics
        hitl_recall = (
            (hitl_corrupted_flagged / hitl_corrupted_total) * 100.0 if hitl_corrupted_total > 0 else 0.0
        )
        hitl_fpr = (
            (hitl_clean_flagged / hitl_clean_total) * 100.0 if hitl_clean_total > 0 else 0.0
        )

        # Latency percentiles
        p50 = np.percentile(latencies_ms, 50)
        p90 = np.percentile(latencies_ms, 90)
        p95 = np.percentile(latencies_ms, 95)
        p99 = np.percentile(latencies_ms, 99)
        avg_latency = statistics.mean(latencies_ms)

        # Format and Print Results
        print("\n" + "=" * 71)
        print(f"{'FIELD':<20} | {'PRECISION (%)':<15} | {'RECALL (%)':<15} | {'F1 SCORE (%)':<15}")
        print("-" * 71)
        for f, m in metrics.items():
            print(f"{f:<20} | {m['precision']:<15.2f} | {m['recall']:<15.2f} | {m['f1']:<15.2f}")
        print("=" * 71)

        print(f"\n--- ARITHMETIC RECONCILIATION & HITL GATING ---")
        print(f"Clean Arithmetic Pass Rate: {arithmetic_pass_rate:.2f}% ({arithmetic_clean_passed}/{arithmetic_clean_total})")
        print(f"Corrupt Document HITL Detection Recall: {hitl_recall:.2f}% ({hitl_corrupted_flagged}/{hitl_corrupted_total})")
        print(f"Clean Document False Review Rate (FPR): {hitl_fpr:.2f}% ({hitl_clean_flagged}/{hitl_clean_total})")

        print(f"\n--- SYSTEM LATENCY BENCHMARK ---")
        print(f"Total Cohort Time: {total_bench_duration:.2f}s (Throughput: {self.num_samples / total_bench_duration:.1f} docs/sec)")
        print(f"P50 Latency: {p50:.2f} ms")
        print(f"P90 Latency: {p90:.2f} ms")
        print(f"P95 Latency: {p95:.2f} ms")
        print(f"P99 Latency: {p99:.2f} ms")
        print(f"Mean Latency: {avg_latency:.2f} ms\n")

        return {
            "metrics": metrics,
            "arithmetic_pass_rate": round(arithmetic_pass_rate, 2),
            "hitl_recall": round(hitl_recall, 2),
            "latency": {
                "p50_ms": round(p50, 2),
                "p90_ms": round(p90, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "mean_ms": round(avg_latency, 2),
            },
        }


def main():
    parser = argparse.ArgumentParser(description="DocuStruct-VLM Extraction & Arithmetic Benchmark")
    parser.add_argument("--num-samples", type=int, default=250, help="Number of synthetic invoices to benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--no-deskew", action="store_true", help="Disable vision deskewing")
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(
        num_samples=args.num_samples,
        seed=args.seed,
        deskew=not args.no_deskew,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
