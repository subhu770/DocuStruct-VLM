"""Deterministic Synthetic Multi-Vendor Invoice & OCR Token Generator.

Generates photorealistic invoice images and LayoutLM-compatible OCR tokens with
exact bounding boxes, multi-layout vendor templates, arithmetic ground truth,
synthetic skew/tilt, noise injection, and configurable edge-case corruptions.
Zero external Tesseract or HuggingFace model dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.api.schemas import BoundingBox, OCRToken


@dataclass
class SyntheticInvoice:
    """Complete bundle of synthetic invoice data, image, and ground-truth OCR tokens."""
    doc_id: str
    ground_truth: Dict[str, Any]
    tokens: List[OCRToken]
    image: np.ndarray
    image_width: int
    image_height: int
    layout_style: str
    applied_skew: float = 0.0
    corruption: Optional[str] = None


class SyntheticInvoiceGenerator:
    """Production-grade generator for synthetic invoice datasets and testing."""

    VENDORS = [
        {
            "name": "Apex Cloud Solutions Inc.",
            "address": "100 Innovation Way, Suite 400, Austin, TX",
            "phone": "+1 (512) 555-0199",
            "email": "billing@apexcloud.io",
            "tax_rate": 0.0825,
            "prefix": "APX-",
        },
        {
            "name": "Metro Logistics & Freight",
            "address": "450 Industrial Parkway, Chicago, IL",
            "phone": "+1 (312) 555-8821",
            "email": "accounts@metrologistics.com",
            "tax_rate": 0.0900,
            "prefix": "MLF-",
        },
        {
            "name": "Starlight Cafe & Roastery",
            "address": "12 Market Square, Seattle, WA",
            "phone": "+1 (206) 555-3410",
            "email": "orders@starlightroast.com",
            "tax_rate": 0.1010,
            "prefix": "SLC-",
        },
        {
            "name": "Global Hardware Supplies",
            "address": "880 Commerce Blvd, Atlanta, GA",
            "phone": "+1 (404) 555-7734",
            "email": "invoices@globalhw.com",
            "tax_rate": 0.0700,
            "prefix": "GHS-",
        },
        {
            "name": "Vertex Consulting Group",
            "address": "77 Wall Street, 18th Floor, New York, NY",
            "phone": "+1 (212) 555-9012",
            "email": "finance@vertexconsulting.com",
            "tax_rate": 0.08875,
            "prefix": "VCG-",
        },
        {
            "name": "Quantum Bio Labs",
            "address": "320 Science Park, Cambridge, MA",
            "phone": "+1 (617) 555-4309",
            "email": "pay@quantumbio.org",
            "tax_rate": 0.0625,
            "prefix": "QBL-",
        },
    ]

    CATALOG = [
        ("Cloud Compute Instance c6i.2xlarge", 140.00, 280.00),
        ("Dedicated NVMe Storage Volume 1TB", 85.00, 170.00),
        ("Managed PostgreSQL Database Cluster", 250.00, 500.00),
        ("High-Bandwidth Network Egress (TB)", 45.00, 90.00),
        ("Freight Shipping Heavy Freight Zone 3", 125.00, 350.00),
        ("Pallet Handling and Storage", 35.00, 105.00),
        ("Single Origin Espresso Beans 5lb", 48.00, 96.00),
        ("Barista Grade Oat Milk Case (12pk)", 36.00, 72.00),
        ("Stainless Steel Fasteners Box (100ct)", 22.50, 45.00),
        ("Industrial Grade Drill Bit Set", 65.00, 130.00),
        ("Senior Cloud Architecture Advisory (Hrs)", 220.00, 880.00),
        ("Document AI Engineering SOW Phase 1", 1500.00, 3000.00),
        ("Cell Culture Reagents Kit Premium", 340.00, 680.00),
    ]

    LAYOUT_STYLES = [
        "standard_two_column",
        "classic_boxed",
        "receipt_compact",
        "modern_minimal",
    ]

    def __init__(self, seed: Optional[int] = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_invoice(
        self,
        doc_id: Optional[str] = None,
        layout_style: Optional[str] = None,
        vendor_idx: Optional[int] = None,
        num_items: Optional[int] = None,
        skew_angle: float = 0.0,
        add_noise: bool = False,
        corruption: Optional[str] = None,
    ) -> SyntheticInvoice:
        """Generates a photorealistic synthetic invoice and LayoutLM-compatible OCR tokens.

        Args:
            doc_id: Unique identifier for document
            layout_style: One of ["standard_two_column", "classic_boxed", "receipt_compact", "modern_minimal"]
            vendor_idx: Index of vendor template (0..5)
            num_items: Number of line items (1..6)
            skew_angle: Document tilt in degrees (-15.0 to +15.0)
            add_noise: Whether to inject synthetic sensor/scanner noise
            corruption: Optional corruption:
                - 'arithmetic_mismatch': corrupt total so subtotal + tax != total
                - 'low_confidence': inject low OCR confidence into total/invoice_no
                - 'missing_tax': omit tax line completely
        """
        if doc_id is None:
            doc_id = f"doc_{self.rng.randint(100000, 999999)}"

        if layout_style is None:
            layout_style = self.rng.choice(self.LAYOUT_STYLES)

        if vendor_idx is None:
            vendor = self.rng.choice(self.VENDORS)
        else:
            vendor = self.VENDORS[vendor_idx % len(self.VENDORS)]

        if num_items is None:
            num_items = self.rng.randint(2, 5)

        # Generate line items
        selected_catalog = self.rng.sample(self.CATALOG, min(num_items, len(self.CATALOG)))
        line_items = []
        subtotal = 0.0

        for desc, min_p, max_p in selected_catalog:
            qty = float(self.rng.choice([1, 2, 3, 5, 10]))
            price = round(self.rng.uniform(min_p, max_p), 2)
            amount = round(qty * price, 2)
            subtotal += amount
            line_items.append({
                "description": desc,
                "quantity": qty,
                "unit_price": price,
                "amount": amount,
            })

        subtotal = round(subtotal, 2)
        tax = round(subtotal * vendor["tax_rate"], 2)
        total = round(subtotal + tax, 2)

        # Apply corruptions if specified
        is_corrupted = False
        if corruption == "arithmetic_mismatch":
            total = round(total + float(self.rng.choice([-15.00, 20.00, 35.50, -10.00])), 2)
            is_corrupted = True
        elif corruption == "missing_tax":
            tax = 0.0

        inv_num = f"{vendor['prefix']}{self.rng.randint(10000, 99999)}"
        days_ago = self.rng.randint(1, 90)
        inv_date = (datetime(2025, 1, 15) - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        ground_truth = {
            "doc_id": doc_id,
            "vendor_name": vendor["name"],
            "invoice_number": inv_num,
            "invoice_date": inv_date,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "currency": "USD",
            "line_items": line_items,
            "is_corrupted": is_corrupted,
            "corruption_type": corruption,
        }

        # Canvas sizing
        if layout_style == "receipt_compact":
            canvas_w, canvas_h = 650, 950
        else:
            canvas_w, canvas_h = 850, 1150

        # Create white background canvas
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        rendered_tokens: List[OCRToken] = []

        # Render layout
        if layout_style == "standard_two_column":
            self._render_two_column(canvas, vendor, inv_num, inv_date, line_items, subtotal, tax, total, rendered_tokens)
        elif layout_style == "classic_boxed":
            self._render_classic_boxed(canvas, vendor, inv_num, inv_date, line_items, subtotal, tax, total, rendered_tokens)
        elif layout_style == "receipt_compact":
            self._render_receipt_compact(canvas, vendor, inv_num, inv_date, line_items, subtotal, tax, total, rendered_tokens)
        else:  # modern_minimal
            self._render_modern_minimal(canvas, vendor, inv_num, inv_date, line_items, subtotal, tax, total, rendered_tokens)

        # Inject low confidence corruption if requested
        if corruption == "low_confidence":
            for t in rendered_tokens:
                if any(kw in t.text.lower() for kw in ["total", str(int(total)), inv_num]):
                    t.confidence = round(float(self.rng.uniform(0.40, 0.65)), 2)

        # Apply synthetic skew angle if specified
        if abs(skew_angle) > 0.01:
            canvas, rot_matrix = self._apply_skew(canvas, skew_angle)
            rendered_tokens = self._transform_tokens(rendered_tokens, rot_matrix, canvas_w, canvas_h)

        # Add noise if requested
        if add_noise:
            canvas = self._add_noise(canvas)

        return SyntheticInvoice(
            doc_id=doc_id,
            ground_truth=ground_truth,
            tokens=rendered_tokens,
            image=canvas,
            image_width=canvas_w,
            image_height=canvas_h,
            layout_style=layout_style,
            applied_skew=skew_angle,
            corruption=corruption,
        )

    def _draw_text_words(
        self,
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        font_scale: float = 0.5,
        thickness: int = 1,
        color: Tuple[int, int, int] = (20, 20, 20),
        tokens_accumulator: Optional[List[OCRToken]] = None,
        default_conf: float = 0.98,
    ) -> int:
        """Draws text word by word, calculating exact bounding box for each word token."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        words = text.split()
        curr_x = x
        h, w = canvas.shape[:2]

        for word in words:
            (tw, th), baseline = cv2.getTextSize(word, font, font_scale, thickness)
            # cv2.putText baseline coordinate
            cv2.putText(canvas, word, (curr_x, y), font, font_scale, color, thickness, cv2.LINE_AA)

            x1 = max(0, curr_x)
            y1 = max(0, y - th)
            x2 = min(w - 1, curr_x + tw)
            y2 = min(h - 1, y + baseline)

            if tokens_accumulator is not None and len(word) > 0:
                conf = round(float(self.rng.uniform(default_conf - 0.04, 1.0)), 3)
                tokens_accumulator.append(OCRToken(text=word, bbox=[x1, y1, x2, y2], confidence=conf))

            # Space between words
            (space_w, _), _ = cv2.getTextSize(" ", font, font_scale, thickness)
            curr_x += tw + space_w

        return curr_x

    def _render_two_column(
        self,
        canvas: np.ndarray,
        vendor: Dict[str, Any],
        inv_num: str,
        inv_date: str,
        line_items: List[Dict[str, Any]],
        subtotal: float,
        tax: float,
        total: float,
        tokens: List[OCRToken],
    ):
        """Standard corporate invoice with left vendor header and right metadata."""
        # Top Header
        self._draw_text_words(canvas, vendor["name"], 50, 70, 0.75, 2, (10, 40, 100), tokens)
        self._draw_text_words(canvas, vendor["address"], 50, 100, 0.45, 1, (80, 80, 80), tokens)
        self._draw_text_words(canvas, vendor["phone"], 50, 125, 0.45, 1, (80, 80, 80), tokens)

        # Right-side Invoice Info
        self._draw_text_words(canvas, "INVOICE", 600, 70, 0.85, 2, (30, 30, 30), tokens)
        self._draw_text_words(canvas, f"Invoice #: {inv_num}", 550, 105, 0.5, 1, (40, 40, 40), tokens)
        self._draw_text_words(canvas, f"Date: {inv_date}", 550, 130, 0.5, 1, (40, 40, 40), tokens)

        # Divider line
        cv2.line(canvas, (50, 160), (800, 160), (200, 200, 200), 2)

        # Table Header
        y = 200
        self._draw_text_words(canvas, "Description", 55, y, 0.55, 2, (30, 30, 30), tokens)
        self._draw_text_words(canvas, "Qty", 480, y, 0.55, 2, (30, 30, 30), tokens)
        self._draw_text_words(canvas, "Unit Price", 570, y, 0.55, 2, (30, 30, 30), tokens)
        self._draw_text_words(canvas, "Amount", 710, y, 0.55, 2, (30, 30, 30), tokens)
        cv2.line(canvas, (50, y + 10), (800, y + 10), (100, 100, 100), 1)

        # Items
        y += 45
        for item in line_items:
            self._draw_text_words(canvas, item["description"], 55, y, 0.5, 1, (50, 50, 50), tokens)
            self._draw_text_words(canvas, f"{item['quantity']:.0f}", 490, y, 0.5, 1, (50, 50, 50), tokens)
            self._draw_text_words(canvas, f"${item['unit_price']:.2f}", 580, y, 0.5, 1, (50, 50, 50), tokens)
            self._draw_text_words(canvas, f"${item['amount']:.2f}", 715, y, 0.5, 1, (50, 50, 50), tokens)
            y += 40

        # Totals section
        cv2.line(canvas, (450, y + 10), (800, y + 10), (180, 180, 180), 1)
        y += 45
        self._draw_text_words(canvas, "Subtotal:", 520, y, 0.55, 1, (40, 40, 40), tokens)
        self._draw_text_words(canvas, f"${subtotal:.2f}", 710, y, 0.55, 1, (40, 40, 40), tokens)

        y += 35
        self._draw_text_words(canvas, "Tax:", 520, y, 0.55, 1, (40, 40, 40), tokens)
        self._draw_text_words(canvas, f"${tax:.2f}", 710, y, 0.55, 1, (40, 40, 40), tokens)

        y += 35
        cv2.line(canvas, (500, y - 10), (800, y - 10), (50, 50, 50), 2)
        self._draw_text_words(canvas, "Total:", 520, y + 15, 0.65, 2, (10, 10, 10), tokens)
        self._draw_text_words(canvas, f"${total:.2f}", 705, y + 15, 0.65, 2, (10, 10, 10), tokens)

    def _render_classic_boxed(
        self,
        canvas: np.ndarray,
        vendor: Dict[str, Any],
        inv_num: str,
        inv_date: str,
        line_items: List[Dict[str, Any]],
        subtotal: float,
        tax: float,
        total: float,
        tokens: List[OCRToken],
    ):
        """Classic bordered invoice with table grid and boxed totals."""
        # Vendor Box
        cv2.rectangle(canvas, (50, 40), (450, 150), (180, 180, 180), 1)
        self._draw_text_words(canvas, vendor["name"], 65, 75, 0.65, 2, (20, 20, 20), tokens)
        self._draw_text_words(canvas, vendor["address"], 65, 105, 0.45, 1, (70, 70, 70), tokens)
        self._draw_text_words(canvas, vendor["email"], 65, 130, 0.45, 1, (70, 70, 70), tokens)

        # Invoice Meta Box
        cv2.rectangle(canvas, (470, 40), (800, 150), (180, 180, 180), 1)
        self._draw_text_words(canvas, "INVOICE RECORD", 490, 70, 0.6, 2, (10, 10, 10), tokens)
        self._draw_text_words(canvas, "Invoice No:", 490, 100, 0.5, 1, (40, 40, 40), tokens)
        self._draw_text_words(canvas, inv_num, 610, 100, 0.5, 2, (20, 20, 20), tokens)
        self._draw_text_words(canvas, "Issue Date:", 490, 130, 0.5, 1, (40, 40, 40), tokens)
        self._draw_text_words(canvas, inv_date, 610, 130, 0.5, 1, (40, 40, 40), tokens)

        # Table Grid
        y_start = 180
        table_h = 50 + len(line_items) * 40
        cv2.rectangle(canvas, (50, y_start), (800, y_start + table_h), (120, 120, 120), 1)
        # Header background
        cv2.rectangle(canvas, (51, y_start + 1), (799, y_start + 38), (240, 240, 240), -1)

        self._draw_text_words(canvas, "Item Description", 65, y_start + 26, 0.5, 2, (30, 30, 30), tokens)
        self._draw_text_words(canvas, "Quantity", 450, y_start + 26, 0.5, 2, (30, 30, 30), tokens)
        self._draw_text_words(canvas, "Rate", 570, y_start + 26, 0.5, 2, (30, 30, 30), tokens)
        self._draw_text_words(canvas, "Total Amount", 680, y_start + 26, 0.5, 2, (30, 30, 30), tokens)

        # Grid column dividers
        cv2.line(canvas, (430, y_start), (430, y_start + table_h), (180, 180, 180), 1)
        cv2.line(canvas, (550, y_start), (550, y_start + table_h), (180, 180, 180), 1)
        cv2.line(canvas, (660, y_start), (660, y_start + table_h), (180, 180, 180), 1)

        y = y_start + 65
        for item in line_items:
            self._draw_text_words(canvas, item["description"], 65, y, 0.45, 1, (40, 40, 40), tokens)
            self._draw_text_words(canvas, f"{item['quantity']:.0f}", 465, y, 0.45, 1, (40, 40, 40), tokens)
            self._draw_text_words(canvas, f"${item['unit_price']:.2f}", 585, y, 0.45, 1, (40, 40, 40), tokens)
            self._draw_text_words(canvas, f"${item['amount']:.2f}", 695, y, 0.45, 1, (40, 40, 40), tokens)
            cv2.line(canvas, (50, y + 12), (800, y + 12), (230, 230, 230), 1)
            y += 40

        # Bottom Totals Box
        tot_box_top = y_start + table_h + 20
        cv2.rectangle(canvas, (470, tot_box_top), (800, tot_box_top + 130), (140, 140, 140), 1)

        ty = tot_box_top + 30
        self._draw_text_words(canvas, "Subtotal", 490, ty, 0.5, 1, (40, 40, 40), tokens)
        self._draw_text_words(canvas, f"${subtotal:.2f}", 690, ty, 0.5, 1, (40, 40, 40), tokens)

        ty += 35
        self._draw_text_words(canvas, "Tax (Sales)", 490, ty, 0.5, 1, (40, 40, 40), tokens)
        self._draw_text_words(canvas, f"${tax:.2f}", 690, ty, 0.5, 1, (40, 40, 40), tokens)

        ty += 40
        cv2.line(canvas, (475, ty - 15), (795, ty - 15), (100, 100, 100), 1)
        self._draw_text_words(canvas, "Grand Total", 490, ty, 0.6, 2, (10, 10, 10), tokens)
        self._draw_text_words(canvas, f"${total:.2f}", 685, ty, 0.6, 2, (10, 10, 10), tokens)

    def _render_receipt_compact(
        self,
        canvas: np.ndarray,
        vendor: Dict[str, Any],
        inv_num: str,
        inv_date: str,
        line_items: List[Dict[str, Any]],
        subtotal: float,
        tax: float,
        total: float,
        tokens: List[OCRToken],
    ):
        """Compact receipt layout with centered headers and dashed line separators."""
        # Centered header
        self._draw_text_words(canvas, vendor["name"], 140, 60, 0.65, 2, (20, 20, 20), tokens)
        self._draw_text_words(canvas, vendor["address"], 120, 90, 0.4, 1, (80, 80, 80), tokens)
        self._draw_text_words(canvas, vendor["phone"], 200, 115, 0.4, 1, (80, 80, 80), tokens)

        self._draw_text_words(canvas, f"Receipt #: {inv_num}", 80, 150, 0.45, 1, (40, 40, 40), tokens)
        self._draw_text_words(canvas, f"Date: {inv_date}", 380, 150, 0.45, 1, (40, 40, 40), tokens)

        # Dashed line
        for dot_x in range(50, 580, 10):
            cv2.line(canvas, (dot_x, 175), (dot_x + 5, 175), (150, 150, 150), 1)

        # Columns
        y = 205
        self._draw_text_words(canvas, "Item", 60, y, 0.45, 2, (30, 30, 30), tokens)
        self._draw_text_words(canvas, "Qty", 320, y, 0.45, 2, (30, 30, 30), tokens)
        self._draw_text_words(canvas, "Price", 400, y, 0.45, 2, (30, 30, 30), tokens)
        self._draw_text_words(canvas, "Total", 500, y, 0.45, 2, (30, 30, 30), tokens)

        y += 35
        for item in line_items:
            # Truncate description slightly for receipt
            desc = item["description"][:24]
            self._draw_text_words(canvas, desc, 60, y, 0.42, 1, (40, 40, 40), tokens)
            self._draw_text_words(canvas, f"{item['quantity']:.0f}", 330, y, 0.42, 1, (40, 40, 40), tokens)
            self._draw_text_words(canvas, f"${item['unit_price']:.2f}", 405, y, 0.42, 1, (40, 40, 40), tokens)
            self._draw_text_words(canvas, f"${item['amount']:.2f}", 505, y, 0.42, 1, (40, 40, 40), tokens)
            y += 32

        for dot_x in range(50, 580, 10):
            cv2.line(canvas, (dot_x, y + 5), (dot_x + 5, y + 5), (150, 150, 150), 1)

        y += 35
        self._draw_text_words(canvas, "Subtotal", 300, y, 0.45, 1, (40, 40, 40), tokens)
        self._draw_text_words(canvas, f"${subtotal:.2f}", 500, y, 0.45, 1, (40, 40, 40), tokens)

        y += 28
        self._draw_text_words(canvas, "Sales Tax", 300, y, 0.45, 1, (40, 40, 40), tokens)
        self._draw_text_words(canvas, f"${tax:.2f}", 500, y, 0.45, 1, (40, 40, 40), tokens)

        y += 35
        cv2.line(canvas, (280, y - 10), (580, y - 10), (30, 30, 30), 2)
        self._draw_text_words(canvas, "TOTAL DUE", 290, y + 5, 0.55, 2, (10, 10, 10), tokens)
        self._draw_text_words(canvas, f"${total:.2f}", 490, y + 5, 0.55, 2, (10, 10, 10), tokens)

    def _render_modern_minimal(
        self,
        canvas: np.ndarray,
        vendor: Dict[str, Any],
        inv_num: str,
        inv_date: str,
        line_items: List[Dict[str, Any]],
        subtotal: float,
        tax: float,
        total: float,
        tokens: List[OCRToken],
    ):
        """Modern minimal layout with high whitespace and subtle lines."""
        # Top Logo/Vendor
        self._draw_text_words(canvas, vendor["name"].upper(), 60, 80, 0.7, 2, (40, 60, 90), tokens)
        self._draw_text_words(canvas, "STATEMENT OF INVOICE", 540, 80, 0.5, 1, (120, 120, 120), tokens)

        # Meta block
        y = 130
        self._draw_text_words(canvas, "Address:", 60, y, 0.45, 1, (100, 100, 100), tokens)
        self._draw_text_words(canvas, vendor["address"], 130, y, 0.45, 1, (50, 50, 50), tokens)

        y += 25
        self._draw_text_words(canvas, "Invoice ID:", 540, y, 0.45, 1, (100, 100, 100), tokens)
        self._draw_text_words(canvas, inv_num, 630, y, 0.45, 2, (20, 20, 20), tokens)

        y += 25
        self._draw_text_words(canvas, "Date:", 540, y, 0.45, 1, (100, 100, 100), tokens)
        self._draw_text_words(canvas, inv_date, 630, y, 0.45, 1, (40, 40, 40), tokens)

        cv2.line(canvas, (60, y + 25), (780, y + 25), (220, 220, 220), 1)

        # Table
        y += 65
        self._draw_text_words(canvas, "SERVICES / PRODUCTS", 60, y, 0.45, 2, (90, 90, 90), tokens)
        self._draw_text_words(canvas, "QTY", 450, y, 0.45, 2, (90, 90, 90), tokens)
        self._draw_text_words(canvas, "RATE", 560, y, 0.45, 2, (90, 90, 90), tokens)
        self._draw_text_words(canvas, "LINE TOTAL", 680, y, 0.45, 2, (90, 90, 90), tokens)

        cv2.line(canvas, (60, y + 10), (780, y + 10), (100, 100, 100), 1)

        y += 40
        for item in line_items:
            self._draw_text_words(canvas, item["description"], 60, y, 0.45, 1, (40, 40, 40), tokens)
            self._draw_text_words(canvas, f"{item['quantity']:.0f}", 460, y, 0.45, 1, (40, 40, 40), tokens)
            self._draw_text_words(canvas, f"${item['unit_price']:.2f}", 565, y, 0.45, 1, (40, 40, 40), tokens)
            self._draw_text_words(canvas, f"${item['amount']:.2f}", 695, y, 0.45, 1, (40, 40, 40), tokens)
            y += 38

        cv2.line(canvas, (500, y + 10), (780, y + 10), (220, 220, 220), 1)

        y += 45
        self._draw_text_words(canvas, "Sub-Total:", 520, y, 0.5, 1, (80, 80, 80), tokens)
        self._draw_text_words(canvas, f"${subtotal:.2f}", 690, y, 0.5, 1, (30, 30, 30), tokens)

        y += 30
        self._draw_text_words(canvas, "Estimated Tax:", 520, y, 0.5, 1, (80, 80, 80), tokens)
        self._draw_text_words(canvas, f"${tax:.2f}", 690, y, 0.5, 1, (30, 30, 30), tokens)

        y += 35
        cv2.line(canvas, (500, y - 10), (780, y - 10), (30, 30, 30), 2)
        self._draw_text_words(canvas, "Total Payable:", 520, y + 10, 0.6, 2, (10, 10, 10), tokens)
        self._draw_text_words(canvas, f"${total:.2f}", 685, y + 10, 0.6, 2, (10, 10, 10), tokens)

    def _apply_skew(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Rotates image around center with white border padding."""
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return rotated, rot_matrix

    def _transform_tokens(
        self,
        tokens: List[OCRToken],
        affine_matrix: np.ndarray,
        img_w: int,
        img_h: int
    ) -> List[OCRToken]:
        """Applies 2x3 affine matrix to tokens' bounding boxes."""
        transformed: List[OCRToken] = []
        for t in tokens:
            x1, y1, x2, y2 = t.bbox
            corners = np.array([
                [x1, y1, 1],
                [x2, y1, 1],
                [x2, y2, 1],
                [x1, y2, 1]
            ], dtype=np.float32).T
            rot_corners = affine_matrix.dot(corners)
            rx = rot_corners[0, :]
            ry = rot_corners[1, :]

            nx1 = max(0, min(img_w - 1, int(np.floor(np.min(rx)))))
            ny1 = max(0, min(img_h - 1, int(np.floor(np.min(ry)))))
            nx2 = max(0, min(img_w, int(np.ceil(np.max(rx)))))
            ny2 = max(0, min(img_h, int(np.ceil(np.max(ry)))))

            transformed.append(OCRToken(text=t.text, bbox=[nx1, ny1, max(nx1 + 1, nx2), max(ny1 + 1, ny2)], confidence=t.confidence))
        return transformed

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Injects mild Gaussian noise and sensor artifacts."""
        noise = self.np_rng.normal(0, 8, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
