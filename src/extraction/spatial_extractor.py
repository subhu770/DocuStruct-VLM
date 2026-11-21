"""Layout-Aware 2D Spatial Bounding-Box Relation Extractor.

Template-free extractor for unstructured invoices and receipts.
Models token 2D geometry, spatial reading order, horizontal/vertical alignment,
semantic anchor detection, and tabular line-item clustering.
Compatible with standard OCR tokens and LayoutLM [0..1000] bounding-box formats.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.api.schemas import BoundingBox, ExtractedField, InvoiceSchema, LineItem, OCRToken


class SpatialLayoutExtractor:
    """Production 2D spatial proximity and layout-aware document extractor."""

    # Semantic anchor keywords with strict word boundaries
    KEYWORD_PATTERNS = {
        "invoice_number": [
            r"\binvoice\s*#",
            r"\binvoice\s*no\.?",
            r"\binvoice\s*id\b",
            r"\binv\s*#",
            r"\binv\s*no\.?",
            r"\breceipt\s*#",
            r"\bstatement\s*#",
            r"\bbill\s*#",
            r"\binvoice:",
        ],
        "invoice_date": [
            r"\binvoice\s*date\b",
            r"\bissue\s*date\b",
            r"\bbill\s*date\b",
            r"\bstatement\s*date\b",
            r"\bdate\s*of\s*issue\b",
            r"\bdate:",
            r"\bdate\b",
        ],
        "subtotal": [
            r"\bsub-?total\b",
            r"\bsub\s*total\b",
            r"\bnet\s*amount\b",
            r"\bsubtotal:",
        ],
        "tax": [
            r"\bsales\s*tax\b",
            r"\bestimated\s*tax\b",
            r"\btax\s*\(sales\)",
            r"\btax\s*total\b",
            r"\bvat\b",
            r"\bgst\b",
            r"\btax:",
            r"\btax\b",
        ],
        "total": [
            r"\bgrand\s*total\b",
            r"\btotal\s*payable\b",
            r"\btotal\s*due\b",
            r"\btotal\s*amount\b",
            r"\bamount\s*due\b",
            r"\bbalance\s*due\b",
            r"\btotal:",
            r"\btotal\b",
        ],
    }

    # Table column header keywords
    TABLE_HEADER_PATTERNS = {
        "description": [r"\bdescription\b", r"\bitem\b", r"\bservices\b", r"\bproducts\b", r"item\s*description", r"\bdetails\b"],
        "quantity": [r"\bqty\b", r"\bquantity\b", r"\bcount\b", r"\bunits\b"],
        "unit_price": [r"unit\s*price", r"\bprice\b", r"\brate\b", r"unit\s*cost"],
        "amount": [r"\bline\s*total\b", r"\bext\s*price\b", r"\bamount\b", r"total\s*amount", r"\btotal\b"],
    }

    DATE_REGEX = re.compile(
        r"(?:\d{4}[-/]\d{2}[-/]\d{2})|(?:\d{2}[-/]\d{2}[-/]\d{4})|(?:[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
    )
    AMOUNT_REGEX = re.compile(r"^\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2})?)$")

    def __init__(
        self,
        line_overlap_threshold: float = 0.45,
        max_h_gap_ratio: float = 0.45,
        max_v_gap_ratio: float = 0.12,
    ):
        self.line_overlap_threshold = line_overlap_threshold
        self.max_h_gap_ratio = max_h_gap_ratio
        self.max_v_gap_ratio = max_v_gap_ratio

    def extract(
        self,
        tokens: List[OCRToken],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> InvoiceSchema:
        """Extracts structured invoice fields and line-items from tokens using 2D spatial layout."""
        if not tokens:
            return InvoiceSchema()

        max_x = max(t.bbox[2] for t in tokens)
        max_y = max(t.bbox[3] for t in tokens)
        doc_w = max(image_width or max_x, 100)
        doc_h = max(image_height or max_y, 100)

        # 1. Structure tokens into reading-order lines
        lines = self._cluster_into_lines(tokens)

        # 2. Extract Vendor Name
        vendor_field = self._extract_vendor(lines, doc_w, doc_h)

        # 3. Detect Table Bounds to prevent table headers from acting as summary anchors
        header_line_idx, table_bottom_y = self._detect_table_bounds(lines, doc_h)

        # 4. Extract Key-Value Pairs via 2D spatial neighbor search
        kv_fields = self._extract_key_values(lines, doc_w, doc_h, header_line_idx)

        # 5. Extract Tabular Line Items
        line_items = self._extract_line_items(lines, doc_w, doc_h, header_line_idx, table_bottom_y)

        return InvoiceSchema(
            vendor_name=vendor_field,
            invoice_number=kv_fields.get("invoice_number", ExtractedField(value=None, confidence=0.0)),
            invoice_date=kv_fields.get("invoice_date", ExtractedField(value=None, confidence=0.0)),
            subtotal=kv_fields.get("subtotal", ExtractedField(value=None, confidence=0.0)),
            tax=kv_fields.get("tax", ExtractedField(value=None, confidence=0.0)),
            total=kv_fields.get("total", ExtractedField(value=None, confidence=0.0)),
            line_items=line_items,
            currency="USD",
        )

    def _cluster_into_lines(self, tokens: List[OCRToken]) -> List[List[OCRToken]]:
        """Groups tokens into horizontal text lines based on vertical overlap and y-centers."""
        if not tokens:
            return []

        sorted_tokens = sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0]))
        lines: List[List[OCRToken]] = []

        for token in sorted_tokens:
            placed = False
            token_box = token.box
            token_h = max(1, token_box.height)
            token_cy = token_box.center_y

            for line in lines:
                line_y1 = min(t.bbox[1] for t in line)
                line_y2 = max(t.bbox[3] for t in line)
                line_h = max(1, line_y2 - line_y1)
                line_cy = (line_y1 + line_y2) / 2.0

                overlap_y1 = max(token.bbox[1], line_y1)
                overlap_y2 = min(token.bbox[3], line_y2)
                overlap = max(0, overlap_y2 - overlap_y1)
                overlap_ratio = overlap / min(token_h, line_h)

                if overlap_ratio >= self.line_overlap_threshold or abs(token_cy - line_cy) <= (min(token_h, line_h) * 0.45):
                    line.append(token)
                    placed = True
                    break

            if not placed:
                lines.append([token])

        for line in lines:
            line.sort(key=lambda t: t.bbox[0])

        lines.sort(key=lambda line: min(t.bbox[1] for t in line))
        return lines

    def _detect_table_bounds(self, lines: List[List[OCRToken]], doc_h: int) -> Tuple[int, int]:
        """Detects the line index of table headers and the bottom y-boundary of items."""
        header_idx = -1
        for idx, line in enumerate(lines):
            detected_cols: Dict[str, List[int]] = {}
            for token in line:
                t_lower = token.text.lower().strip(":")
                for col_name, patterns in self.TABLE_HEADER_PATTERNS.items():
                    if any(re.search(pat, t_lower) for pat in patterns):
                        detected_cols[col_name] = token.bbox
                        break
            if "description" in detected_cols and len(detected_cols) >= 2:
                header_idx = idx
                break

        table_bottom = doc_h
        if header_idx != -1:
            for idx in range(header_idx + 1, len(lines)):
                line = lines[idx]
                line_text = " ".join(t.text for t in line).lower()
                if any(k in line_text for k in ["subtotal", "sub-total", "grand total", "total payable", "sales tax"]):
                    table_bottom = min(t.bbox[1] for t in line) - 5
                    break

        return header_idx, table_bottom

    def _extract_vendor(self, lines: List[List[OCRToken]], doc_w: int, doc_h: int) -> ExtractedField[str]:
        """Extracts vendor name from document header."""
        header_y_limit = doc_h * 0.28
        candidate_lines: List[Tuple[float, List[OCRToken], str]] = []

        ignore_terms = {
            "invoice", "statement", "receipt", "bill", "tax", "date", "number", "due",
            "from", "vendor", "to", "total", "subtotal", "record", "address", "phone", "email",
            "services", "products", "qty", "rate", "amount", "item", "description"
        }
        address_markers = {"suite", "street", "st,", "way,", "parkway", "blvd", "boulevard", "ave", "avenue", "floor"}

        for line in lines:
            line_y1 = min(t.bbox[1] for t in line)
            if line_y1 > header_y_limit:
                continue

            line_text = " ".join(t.text for t in line).strip()
            lower = line_text.lower()

            # Skip address lines with specific markers or US state + 5-digit zip
            if any(marker in lower for marker in address_markers) or re.search(r"\b[A-Z]{2}\b\s+\d{5}", line_text):
                continue

            # Skip table header lines
            header_word_count = sum(1 for w in ["services", "products", "qty", "rate", "amount", "item", "description"] if w in lower)
            if header_word_count >= 2:
                continue

            words = set(re.findall(r"\b[a-zA-Z]+\b", lower))
            if not words or words.issubset(ignore_terms):
                continue

            vendor_tokens = [
                t for t in line
                if not any(w in t.text.lower() for w in ["invoice", "statement", "receipt", "bill", "record", "of"])
            ]
            if not vendor_tokens:
                continue

            clean_text = " ".join(t.text for t in vendor_tokens).strip()
            clean_text = re.sub(r"[:\-#]+$", "", clean_text).strip()

            score = 1.0
            if re.search(r"\b(inc|llc|group|solutions|logistics|roastery|cafe|supplies|corp|labs|equipment)\b", clean_text, re.IGNORECASE):
                score += 5.0
            # Earlier lines in header get higher score
            score += max(0.0, 2.0 * (1.0 - (line_y1 / header_y_limit)))

            candidate_lines.append((score, vendor_tokens, clean_text))

        if not candidate_lines:
            return ExtractedField(value=None, confidence=0.0)

        candidate_lines.sort(key=lambda c: c[0], reverse=True)
        best_score, best_tokens, best_text = candidate_lines[0]

        bbox = [
            min(t.bbox[0] for t in best_tokens),
            min(t.bbox[1] for t in best_tokens),
            max(t.bbox[2] for t in best_tokens),
            max(t.bbox[3] for t in best_tokens),
        ]
        avg_conf = sum(t.confidence for t in best_tokens) / len(best_tokens)
        final_conf = min(1.0, round(avg_conf, 3))

        return ExtractedField(value=best_text, confidence=final_conf, bbox=bbox, raw_text=best_text)

    def _extract_key_values(
        self,
        lines: List[List[OCRToken]],
        doc_w: int,
        doc_h: int,
        table_header_idx: int = -1,
    ) -> Dict[str, ExtractedField]:
        """Extracts invoice metadata and totals using 2D spatial candidate search."""
        results: Dict[str, ExtractedField] = {}

        for field_name, patterns in self.KEYWORD_PATTERNS.items():
            candidates: List[Tuple[float, ExtractedField]] = []

            for line_idx, line in enumerate(lines):
                # Disallow table header row from matching summary totals/subtotals
                if field_name in ["subtotal", "tax", "total"]:
                    if line_idx == table_header_idx:
                        continue
                    line_lower = " ".join(t.text for t in line).lower()
                    if any(th in line_lower for th in ["description", "qty", "unit price", "rate", "services"]):
                        continue

                line_text = " ".join(t.text for t in line)

                for pattern in patterns:
                    regex = re.compile(pattern, re.IGNORECASE)
                    match = regex.search(line_text)
                    if not match:
                        continue

                    key_start_char = match.start()
                    key_end_char = match.end()

                    key_tokens: List[OCRToken] = []
                    val_tokens_same_line: List[OCRToken] = []

                    curr_pos = 0
                    for token in line:
                        tok_len = len(token.text)
                        tok_start = curr_pos
                        tok_end = tok_start + tok_len

                        if tok_start < key_end_char and tok_end > key_start_char:
                            key_tokens.append(token)
                        elif tok_start >= key_end_char:
                            val_tokens_same_line.append(token)

                        curr_pos = tok_end + 1

                    if not key_tokens:
                        continue

                    # If value was attached directly to key token (e.g. #APX-81482 or Invoice#123)
                    if field_name == "invoice_number":
                        last_key_text = key_tokens[-1].text.lstrip("#: ")
                        if len(last_key_text) >= 3 and not last_key_text.lower() in ["invoice", "no", "receipt", "bill", "id", "#"]:
                            val_tokens_same_line = [
                                OCRToken(text=last_key_text, bbox=key_tokens[-1].bbox, confidence=key_tokens[-1].confidence)
                            ] + val_tokens_same_line

                    key_box = [
                        min(t.bbox[0] for t in key_tokens),
                        min(t.bbox[1] for t in key_tokens),
                        max(t.bbox[2] for t in key_tokens),
                        max(t.bbox[3] for t in key_tokens),
                    ]
                    key_conf = sum(t.confidence for t in key_tokens) / len(key_tokens)

                    # Direction A: Rightward candidates on same line
                    if val_tokens_same_line:
                        parsed = self._parse_candidate_value(
                            field_name, val_tokens_same_line, key_box, doc_w, doc_h, direction="horizontal"
                        )
                        if parsed:
                            score, field_obj = parsed
                            # Bonus for summary section
                            bonus = 1.2 if (field_name in ["subtotal", "tax", "total"] and line_idx > table_header_idx) else 1.0
                            candidates.append((score * key_conf * bonus, field_obj))

                    # Direction B: Downward candidate on next line
                    if line_idx + 1 < len(lines):
                        next_line = lines[line_idx + 1]
                        downward_tokens = [
                            t for t in next_line
                            if self._is_vertically_aligned(key_box, t.bbox, h_tolerance=doc_w * 0.15)
                        ]
                        if downward_tokens:
                            parsed = self._parse_candidate_value(
                                field_name, downward_tokens, key_box, doc_w, doc_h, direction="vertical"
                            )
                            if parsed:
                                score, field_obj = parsed
                                candidates.append((score * key_conf * 0.9, field_obj))

            if candidates:
                candidates.sort(key=lambda c: c[0], reverse=True)
                results[field_name] = candidates[0][1]

        return results

    def _parse_candidate_value(
        self,
        field_name: str,
        tokens: List[OCRToken],
        key_box: List[int],
        doc_w: int,
        doc_h: int,
        direction: str,
    ) -> Optional[Tuple[float, ExtractedField]]:
        """Parses and scores candidate tokens for a specific semantic field."""
        if not tokens:
            return None

        full_raw_text = " ".join(t.text for t in tokens).strip()
        val_box = [
            min(t.bbox[0] for t in tokens),
            min(t.bbox[1] for t in tokens),
            max(t.bbox[2] for t in tokens),
            max(t.bbox[3] for t in tokens),
        ]
        avg_tok_conf = sum(t.confidence for t in tokens) / len(tokens)

        if direction == "horizontal":
            h_dist = max(0, val_box[0] - key_box[2])
            dist_score = math.exp(-h_dist / (doc_w * 1.5))
        else:
            v_dist = max(0, val_box[1] - key_box[3])
            dist_score = math.exp(-v_dist / (doc_h * 0.4))

        if field_name in ["subtotal", "tax", "total"]:
            best_num = None
            best_token = None

            for t in reversed(tokens):
                cleaned = t.text.replace("$", "").replace(",", "").strip()
                match = self.AMOUNT_REGEX.match(cleaned)
                if match:
                    try:
                        best_num = float(match.group(1).replace(",", ""))
                        best_token = t
                        break
                    except ValueError:
                        continue

            if best_num is None:
                match = re.search(r"\$?\s*([0-9]+(?:\.[0-9]{2})?)", full_raw_text)
                if match:
                    try:
                        best_num = float(match.group(1))
                    except ValueError:
                        pass

            if best_num is not None:
                final_conf = round(avg_tok_conf * dist_score, 3)
                field_obj = ExtractedField[float](
                    value=best_num,
                    confidence=min(1.0, max(0.1, final_conf)),
                    bbox=val_box if best_token is None else best_token.bbox,
                    raw_text=full_raw_text,
                )
                return (dist_score * 1.5, field_obj)

        elif field_name == "invoice_date":
            date_match = self.DATE_REGEX.search(full_raw_text)
            if date_match:
                date_str = date_match.group(0).strip()
                final_conf = round(avg_tok_conf * dist_score, 3)
                field_obj = ExtractedField[str](
                    value=date_str,
                    confidence=min(1.0, max(0.1, final_conf)),
                    bbox=val_box,
                    raw_text=full_raw_text,
                )
                return (dist_score * 1.5, field_obj)

        elif field_name == "invoice_number":
            cleaned_text = re.sub(r"^[:\-#\s]+", "", full_raw_text).strip()
            candidate_tokens = [
                t.text.strip(":#")
                for t in tokens
                if len(t.text.strip(":#")) >= 3 and not t.text.lower() in ["invoice", "no", "number", "id", "#", "item", "date"]
            ]
            if candidate_tokens:
                inv_id = candidate_tokens[0]
                final_conf = round(avg_tok_conf * dist_score, 3)
                field_obj = ExtractedField[str](
                    value=inv_id,
                    confidence=min(1.0, max(0.1, final_conf)),
                    bbox=val_box,
                    raw_text=cleaned_text,
                )
                return (dist_score * 1.5, field_obj)

        return None

    def _is_vertically_aligned(self, box_a: List[int], box_b: List[int], h_tolerance: float) -> bool:
        """Checks if two boxes are aligned in the same vertical column."""
        center_a = (box_a[0] + box_a[2]) / 2.0
        center_b = (box_b[0] + box_b[2]) / 2.0
        return abs(center_a - center_b) <= h_tolerance

    def _extract_line_items(
        self,
        lines: List[List[OCRToken]],
        doc_w: int,
        doc_h: int,
        header_line_idx: int,
        table_bottom_y: int,
    ) -> List[LineItem]:
        """Extracts tabular line items using column boundary discovery and row clustering."""
        col_anchors: Dict[str, List[int]] = {}

        if header_line_idx != -1:
            for token in lines[header_line_idx]:
                t_lower = token.text.lower().strip(":")
                for col_name, patterns in self.TABLE_HEADER_PATTERNS.items():
                    if any(re.search(pat, t_lower) for pat in patterns):
                        col_anchors[col_name] = token.bbox
                        break

        if header_line_idx == -1:
            return self._extract_line_items_heuristic(lines, doc_w, doc_h)

        header_bottom = max(t.bbox[3] for t in lines[header_line_idx])

        line_items: List[LineItem] = []
        for idx in range(header_line_idx + 1, len(lines)):
            line = lines[idx]
            line_top = min(t.bbox[1] for t in line)

            if line_top >= table_bottom_y:
                break
            if line_top < header_bottom:
                continue

            item = self._parse_table_row(line, col_anchors, doc_w)
            if item:
                line_items.append(item)

        return line_items

    def _parse_table_row(
        self,
        line: List[OCRToken],
        col_anchors: Dict[str, List[int]],
        doc_w: int,
    ) -> Optional[LineItem]:
        """Maps tokens in a single row into description, quantity, unit price, and line amount."""
        if not line:
            return None

        non_punct = [t for t in line if re.search(r"[a-zA-Z0-9]", t.text)]
        if not non_punct:
            return None

        if "amount" in col_anchors and ("quantity" in col_anchors or "unit_price" in col_anchors):
            return self._parse_row_with_column_anchors(line, col_anchors)

        return self._parse_row_positional(line)

    def _parse_row_with_column_anchors(
        self,
        line: List[OCRToken],
        col_anchors: Dict[str, List[int]],
    ) -> Optional[LineItem]:
        """Partitions row tokens into column buckets using spatial horizontal boundaries."""
        qty_box = col_anchors.get("quantity")
        price_box = col_anchors.get("unit_price")
        amt_box = col_anchors.get("amount")

        qty_center = (qty_box[0] + qty_box[2]) / 2.0 if qty_box else None
        price_center = (price_box[0] + price_box[2]) / 2.0 if price_box else None
        amt_center = (amt_box[0] + amt_box[2]) / 2.0 if amt_box else None

        # Determine partition boundaries
        mid_price_amt = (price_center + amt_center) / 2.0 if (price_center and amt_center) else (amt_center - 40 if amt_center else 700)
        mid_qty_price = (qty_center + price_center) / 2.0 if (qty_center and price_center) else (price_center - 40 if price_center else 550)
        qty_left = (qty_center - 80) if qty_center else 400

        desc_tokens: List[OCRToken] = []
        qty_tokens: List[OCRToken] = []
        price_tokens: List[OCRToken] = []
        amt_tokens: List[OCRToken] = []

        for token in line:
            cx = (token.bbox[0] + token.bbox[2]) / 2.0
            if cx >= mid_price_amt:
                amt_tokens.append(token)
            elif cx >= mid_qty_price:
                price_tokens.append(token)
            elif cx >= qty_left:
                qty_tokens.append(token)
            else:
                desc_tokens.append(token)

        # Parse Amount
        amt_val = None
        for t in reversed(amt_tokens or line):
            cleaned = t.text.replace("$", "").replace(",", "").strip()
            if self.AMOUNT_REGEX.match(cleaned):
                try:
                    amt_val = float(cleaned)
                    break
                except ValueError:
                    pass

        if amt_val is None:
            return None

        # Parse Quantity: pick token closest to qty_center if available
        qty_val = 1.0
        if qty_tokens:
            if qty_center:
                best_t = min(qty_tokens, key=lambda t: abs(((t.bbox[0] + t.bbox[2]) / 2.0) - qty_center))
            else:
                best_t = qty_tokens[0]
            cleaned = best_t.text.replace(",", "").strip()
            if re.match(r"^\d+(?:\.0+)?$", cleaned):
                try:
                    qty_val = float(cleaned)
                except ValueError:
                    pass

        # Parse Unit Price
        unit_price_val = None
        for t in price_tokens:
            cleaned = t.text.replace("$", "").replace(",", "").strip()
            if self.AMOUNT_REGEX.match(cleaned):
                try:
                    unit_price_val = float(cleaned)
                    break
                except ValueError:
                    pass

        if unit_price_val is None:
            unit_price_val = round(amt_val / qty_val, 2) if qty_val > 0 else amt_val

        desc_text = " ".join(t.text for t in desc_tokens).strip() or "Item"

        row_box = [
            min(t.bbox[0] for t in line),
            min(t.bbox[1] for t in line),
            max(t.bbox[2] for t in line),
            max(t.bbox[3] for t in line),
        ]
        avg_conf = sum(t.confidence for t in line) / len(line)
        is_arith_valid = math.isclose(qty_val * unit_price_val, amt_val, abs_tol=0.05)
        final_conf = round(avg_conf * (1.0 if is_arith_valid else 0.85), 3)

        return LineItem(
            description=desc_text,
            quantity=qty_val,
            unit_price=unit_price_val,
            amount=amt_val,
            confidence=final_conf,
            bbox=row_box,
        )

    def _parse_row_positional(self, line: List[OCRToken]) -> Optional[LineItem]:
        """Positional fallback when column boundaries cannot be established."""
        numbers_with_indices = []
        for idx, t in enumerate(line):
            cleaned = t.text.replace("$", "").replace(",", "").strip()
            if self.AMOUNT_REGEX.match(cleaned):
                try:
                    val = float(cleaned)
                    numbers_with_indices.append((idx, val, t))
                except ValueError:
                    pass

        if not numbers_with_indices:
            return None

        amount_idx, amount_val, _ = numbers_with_indices[-1]
        qty_val = 1.0
        unit_price_val = amount_val

        if len(numbers_with_indices) >= 3:
            q_idx, q_cand, _ = numbers_with_indices[-3]
            p_idx, p_cand, _ = numbers_with_indices[-2]
            if q_cand > 0:
                qty_val = q_cand
                unit_price_val = p_cand
        elif len(numbers_with_indices) == 2:
            cand_idx, cand_val, _ = numbers_with_indices[0]
            if cand_val.is_integer() and 1 <= cand_val <= 100:
                qty_val = cand_val
                unit_price_val = round(amount_val / qty_val, 2)
            else:
                unit_price_val = cand_val
                qty_val = round(amount_val / unit_price_val, 2) if unit_price_val > 0 else 1.0

        first_num_idx = numbers_with_indices[0][0]
        desc_tokens = line[:first_num_idx]
        if not desc_tokens:
            desc_tokens = [t for idx, t in enumerate(line) if idx not in [n[0] for n in numbers_with_indices]]

        description = " ".join(t.text for t in desc_tokens).strip() or "Item"

        row_box = [
            min(t.bbox[0] for t in line),
            min(t.bbox[1] for t in line),
            max(t.bbox[2] for t in line),
            max(t.bbox[3] for t in line),
        ]
        avg_conf = sum(t.confidence for t in line) / len(line)
        is_arith_valid = math.isclose(qty_val * unit_price_val, amount_val, abs_tol=0.05)
        final_conf = round(avg_conf * (1.0 if is_arith_valid else 0.85), 3)

        return LineItem(
            description=description,
            quantity=qty_val,
            unit_price=unit_price_val,
            amount=amount_val,
            confidence=final_conf,
            bbox=row_box,
        )

    def _extract_line_items_heuristic(
        self,
        lines: List[List[OCRToken]],
        doc_w: int,
        doc_h: int
    ) -> List[LineItem]:
        """Heuristic line-item extraction when no explicit table headers exist."""
        items: List[LineItem] = []
        for line in lines:
            line_y = min(t.bbox[1] for t in line)
            if line_y < doc_h * 0.15 or line_y > doc_h * 0.80:
                continue

            line_text = " ".join(t.text for t in line).lower()
            if any(k in line_text for k in ["total", "subtotal", "tax", "invoice", "date"]):
                continue

            parsed = self._parse_row_positional(line)
            if parsed and parsed.description != "Item":
                items.append(parsed)

        return items
