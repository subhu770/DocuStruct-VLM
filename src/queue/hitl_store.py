"""Thread-Safe In-Memory and JSON-Persistent Human-in-the-Loop (HITL) Review Queue.

Manages low-confidence or arithmetic-mismatch documents pending human audit,
supporting queuing, item retrieval, filtering by review status, and resolution.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from src.api.schemas import (
    InvoiceSchema,
    ReviewQueueItem,
    ReviewStatus,
    ValidationResult,
)


class HITLReviewStore:
    """Thread-safe Human-in-the-Loop review queue repository."""

    def __init__(self, persistence_path: Optional[str] = None):
        self._lock = threading.Lock()
        self._items: Dict[str, ReviewQueueItem] = {}
        self.persistence_path = persistence_path
        if self.persistence_path and os.path.exists(self.persistence_path):
            self._load_from_disk()

    def enqueue(
        self,
        doc_id: str,
        extracted_data: InvoiceSchema,
        validation: ValidationResult,
        overall_confidence: float,
    ) -> ReviewQueueItem:
        """Enqueues an unverified or mismatched document for human review."""
        with self._lock:
            item = ReviewQueueItem(
                doc_id=doc_id,
                created_at=datetime.utcnow().isoformat(),
                status=ReviewStatus.PENDING,
                extracted_data=extracted_data,
                validation=validation,
                overall_confidence=overall_confidence,
            )
            self._items[doc_id] = item
            self._save_to_disk()
            return item

    def get(self, doc_id: str) -> Optional[ReviewQueueItem]:
        """Retrieves a single review queue item by document ID."""
        with self._lock:
            return self._items.get(doc_id)

    def list_items(
        self,
        status: Optional[ReviewStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ReviewQueueItem]:
        """Lists review queue items, optionally filtered by status with pagination."""
        with self._lock:
            items = list(self._items.values())
            if status is not None:
                items = [it for it in items if it.status == status]
            # Order newest first
            items.sort(key=lambda x: x.created_at, reverse=True)
            return items[offset: offset + limit]

    def resolve(
        self,
        doc_id: str,
        corrected_data: InvoiceSchema,
        reviewer_notes: Optional[str] = None,
    ) -> ReviewQueueItem:
        """Resolves a pending review item with human-verified/corrected data."""
        with self._lock:
            if doc_id not in self._items:
                raise KeyError(f"Document ID '{doc_id}' not found in HITL review queue")

            item = self._items[doc_id]
            item.status = ReviewStatus.RESOLVED
            item.resolved_data = corrected_data
            item.reviewer_notes = reviewer_notes
            self._save_to_disk()
            return item

    def reject(
        self,
        doc_id: str,
        reviewer_notes: Optional[str] = None,
    ) -> ReviewQueueItem:
        """Rejects a document as unprocessable/invalid."""
        with self._lock:
            if doc_id not in self._items:
                raise KeyError(f"Document ID '{doc_id}' not found in HITL review queue")

            item = self._items[doc_id]
            item.status = ReviewStatus.REJECTED
            item.reviewer_notes = reviewer_notes
            self._save_to_disk()
            return item

    def stats(self) -> Dict[str, Any]:
        """Returns review queue health metrics."""
        with self._lock:
            total = len(self._items)
            pending = sum(1 for it in self._items.values() if it.status == ReviewStatus.PENDING)
            resolved = sum(1 for it in self._items.values() if it.status == ReviewStatus.RESOLVED)
            rejected = sum(1 for it in self._items.values() if it.status == ReviewStatus.REJECTED)
            return {
                "total": total,
                "pending": pending,
                "resolved": resolved,
                "rejected": rejected,
            }

    def clear(self) -> None:
        """Clears all review items (used in test isolation)."""
        with self._lock:
            self._items.clear()
            if self.persistence_path and os.path.exists(self.persistence_path):
                try:
                    os.remove(self.persistence_path)
                except OSError:
                    pass

    def _save_to_disk(self) -> None:
        """Atomic write to disk if persistence_path is configured."""
        if not self.persistence_path:
            return
        tmp_path = f"{self.persistence_path}.tmp"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.persistence_path)), exist_ok=True)
            data = {k: v.model_dump() for k, v in self._items.items()}
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.persistence_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _load_from_disk(self) -> None:
        """Loads serialized queue items from disk."""
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                self._items[k] = ReviewQueueItem.model_validate(v)
        except Exception:
            self._items = {}


# Global default in-memory store
hitl_queue_store = HITLReviewStore()
