"""FastAPI Async Application for DocuStruct-VLM.

Provides endpoints for layout-aware document extraction, arithmetic reconciliation,
confidence routing, and Human-in-the-Loop review queue management.
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionStatus,
    HealthResponse,
    InvoiceSchema,
    ResolveReviewRequest,
    ReviewQueueItem,
    ReviewStatus,
)
from src.extraction.spatial_extractor import SpatialLayoutExtractor
from src.extraction.validator import ArithmeticValidator
from src.queue.hitl_store import hitl_queue_store
from src.simulation.invoice_generator import SyntheticInvoiceGenerator
from src.vision.preprocessor import DocumentPreprocessor

app = FastAPI(
    title="DocuStruct-VLM",
    description="Production-Grade Layout-Aware Unstructured Invoice & Receipt Extraction Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine components
preprocessor = DocumentPreprocessor()
spatial_extractor = SpatialLayoutExtractor()
validator = ArithmeticValidator()
simulation_generator = SyntheticInvoiceGenerator()


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Returns engine operational status and review queue backlog size."""
    stats = hitl_queue_store.stats()
    return HealthResponse(
        status="healthy",
        service="DocuStruct-VLM",
        version="1.0.0",
        queue_size=stats["pending"],
    )


@app.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract_invoice(request: ExtractionRequest) -> ExtractionResponse:
    """Extracts layout-aware structured invoice data with arithmetic reconciliation and HITL routing.

    Accepts raw OCR tokens, base64 invoice images, or triggers simulation mode.
    """
    start_time = time.perf_counter()
    doc_id = request.doc_id or f"doc_{uuid.uuid4().hex[:10]}"

    tokens = request.tokens
    image = None
    applied_skew = 0.0

    # 1. Handle Image Decoding if provided
    if request.image_base64:
        try:
            img_bytes = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image from provided base64 data")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 image data: {str(e)}",
            )

    # 2. Simulation Mode Fallback (if requested or if environment flag SIMULATION_MODE=true)
    sim_env = os.getenv("SIMULATION_MODE", "false").lower() in ("true", "1", "yes")
    if request.simulate or (tokens is None and image is None and sim_env):
        synth = simulation_generator.generate_invoice(
            doc_id=doc_id,
            layout_style=request.simulation_layout,
            corruption=request.simulation_corruption,
        )
        tokens = synth.tokens
        image = synth.image

    if not tokens and image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'tokens', 'image_base64', or 'simulate=true' must be provided",
        )

    # 3. Vision Preprocessing (Deskewing and Bounding-Box Affine Transformation)
    if image is not None and request.deskew:
        image, tokens, applied_skew = preprocessor.preprocess(image, tokens=tokens, deskew=True)

    img_w = request.image_width or (image.shape[1] if image is not None else None)
    img_h = request.image_height or (image.shape[0] if image is not None else None)

    # 4. Spatial Layout Extraction
    extracted_data = spatial_extractor.extract(
        tokens=tokens or [],
        image_width=img_w,
        image_height=img_h,
    )

    # 5. Arithmetic Cross-Checking & Confidence Evaluation
    route_status, validation_res, overall_conf = validator.evaluate_routing(
        invoice=extracted_data,
        confidence_threshold=request.confidence_threshold,
    )

    # 6. Route to Human-in-the-Loop Queue if verification checks failed
    review_queue_id = None
    if route_status == ExtractionStatus.NEEDS_REVIEW:
        queue_item = hitl_queue_store.enqueue(
            doc_id=doc_id,
            extracted_data=extracted_data,
            validation=validation_res,
            overall_confidence=overall_conf,
        )
        review_queue_id = queue_item.doc_id

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    return ExtractionResponse(
        doc_id=doc_id,
        status=route_status,
        data=extracted_data,
        validation=validation_res,
        overall_confidence=overall_conf,
        processing_time_ms=elapsed_ms,
        review_queue_id=review_queue_id,
    )


@app.get("/review-queue", response_model=Dict[str, Any], tags=["HITL Review"])
async def list_review_queue(
    review_status: Optional[ReviewStatus] = Query(None, alias="status", description="Filter by review status"),
    limit: int = Query(50, ge=1, le=200, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
) -> Dict[str, Any]:
    """Retrieves items in the Human-in-the-Loop review queue."""
    items = hitl_queue_store.list_items(status=review_status, limit=limit, offset=offset)
    stats = hitl_queue_store.stats()
    return {
        "items": [it.model_dump() for it in items],
        "stats": stats,
        "limit": limit,
        "offset": offset,
    }


@app.get("/review-queue/{doc_id}", response_model=ReviewQueueItem, tags=["HITL Review"])
async def get_review_item(doc_id: str) -> ReviewQueueItem:
    """Retrieves a single review queue record by doc_id."""
    item = hitl_queue_store.get(doc_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document ID '{doc_id}' not found in review queue",
        )
    return item


@app.post("/review-queue/{doc_id}/resolve", response_model=ReviewQueueItem, tags=["HITL Review"])
async def resolve_review_item(doc_id: str, request: ResolveReviewRequest) -> ReviewQueueItem:
    """Resolves a flagged document with human-verified corrections."""
    try:
        resolved_item = hitl_queue_store.resolve(
            doc_id=doc_id,
            corrected_data=request.corrected_data,
            reviewer_notes=request.reviewer_notes,
        )
        return resolved_item
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document ID '{doc_id}' not found in review queue",
        )
