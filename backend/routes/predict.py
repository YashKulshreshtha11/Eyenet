"""
FastAPI route handlers for prediction, health, and history.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse
import os
import tempfile
from bson import ObjectId

from backend.models.schema import (
    HealthResponse,
    HistoryResponse,
    PredictionResponse,
)
from backend.services import inference as inf_svc
from backend.services.report_gen import generate_medical_report
from database.db import get_db, get_history, log_event, save_prediction, get_record_by_id, delete_record

logger = logging.getLogger("eyenet.routes")

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe — also reports DB availability."""
    db_alive = get_db() is not None
    log_event("health_check", "Health endpoint called", {"db_alive": db_alive})
    return HealthResponse(status="ok", db_alive=db_alive)


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Retinal fundus image")):
    """
    Classify an uploaded retinal image and return:
    - Disease prediction + probabilities
    - Grad-CAM heatmap (base64 PNG)
    - Heatmap overlay on original image (base64 PNG)
    """
    # Validate MIME
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Use one of {allowed}",
        )

    image_bytes = await file.read()

    try:
        result = inf_svc.predict(image_bytes, filename=file.filename)
    except ValueError as exc:
        log_event("prediction_error", str(exc), {"filename": file.filename})
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        import traceback
        full_tb = traceback.format_exc()
        logger.exception("Unexpected inference error")
        log_event("prediction_error", str(exc) + "\\n---\\n" + full_tb, {"filename": file.filename})
        raise HTTPException(status_code=500, detail="Internal inference error.")

    # Persist to MongoDB (best-effort)
    record_id = save_prediction(
        image_name=file.filename or "unknown",
        predicted_class=result["predicted_class"],
        probabilities=result["probabilities"],
        inference_time_ms=result["inference_time_ms"],
    )
    log_event(
        "prediction_success",
        f"Predicted: {result['predicted_class']}",
        {"file": file.filename, "class_idx": result["class_index"]},
    )

    return PredictionResponse(**result, record_id=record_id)


# ─────────────────────────────────────────────────────────────────────────────
# GET /history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/history", response_model=HistoryResponse, tags=["History"])
async def get_prediction_history(limit: int = 20, skip: int = 0):
    """Return a paginated slice of prediction records stored in MongoDB."""
    records_raw = get_history(limit=limit, skip=skip)

    # Normalise timestamps and ensure ID is present
    records = []
    for r in records_raw:
        r["timestamp"] = str(r.get("timestamp", ""))
        # For schema compatibility if using different models
        if "_id" in r and "id" not in r:
            r["id"] = r["_id"]
        records.append(r)

    return HistoryResponse(records=records, total=len(records))


@router.delete("/history/{record_id}", tags=["History"])
async def delete_prediction_record(record_id: str):
    """Permanently remove a diagnostic record from the database."""
    success = delete_record(record_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record {record_id} not found."
        )
    return {"status": "success", "message": "Record deleted"}


@router.delete("/record/{record_id}", tags=["History"])
async def delete_record_alias(record_id: str):
    """Alias for delete_prediction_record to support both conventions."""
    return await delete_prediction_record(record_id)


# ─────────────────────────────────────────────────────────────────────────────
# GET /report/{record_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/report/{record_id}", tags=["History"])
async def download_report(record_id: str):
    """Generate and return a medical PDF report for a specific scan."""
    try:
        record = get_record_by_id(record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")
        
        # Ensure timestamp is string for PDF
        record["timestamp"] = str(record.get("timestamp", ""))
        
        # Use a temporary file for the PDF
        fd, path = tempfile.mkstemp(suffix=".pdf")
        try:
            generate_medical_report(record, path)
            return FileResponse(
                path, 
                media_type="application/pdf", 
                filename=f"EyeNet_Report_{record_id}.pdf"
            )
        finally:
            os.close(fd)
            # Note: We can't delete the file immediately as FileResponse needs it.
            # In a real app, use BackgroundTasks to cleanup.
    except Exception as e:
        logger.exception("Failed to generate report")
        raise HTTPException(status_code=500, detail=str(e))
