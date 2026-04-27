from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.config import ALLOWED_IMAGE_TYPES
from backend.models.api_schema import (
    HealthResponse,
    HistoryResponse,
    ModelMetadataResponse,
    PredictionResponse,
)
from backend.services import prediction_service as predict_svc
from backend.services.report_gen import generate_medical_report
from database.db import delete_all_records, delete_record, get_db, get_history, get_history_count, get_record_by_id, log_event, save_prediction
import tempfile
import os
import datetime
from fastapi.responses import FileResponse, Response

logger = logging.getLogger("eyenet.routes")
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    model_status = predict_svc.get_model_status()
    db_alive = get_db() is not None
    return HealthResponse(
        status="ok",
        model=model_status["model"],
        model_loaded=model_status["model_loaded"],
        weights_loaded=model_status["weights_loaded"],
        classes=model_status["classes"],
        db_alive=db_alive,
    )


@router.get("/metadata", response_model=ModelMetadataResponse, tags=["System"])
async def metadata():
    return ModelMetadataResponse(**predict_svc.get_metadata())


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Retinal fundus image"),
    state: str = Form("Delhi", description="User's state for localized doctor referral")
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    image_bytes = await file.read()

    try:
        result = predict_svc.predict(image_bytes, filename=file.filename or "")
    except ValueError as exc:
        log_event("prediction_error", str(exc), {"filename": file.filename})
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        log_event("prediction_unavailable", str(exc), {"filename": file.filename})
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        import traceback
        full_tb = traceback.format_exc()
        logger.exception("Unexpected inference error")
        log_event("prediction_error", str(exc) + "\n---\n" + full_tb, {"filename": file.filename})
        raise HTTPException(status_code=500, detail="Internal inference error.")

    record_id = save_prediction(
        image_name=file.filename or "unknown",
        predicted_class=result["predicted_class"],
        probabilities=result["probabilities"],
        inference_time_ms=result["inference_time_ms"],
        gradcam_base64=result.get("gradcam_base64"),
        overlay_base64=result.get("overlay_base64"),
        state=state,
    )

    log_event(
        "prediction_success",
        f"Predicted {result['predicted_class']}",
        {
            "filename": file.filename,
            "weights_loaded": result["weights_loaded"],
            "confidence": result["confidence"],
        },
    )

    return PredictionResponse(**result, record_id=record_id, state=state)


@router.get("/history", tags=["History"])
async def prediction_history(limit: int = 50, skip: int = 0):
    records_raw = get_history(limit=limit, skip=skip)
    records = []
    for record in records_raw:
        # Ensure ID is string and available as 'id' for frontend
        if "_id" in record:
            record["id"] = str(record["_id"])
        
        ts = record.get("timestamp")
        if isinstance(ts, datetime.datetime):
            record["timestamp"] = ts.isoformat() + "Z"
        else:
            record["timestamp"] = str(ts)
            
        # Strip heavy base64 blobs from the list view — fetch them via /record/{id}
        record.pop("gradcam_base64", None)
        record.pop("overlay_base64", None)
        records.append(record)
    return {"records": records, "total": get_history_count()}


@router.get("/record/{record_id}", tags=["History"])
async def get_record(record_id: str):
    """Return the full record including heatmap base64 fields."""
    record = get_record_by_id(record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record not found: {record_id}",
        )
    ts = record.get("timestamp")
    record["_id"] = str(record["_id"])
    if isinstance(ts, datetime.datetime):
        record["timestamp"] = ts.isoformat() + "Z"
    else:
        record["timestamp"] = str(ts)
    return record


@router.delete("/record/{record_id}", tags=["History"])
async def delete_history_record(record_id: str):
    success = delete_record(record_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record not found or failed to delete: {record_id}",
        )
    return {"status": "success", "message": f"Record {record_id} deleted"}


@router.delete("/history", tags=["History"])
async def clear_all_history():
    success = delete_all_records()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear history collection",
        )
    return {"status": "success", "message": "All history records deleted"}


@router.get("/report/{record_id}", tags=["History"])
async def get_report(record_id: str):
    record = get_record_by_id(record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record not found: {record_id}",
        )

    # Use a temporary file to generate the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name

    try:
        generate_medical_report(record, tmp_path)
        return FileResponse(
            tmp_path,
            media_type="application/pdf",
            filename=f"EyeNet_Report_{record_id}.pdf",
        )
    except Exception as exc:
        logger.exception("Failed to generate report")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate medical report.",
        )
