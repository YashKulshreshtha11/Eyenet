"""
Pydantic request / response schemas for EyeNet API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Full prediction result returned by POST /predict."""
    predicted_class:  str                   = Field(..., description="Disease name")
    class_index:      int                   = Field(..., description="0-based class index")
    probabilities:    Dict[str, float]      = Field(..., description="Per-class probabilities")
    gradcam_base64:   str                   = Field(..., description="Grad-CAM heatmap PNG (base64)")
    overlay_base64:   str                   = Field(..., description="Heatmap-on-image overlay PNG (base64)")
    inference_time_ms: float                = Field(..., description="Inference latency in milliseconds")
    record_id:        Optional[str]         = Field(None, description="MongoDB record id (if DB active)")


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status:   str  = "ok"
    model:    str  = "EyeNet Ensemble (ResNet50 + EfficientNetB0 + DenseNet121)"
    db_alive: bool = False


class HistoryRecord(BaseModel):
    """Single entry in prediction history."""
    id:                Optional[str]         = None
    image_name:        str
    timestamp:         str
    predicted_class:   str
    probabilities:     Dict[str, float]
    inference_time_ms: float


class HistoryResponse(BaseModel):
    """Response for GET /history."""
    records: List[HistoryRecord]
    total:   int
