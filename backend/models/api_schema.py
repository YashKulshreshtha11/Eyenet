from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RankedPrediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    class_name: str = Field(..., description="Class label")
    probability: float = Field(..., description="Probability in [0, 1]")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str = Field(..., description="Loaded ensemble model name")
    weights_loaded: bool = Field(..., description="Whether trained weights were loaded")
    predicted_class: str = Field(..., description="Predicted disease label")
    class_index: int = Field(..., description="0-based class index")
    confidence: float = Field(..., description="Top-class probability")
    probabilities: Dict[str, float] = Field(..., description="Per-class probabilities")
    top_predictions: List[RankedPrediction] = Field(..., description="Top ranked classes")
    quality_metrics: Dict[str, float] = Field(..., description="Brightness, contrast, sharpness")
    quality_verdict: str = Field(..., description="Image quality grade")
    quality_advisory: str = Field(..., description="Human-readable capture note")
    preprocess_preview_base64: str = Field(..., description="Preprocessed retinal image preview")
    gradcam_base64: str = Field(..., description="Grad-CAM heatmap PNG")
    overlay_base64: str = Field(..., description="Grad-CAM overlay PNG")
    inference_time_ms: float = Field(..., description="End-to-end inference latency")
    state: str = Field("Delhi", description="User state for localized doctor referral")
    record_id: Optional[str] = Field(None, description="MongoDB record id when available")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str = "ok"
    model: str
    model_loaded: bool = False
    weights_loaded: bool = False
    classes: List[str] = Field(default_factory=list)
    db_alive: bool = False


class HistoryRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str = Field(..., validation_alias="_id", serialization_alias="id")
    image_name: str
    timestamp: str
    predicted_class: str
    probabilities: Optional[Dict[str, float]] = None
    inference_time_ms: Optional[float] = None
    gradcam_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    state: Optional[str] = "Delhi"


class HistoryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    records: List[HistoryRecord]
    total: int


class ModelMetadataResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model: str
    classes: List[str]
    preprocessing_steps: List[str]
    anti_overfitting_features: List[str]
    weights_loaded: bool
    weights_loaded: bool
