from __future__ import annotations

import os
import time
import gc
from typing import Dict, Optional

from backend.config import (
    ANTI_OVERFITTING_FEATURES,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    CONSECUTIVE_PREDICTION_LIMIT,
    MODEL_NAME,
    NORMAL_CLASS_INDEX,
    NORMAL_MIN_CONFIDENCE_FOR_HEALTHY,
    NUM_CLASSES,
    PREPROCESSING_STEPS,
)
from backend.torch_bootstrap import prepare_torch_environment

# --- Global State ---
_model = None
_device = None
_grad_cam = None
_weights_loaded = False
_load_error: Optional[str] = None
_last_predictions = [] # FIFO for anomaly detection

prepare_torch_environment()

try:
    import torch
    import torch.nn.functional as F

    from backend.models.model import build_model
    from backend.services.model_validation import (
        detect_prediction_collapse,
        unwrap_checkpoint_state_dict,
        validate_checkpoint_head_shape,
        validate_class_names,
        validate_model_prediction_diversity,
        validate_model_output,
    )
    from backend.services.vision import (
        EnsembleGradCAM,
        ndarray_to_base64,
        overlay_heatmap,
        preprocess_image,
    )

    HAS_PYTORCH = True
except Exception as exc:  # pragma: no cover
    torch = None
    F = None
    HAS_PYTORCH = False
    _load_error = str(exc)

def load_model(weights_path: Optional[str] = None, device_str: str = "cpu"):
    """
    STRICT Model Loader:
    - Enforces 100% key matching (strict=True)
    - Verifies backbone health on startup
    """
    global _model, _device, _grad_cam, _weights_loaded, _load_error

    if not HAS_PYTORCH:
        _load_error = _load_error or "PyTorch activation failed."
        return None

    try:
        gc.collect()
        _device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
        validate_class_names(CLASS_NAMES, NUM_CLASSES)
        model = build_model(pretrained=False, num_classes=NUM_CLASSES) 
        gc.collect()
        
        if weights_path and os.path.isfile(weights_path):
            payload_raw = torch.load(weights_path, map_location="cpu")
            payload = unwrap_checkpoint_state_dict(payload_raw)
            validate_checkpoint_head_shape(payload, NUM_CLASSES)
            
            model.load_state_dict(payload, strict=True)
            _weights_loaded = True
            
            del payload_raw
            del payload
            gc.collect()
            print(f"EyeNet Integrity Check: 100% Match ({weights_path})")
        else:
            _load_error = f"Weights missing at: {weights_path}"
            print(f"Warning: {_load_error}. Running in uninitialized demo mode.")

        model = model.to(_device)
        model.eval()
        _model = model
        
        gc.collect()
        _grad_cam = EnsembleGradCAM(_model)

        _run_startup_self_test()
        validate_model_prediction_diversity(_model, _device, NUM_CLASSES)
        gc.collect()
        
    except Exception as e:
        _load_error = f"Integrity Failure: {str(e)}"
        print(f"CRITICAL LOAD FAILURE: {str(e)}")
        _model = None
        _weights_loaded = False
    
    return _model

def _run_startup_self_test():
    """Verify that the model isn't returning constant noise or collapsed values."""
    if not _model: return
    dummy = torch.randn(1, 3, 256, 256).to(_device)
    with torch.no_grad():
        out = _model(dummy)
        # Check for NaN or Inf
        validate_model_output(out, NUM_CLASSES)
        # Check for zero variance (dead decision head)
        if out.std() < 1e-4:
             print("Warning: Model output has extremely low variance. Checking weights...")

def get_model_status() -> Dict[str, object]:
    return {
        "model": MODEL_NAME,
        "model_loaded": _model is not None and HAS_PYTORCH,
        "weights_loaded": _weights_loaded,
        "classes": CLASS_NAMES,
        "error": _load_error,
    }


def get_metadata() -> Dict[str, object]:
    status = get_model_status()
    return {
        "model": MODEL_NAME,
        "classes": CLASS_NAMES,
        "preprocessing_steps": PREPROCESSING_STEPS,
        "anti_overfitting_features": ANTI_OVERFITTING_FEATURES,
        "weights_loaded": status["weights_loaded"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "normal_min_confidence_for_healthy": NORMAL_MIN_CONFIDENCE_FOR_HEALTHY,
        "normal_class_index": NORMAL_CLASS_INDEX,
        "gradcam_mode": "ensemble_average",
    }


def _screening_class_index(probs: "torch.Tensor") -> int:
    """Prefer a disease label when 'Normal' is predicted with low confidence."""
    if torch is None:
        return int(probs.argmax().item())
    p = probs.view(-1)
    top = int(p.argmax().item())
    if top == NORMAL_CLASS_INDEX and float(p[NORMAL_CLASS_INDEX].item()) < NORMAL_MIN_CONFIDENCE_FOR_HEALTHY:
        return int(torch.argmax(p[:NORMAL_CLASS_INDEX]).item())
    return top

def predict(image_bytes: bytes, filename: str = "") -> Dict[str, object]:
    """
    STRICT Inference Pipeline:
    - Preprocessing sync (ResNet + EffNet + DenseNet inputs)
    - TTA (Test Time Augmentation) integration
    - Diagnostic logging for anomaly detection
    """
    if not _model:
        raise RuntimeError(f"Prediction inhibited: {_load_error or 'Model not initialized'}")

    start_now = time.perf_counter()
    tensor, art = preprocess_image(image_bytes)
    tensor = tensor.to(_device)

    # 1. Inference with Multi-Sample Averaging (Standard TTA)
    with torch.no_grad():
        logits = _model(tensor)
        flipped = _model(torch.flip(tensor, [3]))
        validate_model_output(logits, NUM_CLASSES)
        validate_model_output(flipped, NUM_CLASSES)
        avg_logits = (logits + flipped) / 2.0
        probs = F.softmax(avg_logits, dim=1).squeeze(0)

    # 2. Extract detailed diagnostics (conservative screening: weak Normal → best disease)
    cls_idx = _screening_class_index(probs)
    confidence = float(probs[cls_idx].item())
    
    # 3. Anomaly & Bias Detection
    _last_predictions.append(cls_idx)
    if len(_last_predictions) > max(10, CONSECUTIVE_PREDICTION_LIMIT):
        _last_predictions.pop(0)
    
    # Detection for single-class mode collapse
    is_anomaly = detect_prediction_collapse(_last_predictions, CONSECUTIVE_PREDICTION_LIMIT)
    if is_anomaly:
        print("Warning: consecutive predictions collapsed to one class.")

    output_mapping = { name: round(float(probs[i].item()), 4) for i, name in enumerate(CLASS_NAMES) }
    top_k = sorted(output_mapping.items(), key=lambda x: x[1], reverse=True)

    # 4. Explainability (Grad-CAM)
    with torch.enable_grad():
        grad_tensor = tensor.clone().detach().requires_grad_(True)
        cam = _grad_cam(grad_tensor, cls_idx)

    # Base64 assets
    overlay_rgb = overlay_heatmap(art["processed_rgb"], cam)
    
    return {
        "model_name": MODEL_NAME,
        "weights_loaded": _weights_loaded,
        "predicted_class": CLASS_NAMES[cls_idx],
        "class_index": cls_idx,
        "confidence": round(confidence, 4),
        "probabilities": output_mapping,
        "top_predictions": [{"class_name": k, "probability": v} for k, v in top_k],
        "quality_metrics": art["quality_metrics"],
        "quality_verdict": art["quality_verdict"],
        "quality_advisory": art["quality_advisory"],
        "preprocess_preview_base64": ndarray_to_base64(art["processed_rgb"]),
        "gradcam_base64": ndarray_to_base64(overlay_heatmap(art["processed_rgb"], cam, alpha=0.9)),
        "overlay_base64": ndarray_to_base64(overlay_rgb),
        "inference_time_ms": round((time.perf_counter() - start_now) * 1000, 2)
    }
    
