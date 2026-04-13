"""
Inference pipeline for EyeNet.
Handles model loading, prediction, and Grad-CAM generation.
Forced production mode – Mocking is disabled to focus on the actual model.
"""

import os
import time
from typing import Dict, Optional
import sys
from collections import deque

from backend.config import (
    CLASS_NAMES,
    CONSECUTIVE_PREDICTION_LIMIT,
    NORMAL_CLASS_INDEX,
    NORMAL_MIN_CONFIDENCE_FOR_HEALTHY,
    NUM_CLASSES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Microsoft Store Python 3.11 DLL Fix
# ─────────────────────────────────────────────────────────────────────────────
if sys.platform == "win32" and sys.version_info >= (3, 8):
    # Potential DLL sites
    dll_sites = [
        # Standard site-packages (Program Files / WindowsApps)
        *[os.path.join(p, "torch", "lib") for p in sys.path if os.path.isdir(os.path.join(p, "torch", "lib"))],
        # Microsoft Store "local-packages" in AppData
        os.path.join(os.environ.get("LOCALAPPDATA", ""), 
                     "Packages", "PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0", 
                     "LocalCache", "local-packages", "Python311", "site-packages", "torch", "lib")
    ]
    for lib_path in dll_sites:
        if os.path.exists(lib_path):
            try:
                os.add_dll_directory(lib_path)
            except Exception:
                pass

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
except ImportError:
    HAS_PYTORCH = False


# Module-level singleton ── loaded once at startup
_model = None
_device = None
_grad_cam = None
_last_predictions = deque(maxlen=max(10, CONSECUTIVE_PREDICTION_LIMIT))


def load_model(weights_path: Optional[str] = None, device_str: str = "cpu"):
    """Build and load pre-trained weights for the production PyTorch model."""
    global _model, _device, _grad_cam

    if not HAS_PYTORCH:
        print("[EyeNet] CRITICAL ERROR: PyTorch cannot be loaded (DLL Missing).")
        return None

    _device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    validate_class_names(CLASS_NAMES, NUM_CLASSES)
    model = build_model(pretrained=False, num_classes=NUM_CLASSES)

    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=_device)
        payload = unwrap_checkpoint_state_dict(state)
        validate_checkpoint_head_shape(payload, NUM_CLASSES)
        model.load_state_dict(payload, strict=True)
        print(f"[EyeNet] Loaded production weights from {weights_path} to {_device}")
    else:
        print(f"[EyeNet] WARNING: No weights found at {weights_path}")

    model = model.to(_device)
    model.eval()
    _model = model

    _grad_cam = EnsembleGradCAM(_model)

    with torch.no_grad():
        sanity_logits = _model(torch.randn(1, 3, 256, 256, device=_device))
        validate_model_output(sanity_logits, NUM_CLASSES)
    validate_model_prediction_diversity(_model, _device, NUM_CLASSES)

    return _model


def predict(image_bytes: bytes, filename: str = "") -> Dict:
    """Run full production PyTorch inference pipeline."""
    t0 = time.perf_counter()

    if not HAS_PYTORCH:
        raise RuntimeError(
            "CRITICAL: PyTorch could not be loaded. This is usually due to missing Windows DLLs. "
            "Please install the 'vc_redist.x64.exe' provided in the project root."
        )

    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    tensor, art = preprocess_image(image_bytes)
    tensor = tensor.to(_device)
    proc_rgb = art["processed_rgb"]

    # Inference with TTA (Test Time Augmentation - Horizontal Flip)
    with torch.no_grad():
        logits1 = _model(tensor)
        tensor_flipped = torch.flip(tensor, dims=[3])
        logits2 = _model(tensor_flipped)
        validate_model_output(logits1, NUM_CLASSES)
        validate_model_output(logits2, NUM_CLASSES)
        logits = (logits1 + logits2) / 2.0

    probs = F.softmax(logits, dim=1).squeeze(0)
    class_idx = int(probs.argmax().item())
    if (
        class_idx == NORMAL_CLASS_INDEX
        and float(probs[NORMAL_CLASS_INDEX].item()) < NORMAL_MIN_CONFIDENCE_FOR_HEALTHY
    ):
        class_idx = int(torch.argmax(probs[:NORMAL_CLASS_INDEX]).item())
    class_name = CLASS_NAMES[class_idx]
    _last_predictions.append(class_idx)
    if detect_prediction_collapse(_last_predictions, CONSECUTIVE_PREDICTION_LIMIT):
        print("[EyeNet] Warning: consecutive predictions collapsed to one class.")

    prob_dict = {CLASS_NAMES[i]: round(float(probs[i].item()), 4)
                 for i in range(len(CLASS_NAMES))}

    # Explainability (ensemble Grad-CAM)
    with torch.enable_grad():
        grad_tensor = tensor.clone().detach().requires_grad_(True)
        cam = _grad_cam(grad_tensor, class_idx)

    overlay = overlay_heatmap(proc_rgb, cam)

    import cv2, numpy as np
    heatmap_uint8 = np.uint8(255 * cam)
    heatmap_bgr   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    gradcam_b64 = ndarray_to_base64(heatmap_rgb)
    overlay_b64 = ndarray_to_base64(overlay)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "predicted_class":   class_name,
        "class_index":       class_idx,
        "probabilities":     prob_dict,
        "gradcam_base64":    gradcam_b64,
        "overlay_base64":    overlay_b64,
        "inference_time_ms": round(elapsed_ms, 2),
    }
    
