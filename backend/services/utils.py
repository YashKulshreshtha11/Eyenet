"""
Preprocessing utilities and Grad-CAM implementation for EyeNet.
albumentations is NOT imported here — inference uses pure cv2/torchvision only.
"""
import base64
import io
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from backend.config import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L-channel of LAB colour space."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_inference_transform() -> transforms.Compose:
    """
    Pure torchvision pipeline for inference — zero albumentations dependency.
    Matches the training pipeline: resize → CLAHE (done manually above) → normalize.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Production inference preprocessing:
      1. Decode bytes → BGR
      2. Crop to fundus circle (removes black borders)
      3. CLAHE contrast enhancement (matches training)
      4. Unsharp mask sharpening for vessel detail
      5. Normalize and tensorise via torchvision

    Returns
    -------
    tensor   : (1, 3, H, W) float32 ready for the model
    orig_rgb : (H, W, 3) uint8 RGB resized for Grad-CAM overlay
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode the uploaded image.")

    # Step 1 — crop the fundus circle
    img_bgr = _crop_fundus_circle(img_bgr)

    # Step 2 — CLAHE on LAB L-channel
    img_bgr = apply_clahe(img_bgr)

    # Step 3 — unsharp mask (subtle sharpening for vessel details)
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=2)
    img_bgr = cv2.addWeighted(img_bgr, 1.4, blur, -0.4, 0)

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    # Step 4 — torchvision normalize + tensorise (no albumentations)
    transform = get_inference_transform()
    tensor = transform(img_rgb).unsqueeze(0).float()   # (1, 3, 256, 256)

    return tensor, orig_rgb


def _crop_fundus_circle(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detect the fundus circle and return a SQUARE crop to preserve aspect ratio.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Square the crop by taking the larger dimension
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    
    nx = max(0, cx - side // 2)
    ny = max(0, cy - side // 2)
    
    # Boundary check and square adjustment
    side = min(side, img_bgr.shape[1] - nx, img_bgr.shape[0] - ny)
    
    cropped = img_bgr[ny:ny+side, nx:nx+side]
    return cropped if cropped.size > 0 else img_bgr


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for a chosen layer.

    Usage
    -----
        cam_engine = GradCAM(model, target_layer)
        heatmap    = cam_engine(input_tensor, class_idx)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks = [target_layer.register_forward_hook(self._save_activation)]

    def _save_activation(self, _module, _input, output):
        self._activations = output.detach()

        def _save_grad(grad: torch.Tensor) -> None:
            self._gradients = grad.detach()

        if torch.is_tensor(output) and output.requires_grad:
            output.register_hook(_save_grad)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Return a (H, W) float32 heatmap in [0, 1]."""
        self.model.eval()                          # ← critical: disables dropout/BN updating
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        # Global average pooling over spatial dims → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)  # (H, W)

        # ReLU: only keep positive contributions
        cam = F.relu(cam).cpu().numpy()

        # --- DIAGNOSTIC SMOOTHING ---
        # 1. Subtle Low-res smoothing
        cam = cv2.GaussianBlur(cam, (3, 3), 0)

        # 2. Resize to input spatial size with high-quality interpolation
        target_size = (input_tensor.shape[3], input_tensor.shape[2])
        cam = cv2.resize(cam, target_size, interpolation=cv2.INTER_CUBIC)

        # 3. Noise suppression thresholding
        cam[cam < 0.2] = 0

        # 4. Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


# ─────────────────────────────────────────────────────────────────────────────
# Overlay + encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(
    orig_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend the Grad-CAM heatmap onto the original image.
    Uses Turbo colormap + Pixel-wise masking for diagnostic clarity.
    """
    HR_SIZE = 512
    orig_hr = cv2.resize(orig_rgb, (HR_SIZE, HR_SIZE), interpolation=cv2.INTER_CUBIC)
    cam_hr  = cv2.resize(cam, (HR_SIZE, HR_SIZE), interpolation=cv2.INTER_LANCZOS4)

    # 1. Turbo Colormap (Superior for medical diagnostics)
    heatmap_uint8 = np.uint8(255 * cam_hr)
    try:
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, 20) # TURBO
    except:
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 2. Sharpening & Smoothing
    heatmap_rgb = cv2.GaussianBlur(heatmap_rgb, (9, 9), 0)

    # 3. Circular Masking
    mask = np.zeros((HR_SIZE, HR_SIZE), dtype=np.uint8)
    cv2.circle(mask, (HR_SIZE // 2, HR_SIZE // 2), int(HR_SIZE * 0.48), 255, -1)
    mask_f = mask.astype(float) / 255.0

    # 4. Diagnostic Pixel-wise Blending
    cam_hr_masked = cam_hr * mask_f
    
    overlay = orig_hr.astype(float)
    blend_indices = cam_hr_masked > 0.05
    
    overlay[blend_indices] = (
        orig_hr[blend_indices] * (1.0 - alpha * cam_hr_masked[blend_indices, None]) + 
        heatmap_rgb[blend_indices] * (alpha * cam_hr_masked[blend_indices, None])
    )

    return np.uint8(np.clip(overlay, 0, 255))


def ndarray_to_base64(img_rgb: np.ndarray) -> str:
    """Encode a (H, W, 3) uint8 RGB array → base64 PNG string."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")