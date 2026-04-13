from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.torch_bootstrap import prepare_torch_environment

prepare_torch_environment()

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from backend.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from backend.services.fundus_ops import decode_image_bytes, preprocess_fundus_bgr, read_image_bgr


def get_base_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def tensorize_rgb_image(image_rgb: np.ndarray) -> torch.Tensor:
    return get_base_transform()(image_rgb).unsqueeze(0).float()


def evaluate_image_quality(image_rgb: np.ndarray) -> Tuple[Dict[str, float], str, str]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    sharpness = float(min(cv2.Laplacian(gray, cv2.CV_64F).var() / 2000.0, 1.0))

    # Heuristic "fundus presence" signal.
    # This is intentionally lightweight and deterministic so it won't impact latency
    # or change preprocessing for valid dataset images.
    _, non_black = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    non_black_ratio = float(non_black.mean() / 255.0)
    contours, _ = cv2.findContours(non_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        peri = float(cv2.arcLength(c, True))
        circularity = float((4.0 * np.pi * area / (peri * peri)) if peri > 1e-8 else 0.0)
        largest_contour_ratio = float(area / max(1.0, float(gray.shape[0] * gray.shape[1])))
    else:
        circularity = 0.0
        largest_contour_ratio = 0.0

    metrics = {
        "brightness": round(brightness * 100.0, 2),
        "contrast": round(contrast * 100.0, 2),
        "sharpness": round(sharpness * 100.0, 2),
        "fundus_coverage": round(non_black_ratio * 100.0, 2),
        "fundus_circularity": round(circularity * 100.0, 2),
    }

    issues = []
    if brightness < 0.20:
        issues.append("image is underexposed")
    elif brightness > 0.85:
        issues.append("image is overexposed")
    if contrast < 0.12:
        issues.append("contrast is low")
    if sharpness < 0.08:
        issues.append("fundus details look blurry")

    # Web / Google images often include screenshots, charts, or non-fundus photos.
    # We don't block inference, but we flag likely invalid inputs so users don't
    # interpret forced predictions as reliable medical outputs.
    likely_not_fundus = (
        non_black_ratio < 0.18
        or largest_contour_ratio < 0.12
        or (contours and circularity < 0.30)
    )
    if likely_not_fundus:
        issues.append("image does not look like a centered retinal fundus photo")

    if not issues:
        verdict = "Excellent"
        advisory = "Image quality is strong enough for reliable analysis."
    elif len(issues) == 1:
        verdict = "Good"
        advisory = f"Analysis is usable, but {issues[0]}."
    else:
        verdict = "Needs Review"
        advisory = "Image quality may reduce confidence because " + ", ".join(issues) + "."

    return metrics, verdict, advisory


def preprocess_image(image_bytes: bytes) -> Tuple[torch.Tensor, Dict[str, object]]:
    original_bgr = decode_image_bytes(image_bytes)
    processed_bgr = preprocess_fundus_bgr(original_bgr)

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.resize(original_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.resize(processed_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    quality_metrics, quality_verdict, quality_advisory = evaluate_image_quality(original_rgb)
    artifacts = {
        "original_rgb": original_rgb,
        "processed_rgb": processed_rgb,
        "quality_metrics": quality_metrics,
        "quality_verdict": quality_verdict,
        "quality_advisory": quality_advisory,
    }
    return tensorize_rgb_image(processed_rgb), artifacts


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = [target_layer.register_forward_hook(self._save_activation)]

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

        def _save_grad(grad: torch.Tensor) -> None:
            self.gradients = grad.detach()

        if torch.is_tensor(output) and output.requires_grad:
            output.register_hook(_save_grad)

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam).cpu().numpy()
        cam = cv2.GaussianBlur(cam, (5, 5), 0)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam.astype(np.float32)


def ensemble_gradcam_target_layers(model: nn.Module) -> List[nn.Module]:
    """Last conv blocks for ResNet50, EfficientNet-B0 (timm), and DenseNet121."""
    layers: List[nn.Module] = [model.resnet.layer4[-1]]
    eff = getattr(model, "effnet", None) or getattr(model, "efficientnet", None)
    if eff is None:
        raise AttributeError("Ensemble model is missing an EfficientNet backbone for Grad-CAM.")
    if hasattr(eff, "blocks"):
        layers.append(eff.blocks[-1])
    elif hasattr(eff, "features"):
        layers.append(eff.features[-1])
    else:
        raise AttributeError("Cannot locate EfficientNet target block for Grad-CAM.")
    db4 = model.densenet.features.denseblock4
    db4_layers = list(db4.children())
    layers.append(db4_layers[-1])
    return layers


def _cam_from_activation_grad(activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1).squeeze(0)
    cam = F.relu(cam).cpu().numpy()
    cam = cv2.GaussianBlur(cam, (5, 5), 0)
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    cmin, cmax = float(cam.min()), float(cam.max())
    if cmax - cmin > 1e-8:
        cam = (cam - cmin) / (cmax - cmin)
    else:
        cam = np.zeros_like(cam)
    return cam.astype(np.float32)


class EnsembleGradCAM:
    """
    Average Grad-CAM maps from multiple backbones so heatmaps align with fused predictions.
    """

    def __init__(self, model: nn.Module, target_layers: Optional[List[nn.Module]] = None):
        self.model = model
        self.target_layers = target_layers or ensemble_gradcam_target_layers(model)
        self.activations: Dict[int, Optional[torch.Tensor]] = {i: None for i in range(len(self.target_layers))}
        self.gradients: Dict[int, Optional[torch.Tensor]] = {i: None for i in range(len(self.target_layers))}
        self.hooks = []
        for idx, layer in enumerate(self.target_layers):

            def save_act(_module, _inputs, output, i=idx):
                self.activations[i] = output.detach()

                def _save_grad(grad: torch.Tensor) -> None:
                    self.gradients[i] = grad.detach()

                if torch.is_tensor(output) and output.requires_grad:
                    output.register_hook(_save_grad)

            self.hooks.append(layer.register_forward_hook(save_act))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        for i in self.activations:
            self.activations[i] = None
            self.gradients[i] = None

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)

        cams: List[np.ndarray] = []
        for i in range(len(self.target_layers)):
            act, grad = self.activations[i], self.gradients[i]
            if act is None or grad is None:
                continue
            cams.append(_cam_from_activation_grad(act, grad))

        if not cams:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        stacked = np.stack(cams, axis=0)
        cam = stacked.mean(axis=0)
        cmin, cmax = float(cam.min()), float(cam.max())
        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        return cam.astype(np.float32)


def overlay_heatmap(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * cam)
    try:
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    except AttributeError:
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0.0)


def ndarray_to_base64(image_rgb: np.ndarray) -> str:
    image = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
