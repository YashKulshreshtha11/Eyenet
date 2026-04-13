from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_bgr(image_path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image_bgr


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode the uploaded retinal image.")
    return image_bgr


def crop_fundus_circle(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_bgr

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    side = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2

    start_x = max(0, center_x - side // 2)
    start_y = max(0, center_y - side // 2)
    side = min(side, image_bgr.shape[1] - start_x, image_bgr.shape[0] - start_y)
    cropped = image_bgr[start_y:start_y + side, start_x:start_x + side]
    return cropped if cropped.size else image_bgr


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(image_bgr: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(image_bgr, 1.40, blurred, -0.40, 0)


def preprocess_fundus_bgr(image_bgr: np.ndarray) -> np.ndarray:
    image_bgr = crop_fundus_circle(image_bgr)
    image_bgr = apply_clahe(image_bgr)
    image_bgr = apply_unsharp_mask(image_bgr)
    return image_bgr
