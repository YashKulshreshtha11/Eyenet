import cv2
import numpy as np

from backend.config import IMAGE_SIZE
from backend.services.vision import preprocess_image


def test_preprocess_image_returns_expected_artifacts():
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 100, (80, 120, 180), -1)
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok

    tensor, artifacts = preprocess_image(encoded.tobytes())

    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert artifacts["original_rgb"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert artifacts["processed_rgb"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert "quality_metrics" in artifacts
