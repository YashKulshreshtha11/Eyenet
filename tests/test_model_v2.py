import torch

from backend.config import NUM_CLASSES
from backend.models.ensemble import build_model


def test_model_output_shape():
    model = build_model(pretrained=False)
    batch = torch.randn(2, 3, 256, 256)
    logits = model(batch)
    assert logits.shape == (2, NUM_CLASSES)


def test_model_probabilities_sum_to_one():
    model = build_model(pretrained=False)
    batch = torch.randn(1, 3, 256, 256)
    probabilities = model.predict_proba(batch)
    total = probabilities.sum(dim=1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
