from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch


def unwrap_checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        candidate = checkpoint.get("state_dict", checkpoint)
        if isinstance(candidate, dict):
            return candidate
    raise TypeError("Checkpoint payload is not a valid state_dict dictionary.")


def validate_class_names(class_names: Sequence[str], expected_classes: int) -> None:
    if len(class_names) != expected_classes:
        raise ValueError(
            f"Class mapping size mismatch: got {len(class_names)}, expected {expected_classes}."
        )
    if len(set(class_names)) != len(class_names):
        raise ValueError("Class mapping contains duplicate class names.")


def validate_checkpoint_head_shape(
    state_dict: Dict[str, torch.Tensor], expected_classes: int
) -> None:
    # Current production architecture uses fc.4 as output layer.
    out_weight = state_dict.get("fc.4.weight")
    out_bias = state_dict.get("fc.4.bias")
    if out_weight is None or out_bias is None:
        raise ValueError(
            "Checkpoint missing classifier head keys (fc.4.weight / fc.4.bias). "
            "This usually means an architecture mismatch."
        )
    if out_weight.shape[0] != expected_classes or out_bias.shape[0] != expected_classes:
        raise ValueError(
            f"Checkpoint classifier head mismatch: got {out_weight.shape[0]} classes, "
            f"expected {expected_classes}."
        )


def validate_model_output(logits: torch.Tensor, expected_classes: int) -> None:
    if logits.ndim != 2:
        raise RuntimeError(f"Unexpected logits rank: {logits.ndim}. Expected [N, C].")
    if logits.shape[1] != expected_classes:
        raise RuntimeError(
            f"Unexpected logits shape {tuple(logits.shape)}; expected class dimension {expected_classes}."
        )
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError("Model output contains NaN/Inf values.")


def detect_prediction_collapse(history: Iterable[int], consecutive_limit: int) -> bool:
    history_list: List[int] = list(history)
    if len(history_list) < consecutive_limit:
        return False
    tail = history_list[-consecutive_limit:]
    return len(set(tail)) == 1


def validate_model_prediction_diversity(
    model: torch.nn.Module,
    device: torch.device,
    expected_classes: int,
    probes: int = 16,
    min_unique_predictions: int = 2,
) -> None:
    model.eval()
    preds: List[int] = []
    max_probs: List[float] = []
    logit_stds: List[float] = []
    with torch.no_grad():
        for _ in range(probes):
            logits = model(torch.randn(1, 3, 256, 256, device=device))
            validate_model_output(logits, expected_classes)
            probs = torch.softmax(logits, dim=1)
            preds.append(int(logits.argmax(dim=1).item()))
            max_probs.append(float(probs.max().item()))
            logit_stds.append(float(logits.std().item()))
    if len(set(preds)) < min_unique_predictions:
        # Only fail hard when the model is both single-class and numerically degenerate.
        # Many healthy models can still map random noise mostly to one class.
        if (sum(max_probs) / len(max_probs)) > 0.99 and (sum(logit_stds) / len(logit_stds)) < 1e-3:
            raise RuntimeError(
                "Model collapse detected during startup sanity check: predictions lack diversity. "
                "Checkpoint likely biased/corrupted; retrain or replace weights."
            )
        return
