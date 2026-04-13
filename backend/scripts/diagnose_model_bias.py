from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

from backend.config import CLASS_NAMES, CLASS_SLUGS, NUM_CLASSES
from backend.models.model import build_model
from backend.services.vision import preprocess_image


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose class-collapse bias for EyeNet weights.")
    parser.add_argument("--weights", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--data_dir", required=True, help="Dataset root with class folders")
    parser.add_argument("--samples_per_class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(weights_path: str) -> torch.nn.Module:
    model = build_model(pretrained=False, num_classes=NUM_CLASSES)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def collect_class_files(root: Path, class_slug: str):
    class_dir = root / class_slug
    if not class_dir.is_dir():
        return []
    files = []
    for name in os.listdir(class_dir):
        path = class_dir / name
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            files.append(path)
    return files


def main():
    args = parse_args()
    random.seed(args.seed)

    model = load_model(args.weights)
    dataset_root = Path(args.data_dir)
    per_class_predictions = defaultdict(Counter)
    total = 0
    correct = 0

    for target_idx, class_slug in enumerate(CLASS_SLUGS):
        files = collect_class_files(dataset_root, class_slug)
        random.shuffle(files)
        files = files[: args.samples_per_class]
        for path in files:
            with open(path, "rb") as handle:
                image_bytes = handle.read()
            tensor, _ = preprocess_image(image_bytes)
            with torch.no_grad():
                pred_idx = int(model(tensor).argmax(dim=1).item())
            per_class_predictions[class_slug][pred_idx] += 1
            total += 1
            correct += int(pred_idx == target_idx)

    print(f"Evaluated samples: {total}")
    if total:
        print(f"Approx accuracy: {correct / total:.4f}")
    print("-" * 60)
    for class_slug in CLASS_SLUGS:
        counts = per_class_predictions[class_slug]
        readable = {CLASS_NAMES[idx]: counts[idx] for idx in sorted(counts)}
        print(f"{class_slug}: {readable}")

    dominant = [max(per_class_predictions[slug].values(), default=0) for slug in CLASS_SLUGS]
    if total and sum(dominant) == total and len({tuple(per_class_predictions[s].items()) for s in CLASS_SLUGS}) == 1:
        print("\nWARNING: strong class-collapse signal detected.")


if __name__ == "__main__":
    main()
