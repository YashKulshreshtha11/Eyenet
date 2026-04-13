from __future__ import annotations

from pathlib import Path

import cv2

from backend.services.fundus_ops import preprocess_fundus_bgr, read_image_bgr


def export_preprocessed_dataset(source_dir: str | Path, output_dir: str | Path) -> int:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    processed_count = 0

    for image_path in source_path.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in valid_exts:
            continue

        relative_path = image_path.relative_to(source_path).with_suffix(".png")
        destination = output_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        image_bgr = read_image_bgr(image_path)
        processed_bgr = preprocess_fundus_bgr(image_bgr)
        cv2.imwrite(str(destination), processed_bgr)
        processed_count += 1

    return processed_count
