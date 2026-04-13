from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from backend.config import CLASS_SLUGS
from backend.services.data_pipeline import collect_samples, detect_pre_split_dataset, stratified_split

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit retinal dataset integrity.")
    parser.add_argument("--data_dir", required=True, help="Dataset root path")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Optional JSON report output path")
    return parser.parse_args()


def file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def distribution(samples):
    counts = Counter(label for _, label in samples)
    return {CLASS_SLUGS[idx]: counts.get(idx, 0) for idx in range(len(CLASS_SLUGS))}


def duplicate_report(samples):
    digest_to_paths = defaultdict(list)
    for file_path, _ in samples:
        path = Path(file_path)
        if path.suffix.lower() not in VALID_EXTS:
            continue
        try:
            digest = file_md5(path)
            digest_to_paths[digest].append(str(path))
        except Exception:
            continue
    duplicates = {k: v for k, v in digest_to_paths.items() if len(v) > 1}
    return duplicates


def class_folder_issues(root: Path):
    missing = []
    for slug in CLASS_SLUGS:
        if not (root / slug).is_dir():
            missing.append(slug)
    return missing


def leakage_count(group_a, group_b):
    files_a = {Path(p).name for p, _ in group_a}
    files_b = {Path(p).name for p, _ in group_b}
    return len(files_a.intersection(files_b))


def main():
    args = parse_args()
    data_root = Path(args.data_dir)
    pre_split = detect_pre_split_dataset(data_root)

    if pre_split:
        train_samples = collect_samples(data_root / "train")
        val_samples = collect_samples(data_root / "val")
        test_samples = collect_samples(data_root / "test") if (data_root / "test").is_dir() else []
        missing_folders = {
            "train": class_folder_issues(data_root / "train"),
            "val": class_folder_issues(data_root / "val"),
            "test": class_folder_issues(data_root / "test") if (data_root / "test").is_dir() else [],
        }
    else:
        all_samples = collect_samples(data_root)
        train_ratio = 1.0 - args.val_ratio - args.test_ratio
        train_samples, val_samples, test_samples = stratified_split(
            all_samples,
            train_ratio=train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        missing_folders = {"root": class_folder_issues(data_root)}

    full_samples = train_samples + val_samples + test_samples
    duplicates = duplicate_report(full_samples)
    report = {
        "dataset_root": str(data_root),
        "pre_split": pre_split,
        "counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "total": len(full_samples),
        },
        "distribution": {
            "train": distribution(train_samples),
            "val": distribution(val_samples),
            "test": distribution(test_samples),
        },
        "folder_issues": missing_folders,
        "leakage_by_filename": {
            "train_vs_val": leakage_count(train_samples, val_samples),
            "train_vs_test": leakage_count(train_samples, test_samples),
            "val_vs_test": leakage_count(val_samples, test_samples),
        },
        "duplicate_files_by_md5": {
            "count_groups": len(duplicates),
            "examples": list(duplicates.values())[:10],
        },
    }

    print(json.dumps(report, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
