from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_SLUGS = ["diabetic_retinopathy", "glaucoma", "cataract", "normal"]


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare production-safe datasets: fixed split + train-augmented split."
    )
    p.add_argument("--original_root", required=True, help="Original flat dataset root (class folders).")
    p.add_argument("--odir_clean_root", required=True, help="ODIR clean dataset root (class folders).")
    p.add_argument("--out_split_root", required=True, help="Output root for fixed split dataset.")
    p.add_argument("--out_merged_root", required=True, help="Output root for merged dataset (ODIR added to train only).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument(
        "--odir_cap_per_class",
        type=int,
        default=400,
        help="Max ODIR images to add per class into TRAIN only (0 = add none).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directories if they already exist.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not copy files; only report intended actions.",
    )
    return p.parse_args()


def iter_images(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]


def validate_flat_root(root: Path) -> None:
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    missing = [slug for slug in CLASS_SLUGS if not (root / slug).is_dir()]
    if missing:
        raise SystemExit(f"Missing class folders in {root}: {missing}")


def stratified_split(
    by_class: Dict[str, List[Path]],
    cfg: SplitConfig,
) -> Dict[str, Dict[str, List[Path]]]:
    if abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) > 1e-6:
        raise SystemExit("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(cfg.seed)
    out = {"train": {}, "val": {}, "test": {}}
    for slug, files in by_class.items():
        items = list(files)
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * cfg.train_ratio))
        n_val = max(1, int(n * cfg.val_ratio))
        train_files = items[:n_train]
        val_files = items[n_train : n_train + n_val]
        test_files = items[n_train + n_val :]
        out["train"][slug] = train_files
        out["val"][slug] = val_files
        out["test"][slug] = test_files
    return out


def ensure_empty_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"Output directory already exists (use --force to overwrite): {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_group(group: Dict[str, List[Path]], dst_root: Path, prefix: str, dry_run: bool) -> int:
    copied = 0
    for slug, files in group.items():
        dst_dir = dst_root / slug
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            # keep original filename but namespace to avoid collisions across sources
            dst = dst_dir / f"{prefix}{src.name}"
            if dry_run:
                copied += 1
                continue
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1
    return copied


def main():
    args = parse_args()
    original_root = Path(args.original_root)
    odir_root = Path(args.odir_clean_root)
    out_split_root = Path(args.out_split_root)
    out_merged_root = Path(args.out_merged_root)

    validate_flat_root(original_root)
    validate_flat_root(odir_root)

    cfg = SplitConfig(
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )

    # Collect original files per class.
    original_by_class: Dict[str, List[Path]] = {}
    for slug in CLASS_SLUGS:
        original_by_class[slug] = iter_images(original_root / slug)

    # Create fixed split directories.
    ensure_empty_dir(out_split_root, force=bool(args.force))
    for split in ("train", "val", "test"):
        (out_split_root / split).mkdir(parents=True, exist_ok=True)

    split_map = stratified_split(original_by_class, cfg)
    split_counts: Dict[str, Dict[str, int]] = {s: {} for s in ("train", "val", "test")}
    copied_split_total = 0
    for split in ("train", "val", "test"):
        copied = copy_group(
            split_map[split],
            dst_root=out_split_root / split,
            prefix="orig_",
            dry_run=bool(args.dry_run),
        )
        copied_split_total += copied
        for slug in CLASS_SLUGS:
            split_counts[split][slug] = len(split_map[split][slug])

    # Create merged dataset: identical val/test to original split, train augmented with ODIR.
    ensure_empty_dir(out_merged_root, force=bool(args.force))
    for split in ("train", "val", "test"):
        (out_merged_root / split).mkdir(parents=True, exist_ok=True)

    copied_merged_total = 0
    # Copy original split groups first.
    for split in ("train", "val", "test"):
        copied_merged_total += copy_group(
            split_map[split],
            dst_root=out_merged_root / split,
            prefix="orig_",
            dry_run=bool(args.dry_run),
        )

    # Add capped ODIR images to TRAIN only.
    rng = random.Random(cfg.seed)
    odir_added_counts: Dict[str, int] = {slug: 0 for slug in CLASS_SLUGS}
    odir_cap = int(args.odir_cap_per_class)
    if odir_cap > 0:
        for slug in CLASS_SLUGS:
            files = iter_images(odir_root / slug)
            rng.shuffle(files)
            files = files[:odir_cap]
            odir_added_counts[slug] = len(files)
            copied_merged_total += copy_group(
                {slug: files},
                dst_root=out_merged_root / "train",
                prefix="odir_",
                dry_run=bool(args.dry_run),
            )

    report = {
        "original_root": str(original_root),
        "odir_clean_root": str(odir_root),
        "out_split_root": str(out_split_root),
        "out_merged_root": str(out_merged_root),
        "split_config": {
            "seed": cfg.seed,
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "test_ratio": cfg.test_ratio,
        },
        "original_counts": {slug: len(original_by_class[slug]) for slug in CLASS_SLUGS},
        "fixed_split_counts": split_counts,
        "odir_added_to_train": odir_added_counts,
        "copied_files": {
            "fixed_split_total": copied_split_total,
            "merged_total": copied_merged_total,
        },
        "dry_run": bool(args.dry_run),
    }

    (out_split_root / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_merged_root / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

