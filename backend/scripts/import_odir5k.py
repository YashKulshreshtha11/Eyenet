from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ODIR_CLASS_CODES = ("N", "D", "G", "C", "A", "H", "M", "O")

# Map ODIR single-label codes to this project's 4-class slugs.
CODE_TO_SLUG = {
    "N": "normal",
    "D": "diabetic_retinopathy",
    "G": "glaucoma",
    "C": "cataract",
}


@dataclass(frozen=True)
class OdirRow:
    left: str
    right: str
    codes: Tuple[str, ...]


def parse_args():
    p = argparse.ArgumentParser(description="Import ODIR-5K into EyeNet 4-class folder structure.")
    p.add_argument("--odir_root", required=True, help="ODIR-5K root folder (contains data.xlsx and image folders).")
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output dataset root with class folders (diabetic_retinopathy/glaucoma/cataract/normal).",
    )
    p.add_argument(
        "--include_split",
        default="train",
        choices=["train", "test", "all"],
        help="Which ODIR image split(s) to import.",
    )
    p.add_argument(
        "--policy",
        default="single",
        choices=["single", "single_or_normal_only"],
        help=(
            "Filtering policy. "
            "'single' keeps only rows with exactly one of N/D/G/C and no other codes. "
            "'single_or_normal_only' additionally allows (N) only; same as 'single' in practice."
        ),
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not copy files; just print what would be imported.",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Print spreadsheet columns and exit.",
    )
    return p.parse_args()


def _read_xlsx(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def _extract_codes(cell: object) -> Tuple[str, ...]:
    """
    ODIR uses a label field containing codes like:
      "N", "D", "G", "C", ... sometimes multiple separated by comma/space.
    We extract valid codes only.
    """
    if cell is None:
        return tuple()
    s = str(cell).strip().upper()
    if not s:
        return tuple()
    # Common separators: ',' ';' ' ' '|'
    parts: List[str] = []
    for chunk in s.replace("|", ",").replace(";", ",").replace(" ", ",").split(","):
        c = chunk.strip()
        if not c:
            continue
        # Sometimes labels appear as words; keep only exact codes.
        if c in ODIR_CLASS_CODES:
            parts.append(c)
    # De-dup while preserving order
    seen = set()
    out: List[str] = []
    for c in parts:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return tuple(out)


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def iter_rows(df: pd.DataFrame) -> Iterable[OdirRow]:
    """
    ODIR variants typically include columns like:
      - Left-Fundus / Right-Fundus (or similar)
      - Labels / Diagnosis / keyword (varies)
    This parser tries common column names and fails loudly if not found.
    """
    left_col = _first_existing(df, ["Left-Fundus", "Left Fundus", "Left-Fundus Image", "Left-Fundus Image Path"])
    right_col = _first_existing(df, ["Right-Fundus", "Right Fundus", "Right-Fundus Image", "Right-Fundus Image Path"])
    for _, row in df.iterrows():
        left = str(row[left_col]).strip()
        right = str(row[right_col]).strip()
        # Prefer one-hot columns if present (ODIR-5K official format).
        if all(c in df.columns for c in ODIR_CLASS_CODES):
            codes = tuple(c for c in ODIR_CLASS_CODES if int(row.get(c, 0) or 0) == 1)
        else:
            label_col = _first_existing(
                df,
                ["Labels", "Label", "Diagnostic Keywords", "diagnostic keywords", "Diagnosis", "Class"],
            )
            if label_col is None:
                raise RuntimeError(
                    "Could not locate label columns in data.xlsx. "
                    f"Found columns: {list(df.columns)}"
                )
            codes = _extract_codes(row[label_col])
        yield OdirRow(left=left, right=right, codes=codes)


def keep_row(row: OdirRow, policy: str) -> Optional[str]:
    """
    Returns class slug if the row should be imported; otherwise None.
    """
    codes = row.codes
    if not codes:
        return None
    # Keep only rows that map cleanly into one of the four classes and contain no other codes.
    supported = [c for c in codes if c in CODE_TO_SLUG]
    unsupported = [c for c in codes if c not in CODE_TO_SLUG]
    if unsupported:
        return None
    if len(supported) != 1:
        return None
    return CODE_TO_SLUG[supported[0]]


def resolve_image_path(odir_root: Path, rel_or_name: str) -> Optional[Path]:
    """
    ODIR sheets may store:
      - filename like "0_left.jpg"
      - or relative paths
    We search both Training Images and Testing Images.
    """
    s = rel_or_name.replace("\\", "/").strip().lstrip("./")
    name = Path(s).name
    candidates = [
        odir_root / "Training Images" / name,
        odir_root / "Testing Images" / name,
        odir_root / s,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def main():
    args = parse_args()
    odir_root = Path(args.odir_root)
    xlsx_path = odir_root / "data.xlsx"
    if not xlsx_path.is_file():
        raise SystemExit(f"Missing data.xlsx at {xlsx_path}")

    df = _read_xlsx(xlsx_path)
    if args.inspect:
        print(json.dumps({"shape": list(df.shape), "columns": list(df.columns)}, indent=2))
        # Avoid Windows console encoding issues from diagnostic keyword columns.
        preview_cols = [c for c in ["ID", "Left-Fundus", "Right-Fundus", "N", "D", "G", "C"] if c in df.columns]
        if preview_cols:
            print(df[preview_cols].head(5).to_string(index=False))
        return

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for slug in sorted(set(CODE_TO_SLUG.values())):
        (out_root / slug).mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped = 0
    missing_files = 0

    for row in iter_rows(df):
        slug = keep_row(row, args.policy)
        if slug is None:
            skipped += 1
            continue

        # Determine whether to import left/right based on include_split and file existence.
        for side in (row.left, row.right):
            src = resolve_image_path(odir_root, side)
            if src is None:
                missing_files += 1
                continue
            if args.include_split != "all":
                if args.include_split == "train" and "Training Images" not in str(src):
                    continue
                if args.include_split == "test" and "Testing Images" not in str(src):
                    continue

            # Keep original filename but namespace it to avoid collisions in merges.
            dst_name = f"odir_{src.name}"
            dst = out_root / slug / dst_name
            if args.dry_run:
                imported += 1
                continue
            if not dst.exists():
                shutil.copy2(src, dst)
            imported += 1

    report = {
        "odir_root": str(odir_root),
        "out_dir": str(out_root),
        "policy": args.policy,
        "include_split": args.include_split,
        "imported_images": imported,
        "skipped_rows": skipped,
        "missing_image_files": missing_files,
        "classes": sorted(set(CODE_TO_SLUG.values())),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

