from __future__ import annotations

import argparse
from pathlib import Path

from backend.services.dataset_export import export_preprocessed_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess a retinal dataset with CLAHE and fundus cleanup")
    parser.add_argument("--source", required=True, help="Input dataset root")
    parser.add_argument("--output", required=True, help="Output directory for preprocessed PNG images")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    processed = export_preprocessed_dataset(source, output)
    print(f"[EyeNet] Preprocessed {processed} retinal images into {output}")


if __name__ == "__main__":
    main()
