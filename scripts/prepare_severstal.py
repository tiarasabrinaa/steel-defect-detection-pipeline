"""
Finds defect-free images in the Severstal Steel Defect Detection dataset
(Kaggle) and stages them into `data/raw/severstal_clean/`, used as:
  - the "Normal" class for stage 1 (binary classifier) -> scripts/build_stage1_binary.py
  - negative/background samples for stage 2 (detector) -> scripts/build_combined_dataset.py --negatives_dir

Severstal is a Kaggle competition dataset; downloading it requires a Kaggle
account and accepting the competition rules at
https://www.kaggle.com/c/severstal-steel-defect-detection/data, then
`kaggle competitions download -c severstal-steel-defect-detection`.

Expected raw layout before running this script:
    data/raw/severstal/train.csv
    data/raw/severstal/train_images/*.jpg

train.csv has two formats across mirrors (a combined `ImageId_ClassId`
column, or separate `ImageId`/`ClassId` columns); this script handles both.
An image is considered defect-free if it has no row with a non-empty
EncodedPixels value.

Example usage:
    python scripts/prepare_severstal.py \
        --raw_dir data/raw/severstal --out_dir data/raw/severstal_clean
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _normalize_train_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the result has ImageId and EncodedPixels columns, regardless
    of the source format."""
    if "ImageId_ClassId" in df.columns:
        df = df.copy()
        df["ImageId"] = df["ImageId_ClassId"].str.rsplit("_", n=1).str[0]
    elif "ImageId" not in df.columns:
        raise ValueError(
            "train.csv has neither an 'ImageId' nor an 'ImageId_ClassId' column - "
            "check the file format, the mirror used may differ."
        )

    if "EncodedPixels" not in df.columns:
        raise ValueError("train.csv has no 'EncodedPixels' column")

    return df


def find_defect_free_images(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "train.csv"
    img_dir = raw_dir / "train_images"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} not found")

    df = _normalize_train_csv(pd.read_csv(csv_path))

    has_defect = set(
        df.loc[df["EncodedPixels"].notna() & (df["EncodedPixels"].astype(str).str.strip() != ""), "ImageId"]
    )

    all_images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    defect_free = [p for p in all_images if p.name not in has_defect]

    print(
        f"Total images: {len(all_images)}, with defects: {len(all_images) - len(defect_free)}, "
        f"defect-free: {len(defect_free)}"
    )
    return defect_free


def main():
    parser = argparse.ArgumentParser(description="Stage defect-free images from Severstal")
    parser.add_argument("--raw_dir", required=True, help="Folder containing train.csv and train_images/")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    defect_free = find_defect_free_images(args.raw_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in defect_free:
        shutil.copy2(img_path, out_dir / img_path.name)

    print(f"Done. {len(defect_free)} defect-free images saved to {out_dir}")


if __name__ == "__main__":
    main()
