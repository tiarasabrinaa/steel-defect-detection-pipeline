"""
Cari gambar defect-free dari Severstal Steel Defect Detection (Kaggle) dan
staging ke `data/raw/severstal_clean/`, siap dipakai sebagai:
  - kelas "Normal" di stage 1 (binary classifier) -> scripts/build_stage1_binary.py
  - negative/background sample di stage 2 (detector) -> scripts/build_combined_dataset.py --negatives_dir

Severstal itu Kaggle COMPETITION dataset (bukan dataset biasa) - download
manual butuh akun Kaggle + accept competition rules dulu di
https://www.kaggle.com/c/severstal-steel-defect-detection/data, baru bisa
`kaggle competitions download -c severstal-steel-defect-detection`.

Asumsi struktur raw SEBELUM script ini dijalankan:
    data/raw/severstal/train.csv
    data/raw/severstal/train_images/*.jpg

train.csv ada 2 varian format yang beredar di mirror berbeda (kolom
`ImageId_ClassId` gabungan, ATAU `ImageId`+`ClassId` terpisah) - script ini
handle keduanya. Gambar dianggap "defect-free" kalau TIDAK ADA baris dengan
EncodedPixels terisi untuk ImageId tersebut (atau ImageId itu nggak
disebut sama sekali di train.csv, tergantung mirror).

Contoh pemakaian:
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
    """Pastikan hasilnya punya kolom ImageId & EncodedPixels, apapun varian format aslinya."""
    if "ImageId_ClassId" in df.columns:
        df = df.copy()
        df["ImageId"] = df["ImageId_ClassId"].str.rsplit("_", n=1).str[0]
    elif "ImageId" not in df.columns:
        raise ValueError(
            "train.csv tidak punya kolom 'ImageId' atau 'ImageId_ClassId' - "
            "cek format file, mungkin mirror yang dipakai beda struktur."
        )

    if "EncodedPixels" not in df.columns:
        raise ValueError("train.csv tidak punya kolom 'EncodedPixels'")

    return df


def find_defect_free_images(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "train.csv"
    img_dir = raw_dir / "train_images"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} tidak ditemukan")
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} tidak ditemukan")

    df = _normalize_train_csv(pd.read_csv(csv_path))

    has_defect = set(
        df.loc[df["EncodedPixels"].notna() & (df["EncodedPixels"].astype(str).str.strip() != ""), "ImageId"]
    )

    all_images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    defect_free = [p for p in all_images if p.name not in has_defect]

    print(
        f"Total gambar: {len(all_images)}, dengan defect: {len(all_images) - len(defect_free)}, "
        f"defect-free: {len(defect_free)}"
    )
    return defect_free


def main():
    parser = argparse.ArgumentParser(description="Staging gambar defect-free dari Severstal")
    parser.add_argument("--raw_dir", required=True, help="Folder berisi train.csv + train_images/")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    defect_free = find_defect_free_images(args.raw_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in defect_free:
        shutil.copy2(img_path, out_dir / img_path.name)

    print(f"Selesai. {len(defect_free)} gambar defect-free tersimpan di {out_dir}")


if __name__ == "__main__":
    main()
