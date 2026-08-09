"""
Bangun dataset stage 1 (README.md v3, bagian 2-3): binary classifier
Defect vs Normal.

"Defect" = SEMUA gambar dari GC10-DET + NEU-CLS + X-SDD (ketiganya
defect-only dataset, jadi gak perlu peduli label kelas aslinya - relevan
di sini cuma "ada defect atau nggak"). "Normal" = subset gambar defect-free
dari Severstal (hasil scripts/prepare_severstal.py).

Balance: total Defect dijadikan acuan, Normal di-sample sepadan (bukan
pakai semua ribuan gambar Normal Severstal - README bagian 2, "Strategi
balance"). Split train/val/test dilakukan TERPISAH per sumber (supaya
GC10/NEU-CLS/X-SDD/Severstal tetap proporsional di ketiga split, bukan
kebetulan satu sumber ngumpul semua di train doang).

Contoh pemakaian:
    python scripts/build_stage1_binary.py \
        --defect_sources gc10=data/raw/gc10 neu_cls=data/raw/neu_cls xsdd=data/raw/xsdd \
        --normal_source data/raw/severstal_clean \
        --out_dir data/combined/stage1_binary
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ["train", "val", "test"]


def _split_list(items: list, val_ratio: float, test_ratio: float, seed: int) -> tuple[list, list, list]:
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    val = items[:n_val]
    test = items[n_val : n_val + n_test]
    train = items[n_val + n_test :]
    return train, val, test


def _gather_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def _parse_source_pairs(pairs: list[str]) -> dict[str, str]:
    return dict(pair.split("=", 1) for pair in pairs)


def build_stage1_binary(
    defect_sources: dict[str, str],
    normal_source: str,
    out_dir: str | Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    out_dir = Path(out_dir)
    for split in SPLITS:
        (out_dir / split / "Defect").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "Normal").mkdir(parents=True, exist_ok=True)

    defect_by_source: dict[str, list[Path]] = {
        name: _gather_images(path) for name, path in defect_sources.items()
    }
    total_defect = sum(len(v) for v in defect_by_source.values())
    if total_defect == 0:
        raise RuntimeError("Tidak ada gambar defect ditemukan, cek --defect_sources")

    all_normal = _gather_images(normal_source)
    if len(all_normal) < total_defect:
        print(
            f"WARNING: Normal cuma {len(all_normal)} gambar, kurang dari total "
            f"Defect ({total_defect}). Pakai semua yang ada -> dataset jadi timpang "
            f"ke sisi Defect, imbang belum tercapai."
        )
        sampled_normal = all_normal
    else:
        sampled_normal = random.Random(seed).sample(all_normal, total_defect)

    print("Distribusi sumber kelas 'Defect':")
    for name, imgs in defect_by_source.items():
        print(f"  {name:15s} {len(imgs):5d}")
    print(f"Total Defect: {total_defect}, Normal (sampled): {len(sampled_normal)}\n")

    # Defect: split per-sumber, supaya proporsi sumber terjaga di tiap split.
    for source_name, imgs in defect_by_source.items():
        train, val, test = _split_list(imgs, val_ratio, test_ratio, seed)
        for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
            dest_dir = out_dir / split_name / "Defect"
            for img_path in split_files:
                shutil.copy2(img_path, dest_dir / f"{source_name}__{img_path.name}")

    # Normal: satu sumber (Severstal), split biasa.
    train, val, test = _split_list(sampled_normal, val_ratio, test_ratio, seed)
    for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
        dest_dir = out_dir / split_name / "Normal"
        for img_path in split_files:
            shutil.copy2(img_path, dest_dir / f"severstal__{img_path.name}")

    for split_name in SPLITS:
        n_defect = len(list((out_dir / split_name / "Defect").iterdir()))
        n_normal = len(list((out_dir / split_name / "Normal").iterdir()))
        print(f"{split_name}: Defect={n_defect} Normal={n_normal}")


def main():
    parser = argparse.ArgumentParser(description="Bangun dataset stage 1 (binary Defect vs Normal)")
    parser.add_argument(
        "--defect_sources", nargs="+", required=True,
        help="Pasangan nama=path, contoh: gc10=data/raw/gc10 neu_cls=data/raw/neu_cls xsdd=data/raw/xsdd",
    )
    parser.add_argument("--normal_source", required=True, help="Folder gambar defect-free (hasil prepare_severstal.py)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_stage1_binary(
        defect_sources=_parse_source_pairs(args.defect_sources),
        normal_source=args.normal_source,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"\nSelesai. Dataset stage 1 tersimpan di {args.out_dir}")


if __name__ == "__main__":
    main()
