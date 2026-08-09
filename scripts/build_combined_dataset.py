"""
Bangun dataset gabungan (skenario A4/B4, README.md bagian 2 & 3) dengan
label sudah diremap ke skema kanonik di src/class_mapping.py.

Classification:
    python scripts/build_combined_dataset.py --task classification \
        --sources gc10=data/processed/gc10_cls neu_cls=data/processed/neu_cls xsdd=data/processed/xsdd \
        --out_dir data/combined/classification

Detection (X-SDD di-drop karena tidak punya bbox, lihat README.md bagian 1 & 4):
    python scripts/build_combined_dataset.py --task detection \
        --sources gc10=data/processed/gc10_det neu_det=data/processed/neu_det \
        --out_dir data/combined/detection

Detection + negative samples (README v3 bagian 2 & 7 - fold gambar
defect-free Severstal sebagai anti false-positive, ~10-15% dari jumlah
gambar positive, JANGAN lebih supaya detector gak jadi terlalu konservatif):
    python scripts/build_combined_dataset.py --task detection \
        --sources gc10=data/processed/gc10_det neu_det=data/processed/neu_det \
        --out_dir data/combined/detection \
        --negatives_dir data/raw/severstal_clean --negative_ratio 0.12

File gambar diberi prefix `<dataset>__` supaya tidak ada collision nama
antar dataset, dan tetap bisa ditelusuri asal datasetnya untuk analisis
cross-dataset generalization (README.md bagian 7, poin 8).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.class_mapping import (
    DATASET_TO_CANONICAL,
    combined_class_names_for,
    get_dataset_classes,
)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ["train", "val", "test"]


def _parse_sources(pairs: list[str]) -> dict[str, str]:
    sources = {}
    for pair in pairs:
        name, path = pair.split("=", 1)
        sources[name] = path
    return sources


def build_combined_classification(sources: dict[str, str], out_dir: str | Path) -> None:
    dataset_names = list(sources.keys())
    canonical_subset = combined_class_names_for(dataset_names)
    out_dir = Path(out_dir)

    for split in SPLITS:
        for cname in canonical_subset:
            (out_dir / split / cname).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {c: 0 for c in canonical_subset}
    for ds_name, src_path in sources.items():
        mapping = DATASET_TO_CANONICAL[ds_name]
        src_root = Path(src_path)
        for split in SPLITS:
            split_dir = src_root / split
            if not split_dir.exists():
                continue
            for local_cls_dir in sorted(split_dir.iterdir()):
                if not local_cls_dir.is_dir():
                    continue
                canonical_name = mapping.get(local_cls_dir.name)
                if canonical_name is None or canonical_name not in canonical_subset:
                    continue
                dest_dir = out_dir / split / canonical_name
                for img in local_cls_dir.iterdir():
                    if img.suffix.lower() not in IMG_EXTENSIONS:
                        continue
                    dest_name = f"{ds_name}__{img.name}"
                    shutil.copy2(img, dest_dir / dest_name)
                    if split == "train":
                        counts[canonical_name] += 1

    print("Distribusi jumlah sampel train per kelas kanonik (cek imbalance sebelum training):")
    for cname, n in counts.items():
        print(f"  {cname:35s} {n:5d}")


def build_combined_detection(sources: dict[str, str], out_dir: str | Path) -> None:
    dataset_names = list(sources.keys())
    canonical_subset = combined_class_names_for(dataset_names)
    canon_to_id = {c: i for i, c in enumerate(canonical_subset)}
    out_dir = Path(out_dir)

    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    box_counts: dict[str, int] = {c: 0 for c in canonical_subset}
    for ds_name, src_path in sources.items():
        local_classes = get_dataset_classes(ds_name)
        mapping = DATASET_TO_CANONICAL[ds_name]
        src_root = Path(src_path)

        for split in SPLITS:
            img_dir = src_root / "images" / split
            label_dir = src_root / "labels" / split
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTENSIONS:
                    continue
                label_path = label_dir / f"{img_path.stem}.txt"
                dest_img_name = f"{ds_name}__{img_path.name}"
                shutil.copy2(img_path, out_dir / "images" / split / dest_img_name)

                new_lines = []
                if label_path.exists():
                    for line in label_path.read_text().strip().splitlines():
                        if not line.strip():
                            continue
                        parts = line.split()
                        local_id = int(parts[0])
                        local_name = local_classes[local_id]
                        canon_name = mapping.get(local_name)
                        if canon_name is None or canon_name not in canon_to_id:
                            continue
                        new_id = canon_to_id[canon_name]
                        new_lines.append(" ".join([str(new_id), *parts[1:]]))
                        if split == "train":
                            box_counts[canon_name] += 1

                dest_label_name = f"{ds_name}__{img_path.stem}.txt"
                (out_dir / "labels" / split / dest_label_name).write_text("\n".join(new_lines))

    print("Distribusi jumlah bounding box train per kelas kanonik (cek imbalance sebelum training):")
    for cname, n in box_counts.items():
        print(f"  {cname:35s} {n:5d}")


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


def add_negative_samples(
    negatives_dir: str | Path,
    out_dir: str | Path,
    ratio: float,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Fold gambar negative/background (defect-free, misal subset Severstal)
    ke dataset detection sebagai anti false-positive sample (README v3
    bagian 2 & 7). Ditulis sebagai label file KOSONG (0 object) -
    YoloDetectionDataset (src/data_loader.py) sudah otomatis treat file
    label yang nggak ada/kosong sebagai "0 box", jadi loader nggak perlu diubah.

    `ratio` dihitung terhadap jumlah gambar POSITIVE yang sudah ada di
    out_dir. Sengaja dibikin kecil (README rekomendasi ~10-15%) - kalau
    kebanyakan negative, detector beresiko jadi terlalu konservatif dan
    malah nurunin recall (README bagian 6, "Error propagation").
    """
    out_dir = Path(out_dir)
    negatives_dir = Path(negatives_dir)

    total_positive = sum(
        len(list((out_dir / "images" / split).glob("*"))) for split in SPLITS
    )
    target_negative = int(round(total_positive * ratio))

    all_negatives = sorted(p for p in negatives_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
    if len(all_negatives) < target_negative:
        print(
            f"WARNING: negative source cuma {len(all_negatives)} gambar, kurang dari target "
            f"{target_negative} ({ratio:.0%} dari {total_positive} positive). Pakai semua yang ada."
        )
        target_negative = len(all_negatives)

    sampled = random.Random(seed).sample(all_negatives, target_negative)
    train, val, test = _split_list(sampled, val_ratio, test_ratio, seed)

    added = {}
    for split_name, files in [("train", train), ("val", val), ("test", test)]:
        img_out = out_dir / "images" / split_name
        label_out = out_dir / "labels" / split_name
        for img_path in files:
            dest_name = f"severstal_neg__{img_path.name}"
            shutil.copy2(img_path, img_out / dest_name)
            (label_out / f"severstal_neg__{img_path.stem}.txt").write_text("")
        added[split_name] = len(files)

    print(f"Negative samples ditambahkan ({ratio:.0%} dari {total_positive} positive): {added}")


def main():
    parser = argparse.ArgumentParser(description="Bangun dataset gabungan (union label kanonik)")
    parser.add_argument("--task", required=True, choices=["classification", "detection"])
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="Pasangan dataset_name=path_ke_processed, contoh: gc10=data/processed/gc10_cls",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--negatives_dir", default=None,
        help="(detection only) folder gambar defect-free buat anti false-positive, misal data/raw/severstal_clean",
    )
    parser.add_argument(
        "--negative_ratio", type=float, default=0.12,
        help="Proporsi negative terhadap jumlah gambar positive (default 0.12 = 12%%)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = _parse_sources(args.sources)

    if args.task == "classification":
        build_combined_classification(sources, args.out_dir)
    else:
        build_combined_detection(sources, args.out_dir)
        if args.negatives_dir:
            add_negative_samples(args.negatives_dir, args.out_dir, args.negative_ratio, seed=args.seed)

    print(f"\nSelesai. Dataset gabungan tersimpan di {args.out_dir}")


if __name__ == "__main__":
    main()
