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

File gambar diberi prefix `<dataset>__` supaya tidak ada collision nama
antar dataset, dan tetap bisa ditelusuri asal datasetnya untuk analisis
cross-dataset generalization (README.md bagian 7, poin 8).
"""

from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Bangun dataset gabungan (union label kanonik)")
    parser.add_argument("--task", required=True, choices=["classification", "detection"])
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="Pasangan dataset_name=path_ke_processed, contoh: gc10=data/processed/gc10_cls",
    )
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    sources = _parse_sources(args.sources)

    if args.task == "classification":
        build_combined_classification(sources, args.out_dir)
    else:
        build_combined_detection(sources, args.out_dir)

    print(f"\nSelesai. Dataset gabungan tersimpan di {args.out_dir}")


if __name__ == "__main__":
    main()
