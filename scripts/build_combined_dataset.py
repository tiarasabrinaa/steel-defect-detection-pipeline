"""
Builds a combined dataset (scenarios A4/B4, README.md sections 2-3) with
labels remapped to the canonical scheme in src/class_mapping.py.

Classification:
    python scripts/build_combined_dataset.py --task classification \
        --sources gc10=data/processed/gc10_cls neu_cls=data/processed/neu_cls xsdd=data/processed/xsdd \
        --out_dir data/combined/classification

Detection (X-SDD is dropped, no bounding boxes available - README.md sections 1 and 4):
    python scripts/build_combined_dataset.py --task detection \
        --sources gc10=data/processed/gc10_det neu_det=data/processed/neu_det \
        --out_dir data/combined/detection

Detection with negative samples (fold in defect-free Severstal images as
anti-false-positive examples, capped at ~10-15% of positive images so the
detector doesn't become overly conservative - README.md sections 2 and 7):
    python scripts/build_combined_dataset.py --task detection \
        --sources gc10=data/processed/gc10_det neu_det=data/processed/neu_det \
        --out_dir data/combined/detection \
        --negatives_dir data/raw/severstal_clean --negative_ratio 0.12

Image filenames are prefixed with `<dataset>__` to avoid collisions and to
keep the source dataset identifiable for cross-dataset analysis.
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

    print("Train sample count per canonical class (check imbalance before training):")
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

    print("Train bounding box count per canonical class (check imbalance before training):")
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
    Folds negative/background images (defect-free, e.g. Severstal) into a
    detection dataset as anti-false-positive samples. Written as empty
    label files (0 objects); YoloDetectionDataset already treats a missing
    or empty label file as "0 boxes", so no loader changes are needed.

    `ratio` is computed against the number of positive images already in
    out_dir. Kept small (~10-15%) since too many negatives can make a
    detector overly conservative and reduce recall.
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
            f"WARNING: only {len(all_negatives)} negative images available, fewer than the target "
            f"{target_negative} ({ratio:.0%} of {total_positive} positives). Using all of them."
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

    print(f"Negative samples added ({ratio:.0%} of {total_positive} positives): {added}")


def main():
    parser = argparse.ArgumentParser(description="Build a combined dataset with canonical labels")
    parser.add_argument("--task", required=True, choices=["classification", "detection"])
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="dataset_name=path_to_processed pairs, e.g. gc10=data/processed/gc10_cls",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--negatives_dir", default=None,
        help="(detection only) folder of defect-free images for anti-false-positive samples",
    )
    parser.add_argument(
        "--negative_ratio", type=float, default=0.12,
        help="Ratio of negatives to positive images (default 0.12 = 12%%)",
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

    print(f"\nDone. Combined dataset saved to {args.out_dir}")


if __name__ == "__main__":
    main()
