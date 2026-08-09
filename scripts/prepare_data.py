"""
Reorganisasi dataset mentah (hasil download manual, lihat README.md bagian 1
& 8) menjadi struktur standar `data/processed/` yang dipakai src/data_loader.py.

Asumsi struktur raw data SEBELUM dijalankan (sesuaikan foldermu ke sini
dulu, karena struktur arsip asli tiap sumber beda-beda dan tidak seragam):

  Classification (GC10 classification variant, NEU-CLS, X-SDD) — folder per kelas:
      data/raw/<dataset>/<class_name>/*.jpg

  Detection (GC10-DET, NEU-DET) — VOC-style:
      data/raw/<dataset>/Annotations/*.xml
      data/raw/<dataset>/JPEGImages/*.jpg

Output:
  Classification -> data/processed/<dataset>/{train,val,test}/<class_name>/*.jpg
  Detection      -> data/processed/<dataset>/images/{train,val,test}/*.jpg
                     data/processed/<dataset>/labels/{train,val,test}/*.txt

Split default 70/15/15 (train/val/test), stratified per kelas, seed
konsisten dengan `seed` di config training supaya reproducible.

Contoh pemakaian:
    python scripts/prepare_data.py --task neu_cls \
        --raw_dir data/raw/neu_cls --out_dir data/processed/neu_cls

    python scripts/prepare_data.py --task gc10_det \
        --raw_dir data/raw/gc10_det --out_dir data/processed/gc10_det
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.voc_to_yolo import parse_voc_xml, voc_to_yolo_lines
from src.class_mapping import get_dataset_classes

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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


def prepare_classification(
    raw_dir: str | Path,
    out_dir: str | Path,
    class_names: list[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    summary = {}

    for cls in class_names:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            print(f"WARNING: folder kelas '{cls}' tidak ditemukan di {raw_dir}, di-skip")
            continue

        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
        if not files:
            print(f"WARNING: tidak ada gambar di {cls_dir}")
            continue

        train, val, test = _split_list(files, val_ratio, test_ratio, seed)
        summary[cls] = {"train": len(train), "val": len(val), "test": len(test)}

        for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
            dest_dir = out_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dest_dir / f.name)

    print(f"\nDistribusi jumlah sampel per kelas ({raw_dir.name}):")
    for cls, counts in summary.items():
        print(f"  {cls:35s} train={counts['train']:4d}  val={counts['val']:4d}  test={counts['test']:4d}")


def _find_image_for_stem(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTENSIONS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def prepare_detection(
    raw_dir: str | Path,
    out_dir: str | Path,
    class_names: list[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    ann_dir = raw_dir / "Annotations"
    img_dir = raw_dir / "JPEGImages"
    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(
            f"Diperlukan {ann_dir} dan {img_dir} (struktur VOC). "
            "Sesuaikan raw_dir kamu dulu ke struktur ini sebelum run script."
        )

    class_to_id = {c: i for i, c in enumerate(class_names)}
    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise RuntimeError(f"Tidak ada file .xml di {ann_dir}")

    # Stratifikasi kasar: pakai kelas object PERTAMA di tiap gambar sebagai
    # "primary label" untuk split. Gambar sering punya >1 object/kelas,
    # jadi ini bukan stratifikasi sempurna, tapi cukup untuk menjaga
    # proporsi kelas antara train/val/test.
    groups: dict[str, list[str]] = defaultdict(list)
    skipped_no_object = 0
    for xml_path in xml_files:
        _, _, objects = parse_voc_xml(xml_path)
        if not objects:
            skipped_no_object += 1
            continue
        primary_label = objects[0][0]
        groups[primary_label].append(xml_path.stem)

    if skipped_no_object:
        print(f"WARNING: {skipped_no_object} file XML tanpa object, di-skip")

    split_stems = {"train": [], "val": [], "test": []}
    summary = {}
    for label, stems in groups.items():
        train, val, test = _split_list(stems, val_ratio, test_ratio, seed)
        split_stems["train"] += train
        split_stems["val"] += val
        split_stems["test"] += test
        summary[label] = {"train": len(train), "val": len(val), "test": len(test)}

    for split_name, stems in split_stems.items():
        img_out = out_dir / "images" / split_name
        label_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            img_path = _find_image_for_stem(img_dir, stem)
            if img_path is None:
                print(f"WARNING: gambar untuk '{stem}' tidak ditemukan di {img_dir}, di-skip")
                continue
            xml_path = ann_dir / f"{stem}.xml"
            lines, _ = voc_to_yolo_lines(xml_path, class_to_id)

            shutil.copy2(img_path, img_out / img_path.name)
            (label_out / f"{stem}.txt").write_text("\n".join(lines))

    print(f"\nDistribusi jumlah object per primary-class ({raw_dir.name}):")
    for label, counts in summary.items():
        print(f"  {label:35s} train={counts['train']:4d}  val={counts['val']:4d}  test={counts['test']:4d}")


TASK_REGISTRY = {
    "gc10_cls": ("gc10", prepare_classification),
    "neu_cls": ("neu_cls", prepare_classification),
    "xsdd_cls": ("xsdd", prepare_classification),
    "gc10_det": ("gc10", prepare_detection),
    "neu_det": ("neu_det", prepare_detection),
}


def main():
    parser = argparse.ArgumentParser(description="Reorganisasi raw dataset -> data/processed/")
    parser.add_argument("--task", required=True, choices=list(TASK_REGISTRY))
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_key, fn = TASK_REGISTRY[args.task]
    class_names = get_dataset_classes(dataset_key)

    fn(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        class_names=class_names,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"\nSelesai. Hasil tersimpan di {args.out_dir}")


if __name__ == "__main__":
    main()
