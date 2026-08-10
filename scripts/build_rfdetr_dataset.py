"""
Converts a YOLO-format detection dataset (output of
scripts/build_combined_dataset.py) into the COCO JSON format required by
RF-DETR (package `rfdetr`).

RF-DETR expects:
    <out_dir>/train/_annotations.coco.json + images
    <out_dir>/valid/_annotations.coco.json + images
    <out_dir>/test/_annotations.coco.json  + images

Class names are read from the detection config yaml (dataset.class_names),
so the category ids match what the rest of the pipeline uses.

Example usage:
    python scripts/build_rfdetr_dataset.py \
        --yolo_dir data/combined/detection \
        --config configs/detection/det_combined.yaml \
        --out_dir data/combined/detection_coco
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from PIL import Image

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


def convert_split(
    yolo_dir: Path, out_dir: Path, split: str, coco_split: str, class_names: list[str]
) -> None:
    img_dir = yolo_dir / "images" / split
    label_dir = yolo_dir / "labels" / split
    out_split_dir = out_dir / coco_split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    images_json = []
    annotations_json = []
    ann_id = 1

    for img_id, img_path in enumerate(sorted(img_dir.iterdir()), start=1):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        shutil.copy2(img_path, out_split_dir / img_path.name)
        images_json.append({"id": img_id, "file_name": img_path.name, "height": h, "width": w})

        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            cls_id, xc, yc, bw, bh = line.split()
            cls_id = int(cls_id)
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            x_min = xc - bw / 2
            y_min = yc - bh / 2
            annotations_json.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,  # COCO category ids start at 1
                "bbox": [x_min, y_min, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    coco_dict = {"images": images_json, "annotations": annotations_json, "categories": categories}
    (out_split_dir / "_annotations.coco.json").write_text(json.dumps(coco_dict))
    print(f"{coco_split}: {len(images_json)} images, {len(annotations_json)} annotations")


def main():
    parser = argparse.ArgumentParser(description="Convert a YOLO detection dataset to COCO JSON for RF-DETR")
    parser.add_argument("--yolo_dir", required=True, help="e.g. data/combined/detection")
    parser.add_argument("--config", required=True, help="Detection config yaml, for dataset.class_names")
    parser.add_argument("--out_dir", required=True, help="e.g. data/combined/detection_coco")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    class_names = config["dataset"]["class_names"]

    yolo_dir = Path(args.yolo_dir)
    out_dir = Path(args.out_dir)
    for split, coco_split in SPLIT_MAP.items():
        convert_split(yolo_dir, out_dir, split, coco_split, class_names)

    print(f"\nDone. COCO-format dataset saved to {out_dir}")


if __name__ == "__main__":
    main()
