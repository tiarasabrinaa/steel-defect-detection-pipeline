"""
Dataset and DataLoader implementations for classification and object
detection, produced by scripts/prepare_data.py and
scripts/build_combined_dataset.py.

Classification layout: <root>/train|val|test/<class_name>/*.jpg
Detection layout (YOLO): <root>/images|labels/{train,val,test}/*

Label indices follow `class_names` order from the config, not alphabetical
folder order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FolderClassificationDataset(Dataset):
    """ImageFolder variant that preserves label order from `class_names`."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        class_names: list[str],
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.class_names = class_names
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        for label_idx, cls_name in enumerate(class_names):
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in IMG_EXTENSIONS and not img_path.name.startswith("."):
                    self.samples.append((img_path, label_idx))

        if not self.samples:
            raise RuntimeError(f"No images found in {split_dir} for classes {class_names}")

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_classification_transforms(img_size: int, train: bool) -> T.Compose:
    if train:
        return T.Compose(
            [
                T.ToImage(),
                T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(int(img_size * 1.15)),
            T.CenterCrop(img_size),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_classification_dataloaders(
    config: dict[str, Any],
) -> dict[str, DataLoader | FolderClassificationDataset]:
    ds_cfg = config["dataset"]
    root = ds_cfg["root"]
    class_names = ds_cfg["class_names"]
    img_size = config.get("img_size", 224)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)

    train_ds = FolderClassificationDataset(
        root, "train", class_names, build_classification_transforms(img_size, train=True)
    )
    val_ds = FolderClassificationDataset(
        root, "val", class_names, build_classification_transforms(img_size, train=False)
    )
    test_ds = FolderClassificationDataset(
        root, "test", class_names, build_classification_transforms(img_size, train=False)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
    }


class YoloDetectionDataset(Dataset):
    """Returns torchvision-style targets: {"boxes": FloatTensor[N,4] (xyxy,
    absolute px), "labels": LongTensor[N]}. `background_offset=1` is used
    for torchvision models, which reserve label 0 for background."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        class_names: list[str],
        img_size: int = 640,
        train: bool = False,
        background_offset: int = 1,
    ):
        self.root = Path(root)
        self.class_names = class_names
        self.img_size = img_size
        self.train = train
        self.background_offset = background_offset

        img_dir = self.root / "images" / split
        label_dir = self.root / "labels" / split
        if not img_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {img_dir}")

        self.image_paths = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS and not p.name.startswith(".")
        )
        self.label_dir = label_dir

        if train:
            self.transform = T.Compose(
                [
                    T.ToImage(),
                    T.Resize((img_size, img_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.2),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.ToImage(),
                    T.Resize((img_size, img_size)),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_label_file(self, img_path: Path, orig_w: int, orig_h: int):
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                cls_id, xc, yc, w, h = line.split()
                cls_id = int(cls_id)
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                x1 = (xc - w / 2) * orig_w
                y1 = (yc - h / 2) * orig_h
                x2 = (xc + w / 2) * orig_w
                y2 = (yc + h / 2) * orig_h
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id + self.background_offset)
        return boxes, labels

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        boxes, labels = self._read_label_file(img_path, orig_w, orig_h)

        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        if boxes:
            boxes = [
                [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                for b in boxes
            ]

        image_t = self.transform(image)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return image_t, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_detection_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    ds_cfg = config["dataset"]
    root = ds_cfg["root"]
    class_names = ds_cfg["class_names"]
    img_size = config.get("img_size", 640)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)

    train_ds = YoloDetectionDataset(root, "train", class_names, img_size, train=True)
    val_ds = YoloDetectionDataset(root, "val", class_names, img_size, train=False)
    test_ds = YoloDetectionDataset(root, "test", class_names, img_size, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=detection_collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=detection_collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=detection_collate_fn, pin_memory=True,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
    }


def build_ultralytics_data_yaml(config: dict[str, Any], out_path: str | Path) -> Path:
    ds_cfg = config["dataset"]
    root = Path(ds_cfg["root"]).resolve()
    data_yaml = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(ds_cfg["class_names"])},
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return out_path
