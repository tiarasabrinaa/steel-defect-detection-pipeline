"""
Metrik evaluasi object detection sesuai README.md bagian 5:
mAP@0.5, mAP@0.5:0.95, per-class AP.

Dipakai oleh training loop torchvision (Faster R-CNN / RetinaNet) yang kita
tulis manual. Untuk Ultralytics (YOLO / RT-DETR), library-nya sudah
menghitung metrik-metrik ini sendiri lewat `model.val()` — lihat
train_detection.py, kita tinggal membaca hasilnya dan log ke MLflow supaya
formatnya konsisten dengan jalur torchvision.
"""

from __future__ import annotations

from typing import Any

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def build_map_metric() -> MeanAveragePrecision:
    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )


def compute_detection_metrics(
    metric: MeanAveragePrecision,
    class_names: list[str],
) -> dict[str, Any]:
    """
    `metric` harus sudah di-update() per-batch dengan preds & targets format
    torchmetrics (list of dict {boxes, scores, labels} / {boxes, labels}).
    Label di sini 0-indexed sesuai class_names (background sudah dibuang
    sebelum masuk sini, lihat train_detection.py `_strip_background`).
    """
    result = metric.compute()

    scalars: dict[str, Any] = {
        "mAP_50_95": result["map"].item(),
        "mAP_50": result["map_50"].item(),
        "mAP_75": result["map_75"].item(),
        "mAR_100": result["mar_100"].item(),
    }

    # per-class AP: torchmetrics mengembalikan tensor sejajar dengan
    # result["classes"] (bukan urut class_names), harus dipetakan balik.
    if "map_per_class" in result and "classes" in result:
        per_class_ap = result["map_per_class"]
        classes_idx = result["classes"]
        if per_class_ap.ndim == 0:
            per_class_ap = per_class_ap.unsqueeze(0)
            classes_idx = classes_idx.unsqueeze(0)
        for ap, cls_idx in zip(per_class_ap.tolist(), classes_idx.tolist()):
            if 0 <= cls_idx < len(class_names):
                safe_name = class_names[cls_idx].replace(" ", "_")
                if ap >= 0:  # torchmetrics pakai -1 untuk kelas tanpa GT
                    scalars[f"AP50_95_per_class.{safe_name}"] = ap

    return scalars


def strip_background(
    boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor | None = None
):
    """
    Torchvision detection API pakai label 0 = background dan foreground
    class mulai dari 1. Sebelum dihitung metriknya (dan sebelum dipetakan
    ke class_names 0-indexed), background harus dibuang dan label
    digeser -1.
    """
    keep = labels > 0
    new_boxes = boxes[keep]
    new_labels = labels[keep] - 1
    if scores is not None:
        return new_boxes, new_labels, scores[keep]
    return new_boxes, new_labels
