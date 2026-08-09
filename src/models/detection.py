"""
Builder untuk model detection berbasis torchvision, dua-tahap classical
detector dari README.md bagian 3 - Task B:
    fasterrcnn_resnet50_fpn_v2  (two-stage)
    retinanet_resnet50_fpn_v2   (one-stage)

YOLO(v8/11) dan RT-DETR TIDAK lewat sini — keduanya dipakai lewat package
`ultralytics` langsung di train_detection.py karena API training/eval-nya
sudah dioptimalkan library tersebut (lihat README.md bagian "Arsitektur
pretrained" Task B).

`num_classes` di sini adalah jumlah kelas FOREGROUND (tanpa background);
torchvision butuh +1 slot untuk background di label index 0.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

TORCHVISION_ARCHITECTURES = ["fasterrcnn_resnet50_fpn_v2", "retinanet_resnet50_fpn_v2"]


def _build_faster_rcnn(num_classes: int, pretrained: bool) -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def _build_retinanet(num_classes: int, pretrained: bool) -> nn.Module:
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = retinanet_resnet50_fpn_v2(weights=weights)
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1,
    )
    return model


def build_model(arch_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch_name == "fasterrcnn_resnet50_fpn_v2":
        return _build_faster_rcnn(num_classes, pretrained)
    if arch_name == "retinanet_resnet50_fpn_v2":
        return _build_retinanet(num_classes, pretrained)
    raise ValueError(
        f"Arsitektur torchvision '{arch_name}' tidak dikenal. "
        f"Pilihan: {TORCHVISION_ARCHITECTURES}"
    )
