"""
Builder untuk model classification berbasis arsitektur pretrained
(README.md bagian 3 - Task A):
    resnet50, tf_efficientnetv2_s, convnext_tiny, swin_tiny_patch4_window7_224,
    mobilenetv3_large_100

Semua lewat `timm` supaya satu API konsisten untuk semua arsitektur
(ImageNet pretrained weights, fine-tune head ke num_classes target).
"""

from __future__ import annotations

import timm
import torch.nn as nn

# Nama arsitektur "ramah manusia" (dipakai di config yaml) -> nama model timm.
ARCHITECTURE_ALIASES = {
    "resnet50": "resnet50",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "mobilenetv3": "mobilenetv3_large_100",
}


def resolve_timm_name(arch_name: str) -> str:
    return ARCHITECTURE_ALIASES.get(arch_name, arch_name)


def build_model(arch_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    timm_name = resolve_timm_name(arch_name)
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model
