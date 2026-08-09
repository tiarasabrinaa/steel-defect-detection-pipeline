"""
Builder untuk model classification berbasis arsitektur pretrained
(README.md bagian 3 - Task A):
    resnet50, tf_efficientnetv2_s, convnext_tiny, swin_tiny_patch4_window7_224,
    mobilenetv3_large_100

Semua backbone lewat `timm` supaya satu API konsisten untuk semua arsitektur
(ImageNet pretrained weights). Classifier HEAD di atas backbone bisa dipilih
lewat `head_cfg` (dari config yaml, key `head:`):

  head:
    type: linear      # default timm: 1 Linear(in_features -> num_classes)
  # atau
  head:
    type: mlp
    hidden_dim: 512
    dropout: 0.3
    use_batchnorm: true

Kenapa perlu opsi "mlp" (bukan cuma linear standar)?
Dataset di project ini KECIL dibanding benchmark ImageNet biasa dipakai buat
transfer learning (NEU-CLS: 300 gambar/kelas, X-SDD: ~194 gambar/kelas rata2,
combined 20-kelas: imbalance parah karena kelas hasil merge - Inclusion,
Scratches - punya sampel 2-3x lebih banyak dari kelas unik per-dataset seperti
Waist Folding). Satu Linear layer langsung di atas pooled feature berdimensi
tinggi (2048 utk resnet50, 1280 utk convnext_tiny/effnetv2-s) gampang overfit
di regime data sekecil ini. Head "mlp" (bottleneck Linear -> BatchNorm -> GELU
-> Dropout -> Linear) dipakai khusus di skenario dataset kecil/imbalance (NEU,
X-SDD, combined) sebagai regularizer murah; skenario GC10 (dataset relatif
besar & balanced, ~2300 gambar utk 10 kelas) tetap pakai "linear" karena tidak
butuh proteksi ekstra itu. Lihat comment di tiap configs/classification/*.yaml
untuk alasan spesifik per skenario.
"""

from __future__ import annotations

from typing import Any

import timm
import torch
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


class MLPHead(nn.Module):
    """Bottleneck MLP head: Linear -> (BatchNorm) -> GELU -> Dropout -> Linear.

    Dipakai sebagai pengganti Linear head standar saat dataset kecil / kelas
    imbalance (lihat docstring modul ini)."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_features, hidden_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierWithHead(nn.Module):
    """Backbone (pooled feature extractor, timm num_classes=0) + custom head."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def build_model(
    arch_name: str,
    num_classes: int,
    pretrained: bool = True,
    head_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    timm_name = resolve_timm_name(arch_name)
    head_cfg = head_cfg or {"type": "linear"}
    head_type = head_cfg.get("type", "linear")

    if head_type == "linear":
        # timm pasang Linear(in_features, num_classes) standar sebagai head.
        return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

    if head_type == "mlp":
        # num_classes=0 -> timm cuma balikin pooled feature vector (backbone
        # tanpa classifier head), supaya head custom kita yang pasang di atasnya.
        backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        in_features = backbone.num_features
        head = MLPHead(
            in_features,
            num_classes,
            hidden_dim=head_cfg.get("hidden_dim", 512),
            dropout=head_cfg.get("dropout", 0.3),
            use_batchnorm=head_cfg.get("use_batchnorm", True),
        )
        return ClassifierWithHead(backbone, head)

    raise ValueError(f"head.type '{head_type}' tidak dikenal. Pilihan: linear, mlp")
