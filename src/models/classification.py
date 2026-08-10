"""Classification model builder. Backbones load through `timm`; the
classifier head is selected via `head_cfg` (config key `head:`, type
"linear" or "mlp")."""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn

ARCHITECTURE_ALIASES = {
    "resnet50": "resnet50",
    "resnet18": "resnet18",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "mobilenetv3": "mobilenetv3_large_100",
    "mobilenetv3_small": "mobilenetv3_small_100",
}


def resolve_timm_name(arch_name: str) -> str:
    return ARCHITECTURE_ALIASES.get(arch_name, arch_name)


class MLPHead(nn.Module):
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
        return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

    if head_type == "mlp":
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

    raise ValueError(f"Unknown head.type '{head_type}'. Options: linear, mlp")
