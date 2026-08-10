"""
Quantization-aware training (QAT) for classification models, via FX Graph
Mode Quantization. Supported only for architectures that FX-trace cleanly:
resnet50, resnet18, efficientnetv2_s, mobilenetv3, mobilenetv3_small.
convnext_tiny and swin_tiny are not supported; requesting QAT for them
falls back to plain fp32 training (see train_classification.py).
"""

from __future__ import annotations

import io

import torch
import torch.nn as nn
from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

QAT_SUPPORTED_ARCHITECTURES = [
    "resnet50", "resnet18", "efficientnetv2_s", "mobilenetv3", "mobilenetv3_small",
]


def is_qat_supported(arch_name: str) -> bool:
    return arch_name in QAT_SUPPORTED_ARCHITECTURES


def prepare_qat_model(
    model: nn.Module, example_input: torch.Tensor, backend: str = "qnnpack"
) -> nn.Module:
    torch.backends.quantized.engine = backend
    model_cpu = model.cpu().train()
    qconfig_mapping = QConfigMapping().set_global(get_default_qat_qconfig(backend))
    prepared = prepare_qat_fx(model_cpu, qconfig_mapping, (example_input.cpu(),))
    return prepared


def convert_to_quantized(prepared_model: nn.Module) -> nn.Module:
    prepared_model = prepared_model.cpu().eval()
    return convert_fx(prepared_model)


def get_state_dict_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)
