"""
Quantization-aware training (QAT) untuk model classification.

Kenapa QAT, bukan cuma post-training quantization (PTQ)? MobileNetV3-Small
dan ResNet18 secara eksplisit dikandidatkan sebagai stage-1 gate classifier
di README (harus ringan & cepat, jalan di setiap gambar sebelum keputusan
lanjut ke detector atau tidak) - kandidat kuat buat deploy edge. Defect
permukaan baja (crazing, pitted_surface, rolled-in_scale, dst.) itu soal
tekstur & edge halus - beda dengan foto natural (ImageNet) yang jadi basis
kalibrasi kebanyakan tooling quantization. Aktivasi di sekitar tekstur
halus itu lebih sensitif ke noise int8, jadi PTQ naif (quantize SETELAH
training selesai, model tidak pernah "lihat" noise itu) beresiko akurasi
drop lebih besar untuk kasus kita dibanding kasus umum. QAT mensimulasikan
noise kuantisasi SELAMA training (fake-quant di forward pass) supaya bobot
sempat beradaptasi duluan, baru di-convert jadi int8 asli di akhir.

Scope: FX Graph Mode Quantization (`torch.ao.quantization.quantize_fx`),
API resmi PyTorch pengganti eager-mode quantization yang sudah deprecated.
Cuma didukung untuk arsitektur CNN yang FX-traceable dengan bersih:
resnet50, resnet18, efficientnetv2_s, mobilenetv3, mobilenetv3_small.
convnext_tiny & swin_tiny TIDAK didukung - keduanya punya control-flow
dinamis (window attention di Swin, banyak functional reshape/permute) yang
bikin `torch.fx.symbolic_trace` gagal atau butuh custom tracer; di luar
scope project ini. Kalau QAT diminta untuk arsitektur yang tidak didukung,
training tetap jalan normal di fp32 (lihat train_classification.py) -
tidak meng-crash run.

Backend quantized kernel (config yaml `quantization.backend`):
  - "qnnpack" -> optimized untuk ARM (Raspberry Pi / edge device)
  - "fbgemm"  -> optimized untuk x86 server
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
    """
    Insert observer + fake-quant modules (FX graph mode) ke `model`, supaya
    training loop berikutnya mensimulasikan noise kuantisasi di forward pass.

    `example_input` dipakai FX buat symbolic trace (butuh 1 batch nyata,
    cukup 1 sample: `images[:1]`). Selalu dijalankan di CPU (persyaratan FX
    prepare) - hasilnya boleh dipindah balik ke GPU untuk lanjut training.
    """
    torch.backends.quantized.engine = backend
    model_cpu = model.cpu().train()
    qconfig_mapping = QConfigMapping().set_global(get_default_qat_qconfig(backend))
    prepared = prepare_qat_fx(model_cpu, qconfig_mapping, (example_input.cpu(),))
    return prepared


def convert_to_quantized(prepared_model: nn.Module) -> nn.Module:
    """
    Convert model hasil QAT training (fake-quant, float weights) jadi int8
    asli. Quantized kernel PyTorch saat ini CPU-only, jadi hasilnya SELALU
    di CPU terlepas dari device training sebelumnya.
    """
    prepared_model = prepared_model.cpu().eval()
    return convert_fx(prepared_model)


def get_state_dict_size_mb(model: nn.Module) -> float:
    """Ukuran state_dict model in-memory (MB) - dipakai buat bandingkan
    fp32 vs int8, metric yang relevan untuk keputusan deploy edge."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)
