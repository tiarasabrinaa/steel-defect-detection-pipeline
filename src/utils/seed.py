"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set seed untuk python/numpy/torch (CPU+CUDA) sekaligus flag cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Pilih device tercepat yang ada: CUDA (Nvidia) > MPS (Apple Silicon) > CPU.

    MPS sering kelewat kalau cuma cek `torch.cuda.is_available()` (device
    selection paling umum di banyak contoh kode PyTorch, tapi itu cuma cover
    Nvidia) - di Mac M-series ini bikin training diam-diam jalan di CPU
    padahal GPU-nya nganggur, jauh lebih lambat tanpa ada warning apapun."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
