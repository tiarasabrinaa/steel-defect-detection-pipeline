"""Class-imbalance helpers (wajib dipakai di skenario gabungan A4/B4, lihat
README.md bagian 4 - Trade-off)."""

from __future__ import annotations

from collections import Counter

import torch


def compute_class_counts(labels: list[int], num_classes: int) -> list[int]:
    counts = Counter(labels)
    return [counts.get(i, 0) for i in range(num_classes)]


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weight, dinormalisasi supaya rata-rata weight = 1.
    Dipakai sebagai `weight=` di CrossEntropyLoss (classification) atau untuk
    weighted sampling.
    """
    counts = compute_class_counts(labels, num_classes)
    total = sum(counts)
    weights = [
        total / (num_classes * c) if c > 0 else 0.0
        for c in counts
    ]
    weights_t = torch.tensor(weights, dtype=torch.float32)
    nonzero = weights_t[weights_t > 0]
    if len(nonzero) > 0:
        weights_t = weights_t / nonzero.mean()
    return weights_t


def compute_sample_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Per-sample weight (dipakai oleh WeightedRandomSampler untuk oversampling)."""
    class_weights = compute_class_weights(labels, num_classes)
    return torch.tensor([class_weights[label] for label in labels], dtype=torch.double)
