"""
Metrik evaluasi classification sesuai README.md bagian 5:
Accuracy, Precision/Recall/F1 (macro & weighted), confusion matrix, AUC.

Semua fungsi menerima numpy array / list biasa (bukan tensor) supaya bisa
dipakai lepas dari training loop (misal buat analysis notebook juga).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """
    y_true: (N,) label index
    y_pred: (N,) label index hasil argmax
    y_prob: (N, C) softmax probability, dipakai untuk AUC
    class_names: nama kelas urut sesuai index

    Return dict berisi:
      - scalar metrics (siap di-log ke mlflow.log_metrics), key datar
        misal "f1_macro", "auc_macro", "f1_per_class.crazing", dst.
      - "confusion_matrix": np.ndarray (C, C)
      - "classification_report": string (sklearn text report, buat artifact)
    """
    num_classes = len(class_names)
    labels_range = list(range(num_classes))

    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, average="weighted", zero_division=0
    )
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_range)

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }

    for i, cname in enumerate(class_names):
        safe_name = cname.replace(" ", "_")
        metrics[f"precision_per_class.{safe_name}"] = precision_pc[i]
        metrics[f"recall_per_class.{safe_name}"] = recall_pc[i]
        metrics[f"f1_per_class.{safe_name}"] = f1_pc[i]
        metrics[f"support_per_class.{safe_name}"] = int(support_pc[i])

    # --- AUC (ROC-AUC one-vs-rest) ---
    # Perlu >= 2 kelas dengan sample & y_prob valid. Kalau ada kelas yang
    # tidak muncul sama sekali di y_true (support=0), roc_auc_score akan
    # error -> hitung per-class secara defensif.
    #
    # Kasus num_classes == 2 (misal stage 1 Defect/Normal) ditangani TERPISAH:
    # `label_binarize(y, classes=[0,1])` sengaja balikin shape (N, 1) untuk
    # binary (konvensi sklearn - 1 kolom aja karena redundan sama negasinya),
    # BUKAN (N, 2) - kalau dipaksa lewat jalur multi-class ovr di bawah,
    # `y_true_bin[:, 1]` bakal IndexError. Buat binary, langsung pakai
    # `roc_auc_score(y_true, y_prob[:, 1])` (probabilitas kelas positif).
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            metrics["auc_macro"] = auc
            metrics["auc_weighted"] = auc
            for cname in class_names:
                # AUC biner cuma ada 1 ROC curve, nilainya sama buat kedua kelas.
                metrics[f"auc_per_class.{cname.replace(' ', '_')}"] = auc
        else:
            y_true_bin = label_binarize(y_true, classes=labels_range)
            auc_macro = roc_auc_score(
                y_true_bin, y_prob, average="macro", multi_class="ovr"
            )
            metrics["auc_macro"] = auc_macro
            auc_weighted = roc_auc_score(
                y_true_bin, y_prob, average="weighted", multi_class="ovr"
            )
            metrics["auc_weighted"] = auc_weighted

            for i, cname in enumerate(class_names):
                safe_name = cname.replace(" ", "_")
                if y_true_bin[:, i].sum() == 0 or y_true_bin[:, i].sum() == len(y_true_bin):
                    continue  # kelas tidak punya kedua sisi (positive & negative) di batch ini
                try:
                    auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                    metrics[f"auc_per_class.{safe_name}"] = auc_i
                except ValueError:
                    continue
    except ValueError:
        # Terjadi kalau cuma ada 1 kelas di y_true (edge case batch/subset kecil)
        metrics["auc_macro"] = None
        metrics["auc_weighted"] = None

    report_str = classification_report(
        y_true, y_pred, labels=labels_range, target_names=class_names, zero_division=0
    )

    return {
        "scalars": metrics,
        "confusion_matrix": cm,
        "classification_report": report_str,
    }
