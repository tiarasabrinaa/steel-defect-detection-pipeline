"""Classification evaluation metrics: accuracy, precision/recall/F1, confusion matrix, AUC."""

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

    # label_binarize returns shape (N, 1) for exactly 2 classes, not (N, 2),
    # so binary AUC needs a separate path from the multi-class OVR one.
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            metrics["auc_macro"] = auc
            metrics["auc_weighted"] = auc
            for cname in class_names:
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
                    continue
                try:
                    auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                    metrics[f"auc_per_class.{safe_name}"] = auc_i
                except ValueError:
                    continue
    except ValueError:
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
