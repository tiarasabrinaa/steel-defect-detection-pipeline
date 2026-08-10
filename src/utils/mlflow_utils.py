"""
Thin wrapper around MLflow so all training scripts log consistently, and so
Databricks credentials live only in environment variables, never hardcoded.

Databricks setup (see README.md, "MLflow + Databricks"):
    export MLFLOW_TRACKING_URI=databricks
    export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
    export DATABRICKS_TOKEN=<personal access token>
    export MLFLOW_EXPERIMENT_BASE_PATH=/Users/<email>/steel-defect-detection

If these are unset, MLflow falls back to local tracking (./mlruns).
"""

from __future__ import annotations

import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv

load_dotenv()

_MAX_PARAMS_PER_BATCH = 100  # mlflow.log_params limit per call


def init_mlflow(experiment_name: str) -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    base_path = os.environ.get("MLFLOW_EXPERIMENT_BASE_PATH", "").rstrip("/")
    if tracking_uri == "databricks" and base_path:
        full_experiment_name = f"{base_path}/{experiment_name}"
    else:
        full_experiment_name = experiment_name

    mlflow.set_experiment(full_experiment_name)
    return full_experiment_name


@contextmanager
def start_run(run_name: str, tags: dict[str, Any] | None = None) -> Iterator[Any]:
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def _flatten(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key))
        elif isinstance(v, (list, tuple)):
            items[new_key] = json.dumps(list(v))
        else:
            items[new_key] = v
    return items


def log_config_params(config: dict[str, Any]) -> None:
    flat = _flatten(config)
    keys = list(flat.keys())
    for i in range(0, len(keys), _MAX_PARAMS_PER_BATCH):
        batch_keys = keys[i : i + _MAX_PARAMS_PER_BATCH]
        batch = {k: str(flat[k]) for k in batch_keys}
        mlflow.log_params(batch)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    clean = {}
    for k, v in metrics.items():
        if v is None:
            continue
        try:
            clean[k] = float(v)
        except (TypeError, ValueError):
            continue
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_confusion_matrix(
    cm,
    class_names: list[str],
    artifact_file: str = "confusion_matrix.png",
    normalize: bool = True,
) -> None:
    import numpy as np

    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = ".0f"

    n = len(class_names)
    fig_size = max(6, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_display, cmap="Blues", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    thresh = cm_display.max() / 2.0 if cm_display.size else 0
    for i in range(n):
        for j in range(n):
            val = cm_display[i, j]
            ax.text(
                j, i, format(val, fmt),
                ha="center", va="center", fontsize=6,
                color="white" if val > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    mlflow.log_figure(fig, artifact_file)
    plt.close(fig)


def log_text_artifact(text: str, artifact_file: str) -> None:
    mlflow.log_text(text, artifact_file)


def log_dict_artifact(data: dict[str, Any], artifact_file: str) -> None:
    mlflow.log_dict(data, artifact_file)


def log_pt_checkpoint(local_path: str | Path, artifact_path: str = "checkpoints") -> None:
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)


def log_artifact_file(local_path: str | Path, artifact_path: str) -> None:
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
