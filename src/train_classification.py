"""
Training pipeline classification (Task A, README.md bagian 3).

Jalankan (dari root project):
    python -m src.train_classification --config configs/classification/cls_gc10.yaml

Per arsitektur di config `architectures: [...]` dilatih sebagai MLflow run
terpisah (semua di bawah experiment yang sama), supaya gampang dibandingkan
di MLflow UI / Databricks. Setiap run:
  - log SEMUA hyperparameter dari config yaml
  - log metric val per epoch: accuracy, precision/recall/f1 (macro &
    weighted + per-class), AUC (macro & per-class)
  - simpan best checkpoint (.pt) berdasarkan val f1_macro -> upload ke MLflow
  - di akhir training, evaluasi ke test set, log confusion matrix +
    classification report sebagai artifact
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data_loader import get_classification_dataloaders
from src.models.classification import build_model
from src.utils import mlflow_utils
from src.utils.class_weights import compute_class_weights
from src.utils.metrics_classification import compute_classification_metrics
from src.utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model: nn.Module, opt_cfg: dict) -> torch.optim.Optimizer:
    name = opt_cfg.get("name", "adamw").lower()
    lr = opt_cfg.get("lr", 3e-4)
    weight_decay = opt_cfg.get("weight_decay", 1e-5)
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(
            model.parameters(), lr=lr, momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Optimizer '{name}' tidak didukung")


@torch.no_grad()
def evaluate(
    model: nn.Module, loader, device: torch.device, class_names: list[str]
) -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    return compute_classification_metrics(y_true, y_pred, y_prob, class_names)


def train_one_epoch(
    model: nn.Module, loader, optimizer, criterion, device: torch.device
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
    return running_loss / max(n_samples, 1)


def train_one_architecture(arch_name: str, config: dict, device: torch.device) -> None:
    ds_cfg = config["dataset"]
    class_names = ds_cfg["class_names"]
    num_classes = ds_cfg["num_classes"]
    assert len(class_names) == num_classes, "class_names dan num_classes tidak konsisten"

    data = get_classification_dataloaders(config)
    train_loader, val_loader, test_loader = (
        data["train_loader"], data["val_loader"], data["test_loader"]
    )

    run_name = f"{ds_cfg['name']}_{arch_name}"
    with mlflow_utils.start_run(run_name=run_name, tags={"task": "classification", "architecture": arch_name, "dataset": ds_cfg["name"]}):
        run_config = copy.deepcopy(config)
        run_config["architecture"] = arch_name
        mlflow_utils.log_config_params(run_config)

        model = build_model(arch_name, num_classes, pretrained=True).to(device)

        loss_cfg = config.get("loss", {})
        weight = None
        if loss_cfg.get("weighted", False):
            weight = compute_class_weights(
                data["train_dataset"].labels, num_classes
            ).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=loss_cfg.get("label_smoothing", 0.0)
        )

        optimizer = build_optimizer(model, config.get("optimizer", {}))
        epochs = config.get("epochs", 50)
        scheduler_cfg = config.get("scheduler", {})
        scheduler = None
        if scheduler_cfg.get("name") == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        early_stop_cfg = config.get("early_stopping", {})
        patience = early_stop_cfg.get("patience", epochs)
        monitor_metric = early_stop_cfg.get("metric", "f1_macro")

        best_score = -float("inf")
        best_state_dict = None
        epochs_without_improve = 0

        output_dir = Path(config.get("output_dir", "results/classification")) / arch_name
        output_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = output_dir / "best.pt"

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_result = evaluate(model, val_loader, device, class_names)
            val_scalars = {f"val_{k}": v for k, v in val_result["scalars"].items()}

            if scheduler is not None:
                scheduler.step()

            epoch_metrics = {
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": time.time() - t0,
                **val_scalars,
            }
            mlflow_utils.log_metrics(epoch_metrics, step=epoch)

            current_score = val_result["scalars"].get(monitor_metric, val_result["scalars"]["f1_macro"])
            print(
                f"[{arch_name}] epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} val_{monitor_metric}={current_score:.4f}"
            )

            if current_score > best_score:
                best_score = current_score
                best_state_dict = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "arch_name": arch_name,
                        "num_classes": num_classes,
                        "class_names": class_names,
                        "epoch": epoch,
                        "state_dict": best_state_dict,
                        f"val_{monitor_metric}": best_score,
                    },
                    best_ckpt_path,
                )
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= patience:
                print(f"[{arch_name}] early stopping di epoch {epoch} (patience={patience})")
                break

        # load bobot terbaik sebelum evaluasi final & logging model
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        test_result = evaluate(model, test_loader, device, class_names)
        test_scalars = {f"test_{k}": v for k, v in test_result["scalars"].items()}
        mlflow_utils.log_metrics(test_scalars)
        mlflow_utils.log_metrics({"best_val_" + monitor_metric: best_score})

        mlflow_utils.log_confusion_matrix(
            test_result["confusion_matrix"], class_names, "test_confusion_matrix.png"
        )
        mlflow_utils.log_text_artifact(
            test_result["classification_report"], "test_classification_report.txt"
        )

        # best checkpoint (.pt) sebagai artifact MLflow
        mlflow_utils.log_pt_checkpoint(best_ckpt_path, artifact_path="checkpoints")

        # full model juga di-log lewat mlflow.pytorch supaya bisa langsung
        # di-register / di-serve dari Databricks Model Registry.
        import mlflow.pytorch

        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"[{arch_name}] selesai. best val_{monitor_metric}={best_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Training classification (Task A)")
    parser.add_argument("--config", required=True, help="Path ke config yaml")
    parser.add_argument(
        "--architectures", nargs="*", default=None,
        help="Override daftar arsitektur dari config (opsional)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mlflow_utils.init_mlflow(config["experiment_name"])

    architectures = args.architectures or config["architectures"]
    for arch_name in architectures:
        print(f"\n=== Training arsitektur: {arch_name} ===")
        train_one_architecture(arch_name, config, device)


if __name__ == "__main__":
    main()
