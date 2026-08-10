"""Object detection training pipeline. --framework selects ultralytics
(YOLOv8/YOLO11/RT-DETR) or torchvision (Faster R-CNN/RetinaNet).

Usage:
    python -m src.train_detection --config configs/detection/det_gc10.yaml --framework all
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from torch.optim import SGD
from tqdm import tqdm

from src.data_loader import build_ultralytics_data_yaml, get_detection_dataloaders
from src.models.detection import build_model as build_torchvision_model
from src.utils import mlflow_utils
from src.utils.metrics_detection import build_map_metric, compute_detection_metrics, strip_background
from src.utils.seed import get_device, set_seed

from ultralytics import RTDETR, YOLO
from ultralytics import settings as ultralytics_settings

ultralytics_settings.update({"mlflow": False})  # logging is handled manually below

RFDETR_MODEL_CLASSES = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}  # rfdetr is imported lazily inside train_rfdetr_architecture (heavy import: torch, transformers, lightning)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_ultralytics_architecture(arch_name: str, config: dict) -> None:
    ds_cfg = config["dataset"]
    class_names = ds_cfg["class_names"]
    ultra_cfg = config.get("frameworks", {}).get("ultralytics", {})

    output_dir = Path(config.get("output_dir", "results/detection")) / arch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_yaml_path = build_ultralytics_data_yaml(config, output_dir / "data.yaml")

    run_name = f"{ds_cfg['name']}_{arch_name}"
    with mlflow_utils.start_run(
        run_name=run_name,
        tags={
            "task": "detection", "architecture": arch_name,
            "dataset": ds_cfg["name"], "framework": "ultralytics",
        },
    ):
        run_config = copy.deepcopy(config)
        run_config["architecture"] = arch_name
        run_config["framework"] = "ultralytics"
        mlflow_utils.log_config_params(run_config)

        model_cls = RTDETR if arch_name.startswith("rtdetr") else YOLO
        model = model_cls(f"{arch_name}.pt")

        train_kwargs = dict(
            data=str(data_yaml_path),
            epochs=config.get("epochs", 100),
            imgsz=config.get("img_size", 640),
            batch=config.get("batch_size", 16),
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
            seed=config.get("seed", 42),
            plots=True,
            verbose=False,
        )
        if ultra_cfg.get("lr0") is not None:
            train_kwargs["lr0"] = ultra_cfg["lr0"]
        if ultra_cfg.get("optimizer"):
            train_kwargs["optimizer"] = ultra_cfg["optimizer"]

        model.train(**train_kwargs)
        save_dir = Path(model.trainer.save_dir)

        test_metrics = model.val(data=str(data_yaml_path), split="test", imgsz=config.get("img_size", 640))
        scalars = {
            "test_mAP_50": float(test_metrics.box.map50),
            "test_mAP_50_95": float(test_metrics.box.map),
            "test_mAP_75": float(test_metrics.box.map75),
            "test_precision": float(test_metrics.box.mp),
            "test_recall": float(test_metrics.box.mr),
        }
        ap_per_class = getattr(test_metrics.box, "maps", None)
        if ap_per_class is not None:
            for i, cname in enumerate(class_names):
                if i < len(ap_per_class):
                    safe = cname.replace(" ", "_")
                    scalars[f"test_AP50_95_per_class.{safe}"] = float(ap_per_class[i])
        mlflow_utils.log_metrics(scalars)

        results_csv = save_dir / "results.csv"
        if results_csv.exists():
            import pandas as pd

            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                step = int(row.get("epoch", 0))
                metric_row = {
                    k: v for k, v in row.items()
                    if k != "epoch" and isinstance(v, (int, float))
                }
                mlflow_utils.log_metrics(metric_row, step=step)

        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.exists():
            mlflow_utils.log_pt_checkpoint(best_pt, artifact_path="checkpoints")

        for fname in [
            "results.csv", "confusion_matrix.png", "confusion_matrix_normalized.png",
            "results.png", "PR_curve.png", "F1_curve.png",
            "val_batch0_pred.jpg", "val_batch0_labels.jpg",
        ]:
            fpath = save_dir / fname
            if fpath.exists():
                mlflow_utils.log_artifact_file(fpath, artifact_path="plots")

        print(f"[{arch_name}] done. test mAP50-95={scalars['test_mAP_50_95']:.4f}")


def train_rfdetr_architecture(arch_name: str, config: dict) -> None:
    import rfdetr

    ds_cfg = config["dataset"]
    class_names = ds_cfg["class_names"]
    rfdetr_cfg = config.get("frameworks", {}).get("rfdetr", {})

    output_dir = Path(config.get("output_dir", "results/detection")) / arch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = rfdetr_cfg.get("dataset_dir", "data/combined/detection_coco")

    run_name = f"{ds_cfg['name']}_{arch_name}"
    with mlflow_utils.start_run(
        run_name=run_name,
        tags={
            "task": "detection", "architecture": arch_name,
            "dataset": ds_cfg["name"], "framework": "rfdetr",
        },
    ):
        run_config = copy.deepcopy(config)
        run_config["architecture"] = arch_name
        run_config["framework"] = "rfdetr"
        mlflow_utils.log_config_params(run_config)

        model_cls_name = RFDETR_MODEL_CLASSES[arch_name]
        model = getattr(rfdetr, model_cls_name)()

        model.train(
            dataset_dir=str(dataset_dir),
            dataset_file="roboflow",  # our train/valid/test + _annotations.coco.json layout, not COCO2017's
            output_dir=str(output_dir),
            epochs=config.get("epochs", 100),
            batch_size=rfdetr_cfg.get("batch_size", 4),
            grad_accum_steps=rfdetr_cfg.get("grad_accum_steps", 4),
            lr=rfdetr_cfg.get("lr", 1e-4),
            class_names=class_names,
            run_test=True,
            early_stopping=True,
            early_stopping_patience=rfdetr_cfg.get("early_stopping_patience", 10),
            seed=config.get("seed", 42),
            tensorboard=False,
            wandb=False,
            mlflow=False,  # our own MLflow run above keeps the metric schema consistent across frameworks
            project=config["experiment_name"],
            run=run_name,
        )

        scalars: dict[str, float] = {}
        metrics_csv = output_dir / "metrics.csv"
        if metrics_csv.exists():
            import pandas as pd

            df = pd.read_csv(metrics_csv)
            for _, row in df.iterrows():
                step = int(row["epoch"]) if "epoch" in row and not pd.isna(row.get("epoch")) else None
                metric_row = {
                    k: v for k, v in row.items()
                    if k not in ("epoch", "step") and isinstance(v, (int, float)) and not pd.isna(v)
                }
                if metric_row:
                    mlflow_utils.log_metrics(metric_row, step=step)

            test_metric_keys = {
                "test/mAP_50_95": "test_mAP_50_95", "test/mAP_50": "test_mAP_50",
                "test/mAP_75": "test_mAP_75", "test/mAR": "test_mAR",
                "test/F1": "test_F1", "test/precision": "test_precision", "test/recall": "test_recall",
            }
            for col, out_key in test_metric_keys.items():
                if col in df.columns:
                    series = df[col].dropna()
                    if len(series):
                        scalars[out_key] = float(series.iloc[-1])

            for cname in class_names:
                col = f"test/AP/{cname}"
                if col in df.columns:
                    series = df[col].dropna()
                    if len(series):
                        safe = cname.replace(" ", "_")
                        scalars[f"test_AP50_95_per_class.{safe}"] = float(series.iloc[-1])

            if scalars:
                mlflow_utils.log_metrics(scalars)

        best_ckpt = output_dir / "checkpoint_best_total.pth"
        if best_ckpt.exists():
            mlflow_utils.log_pt_checkpoint(best_ckpt, artifact_path="checkpoints")

        print(f"[{arch_name}] done. test mAP50-95={scalars.get('test_mAP_50_95', float('nan')):.4f}")


def train_one_epoch_torchvision(model, loader, optimizer, device) -> float:
    model.train()
    running_loss, n_samples = 0.0, 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(images)
        n_samples += len(images)
    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_torchvision(model, loader, device, class_names: list[str]) -> dict:
    model.eval()
    metric = build_map_metric()
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = [img.to(device) for img in images]
        predictions = model(images)

        preds_batch, targets_batch = [], []
        for pred, target in zip(predictions, targets):
            boxes, labels, scores = strip_background(
                pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu()
            )
            preds_batch.append({"boxes": boxes, "labels": labels, "scores": scores})

            gt_boxes, gt_labels = strip_background(target["boxes"], target["labels"])
            targets_batch.append({"boxes": gt_boxes, "labels": gt_labels})

        metric.update(preds_batch, targets_batch)

    scalars = compute_detection_metrics(metric, class_names)
    return {"scalars": scalars}


@torch.no_grad()
def save_qualitative_predictions(
    model, dataset, device, class_names: list[str], out_dir: Path, n: int = 6, score_thresh: float = 0.5
) -> Path | None:
    """Render predictions vs ground truth for a handful of test samples."""
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, len(dataset))
    if n == 0:
        return None

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    thumb = 320
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")

    for i in range(n):
        image_t, target = dataset[i]
        pred = model([image_t.to(device)])[0]

        img_np = (image_t.permute(1, 2, 0).cpu().numpy() * 0.229 + 0.485).clip(0, 1)
        img = Image.fromarray((img_np * 255).astype("uint8")).resize((thumb, thumb))
        draw = ImageDraw.Draw(img)

        gt_boxes, gt_labels = strip_background(target["boxes"], target["labels"])
        scale = thumb / image_t.shape[-1]
        for box, label in zip(gt_boxes.tolist(), gt_labels.tolist()):
            box = [c * scale for c in box]
            draw.rectangle(box, outline="lime", width=2)

        p_boxes, p_labels, p_scores = strip_background(
            pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu()
        )
        for box, label, score in zip(p_boxes.tolist(), p_labels.tolist(), p_scores.tolist()):
            if score < score_thresh:
                continue
            box = [c * scale for c in box]
            draw.rectangle(box, outline="red", width=2)
            cname = class_names[label] if 0 <= label < len(class_names) else str(label)
            draw.text((box[0], max(0, box[1] - 10)), f"{cname}:{score:.2f}", fill="red")

        r, c = divmod(i, cols)
        sheet.paste(img, (c * thumb, r * thumb))

    out_path = out_dir / "test_qualitative_predictions.png"
    sheet.save(out_path)
    return out_path


def train_torchvision_architecture(arch_name: str, config: dict, device: torch.device) -> None:
    ds_cfg = config["dataset"]
    class_names = ds_cfg["class_names"]
    num_classes = ds_cfg["num_classes"]

    data = get_detection_dataloaders(config)
    train_loader, val_loader, test_loader = (
        data["train_loader"], data["val_loader"], data["test_loader"]
    )

    tv_cfg = config.get("frameworks", {}).get("torchvision", {}).get("optimizer", {})

    run_name = f"{ds_cfg['name']}_{arch_name}"
    with mlflow_utils.start_run(
        run_name=run_name,
        tags={
            "task": "detection", "architecture": arch_name,
            "dataset": ds_cfg["name"], "framework": "torchvision",
        },
    ):
        run_config = copy.deepcopy(config)
        run_config["architecture"] = arch_name
        run_config["framework"] = "torchvision"
        mlflow_utils.log_config_params(run_config)

        model = build_torchvision_model(arch_name, num_classes, pretrained=True).to(device)

        optimizer = SGD(
            model.parameters(),
            lr=tv_cfg.get("lr", 0.005),
            momentum=tv_cfg.get("momentum", 0.9),
            weight_decay=tv_cfg.get("weight_decay", 5e-4),
        )
        epochs = config.get("epochs", 50)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(epochs // 3, 1), gamma=0.1
        )

        output_dir = Path(config.get("output_dir", "results/detection")) / arch_name
        output_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = output_dir / "best.pt"

        best_score = -float("inf")
        best_state_dict = None

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch_torchvision(model, train_loader, optimizer, device)
            val_result = evaluate_torchvision(model, val_loader, device, class_names)
            val_scalars = {f"val_{k}": v for k, v in val_result["scalars"].items()}
            scheduler.step()

            mlflow_utils.log_metrics(
                {"train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"], **val_scalars},
                step=epoch,
            )

            current_score = val_result["scalars"]["mAP_50_95"]
            print(f"[{arch_name}] epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_mAP50-95={current_score:.4f}")

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
                        "val_mAP_50_95": best_score,
                    },
                    best_ckpt_path,
                )

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        test_result = evaluate_torchvision(model, test_loader, device, class_names)
        test_scalars = {f"test_{k}": v for k, v in test_result["scalars"].items()}
        mlflow_utils.log_metrics(test_scalars)
        mlflow_utils.log_metrics({"best_val_mAP_50_95": best_score})

        mlflow_utils.log_pt_checkpoint(best_ckpt_path, artifact_path="checkpoints")

        qual_path = save_qualitative_predictions(
            model, data["test_dataset"], device, class_names, output_dir
        )
        if qual_path is not None:
            mlflow_utils.log_artifact_file(qual_path, artifact_path="plots")

        import mlflow.pytorch

        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"[{arch_name}] done. best val_mAP50-95={best_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Object detection training (Task B)")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument(
        "--framework", choices=["ultralytics", "torchvision", "rfdetr", "all"], default="all",
    )
    parser.add_argument(
        "--architectures", nargs="*", default=None,
        help="Override the architecture list from the config (optional)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"Device: {device}")

    mlflow_utils.init_mlflow(config["experiment_name"])

    frameworks_cfg = config.get("frameworks", {})

    if args.framework in ("ultralytics", "all") and "ultralytics" in frameworks_cfg:
        archs = args.architectures or frameworks_cfg["ultralytics"].get("architectures", [])
        for arch_name in archs:
            print(f"\n=== [ultralytics] Training architecture: {arch_name} ===")
            train_ultralytics_architecture(arch_name, config)

    if args.framework in ("torchvision", "all") and "torchvision" in frameworks_cfg:
        archs = args.architectures or frameworks_cfg["torchvision"].get("architectures", [])
        for arch_name in archs:
            print(f"\n=== [torchvision] Training architecture: {arch_name} ===")
            train_torchvision_architecture(arch_name, config, device)

    if args.framework in ("rfdetr", "all") and "rfdetr" in frameworks_cfg:
        archs = args.architectures or frameworks_cfg["rfdetr"].get("architectures", [])
        for arch_name in archs:
            print(f"\n=== [rfdetr] Training architecture: {arch_name} ===")
            train_rfdetr_architecture(arch_name, config)


if __name__ == "__main__":
    main()
