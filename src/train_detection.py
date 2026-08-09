"""
Training pipeline object detection (Task B, README.md bagian 3).

Dua jalur/framework berbeda, dipilih lewat `--framework`:

  ultralytics  -> YOLOv8 / YOLO11 / RT-DETR (fine-tune dari COCO weights),
                  training loop & mAP dihitung oleh package `ultralytics`,
                  kita ambil hasilnya lalu log manual ke MLflow.
  torchvision  -> Faster R-CNN (two-stage) / RetinaNet (one-stage),
                  training loop ditulis manual + mAP dihitung pakai
                  torchmetrics (MeanAveragePrecision).

Kedua jalur logging ke MLflow dengan skema metric yang konsisten:
mAP_50, mAP_50_95, mAP_75, per-class AP, plus best.pt checkpoint.

Jalankan (dari root project):
    python -m src.train_detection --config configs/detection/det_gc10.yaml --framework ultralytics
    python -m src.train_detection --config configs/detection/det_gc10.yaml --framework torchvision
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
from src.utils.seed import set_seed

# Matikan integrasi MLflow bawaan ultralytics: kita mau logging manual yang
# konsisten dengan jalur torchvision, bukan dua run terpisah yang bentrok.
from ultralytics import RTDETR, YOLO
from ultralytics import settings as ultralytics_settings

ultralytics_settings.update({"mlflow": False})


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Jalur Ultralytics: YOLOv8 / YOLO11 / RT-DETR
# ---------------------------------------------------------------------------


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
        model = model_cls(f"{arch_name}.pt")  # pretrained COCO weights, auto-download

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

        # Evaluasi final di test split
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

        # Kurva training per-epoch dari results.csv -> log_metrics per step
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

        # Artifacts: best.pt + plot bawaan ultralytics (PR curve, confusion
        # matrix, contoh prediksi vs GT untuk qualitative check)
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

        print(f"[{arch_name}] selesai. test mAP50-95={scalars['test_mAP_50_95']:.4f}")


# ---------------------------------------------------------------------------
# Jalur torchvision: Faster R-CNN / RetinaNet
# ---------------------------------------------------------------------------


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
    """Visualisasi prediksi vs ground truth untuk beberapa sample test (README.md bagian 5)."""
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

        print(f"[{arch_name}] selesai. best val_mAP50-95={best_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Training object detection (Task B)")
    parser.add_argument("--config", required=True, help="Path ke config yaml")
    parser.add_argument(
        "--framework", choices=["ultralytics", "torchvision", "all"], default="all",
    )
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

    frameworks_cfg = config.get("frameworks", {})

    if args.framework in ("ultralytics", "all") and "ultralytics" in frameworks_cfg:
        archs = args.architectures or frameworks_cfg["ultralytics"].get("architectures", [])
        for arch_name in archs:
            print(f"\n=== [ultralytics] Training arsitektur: {arch_name} ===")
            train_ultralytics_architecture(arch_name, config)

    if args.framework in ("torchvision", "all") and "torchvision" in frameworks_cfg:
        archs = args.architectures or frameworks_cfg["torchvision"].get("architectures", [])
        for arch_name in archs:
            print(f"\n=== [torchvision] Training arsitektur: {arch_name} ===")
            train_torchvision_architecture(arch_name, config, device)


if __name__ == "__main__":
    main()
