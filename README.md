# Steel Surface Defect Detection — Cascade Pipeline (v3, Prototype)

Two-stage pipeline: a binary classifier (Defect / Normal) gates a multi-class detector, which only runs when a defect is found. Built on public datasets since the production camera isn't finalized yet — calibration happens once it is.

---

## 1. System Flow

```
Raw Image
   │
   ▼
[Binary Classifier] ← lightweight, fast, 2 classes: Defect / Normal
   │
   ├── Prediction: NORMAL ──────────► done, report "OK"
   │
   └── Prediction: DEFECT
              │
              ▼
      [Object Detector] ← 15 defect classes + location, runs only when needed
              │
              ▼
      Bounding boxes with defect location and class
```

---

## 2. Dataset per Stage

### Stage 1 — Binary Classification (Defect vs Normal)

All datasets used, stratified to balance Defect vs Normal.

| Source | Contribution to "Defect" | Contribution to "Normal" |
|---|---|---|
| GC10-DET | ✅ all images (defect-only dataset) | – |
| NEU-CLS | ✅ all images (defect-only dataset) | – |
| X-SDD | ✅ all images (defect-only dataset) | – |
| Severstal | optional, can add defective images if more are needed | ✅ defect-free subset (primary source for Normal) |

**Balancing:** GC10 + NEU-CLS + X-SDD gives ~5,460 Defect images (2,300+1,800+1,360); a matching count of Normal images is sampled from Severstal (stratified, not the full pool, to avoid skew). Pipeline: `scripts/prepare_severstal.py` (stages defect-free Severstal images) → `scripts/build_stage1_binary.py` (assembles, balances, splits train/val/test per source) → `configs/classification/cls_stage1_binary.yaml`.

### Stage 2 — Object Detection (15 defect classes)

| Source | Contribution |
|---|---|
| GC10-DET | 10 classes + bounding boxes |
| NEU-DET | 6 classes + bounding boxes (5 unique + 1 overlapping "inclusion") |
| Severstal (defect-free subset) | Negative samples (anti-false-positive), ~10-15% of positive image count |
| ~~X-SDD~~ | Not used — classification-only, no bounding boxes |

`configs/detection/det_combined.yaml` covers this scenario (GC10-DET + NEU-DET union = 15 classes). Build with `scripts/build_combined_dataset.py --task detection --sources gc10=... neu_det=... --negatives_dir data/raw/severstal_clean --negative_ratio 0.12`. Negatives are written as empty label files, handled by `YoloDetectionDataset` in `src/data_loader.py`.

---

## 3. Classes

**Stage 1:** 2 classes — `Defect`, `Normal`

**Stage 2:** 15 classes (union of GC10-DET and NEU-DET, "inclusion" merged):

| # | Class | Source |
|---|---|---|
| 1 | Inclusion | GC10-DET + NEU-DET (merged) |
| 2 | Crazing | NEU-DET |
| 3 | Patches | NEU-DET |
| 4 | Pitted Surface | NEU-DET |
| 5 | Rolled-in Scale | NEU-DET |
| 6 | Scratches | NEU-DET |
| 7 | Punching Hole | GC10-DET |
| 8 | Weld Line | GC10-DET |
| 9 | Crescent Gap | GC10-DET |
| 10 | Water Spot | GC10-DET |
| 11 | Oil Spot | GC10-DET |
| 12 | Silk Spot | GC10-DET |
| 13 | Waist Folding | GC10-DET |
| 14 | Crease | GC10-DET |
| 15 | Rolled Pit | GC10-DET |

---

## 4. Status: Prototype, Not Calibrated to a Real Camera

Every public steel defect dataset is close-range: defect features (fine cracks, small inclusions) need close range/high resolution to be visible and labelable, so these datasets are collected with line-scan cameras positioned close to the material.

Once the production camera is picked:
- **Close-range setup** (similar to industrial line-scan) → current data is likely representative, minor changes only.
- **Wide/distant shot** (e.g. phone photo at uncontrolled distance) → need real-camera samples for validation, plus fine-tuning/augmentation (scale jitter, copy-pasting defect patches onto a wider background) before calling this production-ready.

---

## 5. Experiments — Pretrained Architecture Choices

### Stage 1 — Binary Classification

| Architecture | Notes |
|---|---|
| MobileNetV3-Small | Primary candidate, lightest |
| EfficientNetV2-S | Balance of speed and accuracy |
| ResNet18 | Standard baseline |

### Stage 2 — Object Detection (15 classes)

| Architecture | Notes |
|---|---|
| YOLOv8 / YOLO11 (small/medium) | Primary candidates |
| RF-DETR | Transformer-based alternative |
| Faster R-CNN (ResNet50-FPN) | Classic two-stage baseline |

All four have trained checkpoints under `results/` (see section 8) — final pick is based on the metrics in section 7.

---

## 6. Trade-offs

- **Error propagation** — a stage 1 false negative (defect classified as Normal) never reaches stage 2. Recall is the critical metric at stage 1.
- **Not yet representative of the real camera scale** — see section 4, a known and tracked limitation.
- **Stage 2 class imbalance** — GC10-DET contributes 10 of 15 classes, so NEU-DET-only classes have fewer samples. Needs weighted loss / oversampling.
- **Stage 1 stratified sampling** — Defect:Normal ratio must stay balanced, and source distribution within Defect should be checked (GC10 shouldn't dominate just because it has the most images).
- **Latency budget** — stage 1 must be significantly faster than stage 2 for the cascade to pay off.

---

## 7. Evaluation Metrics

**Stage 1 (Binary Classification):**
- Recall and precision for Defect (recall prioritized — false negatives cost more than false positives)
- F1, confusion matrix
- Recall breakdown by source dataset (GC10 vs NEU-CLS vs X-SDD)
- Inference latency (ms/image)

**Stage 2 (Object Detection):**
- mAP@0.5, mAP@0.5:0.95, per-class AP
- False positive rate on clean images (from Severstal)
- Inference latency (ms/image)

**End-to-end:**
- Overall recall (raw image → box output)
- Average total latency

Run metrics (including `test_*`) are logged to Databricks Managed MLflow (`MLFLOW_EXPERIMENT_BASE_PATH` in `.env`), not committed to this repo.

---

## 8. Project Structure

> `train_classification.py` / `train_detection.py` are generic training loops reused as-is for stages 1 and 2 — only new datasets and configs were added, no restructuring.

```
steel-defect-detection/
├── data/
│   ├── raw/
│   │   ├── gc10/
│   │   ├── neu_cls/           # stage 1 (all images -> Defect label)
│   │   ├── neu_det/           # stage 2 (bounding boxes)
│   │   ├── xsdd/               # stage 1 only (cropped patches, no bounding boxes)
│   │   └── severstal_clean/   # output of scripts/prepare_severstal.py - defect-free, used in stages 1 and 2
│   ├── processed/              # output of scripts/prepare_data.py (VOC->YOLO conversion, splits)
│   └── combined/
│       ├── stage1_binary/      # output of scripts/build_stage1_binary.py (Defect vs Normal, balanced)
│       └── detection/          # output of scripts/build_combined_dataset.py (15-class union + Severstal negatives)
├── src/
│   ├── class_mapping.py        # class harmonization (including the 15-class GC10+NEU-DET union)
│   ├── data_loader.py
│   ├── models/
│   │   ├── classification.py   # timm model builder, includes resnet18/mobilenetv3_small for stage 1
│   │   └── detection.py
│   ├── utils/
│   │   ├── mlflow_utils.py
│   │   ├── quantization.py     # QAT - most relevant for stage 1 (needs to be lightweight)
│   │   └── ...
│   ├── train_classification.py # used for stage 1 (cls_stage1_binary.yaml) and the earlier research scenarios
│   └── train_detection.py      # used for stage 2 (det_combined.yaml, already 15 classes)
├── configs/
│   ├── classification/
│   │   ├── cls_stage1_binary.yaml   # <- STAGE 1
│   │   └── cls_*.yaml                # earlier research scenarios (A1-A4), kept but unused by the production flow
│   └── detection/
│       ├── det_combined.yaml   # <- STAGE 2 (GC10+NEU-DET 15-class union)
│       └── det_*.yaml           # earlier research scenarios (B1/B2)
├── scripts/
│   ├── prepare_severstal.py         # finds and stages defect-free images from the Severstal train.csv
│   ├── build_stage1_binary.py       # assembles and balances the stage 1 dataset
│   ├── build_combined_dataset.py    # assembles the stage 2 dataset (+ --negatives_dir to fold in Severstal)
│   ├── prepare_data.py
│   └── voc_to_yolo.py
├── notebooks/
├── results/                    # trained checkpoints per architecture (classification/, detection/)
├── requirements.txt
└── README.md
```

---

## 9. Roadmap

1. ✅ Download GC10-DET, NEU-CLS, NEU-DET, X-SDD, and the defect-free Severstal subset (Severstal is a Kaggle **competition** dataset — needs a Kaggle account + accepting the competition rules before `kaggle competitions download` works)
2. ✅ `scripts/prepare_severstal.py` — parses `train.csv` (both column format variants), stages defect-free images into `data/raw/severstal_clean/`
3. ✅ `scripts/build_stage1_binary.py` — balances Defect vs Normal for stage 1 (per-source split, checks source distribution within Defect)
4. ✅ `scripts/build_combined_dataset.py --negatives_dir` — folds Severstal negatives into stage 2 (~10-15% of positive image count, section 6)
5. ✅ `src/class_mapping.py` — harmonizes the 15 stage 2 classes ("inclusion" merged)
6. ✅ Train and compare stage 1 architectures — `python -m src.train_classification --config configs/classification/cls_stage1_binary.yaml`, checkpoints in `results/classification/stage1_binary/`
7. ✅ Train and compare stage 2 architectures — `python -m src.train_detection --config configs/detection/det_combined.yaml --framework all`, checkpoints in `results/detection/combined/`
8. Pick final architecture per stage (recall/mAP vs latency, section 7) and build `src/pipeline.py` — chain: stage 1 checkpoint, and if Defect, stage 2 checkpoint (not yet built)
9. Run end-to-end tests with the currently available data (still public datasets, not the real camera)
10. **Key checkpoint:** once the production camera is determined, collect real-camera samples, re-validate accuracy at that scale (section 4), fine-tune/calibrate as needed

> This repository's scope ends at the trained checkpoints (`.pt`) for stages 1 and 2, plus `pipeline.py` for local end-to-end testing. Integration into a web app or API is a separate downstream consumer, not part of this repository.

---

## 10. Source Notes

- GC10-DET: [github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets](https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets) (Baidu Pan link, code: `cdyt`)
- NEU-CLS (mirror): [kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- NEU-DET: [ieee-dataport.org/documents/neu-det](https://ieee-dataport.org/documents/neu-det)
- X-SDD: [ieee-dataport.org/documents/x-sdd](https://ieee-dataport.org/documents/x-sdd) (requires an IEEE subscription/account)
- Severstal Steel Defect Dataset: [kaggle.com/c/severstal-steel-defect-detection](https://www.kaggle.com/c/severstal-steel-defect-detection/data)
