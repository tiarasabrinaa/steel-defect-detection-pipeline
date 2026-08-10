# Steel Surface Defect — Cascade Binary Classification + Detection Pipeline (v3, Prototype Phase)

A two-stage pipeline: a fast binary classifier (Defect vs Normal) acts as an initial gate, followed by multi-class object detection only when a defect is detected. This is a prototype phase — no production camera has been finalized yet, so publicly available datasets are used as-is to build the pipeline mechanics. Calibration to the actual production camera setup follows once it is determined.

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

All datasets are used, with stratified sampling to balance the Defect and Normal classes.

| Source | Contribution to "Defect" | Contribution to "Normal" |
|---|---|---|
| GC10-DET | ✅ all images (defect-only dataset) | – |
| NEU-CLS | ✅ all images (defect-only dataset) | – |
| X-SDD | ✅ all images (defect-only dataset) | – |
| Severstal | (optional, defective images can be added if more are needed) | ✅ defect-free subset (primary source for Normal) |

**Balancing strategy:** the total "Defect" image count from GC10+NEU-CLS+X-SDD (~2,300+1,800+1,360 ≈ 5,460 images) is used as the target, and a matching number of "Normal" images is sampled from Severstal (stratified, not all available images, to avoid skewing the dataset). Implementation: `scripts/prepare_severstal.py` (stages defect-free Severstal images) → `scripts/build_stage1_binary.py` (assembles, balances, and splits into train/val/test per source so proportions are preserved across splits) → config `configs/classification/cls_stage1_binary.yaml`.

### Stage 2 — Object Detection (15 defect classes)

| Source | Contribution |
|---|---|
| GC10-DET | 10 classes + bounding boxes |
| NEU-DET | 6 classes + bounding boxes (5 unique + 1 overlapping "inclusion") |
| Severstal (defect-free subset) | Negative samples (anti-false-positive), ~10-15% of positive image count |
| ~~X-SDD~~ | Not used — classification-only, no bounding boxes |

Implementation: `configs/detection/det_combined.yaml` already matches this scenario (GC10-DET + NEU-DET union = 15 classes). Build the data with `scripts/build_combined_dataset.py --task detection --sources gc10=... neu_det=... --negatives_dir data/raw/severstal_clean --negative_ratio 0.12`. Negative samples are written as empty label files (0 objects), already handled by `YoloDetectionDataset` in `src/data_loader.py`.

---

## 3. Classes

**Stage 1 (Classifier):** 2 classes — `Defect`, `Normal`

**Stage 2 (Detector):** 15 defect classes (union of GC10-DET and NEU-DET, "inclusion" merged):

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

## 4. Status: Prototype, Not Yet Calibrated to a Real Camera

This is not a gap that needs resolving right now — it is a status note to carry forward once the production camera is determined.

**Why every public steel defect dataset is close-up:** likely not a coincidence — defect features (fine cracks, small inclusions, etc.) require close range/high resolution to be visible and labelable, so academic datasets in this domain are typically collected with cameras positioned close to the material (line-scan cameras on production lines), not wide shots from a distance.

**Once the production camera is determined:**
- If it is also positioned close to the material (similar to an industrial line-scan setup) → the current data is likely already representative, no major changes needed.
- If it is a wide/distant shot (e.g. a phone photo at uncontrolled distance) → real-camera samples will need to be collected for validation, and additional fine-tuning or augmentation (scale jitter, copy-pasting defect patches onto a wider background) may be needed before considering the pipeline production-ready.

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
| YOLOv8/YOLO11 (small/medium) | Primary candidates |
| RT-DETR | Transformer-based alternative |
| Faster R-CNN (ResNet50-FPN) | Classic two-stage baseline |
| RetinaNet | One-stage alternative |

---

## 6. Trade-offs

- **Error propagation** — a stage 1 false negative (a defect classified as "Normal") means the image never reaches stage 2. Recall is the most critical metric at stage 1.
- **Not yet representative of the real camera scale** — see section 4; this is a known and tracked limitation, not an oversight.
- **Stage 2 class imbalance** — GC10-DET contributes 10 of 15 classes, so NEU-DET-only classes have fewer samples. Weighted loss/oversampling is required.
- **Stage 1 stratified sampling** — the Defect:Normal ratio must stay reasonably balanced, and the source distribution within the Defect class should also be checked (to avoid GC10 dominating simply because it has the most images).
- **Latency budget** — stage 1 must be significantly faster than stage 2 for the cascade to be worthwhile.

---

## 7. Evaluation Metrics

**Stage 1 (Binary Classification):**
- Recall and precision for the Defect class (recall prioritized — false negatives are more costly than false positives)
- F1, confusion matrix
- Recall breakdown by source dataset (GC10 vs NEU-CLS vs X-SDD), to check whether any source is harder to classify
- Inference latency (ms/image)

**Stage 2 (Object Detection):**
- mAP@0.5, mAP@0.5:0.95, per-class AP
- False positive rate on clean images (from Severstal)
- Inference latency (ms/image)

**End-to-end:**
- Overall recall (raw image → box output)
- Average total latency

---

## 8. Project Structure

> Implementation note: `train_classification.py`/`train_detection.py` (a generic training loop already supporting arbitrary datasets and architectures via config) were intentionally **not** renamed or restructured — only new datasets and configs were added for stages 1/2, reusing the training scripts as-is. See the roadmap (section 9) for context.

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
├── results/
├── requirements.txt
└── README.md
```

---

## 9. Roadmap

1. Download GC10-DET, NEU-CLS, NEU-DET, X-SDD, and the defect-free Severstal subset (Severstal is a Kaggle **competition** dataset — requires a Kaggle account and accepting the competition rules before `kaggle competitions download` works)
2. ✅ `scripts/prepare_severstal.py` — parses `train.csv` (handles both column format variants), stages defect-free images into `data/raw/severstal_clean/`
3. ✅ `scripts/build_stage1_binary.py` — balances Defect vs Normal for stage 1 (per-source split, checks source distribution within the Defect class)
4. ✅ `scripts/build_combined_dataset.py --negatives_dir` — folds Severstal negatives into stage 2 (~10-15% of positive image count, see section 6)
5. ✅ `src/class_mapping.py` — harmonizes the 15 stage 2 classes ("inclusion" merged)
6. Train and compare stage 1 architectures (binary classifier) — `python -m src.train_classification --config configs/classification/cls_stage1_binary.yaml`, select based on Defect-class recall and latency
7. Train and compare stage 2 architectures (detector, 15 classes) — `python -m src.train_detection --config configs/detection/det_combined.yaml --framework all`, select based on mAP and latency
8. Build `src/pipeline.py` — chain: load the best stage 1 checkpoint, and if Defect, load the best stage 2 checkpoint and run it (not yet built)
9. Run end-to-end tests with the currently available data (still public datasets, not the real camera)
10. **Key checkpoint:** once the production camera is determined, collect real-camera samples, re-validate model accuracy at that scale (section 4), fine-tune/calibrate as needed

> This repository's scope ends at the trained checkpoints (`.pt`) for stages 1 and 2, plus `pipeline.py` for local end-to-end testing. Integration into a web app or API is a separate downstream consumer of these models, not part of this repository.

---

## 10. Source Notes

- GC10-DET: [github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets](https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets) (Baidu Pan link, code: `cdyt`)
- NEU-CLS (mirror): [kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- NEU-DET: [ieee-dataport.org/documents/neu-det](https://ieee-dataport.org/documents/neu-det)
- X-SDD: [ieee-dataport.org/documents/x-sdd](https://ieee-dataport.org/documents/x-sdd) (requires an IEEE subscription/account)
- Severstal Steel Defect Dataset: [kaggle.com/c/severstal-steel-defect-detection](https://www.kaggle.com/c/severstal-steel-defect-detection/data)
