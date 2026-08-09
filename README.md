# Steel Surface Defect — Multi-Dataset Benchmark (Classification + Object Detection)

Project untuk membandingkan performa beberapa arsitektur pretrained pada tugas **klasifikasi** dan **object detection** cacat permukaan baja, dievaluasi per-dataset dan pada gabungan dataset (union kelas). Semua eksperimen di-*track* penuh (hyperparameter, metric per-epoch, metric final, confusion matrix, checkpoint terbaik) ke **MLflow** — lokal untuk development, atau ke **Databricks Managed MLflow** untuk kerja beneran.

---

## 1. Dataset

| Dataset | Jumlah Kelas | Jumlah Gambar | Anotasi | Sumber |
|---|---|---|---|---|
| **GC10-DET** | 10 | ~2.300 (3.570 termasuk varian grayscale) | Bounding box + pixel-level | [GitHub (link Baidu Pan, kode: `cdyt`)](https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets) |
| **NEU-CLS** | 6 | 1.800 (300/kelas), 200×200 | Classification-only (label per gambar, tanpa bbox) | [Kaggle mirror](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |
| **X-SDD** | 7 | 1.360 | Classification-only (patch sudah di-crop, tanpa bbox) | [IEEE DataPort](https://ieee-dataport.org/documents/x-sdd) — **butuh subscription/akun IEEE** |

### Daftar kelas per dataset

**GC10-DET (10):** punching_hole, weld_line, crescent_gap, water_spot, oil_spot, silk_spot, inclusion, waist_folding, crease, rolled_pit

**NEU-CLS (6):** crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches

**X-SDD (7):** slag_inclusion, red_iron_sheet, iron_sheet_ash, surface_scratches, oxide_scale_of_plate_system, finishing_roll_printing, oxide_scale_of_temperature

> ⚠️ **Catatan penting soal anotasi untuk object detection:** NEU-CLS dan X-SDD versi yang disebut di atas adalah dataset **classification-only** — tidak ada bounding box bawaan. Untuk task object detection, NEU punya varian terpisah bernama **NEU-DET** (gambar sama, tapi dengan bbox). X-SDD **tidak** punya varian bbox publik — kalau tetap mau dipakai di skenario detection, perlu anotasi manual/semi-otomatis dulu (misal pakai tools seperti ISAT/SAM-assisted labeling), atau X-SDD **di-drop** dari task detection dan hanya dipakai di task classification. Pipeline ini memilih opsi drop (lihat Task B di bawah).

---

## 2. Class Mapping — Union Semua Kelas (bukan cuma yang beririsan)

Skenario gabungan pakai **semua kelas dari ketiga dataset** (union), bukan cuma kelas yang muncul di ketiga-tiganya. Yang di-merge jadi satu kelas kanonik hanya yang namanya/definisinya jelas identik; sisanya tetap dipisah per dataset supaya gak ada asumsi berlebihan soal kesamaan defect. Implementasi single-source-of-truth ada di [`src/class_mapping.py`](src/class_mapping.py) — semua kode lain (data loader, scripts, training) wajib memakai mapping dari file ini.

**Kelas yang di-merge (nama/definisi jelas sama):**
- **Inclusion** ← `inclusion` (NEU) + `inclusion` (GC10) + `slag_inclusion` (X-SDD) → 3 label jadi 1
- **Scratches** ← `scratches` (NEU) + `surface_scratches` (X-SDD) → 2 label jadi 1

**Kelas yang TETAP dipisah** (nama mirip tapi proses/definisi beda, atau memang unik per dataset) — termasuk `rolled-in_scale` (NEU) vs `oxide_scale_of_plate_system` / `oxide_scale_of_temperature` (X-SDD), yang sengaja **tidak** digabung karena mekanisme pembentukan defect-nya beda dan belum diverifikasi visual.

### Daftar 20 kelas kanonik hasil union (classification, A4)

| # | Kelas Kanonik | Sumber |
|---|---|---|
| 1 | Inclusion | NEU + GC10 + X-SDD (merged) |
| 2 | Scratches | NEU + X-SDD (merged) |
| 3 | Crazing | NEU |
| 4 | Patches | NEU |
| 5 | Pitted Surface | NEU |
| 6 | Rolled-in Scale | NEU |
| 7 | Punching Hole | GC10 |
| 8 | Weld Line | GC10 |
| 9 | Crescent Gap | GC10 |
| 10 | Water Spot | GC10 |
| 11 | Oil Spot | GC10 |
| 12 | Silk Spot | GC10 |
| 13 | Waist Folding | GC10 |
| 14 | Crease | GC10 |
| 15 | Rolled Pit | GC10 |
| 16 | Red Iron Sheet | X-SDD |
| 17 | Iron Sheet Ash | X-SDD |
| 18 | Oxide Scale of Plate System | X-SDD |
| 19 | Oxide Scale of Temperature | X-SDD |
| 20 | Finishing Roll Printing | X-SDD |

> Total 20, bukan 23 (6+10+7), karena Inclusion (3→1) dan Scratches (2→1) di-merge.
>
> **Untuk detection (B4)**, X-SDD di-drop (lihat bagian 1), jadi union kelasnya cuma dari GC10 + NEU-DET → **15 kelas** (bukan 16 seperti perkiraan awal di draft rencana — 10 GC10 + 6 NEU − 1 karena `inclusion` merge = 15). Angka pastinya dihitung otomatis oleh `src/class_mapping.py::combined_class_names_for()`, jangan hardcode manual di tempat lain.

---

## 3. Eksperimen

### Task A — Klasifikasi

4 skenario training, tiap skenario dicoba dengan beberapa arsitektur pretrained (transfer learning, fine-tune dari ImageNet weights, lewat [`timm`](https://github.com/huggingface/pytorch-image-models)):

| Skenario | Dataset | Jumlah Kelas | Config |
|---|---|---|---|
| A1 | GC10-DET saja | 10 | `configs/classification/cls_gc10.yaml` |
| A2 | NEU-CLS saja | 6 | `configs/classification/cls_neu.yaml` |
| A3 | X-SDD saja | 7 | `configs/classification/cls_xsdd.yaml` |
| A4 | Gabungan — union semua kelas | 20 | `configs/classification/cls_combined.yaml` |

**Arsitektur pretrained (default 4, bisa ditambah/kurangi lewat config):**
- `resnet50` — baseline standar, banyak referensi di paper steel defect
- `efficientnetv2_s` (`tf_efficientnetv2_s`) — efisien, akurasi tinggi di dataset kecil
- `convnext_tiny` — arsitektur modern, bagus untuk tekstur halus (defect permukaan sering soal tekstur)
- `mobilenetv3` (`mobilenetv3_large_100`) — baseline ringan, buat perbandingan kalau nanti mau deploy edge
- `swin_tiny` (`swin_tiny_patch4_window7_224`) — opsional, tinggal tambahkan ke daftar `architectures` di config

**Model customization (bukan cuma pretrained-as-is):**
- **Classifier head** (`head:` di config, lihat `src/models/classification.py`) — pilihan `linear` (default timm) atau `mlp` (bottleneck Linear→BatchNorm→GELU→Dropout→Linear). Dataset kecil/imbalance (NEU-CLS, X-SDD, combined) pakai `mlp` sebagai regularizer murah; GC10 (dataset terbesar & paling balanced) tetap `linear`. Alasan spesifik per skenario ada di comment masing-masing `configs/classification/cls_*.yaml`.
- **Quantization-aware training (QAT)** (`quantization:` di config, lihat `src/utils/quantization.py`) — FX graph-mode QAT, aktif untuk `resnet50`/`efficientnetv2_s`/`mobilenetv3` (di-skip otomatis untuk `convnext_tiny`/`swin_tiny`, belum FX-traceable dengan stabil). Training jalan fp32 dulu beberapa epoch (bobot stabil dari pretrained weights), baru fake-quant diaktifkan di epoch-epoch terakhir, lalu di-convert jadi int8 asli setelah training selesai. Backend `qnnpack` (target ARM/edge) dipilih karena MobileNetV3 di project ini eksplisit dikandidatkan untuk deploy edge. Hasilnya (`test_quantized_*`, `fp32_model_size_mb` vs `quantized_model_size_mb`, `model_size_reduction_pct`) di-log ke MLflow sebagai basis keputusan "layak di-deploy edge atau tidak", dan checkpoint int8-nya disimpan terpisah (`best_quantized.pt`).

### Task B — Object Detection

3 skenario training (X-SDD di-drop, lihat bagian 1), tiap skenario dicoba beberapa arsitektur pretrained fine-tune dari COCO weights, lewat dua framework:

| Skenario | Dataset | Jumlah Kelas | Config |
|---|---|---|---|
| B1 | GC10-DET saja | 10 | `configs/detection/det_gc10.yaml` |
| B2 | NEU-DET saja (bukan NEU-CLS) | 6 | `configs/detection/det_neu.yaml` |
| B4 | Gabungan GC10 + NEU-DET | 15 | `configs/detection/det_combined.yaml` |

**Arsitektur pretrained:**
- `yolov8s`, `yolo11s` — cepat, mudah fine-tune, banyak dipakai di paper steel defect (lewat `ultralytics`)
- `rtdetr-l` — transformer-based detector, lebih modern (lewat `ultralytics`)
- `fasterrcnn_resnet50_fpn_v2` — baseline klasik two-stage (lewat `torchvision`)
- `retinanet_resnet50_fpn_v2` — baseline one-stage alternatif (lewat `torchvision`)

---

## 4. Trade-off yang Perlu Dipikirkan

- **Class imbalance** — dengan union kelas, timpangnya makin terasa: kelas hasil merge (Inclusion, Scratches) otomatis punya jumlah sampel lebih banyak (gabungan dari 2-3 dataset) dibanding kelas unik yang cuma dari satu dataset (misal Waist Folding cuma dari GC10). `loss.weighted: true` di config combined mengaktifkan weighted CrossEntropyLoss (classification, lihat `src/utils/class_weights.py`); `scripts/build_combined_dataset.py` juga otomatis print distribusi jumlah sampel/box per kelas supaya imbalance-nya kelihatan sebelum training.
- **Domain shift antar dataset** — beda sumber kamera/pencahayaan/resolusi antara GC10, NEU, X-SDD. Model yang bagus di satu dataset belum tentu generalize ke dataset lain — insight ini bisa digali dengan evaluasi cross-dataset (load checkpoint dari satu skenario, `evaluate()` di test set dataset lain).
- **Anotasi tidak konsisten** — GC10 punya bbox+pixel-mask, NEU-CLS/X-SDD cuma classification. Task detection otomatis lebih terbatas datasetnya (lihat Task B).
- **Akses X-SDD** — perlu subscription/akun IEEE DataPort.
- **Reliabilitas class mapping** — mapping "kelas beririsan" di atas berbasis nama, bukan piksel. Kalau setelah dicek visual ternyata beda karakter, hasil skenario gabungan (A4/B4) perlu di-footnote sebagai eksperimen eksploratif, bukan ground truth yang solid.

---

## 5. Metrik Evaluasi (semua di-log otomatis ke MLflow)

**Klasifikasi** (`src/utils/metrics_classification.py`): Accuracy, Precision/Recall/F1 (macro **dan** weighted, plus per-class), AUC ROC one-vs-rest (macro, weighted, per-class), confusion matrix (image artifact), classification report (text artifact).

**Object Detection** (`src/utils/metrics_detection.py` untuk jalur torchvision; native `ultralytics` metrics untuk jalur YOLO/RT-DETR): mAP@0.5, mAP@0.5:0.95, mAP@0.75, per-class AP, precision/recall, plus qualitative check (visualisasi prediksi vs ground truth di sample gambar test, disimpan sebagai image artifact).

---

## 6. Struktur Project

```
steel-defect-detection/
├── data/
│   ├── raw/                  # dataset mentah hasil download manual (lihat bagian 7)
│   │   ├── gc10/
│   │   ├── neu_cls/
│   │   ├── neu_det/
│   │   └── xsdd/
│   ├── processed/             # hasil scripts/prepare_data.py (split train/val/test)
│   └── combined/              # hasil scripts/build_combined_dataset.py (label kanonik)
├── src/
│   ├── class_mapping.py       # single source of truth label harmonization
│   ├── data_loader.py         # Dataset & DataLoader classification + detection
│   ├── models/
│   │   ├── classification.py  # builder timm (resnet50, effnetv2, convnext, dst)
│   │   └── detection.py       # builder torchvision (Faster R-CNN, RetinaNet)
│   ├── utils/
│   │   ├── seed.py
│   │   ├── class_weights.py
│   │   ├── mlflow_utils.py    # semua interaksi ke MLflow/Databricks lewat sini
│   │   ├── metrics_classification.py
│   │   ├── metrics_detection.py
│   │   └── quantization.py    # FX graph-mode QAT (resnet50/effnetv2/mobilenetv3)
│   ├── train_classification.py
│   └── train_detection.py
├── configs/
│   ├── classification/
│   │   ├── cls_gc10.yaml
│   │   ├── cls_neu.yaml
│   │   ├── cls_xsdd.yaml
│   │   └── cls_combined.yaml
│   └── detection/
│       ├── det_gc10.yaml
│       ├── det_neu.yaml
│       └── det_combined.yaml
├── scripts/
│   ├── voc_to_yolo.py           # converter VOC XML -> YOLO txt
│   ├── prepare_data.py          # raw -> processed (split + convert format)
│   └── build_combined_dataset.py  # processed -> combined (remap label kanonik)
├── notebooks/                 # eksplorasi & visualisasi cepat
├── results/                   # checkpoint & output lokal per run (juga di-log ke MLflow)
├── requirements.txt
├── .env.example                # template kredensial Databricks/MLflow
└── README.md
```

---

## 7. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: isi DATABRICKS_HOST, DATABRICKS_TOKEN, MLFLOW_EXPERIMENT_BASE_PATH
```

### MLflow + Databricks

Semua konfigurasi tracking dibaca dari environment variable (lewat `.env`, TIDAK pernah di-hardcode di kode/config), lihat `.env.example`:

```bash
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
DATABRICKS_TOKEN=<personal-access-token>
MLFLOW_EXPERIMENT_BASE_PATH=/Users/<email-databricks-kamu>/steel-defect-detection
```

- Kalau `MLFLOW_TRACKING_URI` tidak diset / dikosongkan, otomatis fallback ke tracking lokal (`./mlruns`) — berguna buat tes pipeline dulu tanpa akun Databricks (`mlflow ui` untuk lihat hasil lokal).
- Tiap kombinasi skenario × arsitektur = 1 MLflow run terpisah, semuanya di bawah experiment yang sama per skenario (misal `<base_path>/steel-defect/classification/gc10`), supaya gampang dibandingkan di MLflow UI / Databricks Experiments.
- Yang di-log per run: **semua hyperparameter** dari config yaml, **semua metric per-epoch** (train loss, val accuracy/precision/recall/F1/AUC atau val mAP), **metric final di test set**, **confusion matrix / qualitative prediction** sebagai image artifact, dan **best checkpoint (`.pt`)** sebagai artifact + full model lewat `mlflow.pytorch.log_model` (bisa langsung diregister ke Databricks Model Registry).

### Download & susun dataset mentah

1. Download GC10-DET, NEU-CLS, NEU-DET, X-SDD dari sumber di bagian 8.
2. Susun ke `data/raw/<dataset>/` sesuai struktur yang diharapkan `scripts/prepare_data.py`:
   - Classification (GC10 classification variant, NEU-CLS, X-SDD): folder per kelas — `data/raw/<dataset>/<class_name>/*.jpg`
   - Detection (GC10-DET, NEU-DET): struktur VOC — `data/raw/<dataset>/Annotations/*.xml` + `data/raw/<dataset>/JPEGImages/*.jpg`
3. Jalankan `prepare_data.py` untuk tiap dataset (lihat contoh di bagian 8 - Cara Pakai).

---

## 8. Cara Pakai

```bash
# 1. Reorganisasi raw -> processed (contoh NEU-CLS & GC10-DET, ulangi untuk dataset lain)
python scripts/prepare_data.py --task neu_cls  --raw_dir data/raw/neu_cls --out_dir data/processed/neu_cls
python scripts/prepare_data.py --task gc10_det --raw_dir data/raw/gc10   --out_dir data/processed/gc10_det

# 2. (Untuk skenario A4/B4) Bangun dataset gabungan dengan label kanonik
python scripts/build_combined_dataset.py --task classification \
    --sources gc10=data/processed/gc10_cls neu_cls=data/processed/neu_cls xsdd=data/processed/xsdd \
    --out_dir data/combined/classification

python scripts/build_combined_dataset.py --task detection \
    --sources gc10=data/processed/gc10_det neu_det=data/processed/neu_det \
    --out_dir data/combined/detection

# 3. Training classification — semua arsitektur di config, tiap arsitektur = 1 MLflow run
python -m src.train_classification --config configs/classification/cls_gc10.yaml
python -m src.train_classification --config configs/classification/cls_combined.yaml --architectures resnet50
# head (linear/mlp) & quantization (QAT) diatur lewat config, lihat bagian 3 "Model customization".
# Buat eksperimen cepat tanpa nunggu fase QAT, set quantization.enabled: false di config yang dipakai.

# 4. Training detection — pilih framework (ultralytics = YOLO/RT-DETR, torchvision = Faster R-CNN/RetinaNet)
python -m src.train_detection --config configs/detection/det_gc10.yaml --framework all
python -m src.train_detection --config configs/detection/det_combined.yaml --framework ultralytics

# 5. Lihat hasil
mlflow ui   # kalau tracking lokal
# atau buka Databricks workspace -> Experiments -> <MLFLOW_EXPERIMENT_BASE_PATH>/steel-defect/...
```

---

## 9. Roadmap

1. Download & organisasi ulang ketiga dataset ke struktur folder standar (`data/raw/`)
2. Spot-check visual untuk validasi class mapping (terutama kelas "Oxide/Rolled Scale")
3. ✅ `src/class_mapping.py` — single source of truth buat label harmonization
4. ✅ Preprocessing pipeline (`scripts/prepare_data.py`, `scripts/build_combined_dataset.py`)
5. ✅ Baseline training — Task A (`src/train_classification.py`, 4 skenario × arsitektur terpilih)
6. ✅ Baseline training — Task B (`src/train_detection.py`, 3 skenario × arsitektur terpilih)
7. ✅ Model customization — custom classifier head (linear/mlp) + QAT untuk resnet50/efficientnetv2_s/mobilenetv3 (lihat bagian 3)
8. Evaluasi & komparasi hasil antar skenario + antar arsitektur (pakai MLflow UI / Databricks Experiments compare-runs)
9. Analisis cross-dataset generalization (train di satu dataset, test di dataset lain) — opsional tapi insight-nya bagus buat laporan
10. Freeze/unfreeze backbone strategy (linear probe / gradual unfreezing) — belum diimplementasikan, nyusul kalau dibutuhkan

---

## 10. Catatan Sumber

- GC10-DET: [github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets](https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets)
- NEU-CLS (mirror): [kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- X-SDD: [ieee-dataport.org/documents/x-sdd](https://ieee-dataport.org/documents/x-sdd) (butuh subscription)
