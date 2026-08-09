# Steel Surface Defect — Cascade Binary Classification + Detection Pipeline (v3, Prototype Phase)

Pipeline 2-tahap: klasifikasi biner cepat (Defect vs Normal) sebagai gate awal, lanjut ke object detection multi-kelas cuma kalau ada defect. Versi prototype — belum ada kamera produksi yang fix, jadi semua dataset publik yang ada dipakai apa adanya dulu buat bangun mekanisme pipeline-nya. Kalibrasi ke scale kamera asli menyusul begitu kamera sudah ditentukan.

---

## 1. Flow Sistem

```
Raw Image
   │
   ▼
[Model Klasifikasi Biner] ← ringan, cepat, 2 kelas: Defect / Normal
   │
   ├── Prediksi: NORMAL ──────────► selesai, tampilkan "OK"
   │
   └── Prediksi: DEFECT
              │
              ▼
      [Model Object Detection] ← 15 kelas defect + lokasi, cuma jalan kalau perlu
              │
              ▼
      Bounding box lokasi defect + kelas
```

---

## 2. Dataset per Stage

### Stage 1 — Klasifikasi Biner (Defect vs Normal)

Semua dataset dipakai, di-stratified sampling supaya kelas Defect vs Normal seimbang.

| Sumber | Kontribusi ke label "Defect" | Kontribusi ke label "Normal" |
|---|---|---|
| GC10-DET | ✅ semua gambar (defect-only dataset) | – |
| NEU-CLS | ✅ semua gambar (defect-only dataset) | – |
| X-SDD | ✅ semua gambar (defect-only dataset) | – |
| Severstal | (opsional, gambar dengan defect bisa ditambah kalau butuh lebih banyak) | ✅ subset no-defect (sumber utama kelas Normal) |

**Strategi balance:** total gambar "Defect" dari GC10+NEU-CLS+X-SDD (~2.300+1.800+1.360 ≈ 5.460 gambar) dijadikan acuan, lalu ambil sample "Normal" dari Severstal dalam jumlah yang sepadan (stratified, bukan asal ambil semua ribuan gambar Normal-nya — supaya gak timpang ke salah satu sisi). Implementasi: `scripts/prepare_severstal.py` (staging gambar defect-free dari Severstal) → `scripts/build_stage1_binary.py` (assemble + balance + split train/val/test, per-sumber supaya proporsi GC10/NEU-CLS/X-SDD/Severstal terjaga di tiap split) → config `configs/classification/cls_stage1_binary.yaml`.

### Stage 2 — Object Detection (15 kelas defect)

Tidak berubah dari rencana sebelumnya:

| Sumber | Kontribusi |
|---|---|
| GC10-DET | 10 kelas + bbox |
| NEU-DET | 6 kelas + bbox (5 unik + 1 overlap "inclusion") |
| Severstal (subset no-defect) | Negative samples (anti false-positive), ~10-15% dari jumlah gambar positive |
| ~~X-SDD~~ | Gak dipakai — classification-only, gak ada bbox |

Implementasi: `configs/detection/det_combined.yaml` udah persis skenario ini (union GC10-DET+NEU-DET = 15 kelas) dari sebelum README v3 ditulis, jadi tinggal dipakai langsung — tinggal build datanya lewat `scripts/build_combined_dataset.py --task detection --sources gc10=... neu_det=... --negatives_dir data/raw/severstal_clean --negative_ratio 0.12`. Negative sample ditulis sebagai label file kosong (0 object) — sudah otomatis dihandle `YoloDetectionDataset` di `src/data_loader.py`, gak perlu ubah loader.

---

## 3. Kelas

**Stage 1 (Classifier):** 2 kelas — `Defect`, `Normal`

**Stage 2 (Detector):** 15 kelas defect (union GC10-DET + NEU-DET, "inclusion" di-merge):

| # | Kelas | Sumber |
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

## 4. Status: Prototype, Belum Dikalibrasi ke Kamera Real

Ini bukan gap yang "harus diputuskan sekarang" — ini catatan status yang wajib dibawa ke tahap berikutnya begitu kamera produksi sudah ditentukan.

**Kenapa semua dataset publik defect steel itu zoom-in:** kemungkinan besar bukan kebetulan — fitur defect (retakan halus, inclusion kecil, dll) butuh resolusi/jarak dekat buat kelihatan dan bisa dilabeli, jadi dataset akademik di domain ini memang lazim dikumpulkan dari kamera dekat ke material (line-scan camera di jalur produksi), bukan wide-shot dari jauh.

**Kalau nanti kamera produksi ditentukan:**
- Kalau kameranya juga dipasang dekat ke material (mirip setup line-scan industrial) → data yang ada sekarang kemungkinan besar udah cukup representatif, gak perlu perubahan besar.
- Kalau kameranya jauh/wide-shot (misal user upload foto dari HP dengan jarak gak terkontrol) → perlu collect sample dari kamera asli buat validasi, dan kemungkinan perlu fine-tuning tambahan atau augmentasi khusus (scale-jitter, copy-paste defect patch ke background yang lebih luas) sebelum dianggap production-ready.

---

## 5. Eksperimen — Pilihan Arsitektur Pretrained

### Stage 1 — Klasifikasi Biner

| Arsitektur | Catatan |
|---|---|
| MobileNetV3-Small | Kandidat utama, paling ringan |
| EfficientNetV2-S | Balance speed & akurasi |
| ResNet18 | Baseline standar |

### Stage 2 — Object Detection (15 kelas)

| Arsitektur | Catatan |
|---|---|
| YOLOv8/YOLO11 (small/medium) | Kandidat utama |
| RT-DETR | Alternatif transformer-based |
| Faster R-CNN (ResNet50-FPN) | Baseline two-stage klasik |
| RetinaNet | Alternatif one-stage |

---

## 6. Trade-off

- **Error propagation** — false negative stage 1 (defect ke-klasifikasi "Normal") bikin gambar gak pernah sampai stage 2. Recall jadi metrik paling kritis di stage 1.
- **Belum representasi scale kamera real** — lihat bagian 4, ini sadar-diketahui, bukan diabaikan.
- **Class imbalance stage 2** — GC10-DET nyumbang 10 dari 15 kelas, jadi kelas dari NEU-DET otomatis lebih sedikit sampelnya. Wajib weighted loss/oversampling.
- **Stratified sampling stage 1** — perlu dipastikan rasio Defect:Normal gak timpang drastis, dan idealnya dicek juga distribusi sumber di dalam kelas Defect (jangan sampai GC10 mendominasi 90% gara-gara paling banyak gambarnya).
- **Latency budget** — stage 1 harus signifikan lebih cepat dari stage 2 supaya cascade-nya worth it.

---

## 7. Metrik Evaluasi

**Stage 1 (Klasifikasi Biner):**
- Recall & Precision untuk kelas Defect (recall diprioritaskan — false negative lebih mahal daripada false positive)
- F1, confusion matrix
- Breakdown recall per sumber dataset asal (GC10 vs NEU-CLS vs X-SDD) — buat lihat apakah ada sumber yang jauh lebih susah dikenali
- Inference latency (ms/gambar)

**Stage 2 (Object Detection):**
- mAP@0.5, mAP@0.5:0.95, per-class AP
- False positive rate di gambar clean (dari Severstal)
- Inference latency (ms/gambar)

**End-to-end:**
- Overall recall (raw image → box keluar)
- Total latency rata-rata

---

## 8. Struktur Project (aktual)

> Catatan implementasi: `train_classification.py`/`train_detection.py` (training loop generic, udah support arbitrary dataset+arsitektur lewat config) **sengaja TIDAK di-rename/dibongkar** — cuma dataset & config yang baru buat stage 1/2, model training scripts-nya reuse langsung. Detail keputusan ini di bagian 9 (Roadmap).

```
steel-defect-detection/
├── data/
│   ├── raw/
│   │   ├── gc10/
│   │   ├── neu_cls/           # stage 1 (semua gambar -> label Defect)
│   │   ├── neu_det/           # stage 2 (bbox)
│   │   ├── xsdd/               # stage 1 saja (crop patch, gak ada bbox)
│   │   └── severstal_clean/   # hasil scripts/prepare_severstal.py - defect-free, dipakai stage 1 & 2
│   ├── processed/              # hasil scripts/prepare_data.py (bbox VOC->YOLO, split)
│   └── combined/
│       ├── stage1_binary/      # hasil scripts/build_stage1_binary.py (Defect vs Normal, balanced)
│       └── detection/          # hasil scripts/build_combined_dataset.py (union 15 kelas + negative Severstal)
├── src/
│   ├── class_mapping.py        # harmonisasi kelas (termasuk union 15 kelas GC10+NEU-DET)
│   ├── data_loader.py
│   ├── models/
│   │   ├── classification.py   # builder timm, termasuk resnet18/mobilenetv3_small buat stage 1
│   │   └── detection.py
│   ├── utils/
│   │   ├── mlflow_utils.py
│   │   ├── quantization.py     # QAT - relevan banget buat stage 1 (gate classifier, harus ringan)
│   │   └── ...
│   ├── train_classification.py # dipakai buat stage 1 (config cls_stage1_binary.yaml) DAN skenario riset lama
│   └── train_detection.py      # dipakai buat stage 2 (config det_combined.yaml, sudah 15 kelas)
├── configs/
│   ├── classification/
│   │   ├── cls_stage1_binary.yaml   # <- STAGE 1 (baru)
│   │   └── cls_*.yaml                # skenario riset lama (A1-A4), tetap ada, gak kepake di flow production
│   └── detection/
│       ├── det_combined.yaml   # <- STAGE 2 (udah GC10+NEU-DET union 15 kelas dari awal, tinggal reuse)
│       └── det_*.yaml           # skenario riset lama (B1/B2), tetap ada
├── scripts/
│   ├── prepare_severstal.py         # cari & staging gambar defect-free dari train.csv Severstal
│   ├── build_stage1_binary.py       # assemble+balance dataset stage 1
│   ├── build_combined_dataset.py    # assemble stage 2 (+ --negatives_dir buat fold Severstal)
│   ├── prepare_data.py
│   └── voc_to_yolo.py
├── notebooks/
├── results/
├── requirements.txt
└── README.md
```

---

## 9. Roadmap

1. Download GC10-DET, NEU-CLS, NEU-DET, X-SDD, subset no-defect Severstal (Severstal: Kaggle **competition** dataset, butuh akun Kaggle + accept competition rules dulu di halaman datanya, baru bisa `kaggle competitions download`)
2. ✅ `scripts/prepare_severstal.py` — parse `train.csv` (handle 2 varian format kolom), staging gambar defect-free ke `data/raw/severstal_clean/`
3. ✅ `scripts/build_stage1_binary.py` — balance Defect vs Normal buat stage 1 (split per-sumber, cek distribusi sumber di dalam kelas Defect)
4. ✅ `scripts/build_combined_dataset.py --negatives_dir` — fold negative Severstal ke stage 2 (~10-15% dari jumlah gambar positive, biar detector gak jadi kelewat konservatif — lihat bagian 6)
5. ✅ `src/class_mapping.py` — harmonisasi 15 kelas stage 2 (merge "inclusion", sudah dipakai skenario B4 sebelumnya, angkanya konsisten)
6. Training & bandingkan arsitektur stage 1 (binary classifier) — `python -m src.train_classification --config configs/classification/cls_stage1_binary.yaml`, pilih berdasarkan recall kelas Defect + latency
7. Training & bandingkan arsitektur stage 2 (detector, 15 kelas) — `python -m src.train_detection --config configs/detection/det_combined.yaml --framework all`, pilih berdasarkan mAP + latency
8. Rakit `src/pipeline.py` — chain load best checkpoint stage 1 -> kalau Defect, load best checkpoint stage 2 -> jalankan (belum dibuat)
9. Uji end-to-end dengan data yang ada (masih pakai dataset publik, belum kamera asli)
10. **Checkpoint penting:** begitu kamera produksi ditentukan, kumpulkan sample dari kamera asli, validasi ulang apakah model masih akurat di scale itu (bagian 4), fine-tune/kalibrasi kalau perlu

> Scope repo ini berhenti di checkpoint (`.pt`) hasil training stage 1 & stage 2 + `pipeline.py` buat uji end-to-end lokal. Integrasi ke web app/API itu konsumen terpisah dari model yang dihasilkan di sini, bukan bagian dari repo ini.

---

## 10. Catatan Sumber

- GC10-DET: [github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets](https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets) (link Baidu Pan, kode: `cdyt`)
- NEU-CLS (mirror): [kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- NEU-DET: [ieee-dataport.org/documents/neu-det](https://ieee-dataport.org/documents/neu-det)
- X-SDD: [ieee-dataport.org/documents/x-sdd](https://ieee-dataport.org/documents/x-sdd) (butuh subscription/akun IEEE)
- Severstal Steel Defect Dataset: [kaggle.com/c/severstal-steel-defect-detection](https://www.kaggle.com/c/severstal-steel-defect-detection/data)