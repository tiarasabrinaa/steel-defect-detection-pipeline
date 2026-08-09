"""
Reorganisasi dataset mentah (hasil download manual, lihat README.md bagian 1
& 8) menjadi struktur standar `data/processed/` yang dipakai src/data_loader.py.

Struktur raw data yang di-handle (per dataset, hasil ngecek langsung isi
`data/raw/` — BUKAN asumsi generik VOC, tiap mirror emang beda-beda):

  GC10-DET (gc10_cls, gc10_det):
      data/raw/gc10/<1-10>/*.jpg     <- gambar, per NOMOR folder (bukan nama kelas!)
      data/raw/gc10/lable/*.xml      <- anotasi flat, <name> di XML itu PINYIN+angka
                                         (misal "3_yueyawan") - dipetakan lewat
                                         GC10_TAG_TO_CLASS di bawah (sumber:
                                         data/raw/gc10/Defects Description.xlsx, kolom "标签").
                                         PENTING: nomor folder tempat gambar berada TIDAK
                                         SAMA dengan kelas semua object di dalamnya - 1 gambar
                                         GC10 bisa punya lebih dari 1 jenis defect (defect
                                         utama sesuai folder + defect sekunder kelas lain).
                                         Buat gc10_det, kelas HARUS dibaca per-object dari
                                         <name> XML (lewat name_remap), BUKAN dipukul rata
                                         pakai kelas foldernya - kalau dipukul rata, defect
                                         sekunder ke-mislabel jadi kelas defect utama folder
                                         itu (bug nyata yang sempet kejadian & ketauan pas
                                         di-visual-check, lihat commit fix-nya).
                                         Buat gc10_cls (classification whole-image), pakai
                                         kelas folder TETAP relevan/benar - itu memang
                                         metodologi standar dataset ini buat task classification
                                         (1 label dominan per gambar, GC10_CLASS_TO_FOLDER).

  NEU-CLS (neu_cls):
      GAK didownload terpisah - gambarnya SAMA kayak NEU-DET, cuma dipakai
      TANPA bbox. Diekstrak dari raw_dir NEU-DET langsung (lihat
      prepare_neu_cls_from_neu_det).

  NEU-DET (neu_det):
      data/raw/neu_det/{train,validation}/images/<class_name>/*.jpg
      data/raw/neu_det/{train,validation}/annotations/*.xml
      (<name> di XML-nya udah bahasa Inggris & konsisten, aman dipakai langsung.
      Mirror ini gak punya split "test" - train+validation digabung lalu
      di-split ulang 70/15/15 sendiri biar konsisten sama dataset lain.)

  X-SDD (xsdd_cls):
      data/raw/xsdd/<nama folder>/*.jpg - nama foldernya PAKAI SPASI dan beda
      kata dari canonical (`src/class_mapping.py::XSDD_CLASSES`), misal
      "red iron" (bukan "red_iron_sheet"), "surface scratch" (bukan
      "surface_scratches"). Dipetakan lewat XSDD_FOLDER_ALIASES di bawah.

Output (semua dataset, format sama):
  Classification -> data/processed/<dataset>/{train,val,test}/<class_name>/*.jpg
  Detection      -> data/processed/<dataset>/images/{train,val,test}/*.jpg
                     data/processed/<dataset>/labels/{train,val,test}/*.txt

Split default 70/15/15, stratified per kelas, seed konsisten biar reproducible.

Contoh pemakaian:
    python scripts/prepare_data.py --task gc10_det --raw_dir data/raw/gc10 --out_dir data/processed/gc10_det
    python scripts/prepare_data.py --task neu_det  --raw_dir data/raw/neu_det --out_dir data/processed/neu_det
    python scripts/prepare_data.py --task neu_cls  --raw_dir data/raw/neu_det --out_dir data/processed/neu_cls
    python scripts/prepare_data.py --task xsdd_cls --raw_dir data/raw/xsdd --out_dir data/processed/xsdd
    python scripts/prepare_data.py --task gc10_cls --raw_dir data/raw/gc10 --out_dir data/processed/gc10_cls
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.voc_to_yolo import parse_voc_xml, voc_to_yolo_lines
from src.class_mapping import get_dataset_classes

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Sumber: data/raw/gc10/Defects Description.xlsx (kolom 英文/标签). Folder
# angka -> nama kelas kanonik (lihat src/class_mapping.py::GC10_CLASSES).
# PENTING: urutan folder angka BUKAN sama dengan urutan GC10_CLASSES index
# (folder 8/9/10 = rolled_pit/crease/waist_folding, bukan waist_folding/
# crease/rolled_pit seperti urutan GC10_CLASSES) - jangan pernah asumsikan
# `folder_number - 1 == index GC10_CLASSES`, selalu lewat mapping nama ini.
GC10_FOLDER_TO_CLASS = {
    "1": "punching_hole",
    "2": "weld_line",
    "3": "crescent_gap",
    "4": "water_spot",
    "5": "oil_spot",
    "6": "silk_spot",
    "7": "inclusion",
    "8": "rolled_pit",
    "9": "crease",
    "10": "waist_folding",
}
GC10_CLASS_TO_FOLDER = {v: k for k, v in GC10_FOLDER_TO_CLASS.items()}

# Sumber SAMA (Defects Description.xlsx, kolom "标签") - ini mapping per-OBJECT
# yang dipakai buat gc10_det, beda dari GC10_FOLDER_TO_CLASS di atas (yang
# per-FILE, cuma valid buat gc10_cls). Tag ini persis isi tag <name> di XML.
GC10_TAG_TO_CLASS = {
    "1_chongkong": "punching_hole",
    "2_hanfeng": "weld_line",
    "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot",
    "5_youban": "oil_spot",
    "6_siban": "silk_spot",
    "7_yiwu": "inclusion",
    "8_yahen": "rolled_pit",
    "9_zhehen": "crease",
    "10_yaozhe": "waist_folding",
    "10_yaozhed": "waist_folding",  # typo di raw XML, TERNYATA lebih umum (131x) dari yang "bener" (12x) - verified langsung scan semua XML
}
# NB: ada juga 1 tag korup "d" doang di seluruh dataset (1 kejadian) - sengaja
# TIDAK dipetakan (dibiarkan ke-skip dengan warning), gak ada cara reliable
# buat nebak itu maksudnya kelas apa tanpa cek visual manual satu-satu.

# Nama folder raw X-SDD (hasil download) -> nama kelas kanonik
# (src/class_mapping.py::XSDD_CLASSES). Beda soal spasi/singular-plural/kata.
XSDD_FOLDER_ALIASES = {
    "slag_inclusion": "slag inclusion",
    "red_iron_sheet": "red iron",
    "iron_sheet_ash": "iron sheet ash",
    "surface_scratches": "surface scratch",
    "oxide_scale_of_plate_system": "oxide scale of plate system",
    "finishing_roll_printing": "finishing roll printing",
    "oxide_scale_of_temperature": "oxide scale of temperature system",
}


def _split_list(items: list, val_ratio: float, test_ratio: float, seed: int) -> tuple[list, list, list]:
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    val = items[:n_val]
    test = items[n_val : n_val + n_test]
    train = items[n_val + n_test :]
    return train, val, test


def prepare_classification(
    raw_dir: str | Path,
    out_dir: str | Path,
    class_names: list[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    folder_aliases: dict[str, str] | None = None,
) -> None:
    """folder_aliases: {nama_kelas_kanonik: nama_folder_raw_asli}, dipakai
    kalau nama folder raw beda dari nama kelas kanonik (misal X-SDD)."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    folder_aliases = folder_aliases or {}
    summary = {}

    for cls in class_names:
        folder_name = folder_aliases.get(cls, cls)
        cls_dir = raw_dir / folder_name
        if not cls_dir.exists():
            print(f"WARNING: folder kelas '{cls}' (dicari: '{folder_name}') tidak ditemukan di {raw_dir}, di-skip")
            continue

        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
        if not files:
            print(f"WARNING: tidak ada gambar di {cls_dir}")
            continue

        train, val, test = _split_list(files, val_ratio, test_ratio, seed)
        summary[cls] = {"train": len(train), "val": len(val), "test": len(test)}

        for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
            dest_dir = out_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dest_dir / f.name)

    print(f"\nDistribusi jumlah sampel per kelas ({raw_dir.name}):")
    for cls, counts in summary.items():
        print(f"  {cls:35s} train={counts['train']:4d}  val={counts['val']:4d}  test={counts['test']:4d}")


def prepare_neu_cls_from_neu_det(
    raw_dir: str | Path,
    out_dir: str | Path,
    class_names: list[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """NEU-CLS gak ada file download terpisah - gambarnya sama persis kayak
    NEU-DET, cuma dipakai tanpa bbox (classification-only). `raw_dir` di sini
    HARUS folder raw NEU-DET (`data/raw/neu_det`), bukan folder NEU-CLS."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    summary = {}

    for cls in class_names:
        files: list[Path] = []
        for split_name in ["train", "validation"]:
            cls_dir = raw_dir / split_name / "images" / cls
            if cls_dir.exists():
                files += [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]

        if not files:
            print(f"WARNING: tidak ada gambar buat kelas '{cls}' di {raw_dir}/{{train,validation}}/images/")
            continue

        train, val, test = _split_list(sorted(files), val_ratio, test_ratio, seed)
        summary[cls] = {"train": len(train), "val": len(val), "test": len(test)}

        for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
            dest_dir = out_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dest_dir / f.name)

    print(f"\nDistribusi jumlah sampel per kelas ({raw_dir.name}, diekstrak dari NEU-DET):")
    for cls, counts in summary.items():
        print(f"  {cls:35s} train={counts['train']:4d}  val={counts['val']:4d}  test={counts['test']:4d}")


# collect_pairs_fn balikin list (xml_path, img_path). Kelas tiap OBJECT di
# dalam XML selalu dibaca dari <name> tag-nya sendiri (lewat voc_to_yolo_lines
# + name_remap kalau perlu) - folder/lokasi file cuma dipakai buat nemuin
# pasangan file, bukan buat nentuin kelas (lihat catatan GC10 di docstring atas).


def _collect_gc10_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    ann_dir = raw_dir / "lable"
    pairs = []
    missing_xml = 0
    for folder_num in GC10_FOLDER_TO_CLASS:
        img_dir = raw_dir / folder_num
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            xml_path = ann_dir / f"{img_path.stem}.xml"
            if xml_path.exists():
                pairs.append((xml_path, img_path))
            else:
                missing_xml += 1
    if missing_xml:
        print(f"WARNING: {missing_xml} gambar GC10 tanpa anotasi XML cocok di {ann_dir}, di-skip")
    return pairs


def _collect_neu_det_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for split_name in ["train", "validation"]:
        img_root = raw_dir / split_name / "images"
        ann_dir = raw_dir / split_name / "annotations"
        if not img_root.exists():
            continue
        for cls_dir in sorted(img_root.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTENSIONS:
                    continue
                xml_path = ann_dir / f"{img_path.stem}.xml"
                if xml_path.exists():
                    pairs.append((xml_path, img_path))
    return pairs


def prepare_detection(
    raw_dir: str | Path,
    out_dir: str | Path,
    class_names: list[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    collect_pairs_fn=None,
    name_remap: dict[str, str] | None = None,
) -> None:
    """`name_remap`: {tag_asli_di_XML: nama_kelas_kanonik} - dipakai buat GC10
    (<name> XML-nya pinyin). Diterapkan PER-OBJECT (lewat voc_to_yolo_lines),
    bukan per-file - 1 gambar boleh punya object dengan kelas berbeda-beda."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    class_to_id = {c: i for i, c in enumerate(class_names)}

    if collect_pairs_fn is not None:
        pairs = collect_pairs_fn(raw_dir)
    else:
        # fallback VOC generik: Annotations/ + JPEGImages/
        ann_dir = raw_dir / "Annotations"
        img_dir = raw_dir / "JPEGImages"
        if not ann_dir.exists() or not img_dir.exists():
            raise FileNotFoundError(
                f"Diperlukan {ann_dir} dan {img_dir} (struktur VOC generik). "
                "Kalau raw_dir kamu strukturnya beda, pakai collect_pairs_fn custom."
            )
        pairs = [
            (ann_dir / f"{p.stem}.xml", p)
            for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTENSIONS
        ]
        pairs = [(x, i) for x, i in pairs if x.exists()]

    if not pairs:
        raise RuntimeError(f"Tidak ada pasangan (xml, gambar) ditemukan di {raw_dir}")

    # Stratifikasi split pakai kelas object PERTAMA di XML (setelah di-remap
    # kalau perlu) sebagai "primary label" - gambar sering >1 object/kelas,
    # jadi ini bukan stratifikasi sempurna, tapi jaga proporsi kelas kasar
    # antara train/val/test. Kelas SEBENARNYA tiap box tetap dibaca ulang
    # per-object pas nulis label (lihat voc_to_yolo_lines di bawah).
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    skipped_no_object = 0
    for xml_path, img_path in pairs:
        _, _, objects = parse_voc_xml(xml_path)
        if not objects:
            skipped_no_object += 1
            continue
        first_name = objects[0][0]
        primary_label = name_remap.get(first_name, first_name) if name_remap else first_name
        groups[primary_label].append((xml_path, img_path))

    if skipped_no_object:
        print(f"WARNING: {skipped_no_object} file XML tanpa object, di-skip")

    split_items: dict[str, list[tuple[Path, Path]]] = {"train": [], "val": [], "test": []}
    summary = {}
    for label, items in groups.items():
        train, val, test = _split_list(items, val_ratio, test_ratio, seed)
        split_items["train"] += train
        split_items["val"] += val
        split_items["test"] += test
        summary[label] = {"train": len(train), "val": len(val), "test": len(test)}

    box_counts: dict[str, int] = defaultdict(int)
    for split_name, items in split_items.items():
        img_out = out_dir / "images" / split_name
        label_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for xml_path, img_path in items:
            lines, objects = voc_to_yolo_lines(xml_path, class_to_id, name_remap=name_remap)
            if split_name == "train":
                for name, *_ in objects:
                    canon = name_remap.get(name, name) if name_remap else name
                    if canon in class_to_id:
                        box_counts[canon] += 1

            shutil.copy2(img_path, img_out / img_path.name)
            (label_out / f"{img_path.stem}.txt").write_text("\n".join(lines))

    print(f"\nDistribusi gambar per primary-class ({raw_dir.name}):")
    for label, counts in summary.items():
        print(f"  {label:35s} train={counts['train']:4d}  val={counts['val']:4d}  test={counts['test']:4d}")
    print(f"\nDistribusi jumlah BOX train per kelas sebenarnya (setelah baca per-object):")
    for label, n in sorted(box_counts.items()):
        print(f"  {label:35s} {n:5d}")


TASK_REGISTRY = {
    "gc10_cls": ("gc10", prepare_classification, {"folder_aliases": GC10_CLASS_TO_FOLDER}),
    "neu_cls": ("neu_cls", prepare_neu_cls_from_neu_det, {}),
    "xsdd_cls": ("xsdd", prepare_classification, {"folder_aliases": XSDD_FOLDER_ALIASES}),
    "gc10_det": ("gc10", prepare_detection, {"collect_pairs_fn": _collect_gc10_pairs, "name_remap": GC10_TAG_TO_CLASS}),
    "neu_det": ("neu_det", prepare_detection, {"collect_pairs_fn": _collect_neu_det_pairs}),
}


def main():
    parser = argparse.ArgumentParser(description="Reorganisasi raw dataset -> data/processed/")
    parser.add_argument("--task", required=True, choices=list(TASK_REGISTRY))
    parser.add_argument(
        "--raw_dir", required=True,
        help="Untuk --task neu_cls, isi dengan folder raw NEU-DET (data/raw/neu_det) - lihat docstring modul ini",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_key, fn, extra_kwargs = TASK_REGISTRY[args.task]
    class_names = get_dataset_classes(dataset_key)

    fn(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        class_names=class_names,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        **extra_kwargs,
    )
    print(f"\nSelesai. Hasil tersimpan di {args.out_dir}")


if __name__ == "__main__":
    main()
