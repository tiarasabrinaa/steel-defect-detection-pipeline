"""
Single source of truth untuk label harmonization lintas dataset.

Berisi:
- Daftar kelas asli per dataset (GC10-DET, NEU-CLS/NEU-DET, X-SDD)
- 20 kelas kanonik hasil union (lihat README.md bagian 2)
- Mapping dari label asli tiap dataset -> kelas kanonik
- Helper untuk membangun label map classification & detection (combined scenario)

Semua modul lain (data_loader, scripts/build_combined_dataset.py, training
scripts) WAJIB memakai mapping dari file ini, jangan hardcode ulang di
tempat lain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Kelas asli per dataset (urutan sesuai README.md)
# ---------------------------------------------------------------------------

GC10_CLASSES = [
    "punching_hole",
    "weld_line",
    "crescent_gap",
    "water_spot",
    "oil_spot",
    "silk_spot",
    "inclusion",
    "waist_folding",
    "crease",
    "rolled_pit",
]

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# NEU-DET pakai gambar yang sama dengan NEU-CLS, kelasnya identik.
NEU_DET_CLASSES = NEU_CLASSES

XSDD_CLASSES = [
    "slag_inclusion",
    "red_iron_sheet",
    "iron_sheet_ash",
    "surface_scratches",
    "oxide_scale_of_plate_system",
    "finishing_roll_printing",
    "oxide_scale_of_temperature",
]

DATASET_CLASSES = {
    "gc10": GC10_CLASSES,
    "neu_cls": NEU_CLASSES,
    "neu_det": NEU_DET_CLASSES,
    "xsdd": XSDD_CLASSES,
}

# ---------------------------------------------------------------------------
# 2. 20 kelas kanonik hasil union (README.md bagian 2, tabel "Daftar 20
#    kelas kanonik hasil union")
# ---------------------------------------------------------------------------

CANONICAL_CLASSES = [
    "inclusion",                     # 1  - merged: NEU.inclusion + GC10.inclusion + XSDD.slag_inclusion
    "scratches",                     # 2  - merged: NEU.scratches + XSDD.surface_scratches
    "crazing",                       # 3  - NEU
    "patches",                       # 4  - NEU
    "pitted_surface",                # 5  - NEU
    "rolled-in_scale",               # 6  - NEU
    "punching_hole",                 # 7  - GC10
    "weld_line",                     # 8  - GC10
    "crescent_gap",                  # 9  - GC10
    "water_spot",                    # 10 - GC10
    "oil_spot",                      # 11 - GC10
    "silk_spot",                     # 12 - GC10
    "waist_folding",                 # 13 - GC10
    "crease",                        # 14 - GC10
    "rolled_pit",                    # 15 - GC10
    "red_iron_sheet",                # 16 - XSDD
    "iron_sheet_ash",                # 17 - XSDD
    "oxide_scale_of_plate_system",   # 18 - XSDD
    "oxide_scale_of_temperature",    # 19 - XSDD
    "finishing_roll_printing",       # 20 - XSDD
]

assert len(CANONICAL_CLASSES) == 20, "Union harus tepat 20 kelas, cek README bagian 2"
assert len(set(CANONICAL_CLASSES)) == 20, "Ada duplikat di CANONICAL_CLASSES"

CANONICAL_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}

# ---------------------------------------------------------------------------
# 3. Mapping (dataset, label_asli) -> kelas_kanonik
#    Hanya "inclusion" dan "scratches" yang di-merge lintas dataset;
#    sisanya 1:1 tapi tetap didaftarkan eksplisit supaya tidak ada asumsi
#    tersembunyi (lihat README.md bagian 2, catatan soal rolled-in_scale
#    vs oxide_scale_*).
# ---------------------------------------------------------------------------

DATASET_TO_CANONICAL: dict[str, dict[str, str]] = {
    "gc10": {
        "punching_hole": "punching_hole",
        "weld_line": "weld_line",
        "crescent_gap": "crescent_gap",
        "water_spot": "water_spot",
        "oil_spot": "oil_spot",
        "silk_spot": "silk_spot",
        "inclusion": "inclusion",
        "waist_folding": "waist_folding",
        "crease": "crease",
        "rolled_pit": "rolled_pit",
    },
    "neu_cls": {
        "crazing": "crazing",
        "inclusion": "inclusion",
        "patches": "patches",
        "pitted_surface": "pitted_surface",
        "rolled-in_scale": "rolled-in_scale",
        "scratches": "scratches",
    },
    "neu_det": {
        "crazing": "crazing",
        "inclusion": "inclusion",
        "patches": "patches",
        "pitted_surface": "pitted_surface",
        "rolled-in_scale": "rolled-in_scale",
        "scratches": "scratches",
    },
    "xsdd": {
        "slag_inclusion": "inclusion",
        "red_iron_sheet": "red_iron_sheet",
        "iron_sheet_ash": "iron_sheet_ash",
        "surface_scratches": "scratches",
        "oxide_scale_of_plate_system": "oxide_scale_of_plate_system",
        "finishing_roll_printing": "finishing_roll_printing",
        "oxide_scale_of_temperature": "oxide_scale_of_temperature",
    },
}

for _ds, _mapping in DATASET_TO_CANONICAL.items():
    _orig = set(DATASET_CLASSES[_ds])
    _mapped = set(_mapping.keys())
    assert _orig == _mapped, f"Mapping {_ds} tidak lengkap: {_orig ^ _mapped}"
    for _canon in _mapping.values():
        assert _canon in CANONICAL_TO_ID, f"{_canon} bukan kelas kanonik valid ({_ds})"


@dataclass
class DatasetSpec:
    """Deskripsi ringkas sebuah dataset untuk keperluan loader/scripts."""

    name: str
    classes: list[str] = field(default_factory=list)
    task: str = "classification"  # "classification" | "detection"


def get_dataset_classes(dataset_name: str) -> list[str]:
    if dataset_name not in DATASET_CLASSES:
        raise KeyError(
            f"Dataset '{dataset_name}' tidak dikenal. Pilihan: {list(DATASET_CLASSES)}"
        )
    return DATASET_CLASSES[dataset_name]


def canonical_id_for(dataset_name: str, original_label: str) -> int:
    """Ubah label asli sebuah dataset menjadi index kelas kanonik (0-19)."""
    mapping = DATASET_TO_CANONICAL[dataset_name]
    canonical_name = mapping[original_label]
    return CANONICAL_TO_ID[canonical_name]


def build_label_remap(dataset_name: str) -> dict[int, int]:
    """
    Bangun dict {local_class_id -> canonical_class_id} berdasarkan urutan
    DATASET_CLASSES[dataset_name]. Dipakai saat remap file label
    (folder classification atau .txt YOLO) ke skema gabungan (combined).
    """
    classes = get_dataset_classes(dataset_name)
    return {
        local_id: canonical_id_for(dataset_name, label)
        for local_id, label in enumerate(classes)
    }


def combined_class_names_for(dataset_names: list[str]) -> list[str]:
    """
    Subset kelas kanonik yang benar-benar dipakai oleh gabungan dataset
    tertentu (misal B4 = GC10 + NEU-DET saja, tanpa X-SDD karena tidak
    punya bbox -> 16 kelas, bukan 20). Urutan tetap mengikuti
    CANONICAL_CLASSES supaya id konsisten dengan skenario classification.
    """
    used = set()
    for ds in dataset_names:
        used.update(DATASET_TO_CANONICAL[ds].values())
    return [c for c in CANONICAL_CLASSES if c in used]
