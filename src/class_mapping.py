"""Single source of truth for label harmonization across datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

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

CANONICAL_CLASSES = [
    "inclusion",
    "scratches",
    "crazing",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "punching_hole",
    "weld_line",
    "crescent_gap",
    "water_spot",
    "oil_spot",
    "silk_spot",
    "waist_folding",
    "crease",
    "rolled_pit",
    "red_iron_sheet",
    "iron_sheet_ash",
    "oxide_scale_of_plate_system",
    "oxide_scale_of_temperature",
    "finishing_roll_printing",
]

assert len(CANONICAL_CLASSES) == 20
assert len(set(CANONICAL_CLASSES)) == 20

CANONICAL_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}

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
    assert _orig == _mapped, f"Incomplete mapping for {_ds}: {_orig ^ _mapped}"
    for _canon in _mapping.values():
        assert _canon in CANONICAL_TO_ID, f"{_canon} is not a valid canonical class ({_ds})"


@dataclass
class DatasetSpec:
    name: str
    classes: list[str] = field(default_factory=list)
    task: str = "classification"


def get_dataset_classes(dataset_name: str) -> list[str]:
    if dataset_name not in DATASET_CLASSES:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Options: {list(DATASET_CLASSES)}")
    return DATASET_CLASSES[dataset_name]


def canonical_id_for(dataset_name: str, original_label: str) -> int:
    mapping = DATASET_TO_CANONICAL[dataset_name]
    canonical_name = mapping[original_label]
    return CANONICAL_TO_ID[canonical_name]


def build_label_remap(dataset_name: str) -> dict[int, int]:
    classes = get_dataset_classes(dataset_name)
    return {
        local_id: canonical_id_for(dataset_name, label)
        for local_id, label in enumerate(classes)
    }


def combined_class_names_for(dataset_names: list[str]) -> list[str]:
    used = set()
    for ds in dataset_names:
        used.update(DATASET_TO_CANONICAL[ds].values())
    return [c for c in CANONICAL_CLASSES if c in used]
