"""
Converter Pascal VOC XML (dipakai GC10-DET & NEU-DET) -> format label YOLO.

VOC XML dianggap punya struktur standar:
    <annotation>
      <size><width>W</width><height>H</height></size>
      <object>
        <name>class_name</name>
        <bndbox><xmin>..</xmin><ymin>..</ymin><xmax>..</xmax><ymax>..</ymax></bndbox>
      </object>
      ...
    </annotation>

Dipakai sebagai library (dipanggil dari scripts/prepare_data.py) maupun CLI
standalone untuk debugging satu file:
    python scripts/voc_to_yolo.py --xml path/to/file.xml --classes crazing inclusion ...
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_voc_xml(xml_path: str | Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = int(float(size.find("width").text))
    height = int(float(size.find("height").text))

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        objects.append((name, xmin, ymin, xmax, ymax))
    return width, height, objects


def voc_to_yolo_lines(xml_path: str | Path, class_to_id: dict[str, int]) -> tuple[list[str], list[tuple]]:
    """Return (yolo_lines, raw_objects). Object dengan nama kelas yang tidak
    dikenal di `class_to_id` di-skip (dengan warning) alih-alih error, supaya
    satu XML yang typo tidak menggagalkan seluruh konversi dataset."""
    width, height, objects = parse_voc_xml(xml_path)
    lines = []
    for name, xmin, ymin, xmax, ymax in objects:
        if name not in class_to_id:
            print(f"WARNING: label '{name}' di {xml_path} tidak ada di class list, di-skip")
            continue
        cls_id = class_to_id[name]
        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, objects


def voc_to_yolo_lines_forced_class(xml_path: str | Path, class_id: int) -> list[str]:
    """
    Sama seperti `voc_to_yolo_lines`, TAPI paksa SEMUA object di file ini
    pakai satu `class_id` yang udah ditentukan dari luar, mengabaikan isi
    tag <name> di XML sepenuhnya.

    Dipakai buat GC10-DET: <name> di XML raw-nya itu pinyin+angka (misal
    "3_yueyawan"), bukan teks yang bisa di-lookup ke class list Inggris -
    ground truth kelas yang reliable buat dataset ini justru dari folder
    angka (1-10) tempat gambarnya berada (lihat GC10_FOLDER_TO_CLASS di
    scripts/prepare_data.py, sumbernya "Defects Description.xlsx").
    """
    width, height, objects = parse_voc_xml(xml_path)
    lines = []
    for _, xmin, ymin, xmax, ymax in objects:
        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Debug: convert 1 VOC XML -> YOLO lines")
    parser.add_argument("--xml", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    args = parser.parse_args()

    class_to_id = {c: i for i, c in enumerate(args.classes)}
    lines, objects = voc_to_yolo_lines(args.xml, class_to_id)
    print(f"Ditemukan {len(objects)} object, {len(lines)} berhasil dikonversi:")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
