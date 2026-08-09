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


def voc_to_yolo_lines(
    xml_path: str | Path,
    class_to_id: dict[str, int],
    name_remap: dict[str, str] | None = None,
) -> tuple[list[str], list[tuple]]:
    """Return (yolo_lines, raw_objects). Object dengan nama kelas yang tidak
    dikenal di `class_to_id` di-skip (dengan warning) alih-alih error, supaya
    satu XML yang typo tidak menggagalkan seluruh konversi dataset.

    `name_remap` (opsional): terjemahkan isi tag <name> ke nama kelas
    kanonik SEBELUM di-lookup ke `class_to_id` - dipakai buat GC10-DET,
    yang <name> XML-nya pinyin+angka (misal "3_yueyawan"), bukan Inggris.
    PENTING: remap ini per-OBJECT, bukan per-file - satu gambar GC10 bisa
    punya lebih dari satu jenis defect berbeda dalam 1 XML (defect utama
    sesuai folder + defect sekunder kelas lain), jadi tiap object HARUS
    dibaca <name>-nya masing-masing, tidak boleh dipukul rata 1 kelas per file
    (lihat commit fix bug ini utk detail kasus nyata yang ketauan)."""
    width, height, objects = parse_voc_xml(xml_path)
    lines = []
    for name, xmin, ymin, xmax, ymax in objects:
        canonical_name = name_remap.get(name, name) if name_remap else name
        if canonical_name not in class_to_id:
            print(f"WARNING: label '{name}' (remap: '{canonical_name}') di {xml_path} tidak ada di class list, di-skip")
            continue
        cls_id = class_to_id[canonical_name]
        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, objects


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
