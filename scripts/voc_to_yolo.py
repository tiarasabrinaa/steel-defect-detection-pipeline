"""
Converts Pascal VOC XML annotations (GC10-DET, NEU-DET) to YOLO label format.

Expected VOC XML structure:
    <annotation>
      <size><width>W</width><height>H</height></size>
      <object>
        <name>class_name</name>
        <bndbox><xmin>..</xmin><ymin>..</ymin><xmax>..</xmax><ymax>..</ymax></bndbox>
      </object>
      ...
    </annotation>

Used as a library (called from scripts/prepare_data.py) and as a standalone
CLI for debugging a single file:
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
    """Return (yolo_lines, raw_objects). Objects with a class name not
    present in `class_to_id` are skipped with a warning rather than raising,
    so a single malformed XML does not fail the whole conversion.

    `name_remap` (optional): translates the <name> tag to a canonical class
    name before the `class_to_id` lookup - used for GC10-DET, whose XML
    <name> values are pinyin (e.g. "3_yueyawan") rather than English. This
    is applied per object, not per file: a single GC10 image can contain
    more than one defect type."""
    width, height, objects = parse_voc_xml(xml_path)
    lines = []
    for name, xmin, ymin, xmax, ymax in objects:
        canonical_name = name_remap.get(name, name) if name_remap else name
        if canonical_name not in class_to_id:
            print(f"WARNING: label '{name}' (remapped: '{canonical_name}') in {xml_path} is not in the class list, skipping")
            continue
        cls_id = class_to_id[canonical_name]
        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, objects


def main():
    parser = argparse.ArgumentParser(description="Debug: convert one VOC XML file to YOLO lines")
    parser.add_argument("--xml", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    args = parser.parse_args()

    class_to_id = {c: i for i, c in enumerate(args.classes)}
    lines, objects = voc_to_yolo_lines(args.xml, class_to_id)
    print(f"Found {len(objects)} objects, {len(lines)} converted successfully:")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
