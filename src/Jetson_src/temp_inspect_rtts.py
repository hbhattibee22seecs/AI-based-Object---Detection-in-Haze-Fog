#!/usr/bin/env python3
"""Quick sanity check for RTTS annotations."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANN_DIR = ROOT / "data" / "RTTS" / "Annotations"
DEFAULT_CACHE = ROOT / "results" / "rtts_class_counts.json"
RTTS_CLASSES = ("person", "car", "bus", "bicycle", "motorbike")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=DEFAULT_ANN_DIR,
        help="Path to RTTS/Annotations (defaults to repo data/RTTS/Annotations)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Where to store/read cached class counts",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only scan the first N XML files"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Force a rescan instead of using cache"
    )
    return parser.parse_args()


def _scan_annotations(ann_dir: Path, limit: Optional[int]):
    ann_dir = ann_dir.expanduser().resolve()
    if not ann_dir.exists():
        raise SystemExit(f"Annotation directory not found: {ann_dir}")

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found inside {ann_dir}")

    counter = Counter()
    images = 0
    for xml_path in xml_files:
        images += 1
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            print(f"Failed to parse {xml_path.name}: {exc}")
            continue
        for obj in tree.findall("object"):
            label = (obj.findtext("name") or "").strip()
            if label:
                counter[label] += 1
        if limit and images >= limit:
            break
    return counter, images


def main():
    args = _parse_args()
    use_cache = args.cache and args.cache.exists() and not args.refresh

    if use_cache:
        print(f"Loading cached counts from {args.cache}")
        data = json.loads(args.cache.read_text())
    else:
        counter, images = _scan_annotations(args.ann_dir, args.limit)
        data = {
            "ann_dir": str(args.ann_dir),
            "images_scanned": images,
            "counts": dict(counter),
        }
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(data, indent=2))
        print(f"Wrote cache to {args.cache}")

    print("\nRTTS class histogram:")
    total = 0
    for cls in RTTS_CLASSES:
        value = data["counts"].get(cls, 0)
        total += value
        print(f"  {cls:10s}: {value}")
    print(f"Total objects counted: {total}")
    print(f"Images scanned: {data['images_scanned']}")


if __name__ == "__main__":
    main()
