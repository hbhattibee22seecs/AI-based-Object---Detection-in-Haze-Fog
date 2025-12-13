"""Fast utility for summarizing RTTS annotation labels."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


def parse_annotations(ann_dir: Path, limit: Optional[int] = None) -> Counter:
    """Stream RTTS xml files and accumulate class counts."""
    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No xml files found under {ann_dir}")
    if limit is not None:
        xml_files = xml_files[:limit]

    classes = Counter()
    for idx, xml_path in enumerate(xml_files, 1):
        # iterparse keeps memory usage low for long files
        for _, elem in ET.iterparse(xml_path):
            if elem.tag == "object":
                name_node = elem.find("name")
                if name_node is not None:
                    name = name_node.text.strip()
                    if name:
                        classes[name] += 1
                elem.clear()
        if idx % 250 == 0 or idx == len(xml_files):
            print(f"[inspect] processed {idx}/{len(xml_files)} files")
    return classes


def load_or_create_counts(ann_dir: Path, cache_path: Path, refresh: bool, limit: Optional[int]) -> Counter:
    if cache_path.exists() and not refresh:
        print(f"[inspect] loading cached counts from {cache_path}")
        return Counter(json.loads(cache_path.read_text(encoding="utf-8")))

    counts = parse_annotations(ann_dir, limit)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"[inspect] wrote cache to {cache_path}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path(__file__).parent / "data" / "RTTS" / "Annotations",
                        help="Path to RTTS Annotations directory")
    parser.add_argument("--cache", type=Path, default=Path("results") / "rtts_class_counts.json",
                        help="Where to store json cache of computed counts")
    parser.add_argument("--refresh", action="store_true", help="Force re-scan even if cache exists")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional number of xml files to scan (for quick smoke tests)")
    args = parser.parse_args()

    counts = load_or_create_counts(args.annotations, args.cache, args.refresh, args.limit)
    print(f"num_classes {len(counts)}")
    print(f"classes {sorted(counts)}")


if __name__ == "__main__":
    main()
