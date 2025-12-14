import argparse
import os
import random
from pathlib import Path


def _read_ids_from_file(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            image_id = line.strip()
            if image_id:
                ids.append(image_id)
    return sorted(set(ids))


def _infer_ids_from_annotations(ann_dir: Path) -> list[str]:
    ids = [p.stem for p in ann_dir.glob("*.xml")]
    return sorted(set(ids))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create RTTS VOC split files (train.txt/val.txt) under ImageSets/Main."
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default=os.environ.get("YOLOX_DATADIR", str(Path.cwd() / "data")),
        help="Path to YOLOX data dir (default: $YOLOX_DATADIR or ./data)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="RTTS",
        help="Dataset folder name under datadir (default: RTTS)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic shuffle (default: 42)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing train.txt/val.txt if present",
    )
    args = parser.parse_args()

    datadir = Path(args.datadir).expanduser().resolve()
    dataset_root = datadir / args.dataset
    splits_dir = dataset_root / "ImageSets" / "Main"
    ann_dir = dataset_root / "Annotations"

    if not splits_dir.is_dir():
        raise SystemExit(f"Missing splits dir: {splits_dir}")
    if not ann_dir.is_dir():
        raise SystemExit(f"Missing annotations dir: {ann_dir}")

    train_path = splits_dir / "train.txt"
    val_path = splits_dir / "val.txt"
    if not args.force and (train_path.exists() or val_path.exists()):
        raise SystemExit(
            f"Refusing to overwrite existing splits. Use --force to overwrite: {train_path}, {val_path}"
        )

    # Prefer basing splits off an existing list, if present.
    source = None
    ids: list[str] = []
    for candidate in (splits_dir / "trainval.txt", splits_dir / "train.txt", splits_dir / "test.txt"):
        if candidate.exists():
            ids = _read_ids_from_file(candidate)
            source = str(candidate)
            break

    if not ids:
        ids = _infer_ids_from_annotations(ann_dir)
        source = str(ann_dir)

    if not ids:
        raise SystemExit("No image ids found. Checked split files and Annotations/*.xml")

    rng = random.Random(args.seed)
    rng.shuffle(ids)

    val_ratio = float(args.val_ratio)
    if not (0.0 < val_ratio < 1.0):
        raise SystemExit("--val-ratio must be between 0 and 1")

    val_n = max(1, int(round(val_ratio * len(ids))))
    val_ids = ids[:val_n]
    train_ids = ids[val_n:]

    train_path.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    print(f"Source: {source}")
    print(f"Total ids: {len(ids)}")
    print(f"Wrote: {train_path} ({len(train_ids)} ids)")
    print(f"Wrote: {val_path} ({len(val_ids)} ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
