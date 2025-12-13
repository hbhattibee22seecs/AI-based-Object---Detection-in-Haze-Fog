#!/usr/bin/env python3
"""Convenience launcher for RTTS fine-tuning, tuned for Colab/T4 setups."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP = ROOT / "src" / "exps" / "example" / "custom" / "rtts_yolox_s.py"
DEFAULT_WEIGHTS = ROOT / "src" / "yolox" / "weights" / "yolox_s.pth"


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _verify_dataset(datadir: Path) -> None:
    annotations = datadir / "RTTS" / "Annotations"
    images = datadir / "RTTS" / "JPEGImages"
    train_split = datadir / "RTTS" / "ImageSets" / "Main" / "train.txt"
    missing = [p for p in (annotations, images, train_split) if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "RTTS dataset is incomplete. Expected files were not found:\n" + missing_str
        )


def _build_train_command(args, exp_path: Path, ckpt_path: Optional[Path]) -> list[str]:
    cmd = [
        sys.executable,
        "src/tools/train.py",
        "-f",
        str(exp_path),
        "-d",
        str(args.devices),
        "-b",
        str(args.batch_size),
        "-o",
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.cache:
        cmd.append("--cache")
    if args.exp_name:
        cmd.extend(["--exp-name", args.exp_name])
    if ckpt_path is not None:
        cmd.extend(["-c", str(ckpt_path)])
    if args.resume:
        cmd.extend(["-r", args.resume])
    if args.extra is not None:
        cmd.extend(args.extra)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp",
        default=str(DEFAULT_EXP),
        help="Path to the RTTS experiment file (create it via docs instructions).",
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU.")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Enable mixed precision (recommended on T4/Jetson).",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable mixed precision.",
    )
    parser.set_defaults(fp16=True)
    parser.add_argument("--cache", action="store_true", help="Enable RAM dataset caching.")
    parser.add_argument(
        "--ckpt",
        default=str(DEFAULT_WEIGHTS),
        help="Path to COCO-pretrained checkpoint (download beforehand).",
    )
    parser.add_argument(
        "--exp-name",
        default="rtts_yolox_s",
        help="Output folder name inside YOLOX_outputs.",
    )
    parser.add_argument(
        "--datadir",
        default=str(ROOT / "data"),
        help="Base data directory that contains the RTTS folder.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint file to resume from (default: latest auto).",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Apply safe defaults for a single T4 in Google Colab.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed verbatim to tools/train.py",
    )
    args = parser.parse_args()

    if args.colab:
        # Colab T4 provides a single GPU with 16 GB RAM; keep defaults modest.
        args.devices = 1
        args.batch_size = min(args.batch_size, 16)
        args.fp16 = True

    exp_path = _resolve(args.exp)
    if not exp_path.exists():
        raise FileNotFoundError(
            f"Experiment file {exp_path} not found. Create it as described in docs/rtts_training.md."
        )

    datadir = _resolve(args.datadir)
    _verify_dataset(datadir)

    ckpt_path: Optional[Path] = None
    if args.ckpt:
        resolved_ckpt = _resolve(args.ckpt)
        if not resolved_ckpt.exists():
            print(
                f"[WARN] Pretrained checkpoint {resolved_ckpt} is missing. Training will start from scratch.",
                file=sys.stderr,
            )
        else:
            ckpt_path = resolved_ckpt

    env = os.environ.copy()
    env.setdefault("YOLOX_DATADIR", str(datadir.resolve()))

    cmd = _build_train_command(args, exp_path, ckpt_path)
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
