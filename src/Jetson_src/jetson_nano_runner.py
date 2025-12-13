#!/usr/bin/env python3
"""TensorRT-friendly inference helper tailored for Jetson Nano."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP = ROOT / "src" / "exps" / "example" / "custom" / "rtts_yolox_s.py"
DEFAULT_EXP_NAME = "rtts_yolox_s"
DEFAULT_CKPT = ROOT / "YOLOX_outputs" / DEFAULT_EXP_NAME / "best_ckpt.pth"
DEFAULT_TRT = ROOT / "YOLOX_outputs" / DEFAULT_EXP_NAME / "model_trt.pth"
DEMO_SCRIPT = ROOT / "src" / "tools" / "demo.py"


def _run(cmd: List[str]) -> None:
    print("[jetson_nano_runner]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def _ensure_trt(args: argparse.Namespace) -> None:
    if args.trt_file.exists() and not args.rebuild:
        print(f"[jetson_nano_runner] Reusing TensorRT weights at {args.trt_file}")
        return

    build_cmd = [
        "python3",
        "src/tools/trt.py",
        "-f",
        str(args.exp),
        "-c",
        str(args.ckpt),
        "-expn",
        args.exp_name,
    ]
    _run(build_cmd)

    built_file = ROOT / "YOLOX_outputs" / args.exp_name / "model_trt.pth"
    if not built_file.exists():
        raise FileNotFoundError(
            f"TensorRT conversion did not produce {built_file}. Check tools/trt.py logs."
        )
    args.trt_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_file, args.trt_file)
    print(f"[jetson_nano_runner] Copied TensorRT weights to {args.trt_file}")


def _launch_demo(args: argparse.Namespace) -> None:
    cmd = [
        "python3",
        str(DEMO_SCRIPT),
        args.mode,
        "-f",
        str(args.exp),
        "--device",
        "gpu",
        "--conf",
        str(args.conf),
        "--nms",
        str(args.nms),
        "--trt",
        "--trt-file",
        str(args.trt_file),
        "--save_result",
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.tsize:
        cmd.extend(["--tsize", str(args.tsize)])

    if args.mode in {"image", "video"}:
        cmd.extend(["--path", str(args.input)])
    else:
        cmd.extend(["--camid", str(args.cam_id)])

    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", type=Path, default=DEFAULT_EXP, help="Experiment file")
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME, help="Experiment/output name")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="Checkpoint path")
    parser.add_argument("--trt-file", type=Path, default=DEFAULT_TRT, help="TensorRT weight file")
    parser.add_argument("--mode", choices=["image", "video", "webcam"], default="image")
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data" / "RTTS" / "JPEGImages" / "AM_Bing_211.png")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--tsize", type=int, default=640, help="Square inference size")
    parser.add_argument("--datadir", type=Path, default=ROOT / "data")
    parser.add_argument("--rebuild", action="store_true", help="Force TensorRT rebuild")
    parser.add_argument("--skip-build", action="store_true", help="Skip TensorRT conversion")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference")
    args = parser.parse_args()

    os.environ.setdefault("YOLOX_DATADIR", str(args.datadir.resolve()))

    if not args.skip_build:
        _ensure_trt(args)

    _launch_demo(args)


if __name__ == "__main__":
    main()
