#!/usr/bin/env python3
"""Helper script for exporting + running RTTS models on Jetson Nano."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP = ROOT / "src" / "exps" / "example" / "custom" / "rtts_yolox_s.py"
DEFAULT_OUTPUT = ROOT / "YOLOX_outputs" / "rtts_yolox_s"


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _run(cmd: list[str], env: Dict[str, str]) -> None:
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def _ensure_trt(
    exp_path: Path,
    exp_name: str,
    ckpt: Optional[str],
    workspace: int,
    batch: int,
    env: Dict[str, str],
) -> None:
    trt_marker = ROOT / "YOLOX_outputs" / exp_name / "model_trt.pth"
    if trt_marker.exists():
        return
    cmd = [
        sys.executable,
        "src/tools/trt.py",
        "-f",
        str(exp_path),
        "-expn",
        exp_name,
        "-w",
        str(workspace),
        "-b",
        str(batch),
    ]
    if ckpt:
        cmd.extend(["-c", ckpt])
    _run(cmd, env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default=str(DEFAULT_EXP), help="Experiment file used for training.")
    parser.add_argument("--exp-name", default="rtts_yolox_s", help="Experiment name / output folder.")
    parser.add_argument(
        "--ckpt",
        default=str(DEFAULT_OUTPUT / "best_ckpt.pth"),
        help="Checkpoint to convert. Leave empty to use YOLOX_outputs/<exp>/best_ckpt.pth",
    )
    parser.add_argument("--mode", choices=["image", "video", "webcam"], default="image")
    parser.add_argument("--input", help="Image or video path for inference.")
    parser.add_argument("--cam-id", type=int, default=0, help="Camera index for webcam mode.")
    parser.add_argument("--save-result", action="store_true", help="Persist annotated outputs.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold.")
    parser.add_argument("--img-size", type=int, default=640, help="Test resolution (square).")
    parser.add_argument("--workspace", type=int, default=32, help="TensorRT max workspace size (power-of-two).")
    parser.add_argument("--batch", type=int, default=1, help="TensorRT batch size.")
    parser.add_argument("--skip-build", action="store_true", help="Assume TensorRT artifacts already exist.")
    parser.add_argument("--datadir", default=str(ROOT / "data"), help="Dataset root (passed via YOLOX_DATADIR).")
    args = parser.parse_args()

    exp_path = _resolve(args.exp)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment file {exp_path} not found.")

    env = os.environ.copy()
    env.setdefault("YOLOX_DATADIR", str(_resolve(args.datadir)))

    ckpt = args.ckpt
    if ckpt:
        ckpt_path = _resolve(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
        ckpt = str(ckpt_path)

    if not args.skip_build:
        _ensure_trt(exp_path, args.exp_name, ckpt, args.workspace, args.batch, env)

    demo_cmd = [
        sys.executable,
        "src/tools/demo.py",
        args.mode,
        "-f",
        str(exp_path),
        "-expn",
        args.exp_name,
        "--device",
        "gpu",
        "--trt",
        "--conf",
        str(args.conf),
        "--nms",
        str(args.nms),
        "--tsize",
        str(args.img_size),
    ]
    if args.save_result:
        demo_cmd.append("--save_result")
    if args.mode in {"image", "video"}:
        if not args.input:
            raise ValueError("--input is required for image/video modes.")
        demo_cmd.extend(["--path", args.input])
    if args.mode == "webcam":
        demo_cmd.extend(["--camid", str(args.cam_id)])

    _run(demo_cmd, env)


if __name__ == "__main__":
    main()
