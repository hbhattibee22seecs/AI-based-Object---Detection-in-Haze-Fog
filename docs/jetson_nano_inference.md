# Jetson Nano Inference Guide

This document walks through exporting a trained RTTS-aware YOLOX checkpoint and running it in real time on an NVIDIA Jetson Nano (4 GB).

## 1. Train & Export on the Host
1. Finish RTTS training (see `docs/rtts_training.md`).
2. Export ONNX from your best checkpoint:
   ```powershell
   python src\tools\export_onnx.py ^
     -f src\exps\example\custom\rtts_yolox_s.py ^
     -c YOLOX_outputs\rtts_yolox_s\best_ckpt.pth ^
     --output-file exports\rtts_yolox_s.onnx ^
     --input [640,640]
   ```
3. (Optional) Build a TensorRT engine on the host if you have matching CUDA/TensorRT versions; otherwise build directly on Jetson.

## 2. Prepare the Jetson Nano
1. Flash JetPack 5.1.2+ (ships with CUDA 11.4, cuDNN 8.6, TensorRT 8.5).
2. Put the board in 10 W mode and lock clocks:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```
3. Install dependencies (Jetson wheels ship with JetPack, only extra packages needed):
   ```bash
   sudo apt update && sudo apt install -y python3-pip libopencv-dev
   pip3 install --upgrade pip
   pip3 install -r requirements.txt --extra-index-url https://pypi.ngc.nvidia.com
   ```
   > On Jetson you must install the matching `torch`/`torchvision` wheels from NVIDIA's index (the default pip wheels target x86_64).

## 3. Copy Artifacts
Use the `rtts_yolox_s_artifacts.zip` bundle produced by the Colab notebook (saved under `results/rtts_yolox_s_artifacts.zip`), or copy files individually:
```bash
scp rtts_yolox_s_artifacts.zip jetson@nano.local:~/haze-yolox/
ssh jetson@nano.local "cd haze-yolox && unzip -o rtts_yolox_s_artifacts.zip"
```
The archive includes `rtts_yolox_s.onnx`, the TensorRT weights, the experiment file, the consolidated `results/` directory (plots, checkpoints, inference samples), and the helper script `src/Jetson_src/jetson_nano_runner.py`. If you prefer manual copying, ensure `exports/`, `YOLOX_outputs/rtts_yolox_s/`, `results/`, and `src/Jetson_src/jetson_nano_runner.py` all land on the Jetson.

## 4. Build a TensorRT Engine (on Jetson)
```bash
python3 src/tools/trt.py \
  -f src/exps/example/custom/rtts_yolox_s.py \
  -c exports/rtts_yolox_s.onnx \
  --trt_fp16 --device 0 \
  --output-name exports/rtts_yolox_s_fp16.trt
```
Alternatively, use `trtexec`:
```bash
/opt/nvidia/tensorrt/bin/trtexec \
  --onnx=exports/rtts_yolox_s.onnx \
  --saveEngine=exports/rtts_yolox_s_fp16.trt \
  --fp16 --workspace=2048
```
> **Why TensorRT?** The Jetson Nano struggles to exceed ~2 FPS with the pure PyTorch model. The helper script (`jetson_nano_runner.py`) requires a TensorRT engine (`model_trt.pth`) and will automatically rebuild it via `tools/trt.py` when `--rebuild` (or the initial run) is invoked. Skip this step only if you already copied a matching engine from another machine.

## 5. Run Inference
### Recommended: `jetson_nano_runner.py`
```bash
python3 src/Jetson_src/jetson_nano_runner.py \
  --mode image \
  --input data/RTTS/JPEGImages/AM_Bing_211.png \
  --conf 0.25 --nms 0.45 --tsize 640
```
- Switch to `--mode video --input fog_drive.mp4` for offline footage.
- Use `--mode webcam --cam-id 0` to connect a CSI/USB camera.
- Pass `--rebuild` when you want to regenerate `YOLOX_outputs/<exp_name>/model_trt.pth`; the script refuses to run without a valid TensorRT engine to guarantee real-time throughput.

### Direct `demo.py` usage (advanced)
```bash
python3 src/tools/demo.py image \
  -f src/exps/example/custom/rtts_yolox_s.py \
  --trt --trt-file YOLOX_outputs/rtts_yolox_s/model_trt.pth \
  --path data/RTTS/JPEGImages/AM_Bing_211.png \
  --conf 0.25 --nms 0.45 --save_result
```
Add `video` or `webcam` in place of `image` plus the relevant `--path/--camid`. Results save to `YOLOX_outputs/trt_demo/`.

## 6. Performance Tips
- Keep input resolution at 640² for ~10 FPS; drop to 512² for smoother feeds.
- Disable MixUp/Mosaic during export (`self.test_size = (640, 640)` in the exp file).
- Pin Jetson to MAX-Q 10 W mode and power via 5 V / 4 A PSU to avoid throttling.
- For repeated deployments, prebuild engines per precision (`FP16` vs `INT8`) and store under `exports/`.

You now have an end-to-end path from RTTS training to Jetson Nano inference.
