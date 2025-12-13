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
Transfer the ONNX/engine, experiment file, and class names:
```bash
scp -r exports/RTTS_yolox_s.onnx jetson@nano.local:~/haze-yolox/
scp src/exps/example/custom/rtts_yolox_s.py jetson@nano.local:~/haze-yolox/exps/
```

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

## 5. Run Inference
### Image or Video file
```bash
python3 src/demo/TensorRT/python/demo.py \
  --trt-file exports/rtts_yolox_s_fp16.trt \
  --input demo/hazy_clip.mp4 \
  --conf 0.25 --nms 0.5 --save_result
```
### Live USB/CSI camera
```bash
python3 src/demo/TensorRT/python/demo.py \
  --trt-file exports/rtts_yolox_s_fp16.trt \
  --cam-id 0 --conf 0.3 --nms 0.45 --fp16
```
Use `--img-size 640 640` to match your training resolution. Results save to `YOLOX_outputs/trt_demo/`.

## 6. Performance Tips
- Keep input resolution at 640² for ~10 FPS; drop to 512² for smoother feeds.
- Disable MixUp/Mosaic during export (`self.test_size = (640, 640)` in the exp file).
- Pin Jetson to MAX-Q 10 W mode and power via 5 V / 4 A PSU to avoid throttling.
- For repeated deployments, prebuild engines per precision (`FP16` vs `INT8`) and store under `exports/`.

You now have an end-to-end path from RTTS training to Jetson Nano inference.
