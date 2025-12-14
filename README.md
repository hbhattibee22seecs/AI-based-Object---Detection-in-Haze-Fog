# 🌫️ “A Lightweight Dehazing-Aware Object Detection Framework for Embedded Vision Systems”

## 📘 Project Overview
This repository contains the **Phase 1: Literature Review & Feasibility** work for the research project conducted at the  
**School of Electrical Engineering and Computer Science (SEECS)**,  
**National University of Sciences and Technology (NUST), Islamabad, Pakistan**.

The study focuses on enhancing **object detection performance under haze and fog** using **lightweight deep-learning architectures** optimized for **embedded deployment** on the **NVIDIA Jetson Nano (4 GB)**.  
The project investigates how modern **dehazing-aware object-detection networks** achieve high accuracy and real-time performance despite limited hardware resources.

---

## 👥 Authors
- **Wassi Haider Kabir**  
- **Muhammad Ashar Javid**  
- **Hamza Irshad Bhatti**  
- **Ammar**  
School of Electrical Engineering and Computer Science (SEECS),  
National University of Sciences and Technology (NUST), Islamabad, Pakistan  

---

## 🧠 Research Summary
The literature review consolidates research from **2019–2025** focusing on haze/fog object detection, highlighting key advances in **joint dehazing–detection frameworks** and **embedded optimization**.

### Key Reviewed Models:
- **AOD-YOLOv5s (Li et al., 2024)** — Hybrid dehazing and YOLOv5 integration achieving **76.8 % mAP** with **42 % parameter reduction**.  
- **TSMD-Net (Raju & Srinivas, 2024)** — Dual-attention dehazing network validated on Jetson Nano (**16.3 M parameters**, **1.9 s/frame**).  
- **YOLOv5s-Fog (Meng et al., 2023)** — Swin-Transformer attention yielding **+5.4 % mAP** improvement on foggy datasets.  
- **IDOD-YOLOv7 (Qiu et al., 2023)** — SAIP-based joint dehazing and detection achieving **+7.9 % mAP** and **71 FPS**.  
- **Improved YOLOX (Liu et al., 2025)** — Dual-branch attention with Focal Loss for foggy vehicle detection, **81.2 % mAP** at **60 FPS**, demonstrating scalability to Jetson Nano.

All reviewed models maintain **< 70 GFLOPs** and **< 20 M parameters**, confirming their **feasibility for Jetson Nano** deployment using FP16 or INT8 quantization.

---

## ⚙️ Repository Structure
📁 /docs
├── Literature_Review_Phase1.pdf # Final IEEE-formatted review
├── Feasibility_Summary.txt # Jetson Nano deployment notes
└── references/ # Verified research papers (PDFs or DOI links)

📁 /src
└── (Phase 2 implementation – to be added)

📁 /results
└── (Future inference benchmarks)

---

## 📥 RTTS Dataset & Pipeline Quickstart
This workspace is already structured to train on RTTS using YOLOX.

### 1) Install dependencies
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Install PyTorch (CPU vs GPU)
PyTorch is intentionally not pinned in `requirements.txt` because the correct wheel depends on your machine.

- **CPU-only** (works everywhere):
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- **CUDA GPU**: install a CUDA-enabled PyTorch build that matches your CUDA driver/toolkit (use the official selector on pytorch.org). Example (choose the right `cu12x` for your setup):
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

Sanity check:
```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

### 3) Point YOLOX at the local dataset
YOLOX resolves datasets via `YOLOX_DATADIR`.
```powershell
$env:PYTHONPATH = "$PWD\src"
$env:YOLOX_DATADIR = "$PWD\data"
```

### 4) Quick smoke test (50 iterations → checkpoint → inference)
This is the fastest end-to-end test (and it shows progress in terminal + TensorBoard):
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train1epoch_and_infer.ps1 -ExpName rtts_yolox_s_smoke -BatchSize 2 -TrainIters 50
```
- TensorBoard URL prints in the terminal (default `http://localhost:6006`).
- Output image with boxes is saved under `YOLOX_outputs\rtts_yolox_s_smoke\vis_res\...`.

### 5) Full training run (GPU if available)
```powershell
# GPU training (default if torch.cuda.is_available() is True)
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

python src\tools\train.py -f src\exps\example\custom\rtts_yolox_s.py -d 1 -b 16 -c src\yolox\weights\yolox_s.pth -expn rtts_yolox_s_train --fp16 -o
```

### 6) Inference on a single image
```powershell
python src\tools\demo.py image -f src\exps\example\custom\rtts_yolox_s.py -c YOLOX_outputs\rtts_yolox_s_train\latest_ckpt.pth -expn rtts_yolox_s_train --device gpu --path data\RTTS\JPEGImages\AM_Bing_211.png --save_result --conf 0.001
```

Detailed guidance lives in `docs/rtts_training.md`.

---

## 🧭 CPU vs GPU Switching
YOLOX will use GPU automatically when **(1)** you installed a CUDA-enabled PyTorch and **(2)** a CUDA device is visible.

- **Force CPU** (useful for debugging):
  ```powershell
  $env:CUDA_VISIBLE_DEVICES = "-1"
  python src\tools\train.py -f src\exps\example\custom\rtts_yolox_s.py -d 1 -b 2 -c src\yolox\weights\yolox_s.pth -expn rtts_cpu_debug max_epoch 1 data_num_workers 0
  ```
- **Force GPU** (if you have CUDA PyTorch):
  ```powershell
  Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
  python src\tools\train.py -f src\exps\example\custom\rtts_yolox_s.py -d 1 -b 16 -c src\yolox\weights\yolox_s.pth -expn rtts_gpu_train
  ```

---

## 🚀 Jetson Nano Inference
- Export your best checkpoint to ONNX: `python src/tools/export_onnx.py ...`.
- Build a TensorRT engine on Jetson with `python src/tools/trt.py --trt_fp16 ...`.
- Run the FP16 engine via `python src/demo/TensorRT/python/demo.py --trt-file exports/rtts_yolox_s_fp16.trt --cam-id 0`.

See `docs/jetson_nano_inference.md` for power-mode tweaks, dependency notes, and sample commands.

---

## 🔍 Key Highlights
- ✅ Comprehensive **IEEE-formatted literature review** (2019–2025)  
- 🌫️ Focus on **joint dehazing and detection** under fog and haze  
- ⚙️ Verified **Jetson Nano 4 GB** feasibility (embedded deployment)  
- 📚 Repository includes **verified research papers and sources**  
- 🚀 Provides a foundation for **Phase 2: Implementation & Benchmarking**

---

## 📚 Verified References
| No. | Paper | Source |
|----:|-------|--------|
| [1] | **Liu et al. (2025)** – *Vehicle Target Detection of Autonomous Driving Vehicles in Foggy Environments Based on an Improved YOLOX Network* | *Sensors 25 (1): 194* |
| [2] | **Qiu et al. (2023)** – *IDOD-YOLOv7: Image-Dehazing YOLOv7 for Object Detection in Low-Light Foggy Traffic Environments* | *Sensors 23 (13): 1347* |
| [3] | **Li et al. (2024)** – *Object Detection in Hazy Environments Based on an All-in-One Dehazing Network and YOLOv5* | *Electronics 13 (1862)* |
| [4] | **Raju & Srinivas (2024)** – *TSMD-Net: Two-Stage Mixed Dehazing Network* | *Digital Signal Processing 155 (104710)* |
| [5] | **Meng et al. (2023)** – *YOLOv5s-Fog: An Improved Model for Object Detection in Foggy Weather* | *Sensors 23 (11): 5321* |

---

## 💻 Hardware Feasibility (Jetson Nano 4 GB)
| Parameter | Specification |
|------------|---------------|
| **GPU** | 128-core Maxwell GPU |
| **RAM** | 4 GB LPDDR4 |
| **Target Models** | YOLOv5s-Fog, IDOD-YOLOv7, Improved YOLOX |
| **Precision** | FP16 / INT8 (TensorRT optimized) |
| **Expected FPS** | 10 – 15 FPS |
| **Power** | 5 V ⎓ 4 A Supply |

---

## 🧾 Mandatory Declaration
> ChatGPT (Deep Research Mode) was used to search, verify, and summarize peer-reviewed research papers and generate the narrative literature review.  
> All cited papers were opened and verified by the team.  
> Verified sources are included in the `/docs/references/` folder.

---

## 🧩 Future Work (Phase 2)
- Implement optimized **dehazing–detection pipelines** (Improved YOLOX / IDOD-YOLOv7 / AOD-YOLOv5).  
- Quantize and deploy on **Jetson Nano 4 GB** using TensorRT acceleration.  
- Benchmark **inference speed, mAP, and power consumption** under fog simulation.  
- Publish comparative results and hardware metrics in `/results/`.

---

## 🧑‍💻 Citation
If referencing this work:
```bibtex
@report{HazeNano2025,
  title   = {Object Detection in Hazy and Foggy Conditions on NVIDIA Jetson Nano},
  author  = {Wassi Haider Kabir and Muhammad Ashar Javid and Hamza Irshad Bhatti and Ammar},
  institution = {School of Electrical Engineering and Computer Science (SEECS), NUST},
  year    = {2025},
  note    = {Phase 1 – Literature Review and Feasibility Report}
}
