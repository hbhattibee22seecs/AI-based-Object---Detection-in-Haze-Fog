# Running the YOLOX Pipeline on RTTS

This guide explains how to fine-tune YOLOX on the Realistic Traffic Scenes (RTTS) subset of the RESIDE benchmark and reproduce the fog-aware training pipeline that powers this repository.

## 0. Before You Run
1. **Dataset in place**: `data/RTTS` must contain `Annotations/`, `JPEGImages/`, and `ImageSets/Main/{train,val,test}.txt` (already laid out in this repo).
2. **Experiment file ready**: use `src/exps/example/custom/rtts_yolox_s.py` (already provided) which wires the RTTS-specific dataset loader and expects the `ImageSets/Main/{train,val,test}.txt` splits.
3. **COCO weights downloaded**: place `yolox_s.pth` under `src/yolox/weights/` so we can fine-tune instead of training from scratch.
4. **Dataset sanity check**: run `python src/Jetson_src/temp_inspect_rtts.py --refresh` once to cache and confirm the five RTTS classes (`person`, `car`, `bus`, `bicycle`, `motorbike`).
5. **Environment hint**: optionally `set YOLOX_DATADIR=%CD%\data` (Windows) or `export YOLOX_DATADIR=$(pwd)/data` so YOLOX automatically locates the dataset tree.

Once the checklist is green, you can either use the standard `src/tools/train.py` CLI or the new helper `src/Jetson_src/train_rtts.py` (described below) to kick off training locally or on Colab T4 GPUs.

## 1. Prerequisites
- Python 3.9+ and CUDA 11.8+ on the training workstation.
- NVIDIA driver that matches your CUDA toolkit.
- Install the repo dependencies:
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- (Optional) Export `YOLOX_DATADIR` so YOLOX automatically locates datasets:
  ```powershell
  $env:YOLOX_DATADIR = "${PWD}/data"
  ```

## 2. Download RTTS
RTTS lives inside the RESIDE benchmark. Download and unzip it into `data/RTTS` (the tree already mirrors VOC style).

- **Windows PowerShell**
  ```powershell
  New-Item -ItemType Directory -Force -Path data | Out-Null
  Invoke-WebRequest -Uri https://residedata.obs.cn-north-4.myhuaweicloud.com/RTTS.zip -OutFile data/RTTS.zip
  Expand-Archive -Path data/RTTS.zip -DestinationPath data -Force
  Remove-Item data/RTTS.zip
  ```
- **Linux / WSL**
  ```bash
  mkdir -p data && cd data
  wget https://residedata.obs.cn-north-4.myhuaweicloud.com/RTTS.zip
  unzip -q RTTS.zip && rm RTTS.zip
  ```

Expected layout:
```
data/RTTS
├── Annotations/*.xml
├── ImageSets/Main/{train.txt,val.txt,test.txt}
└── JPEGImages/*.png
```
Use `python src/Jetson_src/temp_inspect_rtts.py` to list the five RTTS classes (`person`, `car`, `bus`, `bicycle`, `motorbike`).

## 3. Use the RTTS Experiment
The repo now ships with a ready-to-run experiment at `src/exps/example/custom/rtts_yolox_s.py`. It keeps the YOLOX-S depth/width multipliers, sets `num_classes = 5`, and swaps the dataloader to the new `RTTSDataset` that reads `data/RTTS/Annotations` plus the `ImageSets/Main` splits:
```python
from yolox.data import RTTSDataset, TrainTransform, ValTransform, get_yolox_datadir

class Exp(MyExp):
  def __init__(self):
    super().__init__()
    self.num_classes = 5
    self.depth = 0.33
    self.width = 0.50
    self.data_dir = os.path.join(get_yolox_datadir(), "RTTS")
    self.train_splits = ("train",)
    self.val_splits = ("val",)
```
If you need different augmentations or input sizes, edit this file directly and commit the changes so the helper scripts pick them up.

## 4. Launch Training
```powershell
python src\tools\train.py ^
  -f src\exps\example\custom\rtts_yolox_s.py ^
  -d 1 -b 16 --fp16 -o ^
  -c src\yolox\weights\yolox_s.pth ^
  --exp-name rtts_yolox_s
```
Arguments:
- `-d` number of GPUs, `-b` batch size per GPU.
- `--fp16` enables mixed precision; drop it on GPUs without Tensor Cores.
- `-c` points to COCO-pretrained weights (downloaded from the YOLOX model zoo).
- `--exp-name` controls the run directory under `YOLOX_outputs`.

Training artifacts (logs, checkpoints, tensorboard summaries) appear inside `YOLOX_outputs/rtts_yolox_s/`.

### Using the helper launcher
A dedicated wrapper (`src/Jetson_src/train_rtts.py`) keeps the Colab/Windows commands short and automatically wires `YOLOX_DATADIR`:

```bash
python src/Jetson_src/train_rtts.py \
  --exp src/exps/example/custom/rtts_yolox_s.py \
  --ckpt src/yolox/weights/yolox_s.pth \
  --colab  # safe defaults for a single T4
```

- `--colab` clamps the run to 1 GPU, batch size ≤16, and keeps `--fp16` on (perfect for Colab’s free T4).
- `--extra` lets you forward arbitrary options to `src/tools/train.py`, e.g. `--extra --occupy=0.95`.

## 5. Evaluate & Visualize
```powershell
python src\tools\eval.py ^
  -f src\exps\example\custom\rtts_yolox_s.py ^
  -c YOLOX_outputs\rtts_yolox_s\latest_ckpt.pth ^
  -b 1 -d 1 --conf 0.001 --fp16
```
To render qualitative samples:
```powershell
python src\tools\demo.py image ^
  -f src\exps\example\custom\rtts_yolox_s.py ^
  -c YOLOX_outputs\rtts_yolox_s\best_ckpt.pth ^
  --path data\RTTS\JPEGImages\AM_Bing_211.png ^
  --conf 0.25 --nms 0.45
```

## 6. Troubleshooting
- **Process killed early**: reduce `-b`, disable `--fp16`, or set `self.input_size = (512,512)` inside the experiment file.
- **Slow dataloading**: convert RTTS PNGs to JPEG or enable caching (`--cache`) if you have >32 GB RAM.
- **Mismatched class count**: rerun `python src/Jetson_src/temp_inspect_rtts.py --refresh` after editing annotations.

With these steps you can fully reproduce the fog-aware YOLOX pipeline on RTTS.

## 7. Google Colab (T4) Quickstart
```bash
!git clone https://github.com/wassihaiderkabir/Object-Detection-in-Hazy-and-Foggy-Conditions-on-NVIDIA-Jetson-Nano.git
%cd Object-Detection-in-Hazy-and-Foggy-Conditions-on-NVIDIA-Jetson-Nano
!pip install -r requirements.txt
# Copy RTTS into data/RTTS or mount Drive before this step
!python src/Jetson_src/train_rtts.py --colab --ckpt src/yolox/weights/yolox_s.pth
```
- Colab already exposes a single Tesla T4. Leave `--devices` at 1, keep `--fp16` on, and try `--batch-size 16`. If you hit OOM, drop to 12.
- Mount Google Drive (or use `gdown`) to bring in `RTTS.zip` and the COCO weights before invoking `src/Jetson_src/train_rtts.py`.

## 8. Jetson Nano Inference Entry Point
The `src/Jetson_src/jetson_infer.py` script automates TensorRT conversion plus the demo loop once you copy the trained checkpoint to your Jetson Nano.

```bash
# On Jetson Nano (JetPack 5.x)
python3 src/Jetson_src/jetson_infer.py \
  --exp src/exps/example/custom/rtts_yolox_s.py \
  --exp-name rtts_yolox_s \
  --ckpt YOLOX_outputs/rtts_yolox_s/best_ckpt.pth \
  --mode image --input data/RTTS/JPEGImages/AM_Bing_211.png --save-result
```

- First run builds `YOLOX_outputs/rtts_yolox_s/model_trt.pth` via `src/tools/trt.py`; subsequent runs reuse it or pass `--skip-build`.
- Switch to `--mode webcam --cam-id 0` for live CSI/USB feeds, or `--mode video --input path/to/clip.mp4` for recorded fog footage.
- Set `--datadir /path/on/jetson/data` if the dataset lives elsewhere; the script forwards it via `YOLOX_DATADIR`.
