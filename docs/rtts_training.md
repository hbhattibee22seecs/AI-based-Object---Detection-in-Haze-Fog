# Running the YOLOX Pipeline on RTTS

This guide explains how to fine-tune YOLOX on the Realistic Traffic Scenes (RTTS) subset of the RESIDE benchmark and reproduce the fog-aware training pipeline that powers this repository.

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
Use `python temp_inspect_rtts.py` to list the five RTTS classes (`person`, `car`, `bus`, `bicycle`, `motorbike`).

## 3. Create an Experiment File
Duplicate `src/exps/example/custom/yolox_s.py` and update the dataset pointers:
```bash
copy src\exps\example\custom\yolox_s.py src\exps\example\custom\rtts_yolox_s.py
```
Edit the copy so it references RTTS:
```python
class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 5
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "rtts_yolox_s"

        self.data_dir = "data/RTTS"
        self.train_ann = "ImageSets/Main/train.txt"
        self.val_ann = "ImageSets/Main/val.txt"
        self.test_ann = "ImageSets/Main/test.txt"
```
Adjust `max_epoch`, `data_num_workers`, or augmentation hyperparameters as needed.

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
- **Mismatched class count**: rerun `python temp_inspect_rtts.py --refresh` after editing annotations.

With these steps you can fully reproduce the fog-aware YOLOX pipeline on RTTS.
