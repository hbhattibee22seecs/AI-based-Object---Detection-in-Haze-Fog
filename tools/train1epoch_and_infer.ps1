param(
  [string]$ExpName = "rtts_yolox_s_1epoch_cpu",
  [int]$BatchSize = 8,
  [string]$ImagePath = "",
  [string]$RepoRoot = "D:\CV_P\YOLOX",
  [int]$TrainIters = 0,
  [switch]$NoTensorBoard,
  [int]$TensorBoardPort = 6006
)

$ErrorActionPreference = "Stop"

$py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { throw "Python venv not found at $py" }

$env:PYTHONPATH = Join-Path $RepoRoot "src"
$env:YOLOX_DATADIR = Join-Path $RepoRoot "data"
$env:CUDA_VISIBLE_DEVICES = "-1"

$expFile = Join-Path $RepoRoot "src\exps\example\custom\rtts_yolox_s.py"
$pretrained = Join-Path $RepoRoot "src\yolox\weights\yolox_s.pth"

$logDir = Join-Path $RepoRoot "runlogs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "$ExpName.log"

Write-Host "Training 1 epoch (CPU) -> $ExpName"
Write-Host "  log: $log"

if (-not $NoTensorBoard) {
  $tbLogDir = Join-Path $RepoRoot "YOLOX_outputs\$ExpName\tensorboard"
  New-Item -ItemType Directory -Force -Path $tbLogDir | Out-Null
  try {
    $tbArgs = @(
      "-m", "tensorboard",
      "--logdir", $tbLogDir,
      "--port", "$TensorBoardPort"
    )
    $tb = Start-Process -FilePath $py -ArgumentList $tbArgs -WorkingDirectory $RepoRoot -PassThru
    Write-Host "TensorBoard PID=$($tb.Id) -> http://localhost:$TensorBoardPort"
    Write-Host "(If it doesn't open automatically, paste the URL into your browser.)"
  } catch {
    Write-Host "TensorBoard failed to start (continuing without it): $($_.Exception.Message)"
  }
}

$args = @(
  "-u", "src\tools\train.py",
  "-f", "src\exps\example\custom\rtts_yolox_s.py",
  "-d", "1",
  "-b", "$BatchSize",
  "-c", "src\yolox\weights\yolox_s.pth",
  "-expn", "$ExpName",
  "max_epoch", "1",
  "data_num_workers", "0",
  "print_interval", "50",
  "multiscale_range", "0"
)

$printInterval = if ($TrainIters -gt 0) { 10 } else { 50 }

# Rebuild args so print_interval is always consistent.
$args = @(
  "-u", "src\tools\train.py",
  "-f", "src\exps\example\custom\rtts_yolox_s.py",
  "-d", "1",
  "-b", "$BatchSize",
  "-c", "src\yolox\weights\yolox_s.pth",
  "-expn", "$ExpName",
  "max_epoch", "1",
  "data_num_workers", "0",
  "print_interval", "$printInterval",
  "multiscale_range", "0"
)

if ($TrainIters -gt 0) {
  $args += @("train_iters_per_epoch", "$TrainIters")
  # Skip evaluation during smoke-tests; we only need the checkpoint + inference.
  $args += @("eval_interval", "999999")
  Write-Host "Smoke-test mode: train_iters_per_epoch=$TrainIters"
}

Write-Host "Streaming training logs to terminal (also saving to $log)"
Set-Location $RepoRoot

# IMPORTANT (Windows PowerShell): native stderr is treated as the error stream,
# and with $ErrorActionPreference='Stop' it becomes a terminating error even
# for harmless warnings. Use cmd.exe to merge stderr->stdout at the OS level.
$cmdArgs = $args | ForEach-Object {
  if ($_ -match '[\s"]') { '"' + ($_ -replace '"','\\"') + '"' } else { $_ }
}
$cmdLine = '"' + $py + '" ' + ($cmdArgs -join ' ') + ' 2>&1'
cmd.exe /c $cmdLine | Tee-Object -FilePath $log
$trainExit = $LASTEXITCODE
if ($trainExit -ne 0) {
  Write-Host "Training exited with code $trainExit"
  exit $trainExit
}

$ckpt = Join-Path $RepoRoot "YOLOX_outputs\$ExpName\latest_ckpt.pth"
if (!(Test-Path $ckpt)) {
  Write-Host "Training finished but checkpoint not found: $ckpt"
  Write-Host "See logs: $log"
  exit 1
}

if ([string]::IsNullOrWhiteSpace($ImagePath)) {
  $imgDir = Join-Path $RepoRoot "data\RTTS\JPEGImages"
  $splitFile = Join-Path $RepoRoot "data\RTTS\ImageSets\Main\test.txt"
  if (Test-Path $splitFile) {
    $ids = Get-Content $splitFile | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    if ($ids.Count -gt 0) {
      $base = $ids[0].Trim()
      $match = Get-ChildItem -Path $imgDir -Recurse -File | Where-Object { $_.BaseName -eq $base } | Select-Object -First 1
      if ($null -ne $match) {
        $ImagePath = $match.FullName
      }
    }
  }
  if ([string]::IsNullOrWhiteSpace($ImagePath)) {
    $first = Get-ChildItem $imgDir -Recurse -File -Include *.jpg,*.jpeg,*.png | Select-Object -First 1
    if ($null -eq $first) { throw "No images found under $imgDir" }
    $ImagePath = $first.FullName
  }
}

Write-Host "Selected inference image: $ImagePath"

Write-Host "Running inference on: $ImagePath"
& $py -u src\tools\demo.py image -f $expFile -c $ckpt -expn $ExpName --device cpu --path $ImagePath --save_result --conf 0.001

Write-Host "Done. Results under: $(Join-Path $RepoRoot \"YOLOX_outputs\$ExpName\vis_res\")"
