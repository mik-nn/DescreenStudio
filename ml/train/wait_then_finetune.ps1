# Wait for chain_v1 generation, then start fine-tuning automatically.
#
# Completion marker: Data/chain_v1/dataset.json -- make_chain.py writes it last
# (after train and val), so its presence means the dataset is ready.
#
# Run (background, no timeout):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ml/train/wait_then_finetune.ps1
# Status: Temp/wait_finetune.status.log   Training log: Temp/finetune_chain_v1.log
#
# NOTE: keep this file ASCII-only. PowerShell 5.1 reads UTF-8 without BOM as
# ANSI, and non-ASCII text breaks the parser.

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$Marker = Join-Path $Root "Data\chain_v1\dataset.json"
$Log = Join-Path $Root "Temp\finetune_chain_v1.log"
$Status = Join-Path $Root "Temp\wait_finetune.status.log"

function Say([string]$msg) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $msg
    Add-Content -Path $Status -Value $line
    Write-Output $line
}

Say "waiting for $Marker"

$deadline = (Get-Date).AddHours(14)
$nextReport = (Get-Date)
while (-not (Test-Path $Marker)) {
    if ((Get-Date) -gt $deadline) {
        Say "TIMEOUT waiting for generation; training NOT started"
        exit 1
    }
    if ((Get-Date) -gt $nextReport) {
        $n = 0
        $v = 0
        if (Test-Path "Data\chain_v1\train\scan") {
            $n = (Get-ChildItem "Data\chain_v1\train\scan" -Filter *.png).Count
        }
        if (Test-Path "Data\chain_v1\val\scan") {
            $v = (Get-ChildItem "Data\chain_v1\val\scan" -Filter *.png).Count
        }
        Say ("generation progress: train {0}/1500, val {1}/150" -f $n, $v)
        $nextReport = (Get-Date).AddMinutes(10)
    }
    Start-Sleep -Seconds 30
}

# Let generation workers release memory and file handles.
Start-Sleep -Seconds 45

$trainCount = (Get-ChildItem "Data\chain_v1\train\scan" -Filter *.png).Count
$valCount = (Get-ChildItem "Data\chain_v1\val\scan" -Filter *.png).Count
Say "dataset ready: train=$trainCount val=$valCount; starting fine-tune"

if ($trainCount -lt 1000 -or $valCount -lt 100) {
    Say "WARNING: fewer frames than expected (1500/150); training anyway"
}

$env:PYTHONIOENCODING = "utf-8"
$trainArgs = @(
    "-X", "utf8", "ml/train/train.py",
    "--data", "Data/chain_v1",
    "--out", "ml/train/_chain_v1",
    "--epochs", "2",
    "--batch", "2",
    "--grad-accum", "2",
    "--lr", "1e-5",
    "--warm-start", "ml/train/_run27_hyb/best.pt",
    "--cond", "1",
    "--cond-drop", "0.1",
    "--film-lr-mult", "5",
    "--root-weights", "chain_v1:1",
    "--val-every", "1",
    "--val-max", "150",
    "--num-workers", "4",
    "--fourier-w", "0.05",
    "--lpips-w", "0.05",
    "--edge-w", "0.05",
    "--tone-w", "0"
)

Say ("command: C:\Python314\python.exe " + ($trainArgs -join " "))
& C:\Python314\python.exe @trainArgs *>&1 | Tee-Object -FilePath $Log
$code = $LASTEXITCODE
Say "training finished with exit code $code; log: $Log"
exit $code
