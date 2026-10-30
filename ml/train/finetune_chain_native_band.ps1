# Fine-tune at NATIVE resolution with the CARRIER-BAND weighted loss.
# ASCII only: PowerShell 5.1 reads UTF-8 without BOM as ANSI (repo rule).
#
# Why (JOURNAL 2.43): the network is the only stage that can remove the
# lattice, but it leaves ~1/4 of the carrier amplitude, and unsharp then turns
# that residue into visible texture on flat areas. The global Fourier_Mag term
# spreads its weight over the whole spectrum while the carrier ring is a
# fraction of a percent of the coefficients. Every simulated frame knows its
# exact r (manifest.jsonl), so the ring is weighted explicitly:
#   1 + band_k * exp(-((f - r)/band_bw)^2),  band_k = 20, band_bw = 0.02
# Synthetic check: a raster at r=0.15 is penalised 21x inside the band.
#
# Start: ml/train/_chain_native/best.pt (already native-adapted, val PSNR
#   30.56 dB, rel 0.83). This pass only adds the band term, so lr stays low.
#
# Run: powershell -File ml/train/finetune_chain_native_band.ps1

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Get-Location).Path
C:\Python314\python.exe -X utf8 -m ml.train.train `
  --data Data/chain_native `
  --out ml/train/_chain_native_band `
  --epochs 12 `
  --batch 2 `
  --grad-accum 2 `
  --lr 2e-5 `
  --warm-start ml/train/_chain_native/best.pt `
  --arch-base 96 --arch-levels 3 --arch-bottleneck 3 `
  --arch-enc-blocks 1 --arch-dec-blocks 1 `
  --arch-spectral-l0 0 --arch-plain 0 --arch-spectral-mode plain `
  --ema-decay 0.999 --seed 53 `
  --cond 1 --cond-drop 0.1 --film-lr-mult 5 `
  --root-weights "chain_native:1" `
  --val-every 1 `
  --num-workers 4 `
  --fourier-w 0.05 --lpips-w 0.05 --edge-w 0.05 --edge-alpha 2.0 --tone-w 0 `
  --band-w 0.5 --band-bw 0.02 --band-k 20
