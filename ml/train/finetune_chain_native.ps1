# Fine-tune at NATIVE resolution (scale = 1.0, screen period 4-10 px).
# ASCII only: PowerShell 5.1 reads UTF-8 without BOM as ANSI (repo rule).
#
# Why (JOURNAL 2.40): at any working resolution above the screen the chain keeps
# a visible copy of the halftone lattice, and every downsample costs sharpness.
# Measured on 1821146u (same scan, same net):
#   old r_out=0.5   sharpness 0.0376  residual 1.69 %
#   r_out=0.30      sharpness 0.0454  residual 2.55 %   (2.07x upscale)
#   native (1.0)    sharpness 0.0586  residual 2.51 %   (+29 % over 0.30, +56 % over old)
# So the working frame must be the scan itself: no downsample, no upscale, and
# the lattice removal becomes the network's job (V1 already takes ~83 % of the
# carrier band at native).
#
# Input: Data/chain_native, built by
#   make_chain.py --native-scale --train 1500 --val 150
# where r_out = r of every frame, so scale = 1.0 and the chain frame IS the scan.
#
# Start: ml/train/_chain_v1/best.pt (chain-adapted, same arch). The domain is
#   harder, so lr is slightly higher than the 0.30 pass.
#
# Run: powershell -File ml/train/finetune_chain_native.ps1

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Get-Location).Path
C:\Python314\python.exe -X utf8 -m ml.train.train `
  --data Data/chain_native `
  --out ml/train/_chain_native `
  --epochs 10 `
  --batch 2 `
  --grad-accum 2 `
  --lr 2e-5 `
  --warm-start ml/train/_chain_v1/best.pt `
  --arch-base 96 --arch-levels 3 --arch-bottleneck 3 `
  --arch-enc-blocks 1 --arch-dec-blocks 1 `
  --arch-spectral-l0 0 --arch-plain 0 --arch-spectral-mode plain `
  --ema-decay 0.999 --seed 52 `
  --cond 1 --cond-drop 0.1 --film-lr-mult 5 `
  --root-weights "chain_native:1" `
  --val-every 1 `
  --num-workers 4 `
  --fourier-w 0.05 --lpips-w 0.05 --edge-w 0.05 --edge-alpha 2.0 --tone-w 0
