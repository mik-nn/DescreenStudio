# Дообучение run27 на chain_v1 (вход = прод-пре-стейдж с V1).
#
# Зачем: вход обучения теперь буквально совпадает с тем, что видит сеть на
# инференсе (V1 в нативном разрешении -> Найквист-даунсэмпл -> A0 -> notch
# (пропуск при V1) -> LF с гейтом). Замер (JOURNAL 2.36, Temp/dataset_gap.py):
# PSNR к контонному эталону 19.73 (старый вход) / 17.65 (прежняя пре-стадия) /
# 23.54 (новый вход) — задача сети сократилась до остатка.
#
# Требования перед запуском:
#   Data/chain_v1/{train,val}/{scan,gt} + manifest.jsonl + a0.json  (make_chain.py)
#
# Запуск:
#   powershell -File ml/train/finetune_chain_v1.ps1
#
# Гиперпараметры: база — конфиг run27 (ml/train/_run27_hyb/config.json), но
#   lr 1e-5 (дообучение, не с нуля), 2 эпохи по 1500 кадров, batch 2 с
#   накоплением 2 (эффективный 4), кадр 512x1024 — вдвое больше прежнего 512²,
#   поэтому batch уменьшен (на 16 ГБ VRAM batch 4 не влезает).

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Get-Location).Path
# ВАЖНО: arch-параметры обязаны совпадать с чекпойнтом warm start (run27: base=96).
# Без них подставляется base=64 и load_state_dict падает на size mismatch.
C:\Python314\python.exe -X utf8 -m ml.train.train `
  --data Data/chain_v1 `
  --out ml/train/_chain_v1 `
  --epochs 2 `
  --batch 2 `
  --grad-accum 2 `
  --lr 1e-5 `
  --warm-start ml/train/_run27_hyb/best.pt `
  --arch-base 96 --arch-levels 3 --arch-bottleneck 3 `
  --arch-enc-blocks 1 --arch-dec-blocks 1 `
  --arch-spectral-l0 0 --arch-plain 0 --arch-spectral-mode plain `
  --ema-decay 0.999 --seed 52 `
  --cond 1 --cond-drop 0.1 --film-lr-mult 5 `
  --root-weights "chain_v1:1" `
  --val-every 1 `
  --num-workers 4 `
  --fourier-w 0.05 --lpips-w 0.05 --edge-w 0.05 --edge-alpha 2.0 --tone-w 0
