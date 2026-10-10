#!/usr/bin/env python
"""Прокси лосса (2026-09-28): Charbonnier / focal-frequency / deep supervision.

Протокол как search_run7 (warm start, 1000 шагов, score=min по сетам), отличия:
- warm от run10/best.pt (эталон), скелет БЕЗ cond (FiLM закрыт абляцией —
  изолируем эффект лосса; partial load сбрасывает film.*);
- 3 корня (synth + synth_run4 + synth_1200), score = min из трёх;
- сетка из 5 рук: base / charbonnier / focal / aux / combo.

Запуск: python ml/train/search_loss.py (фон, ~2 ч на GPU).
Выход: ml/train/_out/search_loss.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from ml.model import CompositeLoss, FourierUNet, LossConfig
from ml.train.data import DescreenDataset

ROOT = Path(__file__).resolve().parents[2]

TRAIN_SUBSET = 3000  # по 1000 из каждого корня, чередование
VAL_N = 300
PROXY_STEPS = 1500  # opt-шагов batch 4 (=6000 сэмплов, два прохода subset)
PROXY_BATCH = 4
EMA_DECAY = 0.999  # EMA-оценка как в проде
LR = 2e-5

ARMS = [
    ("base", LossConfig(w_fourier=0.05, w_lpips=0.0, w_edge=0.05,
                        w_tone=0.0), False),
    ("charb", LossConfig(w_fourier=0.05, w_lpips=0.0, w_edge=0.05,
                         w_tone=0.0, l1_type="charbonnier"), False),
    ("focal", LossConfig(w_fourier=0.05, w_lpips=0.0, w_edge=0.05,
                         w_tone=0.0, focal_gamma=1.0), False),
    ("aux", LossConfig(w_fourier=0.05, w_lpips=0.0, w_edge=0.05,
                       w_tone=0.0, w_aux=0.3), True),
    ("combo", LossConfig(w_fourier=0.05, w_lpips=0.0, w_edge=0.05,
                         w_tone=0.0, l1_type="charbonnier",
                         focal_gamma=1.0, w_aux=0.3), True),
]

ROOTS = ["Data/synth", "Data/synth_run4", "Data/synth_1200"]


def build_model(use_aux: bool) -> FourierUNet:
    model = FourierUNet(
        in_channels=3, base=96, levels=3,
        enc_blocks=(1, 1), bottleneck_blocks=3, dec_blocks=(1, 1),
        spectral=(False, True, True), cond_dim=0, aux=use_aux,
    )
    ck = torch.load(ROOT / "ml/train/_run10/best.pt", map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    # сбрасываем train-only ключи: aux_heads (свежие при use_aux) и film.*
    # (cond выключен — абляция доказала нулевой вклад, изолируем эффект лосса)
    sd = {k: v for k, v in sd.items()
          if not k.startswith("aux_heads.") and not k.startswith("film.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # свежие головы при use_aux — штатно
    missing = [m for m in missing if not m.startswith("aux_heads.")]
    assert not missing and not unexpected, (missing, unexpected)
    return model


@torch.no_grad()
def val_psnr(model, ds_root: str, n: int, device: str = "cuda") -> float:
    ds = DescreenDataset(ROOT / ds_root, "val")
    ps = []
    for i in range(min(n, len(ds))):
        x, y, _ = ds[i]
        with torch.amp.autocast("cuda", enabled=False):
            pred = model(x.unsqueeze(0).float().to(device))
        mse = torch.mean((pred[0].cpu() - y) ** 2).item()
        ps.append(10.0 * float(np.log10(1.0 / max(mse, 1e-12))))
    return float(np.mean(ps))


def run_arm(name: str, lcfg: LossConfig, use_aux: bool, device: str = "cuda") -> dict:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model = build_model(use_aux).to(device)
    loss_fn = CompositeLoss(lcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # как train.py: scaler + clip (без них fp16-переполнение в spectral-backward
    # даёт inf-grad → Adam NaN; найдено отладкой base-arm 2026-09-28)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    dss = [DescreenDataset(ROOT / r, "train") for r in ROOTS]
    per = TRAIN_SUBSET // len(ROOTS)
    idx = []
    for k, ds in enumerate(dss):
        idx += [(k, i) for i in range(per)]
    random.shuffle(idx)
    model.train()
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    step = 0
    bx, by = [], []
    for k, i in idx * 2:  # два прохода: 1500 opt-шагов batch 4
        if step >= PROXY_STEPS:
            break
        x, y, _ = dss[k][i]
        bx.append(x)
        by.append(y)
        if len(bx) < PROXY_BATCH:
            continue
        xb = torch.stack(bx).float().to(device)
        yb = torch.stack(by).float().to(device)
        bx, by = [], []
        with torch.amp.autocast("cuda", enabled=True):
            if use_aux:
                pred, auxs = model(xb, return_aux=True)
                loss = loss_fn(pred.float(), yb, aux=auxs[0].float())["total"]
            else:
                pred = model(xb)
                loss = loss_fn(pred.float(), yb)["total"]
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            for kk, vv in model.state_dict().items():
                ema[kk].mul_(EMA_DECAY).add_(vv.detach(), alpha=1.0 - EMA_DECAY)
        step += 1
        if step >= PROXY_STEPS:
            break
    model.load_state_dict({k: v.cuda() for k, v in ema.items()})
    model.eval()
    scores = {r: round(val_psnr(model, r, VAL_N, device), 2) for r in ROOTS}
    score = min(scores.values())
    rec = {"arm": name, "scores": scores, "score": round(score, 2)}
    print(f"arm {name}: {scores} score={score:.2f}", flush=True)
    return rec


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = [run_arm(name, lcfg, use_aux, device) for name, lcfg, use_aux in ARMS]
    out = ROOT / "ml/train/_out/search_loss.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
