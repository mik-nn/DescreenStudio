#!/usr/bin/env python
"""Архитектурный прокси-скрининг 2026-09-20 (по запросу: следы растра остаются).

Кандидаты (все в бюджете скорости продакшена, base96/L3):
  run3  — контроль (champion-архитектура, ожидание ~26.3 по протоколу search)
  sres  — FourierSpectralBlockRes: residual в спектре, zero-init
  snorm — FourierSpectralBlockNorm: контраст-норм спектра + residual, zero-init
  ae    — SpectralAE без skip-соединений (автоэнкодер вместо UNet)
  naf   — NAFBlock-блоки в энкодере/декодере, FFC spectral в bottleneck

Протокол = search.py: Data/synth, batch 4, AMP, cosine, loss champion
(fourier 0.05 / lpips 0), 3 эпохи. Поэпоховый resume чекпоинтом
(foreground-ограничение 300 с: один вызов = одна эпоха).

Использование:
    python ml/train/search_arch0920.py --candidate sres --epoch 0
    python ml/train/search_arch0920.py --candidate sres --epoch 1   # resume
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRAIN_SUBSET = 1000
VAL_SUBSET = 100
PROXY_EPOCHS = 3
OUT = ROOT / "ml/train/_arch0920"


def build(candidate: str) -> torch.nn.Module:
    from ml.model.ae import SpectralAE
    from ml.model.unet import FourierUNet

    if candidate == "run3":
        return FourierUNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=3,
                           dec_blocks=(1, 1), spectral=(False, True, True))
    if candidate == "sres":
        return FourierUNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=3,
                           dec_blocks=(1, 1), spectral=(False, True, True), spectral_mode="res")
    if candidate == "snorm":
        return FourierUNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=3,
                           dec_blocks=(1, 1), spectral=(False, True, True), spectral_mode="norm")
    if candidate == "ae":
        return SpectralAE(base=96, enc_blocks=1, bottleneck_blocks=3, dec_blocks=1, block="ffc")
    if candidate == "naf":
        # NAFNet-UNet: NAFBlock на enc/dec, skip-add, spectral нет — чистый NAF
        return SpectralAE(base=96, enc_blocks=1, bottleneck_blocks=2, dec_blocks=1,
                          block="naf", skip=True)
    raise ValueError(candidate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    choices=["run3", "sres", "snorm", "ae", "naf"])
    ap.add_argument("--epoch", type=int, required=True, help="номер эпохи для прогона")
    args = ap.parse_args()

    from ml.model import CompositeLoss, LossConfig
    from ml.train.data import DescreenDataset

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    model = build(args.candidate)
    n_params = model.param_count() / 1e6
    model.cuda()
    loss_fn = CompositeLoss(LossConfig(w_fourier=0.05, w_lpips=0.0)).cuda()

    train_ds = Subset(DescreenDataset(ROOT / "Data" / "synth", "train"), range(TRAIN_SUBSET))
    val_ds = Subset(DescreenDataset(ROOT / "Data" / "synth", "val"), range(VAL_SUBSET))
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROXY_EPOCHS * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    ckpt_path = OUT / args.candidate / f"ep{args.epoch}.pt"
    if args.epoch > 0:
        prev = OUT / args.candidate / f"ep{args.epoch - 1}.pt"
        ck = torch.load(prev, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        print(f"resumed from {prev}")

    OUT.mkdir(exist_ok=True)
    (OUT / args.candidate).mkdir(exist_ok=True)

    if args.epoch == 0:
        x = torch.randn(1, 3, 512, 512, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for _ in range(3):
                model(x)
        torch.cuda.synchronize()
        fwd_ms = (time.time() - t0) / 3 * 1000
        print(f"params={n_params:.2f}M fwd(512²,AMP,batch1)={fwd_ms:.0f}ms", flush=True)

    model.train()
    t0 = time.time()
    nsteps = 0
    for xs, ys, _ in train_dl:
        xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            pred = model(xs)
        L = loss_fn(pred.float(), ys)
        scaler.scale(L["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        nsteps += 1
        if nsteps == 10:
            el = time.time() - t0
            print(f"step {el / 10:.2f}s -> эпоха ~{el / 10 * len(train_dl):.0f}s", flush=True)
    train_el = time.time() - t0

    model.eval()
    ps, nv = 0.0, 0
    with torch.no_grad():
        for xs, ys, _ in val_dl:
            xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                pr = model(xs).float()
            mse = torch.mean((pr - ys) ** 2).item()
            ps += 10.0 * math.log10(1.0 / max(mse, 1e-12))
            nv += 1
    score = ps / nv

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "epoch": args.epoch, "val_psnr": score}, ckpt_path)
    hist_path = OUT / args.candidate / "history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
    hist.append(dict(epoch=args.epoch, val_psnr=round(score, 3),
                     train_s=round(train_el, 1), params_M=round(n_params, 2)))
    hist_path.write_text(json.dumps(hist, indent=1))
    print(f"{args.candidate} ep{args.epoch}: val PSNR={score:.2f} "
          f"({train_el:.0f}s) -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
