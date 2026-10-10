#!/usr/bin/env python
"""Архитектурный прокси-скрининг B (2026-09-20, вторая серия по предложениям
Qwen, отфильтрованным нашими измерениями). Базовая линия — snorm (24.53).

Кандидаты:
  lpf      — ArchV2(use_lpf): learned spectral prefilter (UNet на log-спектре,
             мульт. маска, старт=тождество) перед snorm-UNet, совместное обучение
  lpfedge  — lpf + 4-й канал: градиент люмы на /4 масштабе (контурная маска Qwen)
  slocal   — FourierSpectralBlockLocal: локальный контраст (avg_pool 9x9 магнитуды)
             вместо глобальной std snorm
  vgg      — snorm + LPIPS-VGG w=0.05 (perceptual-абляция; alex в прошлом поиске
             душил PSNR — проверяем VGG на малом весе)

Протокол = search_arch0920.py (1k×3 эп., batch 4, AMP, поэпоховый resume).
Использование:
    python ml/train/search_arch0920b.py --candidate lpf --epoch 0
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
OUT = ROOT / "ml/train/_arch0920b"


def subsets() -> tuple[int, int]:
    import os
    return (int(os.environ.get("ARCH_B_TRAIN", TRAIN_SUBSET)),
            int(os.environ.get("ARCH_B_VAL", VAL_SUBSET)))


def partial_resume() -> tuple[int, int]:
    """(skip_steps, max_steps) для поэтапного прогона эпохи внутри лимита 300 с."""
    import os
    return (int(os.environ.get("ARCH_B_SKIP", 0)),
            int(os.environ.get("ARCH_B_MAX", 10 ** 9)))


def build(candidate: str) -> torch.nn.Module:
    from ml.model.archv2 import ArchV2
    from ml.model.unet import FourierUNet

    if candidate == "snorm":  # базовая линия (результаты из _arch0920 перенесены)
        return FourierUNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=3,
                           dec_blocks=(1, 1), spectral=(False, True, True),
                           spectral_mode="norm")
    if candidate == "lpf":
        return ArchV2(use_lpf=True, use_edge_ch=False, spectral_mode="norm")
    if candidate == "lpfedge":
        return ArchV2(use_lpf=True, use_edge_ch=True, spectral_mode="norm")
    if candidate == "slocal":
        return FourierUNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=3,
                           dec_blocks=(1, 1), spectral=(False, True, True),
                           spectral_mode="local")
    if candidate == "vgg":
        return FourierUNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=3,
                           dec_blocks=(1, 1), spectral=(False, True, True),
                           spectral_mode="norm")
    if candidate == "attn":
        # «щепотка ViT»: Restormer MDTA + GDFN в bottleneck поверх snorm
        return FourierUNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=3,
                           dec_blocks=(1, 1), spectral=(False, True, True),
                           spectral_mode="norm", bottleneck_attn=True)
    raise ValueError(candidate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    choices=["snorm", "lpf", "lpfedge", "slocal", "vgg", "attn"])
    ap.add_argument("--epoch", type=int, required=True)
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
    if args.candidate == "vgg":
        loss_fn = CompositeLoss(LossConfig(w_fourier=0.05, w_lpips=0.05, lpips_net="vgg")).cuda()
    else:
        loss_fn = CompositeLoss(LossConfig(w_fourier=0.05, w_lpips=0.0)).cuda()

    train_ds = Subset(DescreenDataset(ROOT / "Data" / "synth", "train"), range(subsets()[0]))
    val_ds = Subset(DescreenDataset(ROOT / "Data" / "synth", "val"), range(subsets()[1]))
    n_train = len(train_ds) // 4  # drop_last=True
    g = torch.Generator()
    g.manual_seed(args.epoch)  # пермутация своей на каждую эпоху, стабильна при resume
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROXY_EPOCHS * n_train)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    OUT.mkdir(exist_ok=True)
    (OUT / args.candidate).mkdir(exist_ok=True)
    ckpt_path = OUT / args.candidate / f"ep{args.epoch}.pt"
    skip, max_steps = partial_resume()
    if args.epoch > 0 and skip == 0:
        prev = OUT / args.candidate / f"ep{args.epoch - 1}.pt"
        ck = torch.load(prev, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        print(f"resumed from {prev}")
    elif skip > 0:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        assert ck.get("steps_done", 0) == skip, (ck.get("steps_done"), skip)
        print(f"mid-epoch resume: {skip}/{n_train} steps done")

    import os
    if args.epoch == 0 and skip == 0 and os.environ.get("ARCH_B_FWD") == "1":
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
    trained = 0
    it = iter(train_dl)
    for _ in range(skip):
        next(it)
    while trained < max_steps:
        try:
            xs, ys, _ = next(it)
        except StopIteration:
            break
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
        trained += 1
        if nsteps == 10:
            el = time.time() - t0
            print(f"step {el / 10:.2f}s -> эпоха ~{el / 10 * n_train:.0f}s", flush=True)
    done_total = skip + trained
    train_el = time.time() - t0
    print(f"trained {trained} steps (total {done_total}/{n_train})", flush=True)

    if done_total < n_train:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": args.epoch, "steps_done": done_total,
                    "val_psnr": None}, ckpt_path)
        print(f"partial save -> {ckpt_path}; продолжение: ARCH_B_SKIP={done_total}", flush=True)
        return

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
                "epoch": args.epoch, "steps_done": n_train, "val_psnr": score}, ckpt_path)
    hist_path = OUT / args.candidate / "history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
    hist.append(dict(epoch=args.epoch, val_psnr=round(score, 3),
                     train_s=round(train_el, 1), params_M=round(n_params, 2),
                     subset=list(subsets())))
    hist_path.write_text(json.dumps(hist, indent=1))
    print(f"{args.candidate} ep{args.epoch}: val PSNR={score:.2f} "
          f"({train_el:.0f}s) -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
