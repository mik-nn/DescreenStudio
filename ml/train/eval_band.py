#!/usr/bin/env python
"""Сравнение чекпоинтов по остатку НЕСУЩЕЙ в предсказании (2.43).

Почему не хватает val_psnr: PSNR считает весь кадр, а растр — доли процента
энергии, поэтому «стало на 0.7 дБ лучше» не отвечает на вопрос, снят ли
растр. Здесь меряется отношение энергии кольца несущей в предсказании к
энергии того же кольца во ВХОДНОМ скане: 1.0 = растр не тронут, 0.0 = снят.

  C:\\Python314\\python.exe -X utf8 -m ml.train.eval_band ^
      ml/train/_chain_native/best.pt ml/train/_chain_native_band/best.pt ^
      --data Data/chain_native --n 100

Плюс `flat_hf` — широкая ВЧ-энергия: растёт, когда сеть подменяет решётку
зерном.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ml.model import FourierUNet
from ml.train.data import DescreenDataset

COND_DIM = 4


def luma(t: torch.Tensor) -> torch.Tensor:
    return 0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]


def radial(h: int, w: int, device) -> torch.Tensor:
    fy = torch.fft.fftfreq(h, device=device)[:, None]
    fx = torch.fft.rfftfreq(w, device=device)[None, :]
    return torch.sqrt(fx * fx + fy * fy)


def band_amp(x: torch.Tensor, r: float, bw: float) -> float:
    """Амплитуда кольца |f - r| < bw в лума-канале (один кадр, (1,C,H,W))."""
    g = luma(x)
    g = g - g.mean()
    f = torch.abs(torch.fft.rfft2(g))
    ring = (radial(g.shape[1], g.shape[2], g.device) - r).abs() < bw
    return float(f[:, ring].pow(2).sum(dim=1).sqrt())


def _blur1(g: torch.Tensor, k: float = 1.2) -> torch.Tensor:
    rad = int(4 * k + 0.5)
    xs = torch.arange(-rad, rad + 1, device=g.device, dtype=g.dtype)
    ker = torch.exp(-0.5 * (xs / k) ** 2)
    ker = (ker / ker.sum()).view(1, 1, 1, -1)
    pad = torch.nn.functional.pad
    b = pad(g, (rad, rad, 0, 0), mode="reflect")
    b = torch.nn.functional.conv2d(b, ker)
    b = pad(b, (0, 0, rad, rad), mode="reflect")
    b = torch.nn.functional.conv2d(b, ker.transpose(-1, -2))
    return b


def flat_hf(x: torch.Tensor) -> float:
    """RMS высоких частот по кадру (g - gauss(g, 1.2))."""
    g = luma(x).unsqueeze(1)
    diff = g - _blur1(g)
    return float(diff.pow(2).mean().sqrt())


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(torch.mean((pred - target) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def build(sd: dict, cond: bool) -> FourierUNet:
    model = FourierUNet(
        in_channels=3, base=96, levels=3,
        enc_blocks=(1, 1), bottleneck_blocks=3, dec_blocks=(1, 1),
        spectral=(False, True, True), spectral_mode="plain",
        cond_dim=COND_DIM if cond else 0,
    )
    model.load_state_dict({k: v.float().cuda() for k, v in sd.items()})
    return model.cuda().eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data", default="Data/chain_native")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--bw", type=float, default=0.02)
    ap.add_argument("--cond", type=int, default=1)
    ap.add_argument("--ema", type=int, default=1, help="1 = если есть ema, мерить и его")
    args = ap.parse_args()

    ds = DescreenDataset(args.data, args.split, cond=bool(args.cond), with_r=True)
    n = min(args.n, len(ds))
    samples = [ds[i] for i in range(n)]
    scans = torch.stack([s[0] for s in samples]).cuda()
    gts = torch.stack([s[1] for s in samples]).cuda()
    stems = [s[2] for s in samples]
    man = Path(args.data) / args.split / "manifest.jsonl"
    r_map: dict[str, float] = {}
    if man.exists():
        for line in man.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                r_map[row["id"]] = float(row["r"])
    rr = [r_map.get(s, float(sm[-1])) for s, sm in zip(stems, samples)]

    in_band = [band_amp(scans[i:i + 1], rr[i], args.bw) for i in range(n)]
    hf_gt = float(np.mean([flat_hf(gts[i:i + 1]) for i in range(n)]))
    hf_scan = float(np.mean([flat_hf(scans[i:i + 1]) for i in range(n)]))
    print(f"{'вариант':32s} {'PSNR':>7s} {'band/in':>8s} {'flat_hf':>9s}")
    print(f"{'gt (решётки нет)':32s} {'-':>7s} {'-':>8s} {hf_gt:9.5f}")
    print(f"{'scan (вход)':32s} {'-':>7s} {'1.0000':>8s} {hf_scan:9.5f}")

    cond = torch.zeros(1, COND_DIM, device="cuda")
    for cp in args.ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        variants = []
        if "model" in ck:
            variants.append(("model", ck["model"]))
        if args.ema and "ema" in ck:
            variants.append(("ema", ck["ema"]))
        for vname, sd in variants:
            model = build(sd, bool(args.cond))
            ps, bnd, hf = 0.0, [], []
            with torch.no_grad():
                for i in range(n):
                    x = scans[i:i + 1]
                    pr = model(x, cond) if args.cond else model(x)
                    pr = pr.float().clamp(0, 1)
                    ps += psnr(pr, gts[i:i + 1])
                    bnd.append(band_amp(pr, rr[i], args.bw) / max(in_band[i], 1e-12))
                    hf.append(flat_hf(pr))
            tag = f"{Path(cp).parent.name}/{vname}"
            print(f"{tag:32s} {ps/n:7.2f} {np.mean(bnd):8.4f} {np.mean(hf):9.5f}")


if __name__ == "__main__":
    main()
