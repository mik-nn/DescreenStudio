#!/usr/bin/env python
"""Честное сравнение чекпоинтов run3 vs run4'' (2026-09-20).

Проблема приёмки: run3 валидировался на Data/synth/val (32.88/0.907), а run4'' —
на Data/synth_run4/val (28.14/0.849). Разные вал-сеты, разная сложность (P1:
tint/banding/show-through/LPI 75-190). Прямое сравнение некорректно.

Здесь обе модели оцениваются на ОБОИХ вал-сетах одним протоколом
(полные 512x512 пары, mean PSNR/SSIM, без EMA-хитростей — чекпоинт как есть).

Использование:
    python ml/benchmark/eval_models.py --n 300
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import SSIM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.model.unet import FourierUNet  # noqa: E402
from ml.train.dctfeat import dct_channels  # noqa: E402


def load_model(run_dir: str, weights: str | None = None):
    cfg = json.loads((ROOT / run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("use_lpf") or cfg.get("arch_spectral_mode", "plain") != "plain":
        from ml.model.archv2 import ArchV2
        model = ArchV2(
            use_lpf=bool(cfg.get("use_lpf")),
            spectral_mode=cfg.get("arch_spectral_mode", "norm"),
            base=cfg.get("arch_base", 96),
        )
        ck = torch.load(ROOT / run_dir / "best.pt", map_location="cpu", weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        model.load_state_dict(sd)
        model.eval()
        return model, cfg
    in_ch = 5 if cfg.get("extra_dct") else 3
    if cfg.get("arch_plain"):
        spectral = (False,) * cfg.get("arch_levels", 3)
    else:
        spectral = (bool(cfg.get("arch_spectral_l0")), True, True)
    ck = torch.load(ROOT / (weights or run_dir + "/best.pt"), map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    # aux-головы живут только в трейне: для eval отбрасываем их ключи явно
    sd = {k: v for k, v in sd.items() if not k.startswith("aux_heads.")}
    has_cond = any(k.startswith("film.") for k in sd)
    has_ffnorm = any(".norm_f." in k for k in sd)
    if has_cond:
        from ml.model.film import COND_DIM
    model = FourierUNet(
        in_channels=in_ch,
        base=cfg.get("arch_base", 64),
        levels=cfg.get("arch_levels", 3),
        enc_blocks=(cfg.get("arch_enc_blocks", 1),) * (cfg.get("arch_levels", 3) - 1),
        bottleneck_blocks=cfg.get("arch_bottleneck", 2),
        dec_blocks=(cfg.get("arch_dec_blocks", 1),) * (cfg.get("arch_levels", 3) - 1),
        spectral=spectral,
        cond_dim=COND_DIM if has_cond else 0,
        ff_norm=has_ffnorm,
    )
    model.load_state_dict(sd)
    model.eval()
    return model, cfg, has_cond


@torch.no_grad()
def eval_dataset(model: FourierUNet, root: Path, n: int, use_dct: bool,
                 device: str, batch: int = 4, has_cond: bool = False,
                 zero_cond: bool = False, tone_res: bool = False) -> dict:
    scan_dir, gt_dir = root / "val" / "scan", root / "val" / "gt"
    scans = sorted(scan_dir.glob("*.png"))[:n]
    cond_map: dict[str, list[float]] = {}
    if has_cond:
        from ml.model.film import build_cond_vec
        a0p = root / "val" / "a0.json"
        if a0p.exists():
            for k, v in json.loads(a0p.read_text(encoding="utf-8")).items():
                cond_map[k] = build_cond_vec(*v)
    # tone-resolution (Найквист): цель — 2 px на ячейку растра; фактор по манифесту
    tone_map: dict[str, float] = {}
    if tone_res:
        man = root / "val" / "manifest.jsonl"
        if man.exists():
            for line in open(man, encoding="utf-8"):
                r = json.loads(line)
                tone_map[r["id"]] = 2.0 * r["lpi"] / r["scan_dpi"]
    ssim_fn = SSIM(data_range=1.0, channel=3).to(device)
    psnrs, ssims, tone_psnrs = [], [], []
    for i in range(0, len(scans), batch):
        chunk = scans[i : i + batch]
        xs, ys = [], []
        for p in chunk:
            x = np.asarray(__import__("PIL.Image", fromlist=["open"]).open(p), dtype=np.float32) / 255.0
            y = np.asarray(__import__("PIL.Image", fromlist=["open"]).open(gt_dir / p.name), dtype=np.float32) / 255.0
            if use_dct:
                from PIL import Image
                luma = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
                d = dct_channels(luma)
                x = np.concatenate([x, d], axis=-1)
            xs.append(x.transpose(2, 0, 1))
            ys.append(y.transpose(2, 0, 1))
        xb = torch.from_numpy(np.stack(xs)).to(device)
        yb = torch.from_numpy(np.stack(ys)).to(device)
        with torch.amp.autocast("cuda", enabled=False):
            if has_cond:
                if zero_cond:
                    cb = torch.zeros(len(chunk), 4, dtype=torch.float32).to(device)
                else:
                    cb = torch.tensor([cond_map.get(p.stem, [0.0, 0.0, 1.0, 0.0]) for p in chunk],
                                      dtype=torch.float32).to(device)
                pred = model(xb.float(), cb)
            else:
                pred = model(xb.float())
        for j in range(len(chunk)):
            psnrs.append(10.0 * math.log10(1.0 / max(torch.mean((pred[j] - yb[j]) ** 2).item(), 1e-12)))
            ssims.append(ssim_fn(pred[j : j + 1], yb[j : j + 1]).item())
            if tone_res and (f := tone_map.get(chunk[j].stem)):
                th, tw = max(8, round(pred.shape[2] * f)), max(8, round(pred.shape[3] * f))
                pt = torch.nn.functional.interpolate(
                    pred[j : j + 1].float(), size=(th, tw), mode="bicubic",
                    align_corners=False, antialias=True)
                gt = torch.nn.functional.interpolate(
                    yb[j : j + 1].float(), size=(th, tw), mode="bicubic",
                    align_corners=False, antialias=True)
                tone_psnrs.append(10.0 * math.log10(1.0 / max(torch.mean((pt - gt) ** 2).item(), 1e-12)))
    out = dict(n=len(scans), psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)),
               psnr_p10=float(np.percentile(psnrs, 10)))
    if tone_psnrs:
        out["tone_psnr"] = round(float(np.mean(tone_psnrs)), 3)
        out["tone_n"] = len(tone_psnrs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default="run3,run4",
                    help="список через запятую: имя=путь или имя (путь ml/train/_<имя>)")
    ap.add_argument("--sets", default="synth,synth_run4",
                    help="список через запятую: имя=путь или имя (путь Data/<имя>)")
    ap.add_argument("--tag", default="", help="суффикс имени выходного JSON")
    ap.add_argument("--zero-cond", type=int, default=0,
                    help="абляция FiLM: нулевой cond вместо замера A0")
    ap.add_argument("--tone-res", type=int, default=0,
                    help="тоновая метрика Найквиста: PSNR после даунскейла к 2 px/ячейку")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    models = []
    for item in args.models.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, run_dir = item.split("=", 1)
        else:
            name, run_dir = item, f"ml/train/_{item}"
        models.append((name.strip(), run_dir.strip()))
    sets = []
    for item in args.sets.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            ds_name, ds_root = item.split("=", 1)
        else:
            ds_name, ds_root = item, f"Data/{item}"
        sets.append((ds_name.strip(), ROOT / ds_root.strip()))
    results = {}
    for name, run_dir in models:
        model, cfg, has_cond = load_model(run_dir)
        model = model.to(device)
        use_dct = bool(cfg.get("extra_dct"))
        for ds_name, ds_root in sets:
            r = eval_dataset(model, ds_root, args.n, use_dct, device,
                             has_cond=has_cond, zero_cond=bool(args.zero_cond),
                             tone_res=bool(args.tone_res))
            results[f"{name}@{ds_name}"] = r
            extra = f" tone={r['tone_psnr']:.2f}" if "tone_psnr" in r else ""
            print(f"{name:5s} {ds_name:16s} PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} "
                  f"(p10={r['psnr_p10']:.2f}, n={r['n']}){extra}", flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    names = "_vs_".join(n for n, _ in models)
    out = ROOT / f"ml/benchmark/_out/eval_{names}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
