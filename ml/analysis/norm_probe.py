#!/usr/bin/env python
"""Zero-shot проба нормализации к линеатуре (без обучения).

Гипотеза: сеть страдает от ширины распределения r (0.06-0.63); приведение
входа к канонической частоте r=0.25 должно помочь даже необученной на это сети.
Проба: val-сэмплы с theory-r в [0.10, 0.25] апсемплятся (scan И gt одним ядром,
пары остаются попиксельными) к r=0.25, прогон run10/run7, даунскейл выхода
обратно, парный PSNR против baseline на тех же сэмплах без нормализации.

Использование:
    python ml/analysis/norm_probe.py --models run10,run7 --sets synth,synth_run4,synth_1200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.benchmark.eval_models import load_model  # noqa: E402

R_TARGET = 0.25
R_LO, R_HI = 0.10, 0.25
N = 300


def theory_r(root: Path, sid: str) -> float | None:
    man = root / "val" / "manifest.jsonl"
    if not man.exists():
        return None
    for line in open(man, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] == sid:
            return r["lpi"] / r["scan_dpi"] * r.get("freq_mul", 1.0)
    return None


def load_pair(root: Path, sid: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.asarray(Image.open(root / "val" / "scan" / f"{sid}.png"), dtype=np.float32) / 255.0
    y = np.asarray(Image.open(root / "val" / "gt" / f"{sid}.png"), dtype=np.float32) / 255.0
    to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)
    return to_t(x), to_t(y)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="run10,run7")
    ap.add_argument("--sets", default="synth,synth_run4,synth_1200")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    models = {}
    for name in args.models.split(","):
        name = name.strip()
        m, cfg, has_cond = load_model(f"ml/train/_{name}")
        models[name] = (m.to(device).eval(), has_cond)
    table: dict = {}
    for s in [x.strip() for x in args.sets.split(",") if x.strip()]:
        root = ROOT / "Data" / s
        scans = sorted((root / "val" / "scan").glob("*.png"))[:N]
        # cond-таблица для run-моделей с film (a0.json)
        cond_map: dict[str, list[float]] = {}
        a0p = root / "val" / "a0.json"
        if a0p.exists():
            from ml.model.film import build_cond_vec
            for k, v in json.loads(a0p.read_text(encoding="utf-8")).items():
                cond_map[k] = build_cond_vec(*v)
        res: dict[str, list[float]] = {}
        n_used = 0
        for p in scans:
            r = theory_r(root, p.stem)
            if r is None or not (R_LO <= r <= R_HI):
                continue
            n_used += 1
            x, y = load_pair(root, p.stem)
            k = R_TARGET / r
            h, w = x.shape[1:]
            # UNet L3 требует кратности 8 — снап (r получается 0.25±1.5%, для пробы ок)
            nh, nw = max(8, round(h * k / 8) * 8), max(8, round(w * k / 8) * 8)
            xu = torch.nn.functional.interpolate(
                x[None], size=(nh, nw), mode="bicubic", align_corners=False)[0]
            yu = torch.nn.functional.interpolate(
                y[None], size=(nh, nw), mode="bicubic", align_corners=False)[0]
            for name, (m, has_cond) in models.items():
                kw = {}
                if has_cond:
                    kw["cond"] = torch.tensor(
                        [cond_map.get(p.stem, [0.0, 0.0, 1.0, 0.0])],
                        dtype=torch.float32).to(device)
                with torch.amp.autocast("cuda", enabled=False):
                    pred = m(xu[None].to(device).float(), **kw)[0].cpu()
                # baseline: тот же вход без нормализации
                with torch.amp.autocast("cuda", enabled=False):
                    pred_b = m(x[None].to(device).float(), **kw)[0].cpu()
                back = torch.nn.functional.interpolate(
                    pred[None], size=(h, w), mode="bicubic",
                    align_corners=False, antialias=True)[0].clamp(0, 1)
                res.setdefault(f"{name}_norm", []).append(psnr(back, y))
                res.setdefault(f"{name}_base", []).append(psnr(pred_b.clamp(0, 1), y))
        summ = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
        summ["n"] = n_used
        table[s] = summ
        print(f"{s}: n={n_used} " + " ".join(f"{k}={v:.2f}" for k, v in summ.items() if k != "n"),
              flush=True)
    outp = ROOT / "ml" / "analysis" / "_norm_probe.json"
    outp.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print("saved:", outp)


if __name__ == "__main__":
    main()
