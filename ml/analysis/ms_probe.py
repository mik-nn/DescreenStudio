#!/usr/bin/env python
"""MS-проба нормализации (zero-shot, без обучения).

Пирамида ТОЛЬКО ВНИЗ: native 512 -> 256 -> 128 (Lanczos+antialias, snap x8).
run10 на каждом уровне; мелкие выходы апсемплим обратно (чистый контент).
Сравнение с baseline на нативном GT: PSNR + tone-PSNR + sharp, по полосам r.
Правило вердикта: +0.5 дБ mean хотя бы в одной полосе r без просадки sharp.
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
from ml.model.film import build_cond_vec  # noqa: E402

R_LO, R_HI = 0.10, 0.50
SETS = ["Data/synth", "Data/synth_run4", "Data/synth_1200"]
N_PER_SET = 20
SCALES = [1.0, 0.5, 0.25]


def snap8(x: int) -> int:
    return max(64, round(x / 8) * 8)


def load_pair(root: Path, sid: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.asarray(Image.open(root / "val" / "scan" / f"{sid}.png"), dtype=np.float32) / 255.0
    y = np.asarray(Image.open(root / "val" / "gt" / f"{sid}.png"), dtype=np.float32) / 255.0
    to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)
    return to_t(x), to_t(y)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def lapvar(g: torch.Tensor) -> float:
    gg = g.mean(dim=0) if g.dim() == 3 else g
    lap = gg[:-2, :-2] + gg[:-2, 2:] + gg[2:, :-2] + gg[2:, 2:] - 4.0 * gg[1:-1, 1:-1]
    return float(lap.var().item())


def theory_map(root: Path) -> dict[str, float]:
    man = root / "val" / "manifest.jsonl"
    assert man.exists(), f"no manifest: {man}"
    out = {}
    for line in open(man, encoding="utf-8"):
        r = json.loads(line)
        out[r["id"]] = r["lpi"] / r["scan_dpi"] * r.get("freq_mul", 1.0)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="run10")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    models = {}
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        assert name, "empty model name"
        m, cfg, has_cond = load_model(f"ml/train/_{name}")
        models[name] = (m.to(device).eval(), has_cond)
    assert models, "no models loaded"
    table: dict = {}
    for s in SETS:
        root = ROOT / s
        scans = sorted((root / "val" / "scan").glob("*.png"))
        assert scans, f"no val scans in {root}"
        tmap = theory_map(root)
        stems = [p.stem for p in scans]
        missing = [st for st in stems if st not in tmap]
        assert not missing, f"{s}: {len(missing)} stems w/o manifest, e.g. {missing[:5]}"
        cand = [p for p in scans if R_LO <= tmap[p.stem] <= R_HI][:N_PER_SET]
        assert cand, f"{s}: no samples in r-band [{R_LO},{R_HI}]"
        print(f"{s}: pool={len(scans)} in-band, probing n={len(cand)}", flush=True)
        cond_map: dict[str, list[float]] = {}
        a0p = root / "val" / "a0.json"
        if a0p.exists():
            for k, v in json.loads(a0p.read_text(encoding="utf-8")).items():
                cond_map[k] = build_cond_vec(*v)
        rows = []
        for p in cand:
            x, y = load_pair(root, p.stem)
            h, w = x.shape[1:]
            rec: dict = {"id": p.stem, "r": round(tmap[p.stem], 4)}
            outs: dict[float, torch.Tensor] = {}
            for sc in SCALES:
                nh, nw = (snap8(round(h * sc)), snap8(round(w * sc))) if sc < 1.0 else (h, w)
                xi = x if sc == 1.0 else torch.nn.functional.interpolate(
                    x[None], size=(nh, nw), mode="bicubic",
                    align_corners=False, antialias=True)[0]
                for name, (m, has_cond) in models.items():
                    kw = {}
                    if has_cond:
                        kw["cond"] = torch.tensor(
                            [cond_map.get(p.stem, [0.0, 0.0, 1.0, 0.0])],
                            dtype=torch.float32).to(device)
                    with torch.amp.autocast("cuda", enabled=False):
                        pr = m(xi[None].to(device).float(), **kw)[0].cpu()
                    if sc < 1.0:
                        pr = torch.nn.functional.interpolate(
                            pr[None], size=(h, w), mode="bicubic",
                            align_corners=False, antialias=True)[0].clamp(0, 1)
                    else:
                        pr = pr.clamp(0, 1)
                    outs[(sc, name)] = pr
            for name in models:
                pb = outs[(1.0, name)]
                rec[f"{name}_base"] = round(psnr(pb, y), 2)
                rec[f"{name}_sharp_base"] = round(lapvar(pb) / max(lapvar(y), 1e-12), 3)
                for sc in SCALES[1:]:
                    pn = outs[(sc, name)]
                    rec[f"{name}_s{sc}"] = round(psnr(pn, y), 2)
                    rec[f"{name}_sharp_s{sc}"] = round(lapvar(pn) / max(lapvar(y), 1e-12), 3)
            rows.append(rec)
        table[s] = rows
        name0 = list(models)[0]
        for sc in SCALES:
            tag = "base" if sc == 1.0 else f"s{sc}"
            vals = [r[f"{name0}_{tag}"] for r in rows]
            print(f"{s} {tag}: n={len(vals)} mean={np.mean(vals):.2f}", flush=True)
    outp = ROOT / "ml" / "analysis" / "_msprobe.json"
    outp.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print("saved:", outp)


if __name__ == "__main__":
    main()
