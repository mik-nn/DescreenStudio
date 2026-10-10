#!/usr/bin/env python
"""RealScan-оценка чекпоинтов: полная цепочка P0/E (notch -> anotch -> lf ->
net -> blend -> bgswap -> lintone) с заменой модели (run3 vs run4'', torch GPU).

Метрики без GT: resid / sharp / DC / LF (NoGTReport, center ROI 1024).

Использование:
    python ml/benchmark/eval_realscan.py --scans Data/RealScan/1821145u_1.tif ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.benchmark.eval_models import load_model  # noqa: E402
from ml.preproc.blend import adaptive_blend  # noqa: E402
from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.edgenotch import adaptive_notch_rgb  # noqa: E402
from ml.preproc.lfband import detect_lf_bands, subtract_lf_rgb  # noqa: E402
from ml.preproc.metrics import NoGTReport, evaluate  # noqa: E402
from ml.preproc.notch import auto_notch_rgb  # noqa: E402
from ml.preproc.paperbg import default_sigma, linear_tone, return_background  # noqa: E402
from ml.preproc.run_p0 import build_notch_lists, hann_1d, load_rgb, save_rgb  # noqa: E402
from ml.preproc.spectrum import estimate_raster, luma  # noqa: E402
from ml.train.dctfeat import dct_channels  # noqa: E402


def net_tiled(img: np.ndarray, model: torch.nn.Module, use_dct: bool,
              tile: int, overlap: int, device: str,
              cond: torch.Tensor | None = None) -> np.ndarray:
    c = 5 if use_dct else 3
    h, w = img.shape[:2]
    stride = tile - overlap
    ty = max(0, int(np.ceil((h - tile) / stride)))
    tx = max(0, int(np.ceil((w - tile) / stride)))
    ph, pw = ty * stride + tile, tx * stride + tile
    padded = np.zeros((ph, pw, 3), dtype=np.float32)
    padded[:h, :w] = img
    padded[h:, :] = padded[h - 1 : h, :]
    padded[:, w:] = padded[:, w - 1 : w]
    if use_dct:
        luma_full = (padded @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
        d = dct_channels(luma_full)
        padded = np.concatenate([padded, d], axis=-1)
    win = hann_1d(tile).astype(np.float32)
    w2 = win[:, None] * win[None, :]
    acc = np.zeros((3, ph, pw), dtype=np.float64)
    weight = np.zeros((ph, pw), dtype=np.float64)
    nty, ntx = (ph - tile) // stride + 1, (pw - tile) // stride + 1
    total = nty * ntx
    with torch.no_grad():
        for yy in range(nty):
            for xx in range(ntx):
                y0, x0 = yy * stride, xx * stride
                patch = torch.from_numpy(
                    padded[y0 : y0 + tile, x0 : x0 + tile].transpose(2, 0, 1)[None]
                ).to(device)
                with torch.amp.autocast("cuda", enabled=False):
                    if cond is not None:
                        out = model(patch.float(), cond.to(patch.device)).cpu().numpy()[0]
                    else:
                        out = model(patch.float()).cpu().numpy()[0]
                acc[:, y0 : y0 + tile, x0 : x0 + tile] += out * w2
                weight[y0 : y0 + tile, x0 : x0 + tile] += w2
    res = (acc / np.maximum(weight, 1e-9))[:, :h, :w]
    return np.clip(res.transpose(1, 2, 0), 0.0, 1.0).astype(np.float32)


def fmt(r: NoGTReport) -> str:
    s = (f"resid={r.residual_peak_xmean:7.1f}@r={r.residual_peak_r:.3f} "
         f"sharp={r.sharp_ratio:.2f} DC=({r.dc_shift[0]:+.3f},{r.dc_shift[1]:+.3f},{r.dc_shift[2]:+.3f}) "
         f"LF={r.lf_ratio:.2f}")
    if r.resid_top4_sum or r.de_mean:
        s += (f" top4={r.resid_top4_sum:.1f} lfmoire={r.resid_lf_moire:.1f} "
              f"quad={r.resid_max_quad:.1f} edgeHF={r.edge_hf_ratio:.2f} "
              f"flatHF={r.flat_hf_ratio:.2f} dE={r.de_mean:.2f}/{r.de_p90:.2f}")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", nargs="+", required=True)
    ap.add_argument("--outdir", default=r"Data\RealScan\_eval0920")
    ap.add_argument("--models", nargs="+", default=["run3", "run4"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--crop", type=int, default=0,
                    help="центральный кроп NxN перед цепочкой (0 = полный кадр)")
    ap.add_argument("--unsharp", type=int, default=0,
                    help="дополнительно оценить цепочку с adaptive_unsharp (1 = да)")
    ap.add_argument("--zero-cond", type=int, default=0,
                    help="абляция FiLM: нулевой cond вместо замера A0")
    ap.add_argument("--weights", default=None,
                    help="путь к весам (только при одном --models)")
    ap.add_argument("--backoff", type=int, default=0,
                    help="1 = backoff коррекции сети (1821146u)")
    ap.add_argument("--backoff-mode", choices=["flat", "global", "gated"], default="flat",
                    help="flat = блочный по плоским зонам, global = скалярный, "
                         "gated = скалярный, если top4 вырос к префильтру > gate")
    ap.add_argument("--backoff-w", type=float, default=1.0,
                    help="глобальный вес коррекции (mode=global/gated)")
    ap.add_argument("--backoff-gate", type=float, default=0.25,
                    help="gated: срабатывать при top4_out > top4_pre*(1+gate)")
    ap.add_argument("--backoff-wmin", type=float, default=0.25,
                    help="вес коррекции в блоках с выросшей периодикой")
    ap.add_argument("--backoff-eps", type=float, default=0.10,
                    help="порог роста периодики (доля) для backoff")
    args = ap.parse_args()
    use_unsharp = bool(args.unsharp)
    zero_cond = bool(args.zero_cond)
    use_backoff = bool(args.backoff)

    device = args.device if torch.cuda.is_available() else "cpu"
    cfg = P0Config()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict[str, str]] = {}

    models = {}
    for name in args.models:
        w = args.weights if args.weights and len(args.models) == 1 else None
        m, mcfg, mcond = load_model(f"ml/train/_{name}", weights=w)
        models[name] = (m.to(device), bool(mcfg.get("extra_dct")), mcond)

    for scan_path in args.scans:
        src = load_rgb(str(ROOT / scan_path))
        if args.crop:
            s = min(args.crop, src.shape[0], src.shape[1])
            y, x = (src.shape[0] - s) // 2, (src.shape[1] - s) // 2
            src = src[y : y + s, x : x + s]
        stem = Path(scan_path).stem.replace(" ", "_")
        est = estimate_raster(luma(src), cfg.spectrum)
        ch_peaks = build_notch_lists(src, cfg, cfg.spectrum.measure_roi)
        print(f"\n=== {scan_path} (A0: r={est.r:.4f} ang={est.angle_deg:.1f} conf={est.confidence:.2f})",
              flush=True)

        cur = auto_notch_rgb(src, ch_peaks, cfg.notch, cfg.spectrum.measure_roi)
        cur, _ = adaptive_notch_rgb(cur, ch_peaks, cfg.edge_notch, cfg.edge_notch.radius_ref_roi)
        bands = detect_lf_bands(luma(cur), cfg.lf)
        cur = subtract_lf_rgb(cur, bands, cfg.lf)
        save_rgb(str(outdir / f"{stem}_prefilter.png"), cur)
        pre = cur.copy()
        rep_pre = evaluate(src, pre, cfg.spectrum)
        print("  prefilter:", fmt(rep_pre), flush=True)

        def finish(nr: np.ndarray) -> tuple[np.ndarray, NoGTReport]:
            blended = adaptive_blend(pre, nr, src, cfg.blend)
            o = return_background(src, blended, default_sigma(cfg))
            o = linear_tone(o, src)
            return o, evaluate(src, o, cfg.spectrum)

        for name, (model, use_dct, has_cond) in models.items():
            if has_cond:
                if zero_cond:
                    cond = torch.zeros(1, 4, dtype=torch.float32)
                else:
                    from ml.model.film import build_cond_vec
                    cond = torch.tensor([build_cond_vec(est.r, est.angle_deg, est.confidence)],
                                        dtype=torch.float32)
            else:
                cond = None
            net_raw = net_tiled(pre, model, use_dct, cfg.tile.tile, cfg.tile.overlap, device, cond)
            if use_backoff and args.backoff_mode == "flat":
                from ml.preproc.backoff import flat_periodicity_backoff
                net_raw, bo = flat_periodicity_backoff(
                    pre, net_raw, block=128,
                    wmin=args.backoff_wmin, eps=args.backoff_eps)
                if bo.get("applied"):
                    print(f"    backoff: {bo['backoff']}/{bo['blocks']} blocks, "
                          f"flat={bo['flat']}, wmin={bo['wmin']}", flush=True)
            elif use_backoff and args.backoff_mode == "global":
                from ml.preproc.backoff import apply_global
                net_raw = apply_global(pre, net_raw, args.backoff_w)
                print(f"    backoff: global w={args.backoff_w}", flush=True)
            out, rep = finish(net_raw)
            if use_backoff and args.backoff_mode == "gated":
                t4o, t4p = rep.resid_top4_sum, rep_pre.resid_top4_sum
                if t4o and t4p and t4o > t4p * (1.0 + args.backoff_gate):
                    from ml.preproc.backoff import apply_global
                    out, rep = finish(apply_global(pre, net_raw, args.backoff_w))
                    print(f"    backoff FIRED: top4 {t4o:.1f} > {t4p:.1f}*"
                          f"{1.0 + args.backoff_gate:.2f} -> w={args.backoff_w}", flush=True)
            save_rgb(str(outdir / f"{stem}_{name}.png"), out)
            table.setdefault(stem, {})[name] = fmt(rep)
            print(f"  {name}: {fmt(rep)}", flush=True)
            if use_unsharp:
                from ml.preproc.unsharp import adaptive_unsharp
                out_u, amt, sharp_u = adaptive_unsharp(out, src, cfg.spectrum)
                save_rgb(str(outdir / f"{stem}_{name}_unsharp.png"), out_u)
                rep_u = evaluate(src, out_u, cfg.spectrum)
                table.setdefault(stem, {})[name + "+unsharp"] = fmt(rep_u)
                print(f"  {name}+unsharp(amt={amt:.2f}): {fmt(rep_u)}", flush=True)

    (outdir / "metrics.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print("\nsaved:", outdir / "metrics.json")


if __name__ == "__main__":
    main()
