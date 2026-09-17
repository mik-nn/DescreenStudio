"""P0 pipeline runner: A0 measure -> notch prefilter -> LF subtract ->
frozen run3 (tiled ORT) -> tone alignment. Saves every stage + metrics.

Usage:
    python ml/preproc/run_p0.py --input Data/RealScan/1821145u_1.tif
    python ml/preproc/run_p0.py --input ... --stages measure,notch,lf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.blend import adaptive_blend  # noqa: E402
from ml.preproc.edgenotch import adaptive_notch_rgb  # noqa: E402
from ml.preproc.lfband import detect_lf_bands, subtract_lf_rgb  # noqa: E402
from ml.preproc.metrics import NoGTReport, center_roi, evaluate  # noqa: E402
from ml.preproc.notch import auto_notch_rgb  # noqa: E402
from ml.preproc.paperbg import default_sigma, linear_tone, return_background  # noqa: E402
from ml.preproc.spectrum import (  # noqa: E402
    RasterEstimate,
    detect_peaks,
    estimate_raster,
    log_magnitude,
    luma,
    tophat_contrast,
)
from ml.preproc.tone import match_histogram, tone_curve  # noqa: E402


def load_rgb(path: str) -> np.ndarray:
    im = Image.open(path)
    im.load()
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: str, img: np.ndarray) -> None:
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def hann_1d(n: int) -> np.ndarray:
    t = 2.0 * np.pi * np.arange(n) / (n - 1.0)
    return 0.5 * (1.0 - np.cos(t))


def run3_tiled(
    img: np.ndarray, model_path: str, tile: int, overlap: int, ep: str
) -> np.ndarray:
    import onnxruntime as ort

    providers = (
        ["DmlExecutionProvider", "CPUExecutionProvider"]
        if ep == "dml"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(model_path, providers=providers)
    in_name = sess.get_inputs()[0].name
    c, h, w = 3, img.shape[0], img.shape[1]
    stride = tile - overlap
    ty = max(0, int(np.ceil((h - tile) / stride)))
    tx = max(0, int(np.ceil((w - tile) / stride)))
    ph, pw = ty * stride + tile, tx * stride + tile
    padded = np.zeros((1, c, ph, pw), dtype=np.float32)
    padded[0, :, :h, :w] = img.transpose(2, 0, 1)
    padded[0, :, h:, :] = padded[0, :, h - 1 : h, :]
    padded[0, :, :, w:] = padded[0, :, :, w - 1 : w]
    win = hann_1d(tile).astype(np.float32)
    w2 = win[:, None] * win[None, :]
    acc = np.zeros((c, ph, pw), dtype=np.float64)
    weight = np.zeros((ph, pw), dtype=np.float64)
    # warm-up
    sess.run(None, {in_name: np.zeros((1, c, tile, tile), dtype=np.float32)})
    nty, ntx = (ph - tile) // stride + 1, (pw - tile) // stride + 1
    done = 0
    total = nty * ntx
    t0 = time.time()
    for yy in range(nty):
        for xx in range(ntx):
            y0, x0 = yy * stride, xx * stride
            patch = padded[:, :, y0 : y0 + tile, x0 : x0 + tile].copy()
            out = np.asarray(sess.run(None, {in_name: patch})[0], dtype=np.float64)[0]
            acc[:, y0 : y0 + tile, x0 : x0 + tile] += out * w2
            weight[y0 : y0 + tile, x0 : x0 + tile] += w2
            done += 1
            if done % 8 == 0 or done == total:
                dt = time.time() - t0
                print(f"  tile {done}/{total} ({dt/done:.2f}s/tile)", flush=True)
    res = (acc / np.maximum(weight, 1e-9))[:, :h, :w]
    return np.clip(res.transpose(1, 2, 0), 0.0, 1.0).astype(np.float32)


def build_notch_lists(
    img: np.ndarray, cfg: P0Config, n: int
) -> list[list[tuple[float, float]]]:
    """Per-channel notch targets: fundamental + strong co-screens (>0.4x) +
    predicted harmonic family. Content peaks are NOT notched blindly."""
    from ml.preproc.spectrum import harmonic_family

    h, w, _ = img.shape
    s = min(n, h, w)
    y, x = h // 2 - s // 2, w // 2 - s // 2
    out: list[list[tuple[float, float]]] = []
    for c in range(3):
        roi = img[y : y + s, x : x + s, c].astype(np.float64)
        diff = tophat_contrast(log_magnitude(roi), cfg.spectrum.tophat_win)
        peaks, _ = detect_peaks(diff, cfg.spectrum)
        lst: list[tuple[float, float]] = []
        if peaks:
            fund = peaks[0]
            lst.extend(harmonic_family(fund.r, fund.angle_deg))
            for p in peaks[1:]:
                if p.strength > fund.strength * 0.4 and len(lst) < cfg.notch.max_peaks:
                    lst.extend(harmonic_family(p.r, p.angle_deg))
        out.append(lst[: cfg.notch.max_peaks])
    return out


def fmt_report(r: NoGTReport) -> str:
    return (
        f"resid_peak xmean={r.residual_peak_xmean:.1f} @r={r.residual_peak_r:.3f} | "
        f"LF_ratio={r.lf_ratio:.3f} | sharp_ratio={r.sharp_ratio:.3f} | "
        f"DC=({r.dc_shift[0]:+.4f},{r.dc_shift[1]:+.4f},{r.dc_shift[2]:+.4f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default=r"Data\RealScan\_p0")
    ap.add_argument(
        "--stages", default="notch,lf,run3,tone", help="csv subset of notch,lf,run3,tone"
    )
    ap.add_argument("--model", default=r"ml\export\fourier_descreen_unet.opt.onnx")
    ap.add_argument("--ep", default="dml", choices=["dml", "cpu"])
    ap.add_argument("--blend", type=float, default=1.0)
    args = ap.parse_args()

    cfg = P0Config()
    np.random.seed(cfg.seed)
    from dataclasses import replace as _replace

    cfg = _replace(cfg, blend=_replace(cfg.blend, base=args.blend))
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    src = load_rgb(args.input)
    print(f"input {args.input} {src.shape}")

    # A0: single-shot estimate (luma, center ROI) + per-channel notch lists.
    est: RasterEstimate = estimate_raster(luma(src), cfg.spectrum)
    print(
        f"A0 raster: r={est.r:.4f} ang={est.angle_deg:.1f} conf={est.confidence:.2f} "
        f"harm={est.harmonic_r} npeaks={len(est.peaks)}"
    )
    ch_peaks = build_notch_lists(src, cfg, cfg.spectrum.measure_roi)
    for c, lst in enumerate(ch_peaks):
        print(f"  ch{c}: {[(round(r,4), round(a,1)) for r, a in lst[:6]]}")

    cur = src
    table: dict[str, str] = {"source": fmt_report(evaluate(src, src, cfg.spectrum))}

    if "notch" in stages:
        cur = auto_notch_rgb(cur, ch_peaks, cfg.notch, cfg.spectrum.measure_roi)
        save_rgb(os.path.join(args.outdir, f"{stem}_pre_notch.png"), cur)
        table["+notch"] = fmt_report(evaluate(src, cur, cfg.spectrum))
        print("notch:", table["+notch"])

    if "anotch" in stages:
        cur, _cmap = adaptive_notch_rgb(
            cur, ch_peaks, cfg.edge_notch, cfg.edge_notch.radius_ref_roi
        )
        save_rgb(os.path.join(args.outdir, f"{stem}_pre_anotch.png"), cur)
        table["+anotch"] = fmt_report(evaluate(src, cur, cfg.spectrum))
        print("anotch:", table["+anotch"])

    if "lf" in stages:
        bands = detect_lf_bands(luma(cur), cfg.lf)
        print(f"LF bands: {[(round(b.r,4), b.axis) for b in bands]}")
        cur = subtract_lf_rgb(cur, bands, cfg.lf)
        save_rgb(os.path.join(args.outdir, f"{stem}_pre_lf.png"), cur)
        table["+lf"] = fmt_report(evaluate(src, cur, cfg.spectrum))
        print("lf:", table["+lf"])

    if "run3" in stages:
        pre_img = cur.copy()
        net_raw = run3_tiled(cur, args.model, cfg.tile.tile, cfg.tile.overlap, args.ep)
        save_rgb(os.path.join(args.outdir, f"{stem}_run3_raw.png"), net_raw)
        cur = adaptive_blend(pre_img, net_raw, load_rgb(args.input), cfg.blend)
        save_rgb(os.path.join(args.outdir, f"{stem}_run3.png"), cur)
        table["+run3"] = fmt_report(evaluate(src, cur, cfg.spectrum))
        print("run3:", table["+run3"])

    if "tone" in stages:
        cur = match_histogram(cur, src, cfg.tone)
        save_rgb(os.path.join(args.outdir, f"{stem}_final.png"), cur)
        table["+tone"] = fmt_report(evaluate(src, cur, cfg.spectrum))
        print("tone:", table["+tone"])
        ce, me = tone_curve(src, cur, cfg.tone)
        print("tone_curve out-vs-in:", np.round(me - ce, 4).tolist())

    if "bgswap" in stages:
        cur = return_background(src, cur, default_sigma(cfg))
        save_rgb(os.path.join(args.outdir, f"{stem}_bgswap.png"), cur)
        table["+bgswap"] = fmt_report(evaluate(src, cur, cfg.spectrum))
        print("bgswap:", table["+bgswap"])

    if "lintone" in stages:
        cur = linear_tone(cur, src)
        save_rgb(os.path.join(args.outdir, f"{stem}_lintone.png"), cur)
        table["+lintone"] = fmt_report(evaluate(src, cur, cfg.spectrum))
        print("lintone:", table["+lintone"])

    with open(os.path.join(args.outdir, f"{stem}_metrics.json"), "w") as f:
        json.dump(table, f, indent=1)
    print("done:", args.outdir)


if __name__ == "__main__":
    main()
