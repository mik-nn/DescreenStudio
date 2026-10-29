#!/usr/bin/env python
"""Chain-geometry dataset: пары «вход сети / контонный эталон» в разрешении
Найквиста–Котельникова (продукт 2.29: период растра ровно 2 px, тайл 512).

Отличия от `generate.py` (соответствие продакшн-инференсу):

1. **GT — контонный оттиск**, а не исходная цифровая картинка: та же
   сепарация/ICC, тот же TVI, та же бумага, тот же поворот и PSF, но без
   растровой маски (`compose.simulate_print_pair`). Сеть учится убирать
   растр и шум сканера, а не обращать цвет/тон/тинт бумаги.
2. **Вход — ровно прод-пре-стейдж** в chain-разрешении: A0 (`estimate_raster`) →
   адаптивный notch по гармоникам → LF-вычитание, затем тайлы 512.
   Раньше train брался из `prefiltered/` (глобальный notch, без LF, без
   Найквист-нормализации), а inference шёл по полной цепочке.
3. **Геометрия из линиатуры** (`simulator/chain.py`): 512 chain px = 512/(2·LPI)
   дюймов, т.е. та же физическая площадь, что видит сеть на инференсе.

Кадр шире тайла (`--chain-w`), чтобы A0 и LF считались на кадре, а не на
одном тайле; DataLoader режет случайные 512-кроны.

Выход: <root>/{train,val}/{scan,gt}/*.png + manifest.jsonl + a0.json + dataset.json
  scan/<id>.png — вход сети (pre, после notch+LF), chain-разрешение
  gt/<id>.png   — контонный эталон, chain-разрешение
  raw/<id>.png  — chain-скан до пре-фильтра (--keep-raw; для ablations)

Использование:
  $env:PYTHONIOENCODING='utf-8'
  C:\\Python314\\python.exe -X utf8 ml/dataset/make_chain.py --root Data/chain_pilot --train 8 --val 2
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.edgenotch import adaptive_notch_rgb  # noqa: E402
from ml.preproc.lfband import detect_lf_bands, subtract_lf_rgb  # noqa: E402
from ml.preproc.run_p0 import build_notch_lists  # noqa: E402
from ml.preproc.spectrum import estimate_raster, luma  # noqa: E402
from ml.simulator.chain import (  # noqa: E402
    ChainGeometry, box_down_rgb, build_chain_pair, chain_geometry,
)
from ml.simulator.config import ScanConfig, preset  # noqa: E402

from ml.preproc.v1 import V1Config, apply_v1, axis_peakiness, band_energy_native  # noqa: E402

GT_PHOTOS = ROOT / "Data" / "gt" / "unsplash"
GT_DOCS = ROOT / "Data" / "gt" / "documents"

# Паритет с `prefilter::LF_PEAK_DROP` (Rust): LF применяется, только если
# пиковость осевой полосы упала хотя бы на 15%.
LF_PEAK_DROP = 0.85

# Пресет -> линиатура. Диапазоны выведены из 11 приёмочных сканов
# (r = 0.099..0.146 при 800/1200 dpi), а не из каталога пресетов.
LPI_RANGE = {"newspaper": (110.0, 130.0), "magazine": (133.0, 160.0), "fine": (160.0, 185.0)}
PRESET_WEIGHTS = [("magazine", 0.5), ("fine", 0.3), ("newspaper", 0.2)]
INK_WEIGHTS = [("cmyk", 0.85), ("k", 0.1), ("duotone", 0.05)]
SCAN_DPI_CHOICES = [(1200, 0.7), (800, 0.3)]
SHOWTHROUGH_PROB = 0.15
PHOTO_PROB = 0.65

_cfg: P0Config | None = None


def _cfg_once() -> P0Config:
    global _cfg
    if _cfg is None:
        _cfg = P0Config()
    return _cfg


def pick_weighted(rng: random.Random, choices: list[tuple]) -> object:
    x = rng.random()
    acc = 0.0
    for val, w in choices:
        acc += w
        if x <= acc:
            return val
    return choices[-1][0]


def sample_paper_tint(rng: random.Random) -> tuple[float, float, float]:
    def ch(low: float, p_white: float) -> float:
        return 1.0 if rng.random() < p_white else round(rng.uniform(low, 1.0), 3)

    return (ch(0.94, 0.7), ch(0.94, 0.7), ch(0.85, 0.5))


def crop_at(path: str, w: int, h: int, rng: random.Random) -> np.ndarray:
    """Кроп w×h из источника; при нехватке — Lanczos до точного размера.

    Масштаб контента физически осмыслен: в журнальной печати фото имеет
    ~300 ppi, т.е. 1–2 пикселя контента на chain-пиксель (chain = 2·LPI ppi).
    Поэтому «мягкий» контент относительно скана 1200 dpi — норма, а не дефект.
    """
    im = Image.open(path).convert("RGB")
    cw, ch = min(im.width, w), min(im.height, h)
    x0 = rng.randint(0, im.width - cw)
    y0 = rng.randint(0, im.height - ch)
    crop = im.crop((x0, y0, x0 + cw, y0 + ch))
    if (cw, ch) != (w, h):
        crop = crop.resize((w, h), Image.LANCZOS)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    return arr


def prefilter_chain(scan_chain: np.ndarray, v1_acted: bool = False,
                    v1_info: dict | None = None) -> tuple[np.ndarray, dict]:
    """Прод-пре-стейдж ПОСЛЕ V1 и даунсэмпла: A0 -> notch -> LF с гейтом.

    V1 применяется раньше и в нативном разрешении (`make_one`), как в
    `descreen_file`; здесь только то, что идёт после Найквист-нормализации.
    """
    cfg = _cfg_once()
    info: dict = dict(v1_info or {})
    cur = scan_chain
    est = estimate_raster(luma(cur), cfg.spectrum)
    peaks = build_notch_lists(cur, cfg, cfg.spectrum.measure_roi)
    if v1_acted:
        pre = cur
        info["notch"] = "skipped(v1)"
    else:
        pre, _cmap = adaptive_notch_rgb(
            cur, peaks, cfg.edge_notch, cfg.edge_notch.radius_ref_roi
        )
        info["notch"] = "applied"
    bands = detect_lf_bands(luma(pre), cfg.lf)
    if bands:
        cand = subtract_lf_rgb(pre, bands, cfg.lf)
        pk0 = axis_peakiness(luma(pre))
        pk1 = axis_peakiness(luma(cand))
        keep = pk1 < pk0 * LF_PEAK_DROP
        info["lf_gate"] = [round(pk0, 1), round(pk1, 1), bool(keep)]
        pre = cand if keep else pre
    else:
        info["lf_gate"] = None
    info.update({
        "a0_r": round(float(est.r), 5),
        "a0_angle": round(float(est.angle_deg), 2),
        "a0_conf": round(float(est.confidence), 4),
        "n_peaks": [len(p) for p in peaks],
        "lf_bands": [[round(b.r, 5), b.axis] for b in bands],
    })
    return pre, info


def make_one(spec: dict) -> dict:
    t_start = time.perf_counter()
    rng = random.Random(spec["seed"])
    profile = spec["profile"]
    geom: ChainGeometry = chain_geometry(
        lpi=spec["lpi"],
        scan_dpi=spec["scan_dpi"],
        print_dpi=spec["print_dpi"],
        chain_h=spec["chain_h"],
        chain_w=spec["chain_w"],
        r_out=spec.get("r_out", 0.5),
    )
    scan_h, scan_w = geom.scan_h, geom.scan_w

    # Тон контента: экстремально тёмные/светлые кадры дают вырожденный растр
    # (точки сливаются либо их нет), и детектор растра уходит на суммарные
    # продукты между красками. Реальные страницы лежат в средних тонах, поэтому
    # кадр перекропывается, пока средняя яркость не попадёт в диапазон.
    best = None
    for _ in range(spec.get("luma_tries", 12)):
        cand = crop_at(spec["file"], scan_w, scan_h, rng)
        m = float(cand.mean())
        if best is None or abs(m - 0.5) < abs(best[1] - 0.5):
            best = (cand, m)
        if spec["luma_min"] <= m <= spec["luma_max"]:
            best = (cand, m)
            break
    gt_crop, content_luma = best

    pcfg = preset(spec["preset"], seed=spec["print_seed"]).with_overrides(        lpi=spec["lpi"], ink_set=spec["ink_set"], paper_tint=tuple(spec["paper_tint"]),
        print_dpi=spec["print_dpi"],
        registration_px_max=spec["registration_px_max"],
        warp_amp_px=spec["warp_amp_px"],
        warp_corr_px=spec["warp_corr_px"],
    )
    pcfg = pcfg.for_scan_dpi(spec["scan_dpi"])
    scfg = ScanConfig(seed=spec["scan_seed"])
    # Калибровка шума по 11 реальным сканам (JOURNAL 2.37, Temp/noise_calib.py):
    # после V1 остаточный ВЧ-шум у реальных сканов ≈31 дБ, у синтетики ≈20 дБ,
    # т.е. симулятор в 3–4 раза шумнее по σ. Переопределения — CLI, дефолты не
    # меняются, чтобы старые датасеты воспроизводились бит-в-бит.
    scfg = scfg.with_overrides(
        poisson_peak_min=spec.get("poisson_min", scfg.poisson_peak_min),
        poisson_peak_max=spec.get("poisson_max", scfg.poisson_peak_max),
        gauss_sigma_min=spec.get("gauss_min", scfg.gauss_sigma_min),
        gauss_sigma_max=spec.get("gauss_max", scfg.gauss_sigma_max),
    )

    ghost = None
    show_alpha = 0.0
    if rng.random() < SHOWTHROUGH_PROB:
        ghost = np.asarray(
            Image.fromarray(
                (gt_crop[:, ::-1] * 255.0 + 0.5).astype(np.uint8)
            ).filter(ImageFilter.GaussianBlur(2.0)),
            dtype=np.float32,
        ) / 255.0
        show_alpha = round(rng.uniform(scfg.show_alpha_min, scfg.show_alpha_max), 4)

    from ml.simulator.scan import sample_scan_params

    sparams = sample_scan_params(scfg, np.random.default_rng(spec["scan_seed"]))
    sparams["show_alpha"] = show_alpha

    scan_native, gt_native, scan_chain, gt_chain = build_chain_pair(
        gt_crop, geom, pcfg, scfg, sparams, ghost=ghost
    )
    scan_native_saved = scan_native if spec.get("keep_native") else None

    # V1 в НАТИВНОМ разрешении, до Найквист-даунсэмпла — как в продукте.
    # Вход сети обязан совпадать с тем, что видит инференс (JOURNAL 2.32–2.34).
    v1_info: dict = {}
    scan_v1 = scan_native
    if spec.get("v1", True):
        band0 = band_energy_native(scan_native, geom.r)
        scan_v1, v1_rep = apply_v1(scan_native, geom.r, V1Config(), _cfg_once())
        band1 = band_energy_native(scan_v1, geom.r)
        v1_info = {
            "v1_acted": bool(v1_rep["acted"]),
            "v1_coh": round(float(v1_rep["coh"]), 3),
            "v1_carriers": v1_rep["carriers"],
            "v1_band_drop": round(100.0 * (1.0 - band1 / max(band0, 1e-30)), 1),
        }
    pre, pinfo = prefilter_chain(
        box_down_rgb(scan_v1, geom.chain_w, geom.chain_h),
        v1_acted=bool(v1_info.get("v1_acted")),
        v1_info=v1_info,
    )
    gt_chain = box_down_rgb(gt_native, geom.chain_w, geom.chain_h)
    scan_chain = box_down_rgb(scan_native, geom.chain_w, geom.chain_h)

    pid = spec["id"]
    scan_dir, gt_dir, raw_dir = (Path(spec["out"]) / "scan", Path(spec["out"]) / "gt",
                                 Path(spec["out"]) / "raw")
    scan_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    def save(arr: np.ndarray, path: Path) -> None:
        Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)

    save(pre, scan_dir / f"{pid}.png")
    save(gt_chain, gt_dir / f"{pid}.png")
    if spec.get("keep_raw"):
        raw_dir.mkdir(parents=True, exist_ok=True)
        save(scan_chain, raw_dir / f"{pid}.png")
    if spec.get("keep_native"):
        native_dir = Path(spec["out"]) / "native"
        native_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(scan_native_saved, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(
            native_dir / f"{pid}.png"
        )

    return {
        "id": pid,
        "profile": profile,
        "src": Path(spec["file"]).name,
        "kind": spec["kind"],
        "scan_dpi": spec["scan_dpi"],
        "print_dpi": spec["print_dpi"],
        "lpi": round(spec["lpi"], 2),
        "r": round(geom.r, 5),
        "r_out": round(geom.r_out, 4),
        "scale": round(geom.scale, 5),
        "inches": [round(geom.inches_h, 3), round(geom.inches_w, 3)],
        "chain": [geom.chain_h, geom.chain_w],
        "scan_px": [geom.scan_h, geom.scan_w],
        "print_px": [geom.print_h, geom.print_w],
        "preset": spec["preset"],
        "ink_set": spec["ink_set"],
        "paper_tint": [round(float(v), 3) for v in spec["paper_tint"]],
        "content_luma": round(content_luma, 4),
        "angle_deg": spec["angle_deg"],
        "show_alpha": show_alpha,
        "print_seed": spec["print_seed"],
        "scan_seed": spec["scan_seed"],
        "scan_params": {k: (round(float(v), 4) if isinstance(v, float) else v)
                        for k, v in sparams.items()},
        **pinfo,
        "secs": round(time.perf_counter() - t_start, 2),
    }


def build_plan(n: int, photos: list[str], docs: list[str], seed: int, start: int,
               args: argparse.Namespace, profile: str) -> list[dict]:
    rng = random.Random(seed)
    plan = []
    for i in range(n):
        kind = "photo" if rng.random() < PHOTO_PROB else "doc"
        pool = photos if kind == "photo" else docs
        scan_dpi = pick_weighted(rng, SCAN_DPI_CHOICES)
        preset_name = pick_weighted(rng, PRESET_WEIGHTS)
        lo, hi = LPI_RANGE[preset_name]
        lpi = rng.uniform(lo, hi)
        if args.lpi_min is not None:
            lpi = rng.uniform(args.lpi_min, args.lpi_max)
        # r = LPI/scan_dpi должен лежать в измеренном диапазоне реальных сканов.
        if args.r_range is not None:
            r_lo, r_hi = args.r_range
            scan_dpi = int(args.dpi_force or scan_dpi)
            lpi = rng.uniform(r_lo, r_hi) * scan_dpi
        angle = rng.uniform(0.1, 2.0)
        r_out = args.r_out
        if args.native_scale:
            # Нативный режим: рабочий кадр = скан (scale 1.0), сеть видит растр
            # с его настоящим периодом 4–10 px. Так продукт и работает при
            # `r_out <= r_in`: даунсэмпла нет, апскейла нет — резкость 1:1.
            r_out = lpi / scan_dpi
        elif args.r_out_jitter > 0:
            r_out = rng.uniform(max(0.15, r_out - args.r_out_jitter),
                                min(0.5, r_out + args.r_out_jitter))
        plan.append({
            "id": f"{profile[0]}{start + i:05d}",
            "profile": profile,
            "kind": kind,
            "file": rng.choice(pool),
            "scan_dpi": int(scan_dpi),
            "print_dpi": args.print_dpi,
            "chain_h": args.chain_h,
            "chain_w": args.chain_w,
            "r_out": round(r_out, 4),
            "preset": preset_name,
            "lpi": round(lpi, 2),
            "ink_set": pick_weighted(rng, INK_WEIGHTS),
            "paper_tint": sample_paper_tint(rng),
            "registration_px_max": args.registration_px,
            "warp_amp_px": args.warp_px,
            "warp_corr_px": args.warp_corr_px,
            "luma_min": args.luma_min,
            "luma_max": args.luma_max,
            "poisson_min": args.poisson_min,
            "poisson_max": args.poisson_max,
            "gauss_min": args.gauss_min,
            "gauss_max": args.gauss_max,
            "angle_deg": round(angle, 3),
            "seed": rng.randint(0, 2**31 - 1),
            "print_seed": rng.randint(0, 2**31 - 1),
            "scan_seed": rng.randint(0, 2**31 - 1),
            "keep_raw": args.keep_raw,
            "keep_native": args.keep_native,
            "out": str(Path(args.root) / profile),
        })
    return plan


def run_profile(args: argparse.Namespace, profile: str, n: int, seed: int, start: int,
                photos: list[str], docs: list[str]) -> None:
    out = Path(args.root) / profile
    (out / "scan").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    plan = build_plan(n, photos, docs, seed, start, args, profile)
    man_path = out / "manifest.jsonl"
    done: set[str] = set()
    if man_path.exists():
        for line in io.open(man_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [s for s in plan if s["id"] not in done]
    print(f"{profile}: total={n} todo={len(todo)} workers={args.workers}", flush=True)
    a0: dict = {}
    a0_path = out / "a0.json"
    if a0_path.exists():
        a0 = json.loads(a0_path.read_text(encoding="utf-8"))
    with io.open(man_path, "a", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(make_one, todo)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            a0[rec["id"]] = [rec["a0_r"], rec["a0_angle"], rec["a0_conf"]]
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"{profile}: {i + 1}/{len(todo)} ({rec['secs']}s/last)", flush=True)
            a0_path.write_text(json.dumps(a0), encoding="utf-8")
    print(f"{profile}: done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Data/chain")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chain-w", type=int, default=1024, help="ширина chain-кадра, px")
    ap.add_argument("--chain-h", type=int, default=512, help="высота chain-кадра, px")
    ap.add_argument("--r-out", type=float, default=0.5,
                    help="частота растра в рабочем кадре (0.5 = Котельников, 2 px)")
    ap.add_argument("--r-out-jitter", type=float, default=0.0,
                    help="разброс r_out в обе стороны (устойчивость сети к оценке растра)")
    ap.add_argument("--native-scale", action="store_true",
                    help="рабочий кадр = скан (scale 1.0): r_out = r каждого кадра")
    ap.add_argument("--print-dpi", type=int, default=2400)
    ap.add_argument("--dpi-force", type=int, default=None)
    ap.add_argument("--lpi-min", type=float, default=None)
    ap.add_argument("--lpi-max", type=float, default=None)
    ap.add_argument("--r-min", type=float, default=None)
    ap.add_argument("--r-max", type=float, default=None)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--keep-native", action="store_true",
                    help="сохранить нативный скан (для паритета с Rust-цепочкой)")
    ap.add_argument("--registration-px", type=float, default=8.0,
                    help="постоянная приводка плиты, px печати (0.05-0.1 мм)")
    ap.add_argument("--warp-px", type=float, default=1.5,
                    help="амплитуда медленного дрейфа плиты, px печати")
    ap.add_argument("--warp-corr-px", type=int, default=256)
    # Шум сканера. Дефолты = исторические (peak 40–120, gauss 0.004–0.012);
    # калибровка по 11 реальным сканам даёт peak ~700–900 при gauss ~0.003–0.008
    # (JOURNAL 2.37).
    ap.add_argument("--poisson-min", type=float, default=40.0)
    ap.add_argument("--poisson-max", type=float, default=120.0)
    ap.add_argument("--gauss-min", type=float, default=0.004)
    ap.add_argument("--gauss-max", type=float, default=0.012)
    ap.add_argument("--luma-min", type=float, default=0.22,
                    help="нижняя граница средней яркости кадра (вырожденный растр)")
    ap.add_argument("--luma-max", type=float, default=0.80,
                    help="верхняя граница средней яркости кадра")
    ap.add_argument("--val-photos", type=int, default=200)
    ap.add_argument("--val-docs", type=int, default=50)
    args = ap.parse_args()

    r_range = None
    if args.r_min is not None or args.r_max is not None:
        assert args.r_min is not None and args.r_max is not None, "--r-min/--r-max задаются парой"
        r_range = (args.r_min, args.r_max)
    args.r_range = r_range
    if args.lpi_min is not None:
        assert args.lpi_max is not None, "--lpi-min/--lpi-max задаются парой"

    photos = sorted(str(p) for p in GT_PHOTOS.glob("*.jpg"))
    docs = sorted(str(p) for p in GT_DOCS.glob("*.png"))
    assert len(photos) > args.val_photos and len(docs) > args.val_docs
    train_photos, val_photos = photos[: -args.val_photos], photos[-args.val_photos :]
    train_docs, val_docs = docs[: -args.val_docs], docs[-args.val_docs :]

    t0 = time.perf_counter()
    run_profile(args, "train", args.train, args.seed, 0, train_photos, train_docs)
    run_profile(args, "val", args.val, args.seed + 1, 0, val_photos, val_docs)
    summary = {
        "generator": "make_chain",
        "gt": "contone print (same separation/TVI/paper/PSF/rotation, no screen)",
        "input": "prod pre-stage at chain res (A0 -> adaptive notch -> LF)",
        "train": args.train,
        "val": args.val,
        "seed": args.seed,
        "chain": [args.chain_h, args.chain_w],
        "print_dpi": args.print_dpi,
        "dpi_force": args.dpi_force,
        "registration_px": args.registration_px,
        "warp_px": args.warp_px,
        "warp_corr_px": args.warp_corr_px,
        "luma": [args.luma_min, args.luma_max],
        "lpi": [args.lpi_min, args.lpi_max],
        "r": list(r_range) if r_range else None,
        "presets": PRESET_WEIGHTS,
        "inks": INK_WEIGHTS,
        "elapsed_secs": round(time.perf_counter() - t0, 1),
    }
    Path(args.root).mkdir(parents=True, exist_ok=True)
    (Path(args.root) / "dataset.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
