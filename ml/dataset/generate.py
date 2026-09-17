from __future__ import annotations

import argparse
import io
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ml.simulator.color import rgb_to_cmyk  # noqa: F401 (keeps package import warm in workers)
from ml.simulator.compose import simulate_print
from ml.simulator.config import PRESETS, PrintConfig, ScanConfig, preset
from ml.simulator.scan import rotate_rgb, sample_scan_params, simulate_scan

ROOT = Path(__file__).resolve().parents[2]
GT_PHOTOS = ROOT / "Data" / "gt" / "unsplash"
GT_DOCS = ROOT / "Data" / "gt" / "documents"
PATCH = 512

PRESET_WEIGHTS = [("magazine", 0.5), ("newspaper", 0.25), ("fine", 0.25)]
LPI_RANGE = {"magazine": (120.0, 160.0), "newspaper": (75.0, 100.0), "fine": (150.0, 190.0)}
INK_WEIGHTS = [("cmyk", 0.8), ("k", 0.1), ("duotone", 0.1)]
SHOWTHROUGH_PROB = 0.15
PHOTO_PROB = 0.65
MIXED_PROB = 0.20
DPI_CHOICES = [(300, 0.6), (600, 0.4)]

_CACHE: dict = {}


def pick_weighted(rng: random.Random, choices: list[tuple]) -> object:
    x = rng.random()
    acc = 0.0
    for val, w in choices:
        acc += w
        if x <= acc:
            return val
    return choices[-1][0]


def load_source(kind: str, path: str, scan_dpi: int) -> np.ndarray:
    key = (kind, path, scan_dpi)
    img = _CACHE.get(key)
    if img is None:
        im = Image.open(path).convert("RGB")
        if kind == "doc" and scan_dpi == 300:
            im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
        img = np.asarray(im, dtype=np.float32) / 255.0
        if len(_CACHE) < 8:
            _CACHE[key] = img
    return img


def overlay_headline(crop: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w, _ = crop.shape
    im = Image.fromarray((crop * 255 + 0.5).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    fonts = ["arialbd.ttf", "timesbd.ttf", "verdanab.ttf", "DejaVuSans-Bold.ttf"]
    words = "SALE NEW ISSUE SPECIAL REPORT PHOTO STORY EXCLUSIVE WEEKEND EDITION".split(" ")
    y = rng.randint(0, h - 120)
    for _ in range(rng.randint(1, 3)):
        size = rng.randint(36, 110)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/" + rng.choice(fonts), size)
        except OSError:
            font = ImageFont.load_default()
        text = " ".join(rng.sample(words, rng.randint(1, 3)))
        tw = draw.textlength(text, font=font)
        x = rng.randint(0, max(w - int(tw), 0))
        col = rng.choice([(15, 15, 15), (200, 30, 30), (30, 60, 160), (255, 255, 255)])
        draw.text((x, y), text, font=font, fill=col)
        y += size + rng.randint(4, 20)
        if y >= h - 40:
            break
    return np.asarray(im, dtype=np.float32) / 255.0


def gt_top_peak(gt: np.ndarray) -> float:
    g = gt.mean(axis=-1).astype(np.float64)
    spec = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    mag = np.abs(spec)
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w // 2) / w) ** 2 + ((yy - h // 2) / h) ** 2)
    mag[r < 0.02] = 0.0
    return round(float(mag.max() / (mag.mean() + 1e-9)), 1)


def sample_paper_tint(rng: random.Random) -> tuple[float, float, float]:
    """Mostly white paper, sometimes aged/yellowed (old books)."""

    def ch(low: float, p_white: float) -> float:
        return 1.0 if rng.random() < p_white else round(rng.uniform(low, 1.0), 3)

    return (ch(0.94, 0.7), ch(0.94, 0.7), ch(0.85, 0.5))


def _scan_params_with_alpha(scfg: ScanConfig, seed: int, show_alpha: float) -> dict:
    """Reproduce simulate_scan's default sampling, then inject show-through alpha."""
    from ml.simulator.scan import sample_scan_params as _sample

    params = _sample(scfg, np.random.default_rng(seed))
    params["show_alpha"] = show_alpha
    return params


def make_patch(spec: dict, out_dirs: tuple[Path, Path]) -> dict:
    rng = random.Random(spec["seed"])
    img = load_source(spec["kind"], spec["file"], spec["scan_dpi"])
    h, w, _ = img.shape
    x0 = rng.randint(0, w - PATCH)
    y0 = rng.randint(0, h - PATCH)
    gt = img[y0 : y0 + PATCH, x0 : x0 + PATCH].copy()
    mixed = spec["kind"] == "photo" and rng.random() < MIXED_PROB
    if mixed:
        gt = overlay_headline(gt, rng)
    lpi = rng.uniform(*LPI_RANGE[spec["preset"]])
    tint = sample_paper_tint(rng)
    pcfg = preset(spec["preset"], seed=spec["print_seed"]).with_overrides(
        lpi=lpi, ink_set=spec["ink_set"], paper_tint=tint
    )
    pcfg = pcfg.for_scan_dpi(spec["scan_dpi"])
    scfg = ScanConfig(seed=spec["scan_seed"])
    printed = simulate_print(gt, pcfg)
    ghost_src: str | None = None
    show_alpha = 0.0
    if rng.random() < SHOWTHROUGH_PROB:
        gx = rng.randint(0, w - PATCH)
        gy = rng.randint(0, h - PATCH)
        ghost = img[gy : gy + PATCH, gx : gx + PATCH][:, ::-1]
        ghost = np.asarray(
            Image.fromarray((ghost * 255 + 0.5).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(2.0)
            ),
            dtype=np.float32,
        ) / 255.0
        show_alpha = round(rng.uniform(scfg.show_alpha_min, scfg.show_alpha_max), 4)
        ghost_src = f"{Path(spec['file']).name}@{gx},{gy}"
    else:
        ghost = None
    scan, sparams = simulate_scan(
        printed,
        pcfg,
        scfg,
        params=_scan_params_with_alpha(scfg, spec["scan_seed"], show_alpha),
        ghost=ghost,
    )
    gt_aligned = rotate_rgb(gt, sparams["angle"])
    pid = spec["id"]
    scan_dir, gt_dir = out_dirs
    Image.fromarray((scan * 255 + 0.5).astype(np.uint8)).save(scan_dir / f"{pid}.png")
    Image.fromarray((gt_aligned * 255 + 0.5).astype(np.uint8)).save(gt_dir / f"{pid}.png")
    return {
        "id": pid,
        "kind": spec["kind"] + ("+headline" if mixed else ""),
        "src": Path(spec["file"]).name,
        "crop": [x0, y0],
        "scan_dpi": spec["scan_dpi"],
        "preset": spec["preset"],
        "lpi": round(lpi, 2),
        "ink_set": spec["ink_set"],
        "paper_tint": [round(float(v), 3) for v in pcfg.paper_tint],
        "ghost_src": ghost_src,
        "print_seed": spec["print_seed"],
        "scan_seed": spec["scan_seed"],
        "scan_params": {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in sparams.items()},
        "gt_top_peak": gt_top_peak(gt_aligned),
    }


def build_plan(n: int, photos: list[str], docs: list[str], seed: int, start: int = 0) -> list[dict]:
    rng = random.Random(seed)
    plan = []
    for i in range(n):
        kind = "photo" if rng.random() < PHOTO_PROB else "doc"
        pool = photos if kind == "photo" else docs
        plan.append(
            {
                "id": f"p{start + i:05d}",
                "kind": kind,
                "file": rng.choice(pool),
                "scan_dpi": pick_weighted(rng, DPI_CHOICES),
                "preset": pick_weighted(rng, PRESET_WEIGHTS),
                "ink_set": pick_weighted(rng, INK_WEIGHTS),
                "seed": rng.randint(0, 2**31 - 1),
                "print_seed": rng.randint(0, 2**31 - 1),
                "scan_seed": rng.randint(0, 2**31 - 1),
            }
        )
    return plan


def run_split(root: Path, name: str, n: int, photos: list[str], docs: list[str], seed: int, workers: int) -> None:
    out = root / name
    scan_dir, gt_dir = out / "scan", out / "gt"
    scan_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(n, photos, docs, seed)
    todo = [s for s in plan if not (scan_dir / f"{s['id']}.png").exists() or not (gt_dir / f"{s['id']}.png").exists()]
    print(f"{name}: total={n} todo={len(todo)} workers={workers}", flush=True)
    manifest = out / "manifest.jsonl"
    done_ids = set()
    if manifest.exists():
        with io.open(manifest, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [s for s in todo if s["id"] not in done_ids]
    with io.open(manifest, "a", encoding="utf-8") as f, ProcessPoolExecutor(max_workers=workers) as ex:
        for i, rec in enumerate(ex.map(make_patch, todo, [((scan_dir, gt_dir))] * len(todo))):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"{name}: {i + 1}/{len(todo)}", flush=True)
    print(f"{name}: done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Data/synth")
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val-photos", type=int, default=200)
    ap.add_argument("--val-docs", type=int, default=50)
    args = ap.parse_args()
    root = Path(args.root)
    photos = sorted(str(p) for p in GT_PHOTOS.glob("*.jpg"))
    docs = sorted(str(p) for p in GT_DOCS.glob("*.png"))
    assert len(photos) > args.val_photos and len(docs) > args.val_docs
    train_photos, val_photos = photos[:- args.val_photos], photos[-args.val_photos :]
    train_docs, val_docs = docs[:- args.val_docs], docs[-args.val_docs :]
    run_split(root, "train", args.train, train_photos, train_docs, args.seed, args.workers)
    run_split(root, "val", args.val, val_photos, val_docs, args.seed + 1, args.workers)
    summary = {
        "train": args.train,
        "val": args.val,
        "seed": args.seed,
        "photo_prob": PHOTO_PROB,
        "presets": PRESET_WEIGHTS,
        "dpi": DPI_CHOICES,
    }
    (root / "dataset.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
