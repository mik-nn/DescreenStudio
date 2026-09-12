from __future__ import annotations

import argparse
import io
import json
import random
import re
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
TEXTS_DIR = ROOT / "Data" / "gt" / "doc_texts"
PAGES_DIR = ROOT / "Data" / "gt" / "documents"
FONTS_DIR = Path("C:/Windows/Fonts")

BOOKS = [1342, 11, 84, 1661, 74, 98, 43, 2701, 2600, 2554, 1259, 1400, 1080, 244, 76, 5200]

FONTS = {
    "serif": ["times.ttf", "timesbd.ttf", "timesi.ttf", "georgia.ttf"],
    "sans": ["arial.ttf", "arialbd.ttf", "verdana.ttf", "calibri.ttf"],
    "mono": ["cour.ttf", "courbd.ttf", "DejaVuSansMono.ttf"],
}

DPI = 600
PAGE_W, PAGE_H = int(8.27 * DPI), int(11.69 * DPI)
MARGIN = 400


def pt(size_pt: float) -> int:
    return max(int(size_pt / 72 * DPI), 8)


def fetch_books() -> list[Path]:
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for bid in BOOKS:
        dest = TEXTS_DIR / f"pg{bid}.txt"
        if not dest.exists() or dest.stat().st_size == 0:
            try:
                r = requests.get(f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt", timeout=120)
                r.raise_for_status()
                dest.write_bytes(r.content)
            except requests.RequestException:
                continue
        if dest.exists() and dest.stat().st_size > 10000:
            out.append(dest)
    return out


def clean_text(path: Path) -> list[str]:
    raw = path.read_bytes().decode("utf-8", "replace")
    start = re.search(r"\*\*\* START OF .*?\*\*\*", raw, re.S)
    end = re.search(r"\*\*\* END OF .*?\*\*\*", raw, re.S)
    if start and end:
        raw = raw[start.end() : end.start()]
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw)]
    return [p for p in paras if len(p) > 40]


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_table(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, rng: random.Random) -> int:
    rows, cols = rng.randint(3, 7), rng.randint(3, 5)
    rh = pt(11)
    font = ImageFont.truetype(str(FONTS_DIR / "arial.ttf"), pt(8))
    cw = w // cols
    for r in range(rows + 1):
        draw.line([(x, y + r * rh), (x + w, y + r * rh)], fill=(20, 20, 20), width=3)
    for c in range(cols + 1):
        draw.line([(x + c * cw, y), (x + c * cw, y + rows * rh)], fill=(20, 20, 20), width=3)
    for r in range(rows):
        for c in range(cols):
            draw.text((x + c * cw + 12, y + r * rh + 4), f"{rng.randint(10, 9999)}", font=font, fill=(20, 20, 20))
    return y + rows * rh + pt(10)


def draw_figure(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, rng: random.Random) -> int:
    h = rng.randint(pt(60), pt(140))
    shade = rng.randint(90, 200)
    draw.rectangle([x, y, x + w, y + h], fill=(shade, shade, shade), outline=(20, 20, 20), width=3)
    for _ in range(rng.randint(2, 5)):
        x0 = rng.randint(x, x + w)
        draw.line([(x0, y), (x0 + rng.randint(-w // 3, w // 3), y + h)], fill=(40, 40, 40), width=4)
    font = ImageFont.truetype(str(FONTS_DIR / "ariali.ttf"), pt(8))
    draw.text((x, y + h + 8), f"Fig. {rng.randint(1, 40)} — illustration plate.", font=font, fill=(20, 20, 20))
    return y + h + pt(24)


def render_page(paras: list[str], rng: random.Random, page_no: int) -> tuple[Image.Image, dict]:
    img = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    fam = rng.choice(["serif", "serif", "sans", "mono"])
    body_name = rng.choice(FONTS[fam])
    body_pt = rng.choice([9, 10, 10, 11, 12])
    body = ImageFont.truetype(str(FONTS_DIR / body_name), pt(body_pt))
    ncols = 2 if rng.random() < 0.45 else 1
    gutter = 120
    col_w = (PAGE_W - 2 * MARGIN - (ncols - 1) * gutter) // ncols
    meta = {"family": fam, "font": body_name, "body_pt": body_pt, "columns": ncols, "fine_print": False, "extras": []}
    if rng.random() < 0.15:
        hf = ImageFont.truetype(str(FONTS_DIR / rng.choice(FONTS["serif"])), pt(rng.choice([16, 20, 24])))
        head = rng.choice(paras)[:60].upper()
        draw.text((MARGIN, MARGIN), head, font=hf, fill=(15, 15, 15))
        top = MARGIN + pt(30)
        meta["extras"].append("heading")
    else:
        top = MARGIN
    pi = rng.randrange(len(paras))
    for c in range(ncols):
        x = MARGIN + c * (col_w + gutter)
        y = top
        bottom = PAGE_H - MARGIN - pt(14)
        while y < bottom - pt(20):
            roll = rng.random()
            if roll < 0.10:
                y = draw_table(draw, x, y, col_w, rng)
                meta["extras"].append("table")
            elif roll < 0.25 and y < bottom - pt(160):
                y = draw_figure(draw, x, y, col_w, rng)
                meta["extras"].append("figure")
            else:
                if rng.random() < 0.30:
                    fp = rng.choice([4, 5, 6])
                    font = ImageFont.truetype(str(FONTS_DIR / body_name), pt(fp))
                    meta["fine_print"] = True
                else:
                    font = body
                for line in wrap(draw, paras[pi % len(paras)], font, col_w):
                    lh = pt(body_pt) + 14 if font is body else pt(fp) + 10
                    if y + lh > bottom:
                        break
                    draw.text((x, y), line, font=font, fill=(15, 15, 15))
                    y += lh
                y += pt(6)
                pi += 1
    foot = ImageFont.truetype(str(FONTS_DIR / "arial.ttf"), pt(8))
    draw.text((PAGE_W // 2, PAGE_H - MARGIN + 20), str(page_no), font=foot, fill=(15, 15, 15), anchor="ma")
    return img, meta


def render(pages: int = 500, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    books = fetch_books()
    if not books:
        raise RuntimeError("no book texts downloaded")
    corpora = [(b, clean_text(b)) for b in books]
    corpora = [(b, c) for b, c in corpora if c]
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    recs = []
    for i in range(1, pages + 1):
        book, paras = corpora[(i - 1) % len(corpora)]
        img, meta = render_page(paras, rng, i)
        name = f"doc_{i:04d}.png"
        img.save(PAGES_DIR / name)
        recs.append({"file": name, "book": book.stem, "page": i, **meta})
        if i % 50 == 0:
            print(f"rendered {i}/{pages}", flush=True)
    with io.open(PAGES_DIR / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    recs = render(args.pages, args.seed)
    print(f"pages={len(recs)} -> {PAGES_DIR}")


if __name__ == "__main__":
    main()
