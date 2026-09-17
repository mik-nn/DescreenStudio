from __future__ import annotations

import argparse
import io
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.model import CompositeLoss, FourierUNet, LossConfig
from ml.train.data import DescreenDataset


@dataclass(frozen=True)
class TrainConfig:
    data: str = "Data/synth"
    out: str = "ml/train/_run1"
    epochs: int = 10
    batch: int = 4
    lr: float = 2e-4
    seed: int = 42
    val_every: int = 2
    val_max: int = 1000
    num_workers: int = 2
    amp: bool = True
    arch_base: int = 64
    arch_levels: int = 3
    arch_bottleneck: int = 2
    arch_enc_blocks: int = 1
    arch_dec_blocks: int = 1
    arch_spectral_l0: bool = True
    fourier_w: float = 0.1
    lpips_w: float = 0.1
    edge_w: float = 0.0
    edge_alpha: float = 2.0
    tone_w: float = 0.0
    extra_dct: bool = False
    ema_decay: float = 0.999
    hard_boost: bool = False
    warm_start: str = ""


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def hard_weights(data_root: str | Path, ds: DescreenDataset, seed: int) -> torch.DoubleTensor:
    man = [json.loads(l) for l in open(Path(data_root) / "train" / "manifest.jsonl", encoding="utf-8")]
    by_id = {r["id"]: r for r in man}
    edge_path = Path(data_root) / "train" / "edge_q.json"
    edge = {}
    if edge_path.exists():
        edge = json.loads(edge_path.read_text(encoding="utf-8"))
        qs = sorted(edge.values())
        q75 = qs[int(0.75 * len(qs))]
    else:
        q75 = float("inf")
    w = []
    for p in ds.scans:
        r = by_id[p.stem]
        x = 1.0
        if r["preset"] == "newspaper":
            x *= 2.0
        if r["kind"] == "photo+headline":
            x *= 2.0
        if r["scan_dpi"] == 300:
            x *= 1.5
        if edge and edge.get(p.stem, 0.0) >= q75:
            x *= 2.0
        w.append(x)
    return torch.DoubleTensor(w)


def save_ckpt(path: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, best: float, ema: dict | None = None) -> None:
    torch.save(
        {"epoch": epoch, "best": best, "model": model.state_dict(), "opt": opt.state_dict(), "ema": ema},
        path,
    )


@torch.no_grad()
def apply_ema(model: torch.nn.Module, ema: dict, decay: float) -> None:
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            ema[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
        else:
            ema[k].copy_(v)


@torch.no_grad()
def load_ema(model: torch.nn.Module, ema: dict) -> dict:
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema)
    return backup


@torch.no_grad()
def restore_model(model: torch.nn.Module, backup: dict) -> None:
    model.load_state_dict(backup)


def train(cfg: TrainConfig) -> dict:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")
    train_ds = DescreenDataset(cfg.data, "train", extra_dct=cfg.extra_dct)
    val_ds = DescreenDataset(cfg.data, "val", extra_dct=cfg.extra_dct)
    if cfg.hard_boost:
        sampler = torch.utils.data.WeightedRandomSampler(
            hard_weights(cfg.data, train_ds, cfg.seed), len(train_ds), replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        train_dl = DataLoader(train_ds, batch_size=cfg.batch, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    model = FourierUNet(
        in_channels=5 if cfg.extra_dct else 3,
        base=cfg.arch_base,
        levels=cfg.arch_levels,
        enc_blocks=tuple([cfg.arch_enc_blocks] * (cfg.arch_levels - 1)),
        bottleneck_blocks=cfg.arch_bottleneck,
        dec_blocks=tuple([cfg.arch_dec_blocks] * (cfg.arch_levels - 1)),
        spectral=tuple([cfg.arch_spectral_l0] + [True] * (cfg.arch_levels - 1)),
    ).cuda()
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if cfg.warm_start:
        w = torch.load(cfg.warm_start, map_location="cpu", weights_only=False)
        model.load_state_dict(w["model"] if "model" in w else w)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"warm start from {cfg.warm_start}", flush=True)
    loss_fn = CompositeLoss(
        LossConfig(
            w_fourier=cfg.fourier_w,
            w_lpips=cfg.lpips_w,
            w_edge=cfg.edge_w,
            edge_alpha=cfg.edge_alpha,
            w_tone=cfg.tone_w,
        )
    ).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * len(train_dl))
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3).cuda()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    start_epoch, best = 0, -1.0
    last_ckpt = out / "last.pt"
    if last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ck.get("ema"):
            ema = {k: v.cuda() for k, v in ck["ema"].items()}
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed epoch={start_epoch} best={best:.2f}", flush=True)
    log = out / "log.jsonl"
    writer = SummaryWriter(str(out / "tb"))
    f = io.open(log, "a", encoding="utf-8")
    step = start_epoch * len(train_dl)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        tot, n = 0.0, 0
        for xs, ys, _ in train_dl:
            xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                pred = model(xs)
            L = loss_fn(pred.float(), ys)
            scaler.scale(L["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            apply_ema(model, ema, cfg.ema_decay)
            tot += float(L["total"].detach())
            n += 1
            writer.add_scalar("train/loss", float(L["total"].detach()), step)
            writer.add_scalar("train/l1", float(L["l1"]), step)
            writer.add_scalar("train/edge", float(L["edge"]), step)
            writer.add_scalar("train/tone", float(L["tone"]), step)
            step += 1
        rec = {"epoch": epoch, "train_loss": round(tot / max(n, 1), 4)}
        if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            backup = load_ema(model, {k: v.cuda() for k, v in ema.items()})
            ps, ss, nv = 0.0, 0.0, 0
            with torch.no_grad():
                for xs, ys, _ in val_dl:
                    if nv >= cfg.val_max:
                        break
                    xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=cfg.amp):
                        pr = model(xs)
                    pr = pr.float()
                    ps += psnr(pr, ys)
                    ss += float(ssim_fn(pr, ys))
                    nv += 1
                    if nv == 1:
                        grid = torch.cat([xs[:1], pr[:1], ys[:1]], dim=0)
                        writer.add_images("val/scan_pred_gt", grid, epoch)
            rec["val_psnr"] = round(ps / nv, 2)
            rec["val_ssim"] = round(ss / nv, 4)
            writer.add_scalar("val/psnr", rec["val_psnr"], epoch)
            writer.add_scalar("val/ssim", rec["val_ssim"], epoch)
            score = rec["val_psnr"]
            restore_model(model, backup)
            save_ckpt(last_ckpt, model, opt, epoch, best, ema)
            if score > best:
                best = score
                save_ckpt(out / "best.pt", model, opt, epoch, best, ema)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(rec, flush=True)
    f.close()
    writer.close()
    return {"best_psnr": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data/synth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="ml/train/_run1")
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--arch-base", type=int, default=64)
    ap.add_argument("--arch-levels", type=int, default=3)
    ap.add_argument("--arch-bottleneck", type=int, default=2)
    ap.add_argument("--arch-enc-blocks", type=int, default=1)
    ap.add_argument("--arch-dec-blocks", type=int, default=1)
    ap.add_argument("--arch-spectral-l0", type=int, default=1)
    ap.add_argument("--fourier-w", type=float, default=0.1)
    ap.add_argument("--lpips-w", type=float, default=0.1)
    ap.add_argument("--edge-w", type=float, default=0.0)
    ap.add_argument("--edge-alpha", type=float, default=2.0)
    ap.add_argument("--tone-w", type=float, default=0.0)
    ap.add_argument("--extra-dct", type=int, default=0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--hard-boost", type=int, default=0)
    ap.add_argument("--warm-start", type=str, default="")
    args = ap.parse_args()
    cfg = TrainConfig(
        data=args.data,
        seed=args.seed,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        out=args.out,
        val_every=args.val_every,
        arch_base=args.arch_base,
        arch_levels=args.arch_levels,
        arch_bottleneck=args.arch_bottleneck,
        arch_enc_blocks=args.arch_enc_blocks,
        arch_dec_blocks=args.arch_dec_blocks,
        arch_spectral_l0=bool(args.arch_spectral_l0),
        fourier_w=args.fourier_w,
        lpips_w=args.lpips_w,
        edge_w=args.edge_w,
        edge_alpha=args.edge_alpha,
        tone_w=args.tone_w,
        extra_dct=bool(args.extra_dct),
        ema_decay=args.ema_decay,
        hard_boost=bool(args.hard_boost),
        warm_start=args.warm_start,
    )
    print(train(cfg), flush=True)


if __name__ == "__main__":
    main()
