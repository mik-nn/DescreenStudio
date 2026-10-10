from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.model import CompositeLoss, FourierUNet, LossConfig
from ml.model.film import COND_DIM
from ml.train.data import DescreenDataset


@dataclass(frozen=True)
class TrainConfig:
    data: str = "Data/synth"
    data_extra: str = ""  # доп. корни через ";", смешиваются в train (run7, 2026-09-21)
    data_val: str = ""  # корень валидации; пусто = cfg.data
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
    arch_plain: bool = False  # чисто свёрточный UNet без FFC (гибридный трек 2.22)
    arch_spectral_mode: str = "plain"  # plain|res|norm|local (run5: norm)
    use_lpf: bool = False  # learned spectral prefilter (run5, спека 2026-09-20)
    fourier_w: float = 0.1
    lpips_w: float = 0.1
    edge_w: float = 0.0
    edge_alpha: float = 2.0
    tone_w: float = 0.0
    extra_dct: bool = False
    grad_accum: int = 1  # micro-batches per optimizer step (VRAM safety)
    ema_decay: float = 0.999
    hard_boost: bool = False
    warm_start: str = ""
    root_weights: str = ""  # баланс корней "synth:1,synth_run4:1,synth_1200:3"
    cond: bool = False  # FiLM-conditioning bottleneck на замер A0 (фаза 2)
    cond_drop: float = 0.1  # доля train-батчей с нулевым cond (robustness)
    film_lr_mult: float = 5.0  # множитель LR для film.* (иначе identity-init не сдвигается)
    l1_type: str = "l1"  # "l1" | "charbonnier"
    focal_gamma: float = 0.0  # focal-frequency вес (0 = обычный Fourier_Mag)
    grad_w: float = 0.0  # Sobel-градиент L1: привязка границ pred к GT (против извилистых контуров)
    aux_loss: float = 0.0  # deep supervision: вес aux-головы 1/2 (0 = выкл)
    ff_norm: bool = False
    ff_zero_init: bool = False
    guard_l1: float = 0.2
    guard_start: int = 1000
    guard_every: int = 500
    guard_patience: int = 2
    debug_trace: bool = False  # per-step трейс: loss-компоненты, grad-norm по группам, batch stems
    swa_tail: int = 0  # SWA: усреднить best+last в swa.pt + финальный val (0 = выкл)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def hard_weights(ds: DescreenDataset, seed: int, root_weights: dict[str, float] | None = None) -> torch.DoubleTensor:
    # per-root кэши манифестов: корень -> (by_id, edge, q75)
    caches: dict[Path, tuple] = {}
    for r in set(ds.roots):
        man = [json.loads(l) for l in open(Path(r) / "train" / "manifest.jsonl", encoding="utf-8")]
        edge_path = Path(r) / "train" / "edge_q.json"
        if edge_path.exists():
            edge = json.loads(edge_path.read_text(encoding="utf-8"))
            qs = sorted(edge.values())
            q75 = qs[int(0.75 * len(qs))]
        else:
            edge, q75 = {}, float("inf")
        caches[Path(r)] = ({row["id"]: row for row in man}, edge, q75)
    rw = root_weights or {}
    w = []
    for p, r in zip(ds.scans, ds.roots):
        by_id, edge, q75 = caches[Path(r)]
        row = by_id[p.stem]
        x = float(rw.get(Path(r).name, 1.0))
        if row["preset"] == "newspaper":
            x *= 2.0
        if row["kind"] == "photo+headline":
            x *= 2.0
        if row["scan_dpi"] == 300:
            x *= 1.5
        if edge and edge.get(p.stem, 0.0) >= q75:
            x *= 2.0
        w.append(x)
    return torch.DoubleTensor(w)


def save_ckpt(path: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, best: float,
              ema: dict | None = None, override: dict | None = None) -> None:
    torch.save(
        {"epoch": epoch, "best": best,
         "model": override if override is not None else model.state_dict(),
         "opt": opt.state_dict(), "ema": ema},
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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_lock(out: Path) -> None:
    """Замок от дабл-запусков (урок run9/run18: два процесса в один out портят
    логи, чекпоинты и делят GPU). Stale-замок мёртвого PID перезаписывается."""
    lock = out / "RUNNING"
    if lock.exists():
        try:
            old = int(lock.read_text(encoding="utf-8").strip().split()[0])
        except (ValueError, OSError):
            old = -1
        if old > 0 and _pid_alive(old):
            raise SystemExit(f"refusing to start: {out} locked by live PID {old}")
        print(f"stale lock (PID {old}), overwriting", flush=True)
    try:
        lock.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as e:
        print(f"lock write failed: {e}", flush=True)


def release_lock(out: Path) -> None:
    try:
        cur = (out / "RUNNING").read_text(encoding="utf-8").strip().split()[0]
        if cur == str(os.getpid()):
            (out / "RUNNING").unlink()
    except OSError:
        pass


def _num(x) -> float | str:
    try:
        v = float(x)
    except (ValueError, OverflowError):
        return "nonfinite"
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return round(v, 5)


def train(cfg: TrainConfig) -> dict:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    acquire_lock(out)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")
    train_roots = [cfg.data] + [r for r in cfg.data_extra.split(";") if r]
    train_ds = DescreenDataset(train_roots, "train", extra_dct=cfg.extra_dct,
                               cond=cfg.cond, cond_drop=cfg.cond_drop if cfg.cond else 0.0)
    val_ds = DescreenDataset(cfg.data_val or cfg.data, "val", extra_dct=cfg.extra_dct,
                             cond=cfg.cond, cond_drop=0.0)
    if cfg.hard_boost:
        rw: dict[str, float] = {}
        for item in cfg.root_weights.split(","):
            item = item.strip()
            if item and ":" in item:
                k, v = item.split(":", 1)
                rw[k.strip()] = float(v)
        sampler = torch.utils.data.WeightedRandomSampler(
            hard_weights(train_ds, cfg.seed, rw or None), len(train_ds), replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        train_dl = DataLoader(train_ds, batch_size=cfg.batch, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    # val с num_workers=0: на стыке эпох ловился дедлок воркеров DataLoader (run8)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    if cfg.use_lpf or cfg.arch_spectral_mode != "plain":
        from ml.model.archv2 import ArchV2

        model = ArchV2(
            use_lpf=cfg.use_lpf,
            use_edge_ch=False,
            spectral_mode=cfg.arch_spectral_mode,
            base=cfg.arch_base,
        )
    else:
        model = FourierUNet(
            in_channels=5 if cfg.extra_dct else 3,
            base=cfg.arch_base,
            levels=cfg.arch_levels,
            enc_blocks=tuple([cfg.arch_enc_blocks] * (cfg.arch_levels - 1)),
            bottleneck_blocks=cfg.arch_bottleneck,
            dec_blocks=tuple([cfg.arch_dec_blocks] * (cfg.arch_levels - 1)),
            spectral=(False,) * cfg.arch_levels if cfg.arch_plain else
            tuple([cfg.arch_spectral_l0] + [True] * (cfg.arch_levels - 1)),
            cond_dim=COND_DIM if cfg.cond else 0,
            aux=cfg.aux_loss > 0,
            ff_norm=cfg.ff_norm,
            ff_zero_init=cfg.ff_zero_init,
        )
    model = model.cuda()
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if cfg.warm_start:
        w = torch.load(cfg.warm_start, map_location="cpu", weights_only=False)
        sd = w["model"] if "model" in w else w
        try:
            model.load_state_dict(sd)
            print(f"warm start from {cfg.warm_start} (strict)", flush=True)
        except RuntimeError as e:
            # смена поколений (film <-> no-film): грузим пересечение ключей
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"warm start from {cfg.warm_start} (partial: missing={sorted(missing)} "
                  f"unexpected={len(unexpected)})", flush=True)
            _ = e
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss_fn = CompositeLoss(
        LossConfig(
            w_fourier=cfg.fourier_w,
            w_lpips=cfg.lpips_w,
            w_edge=cfg.edge_w,
            edge_alpha=cfg.edge_alpha,
            w_tone=cfg.tone_w,
            l1_type=cfg.l1_type,
            focal_gamma=cfg.focal_gamma,
            w_aux=cfg.aux_loss,
            w_grad=cfg.grad_w,
        )
    ).cuda()
    if cfg.cond:
        film_params = [p for n, p in model.named_parameters() if n.startswith("film.")]
        base_params = [p for n, p in model.named_parameters() if not n.startswith("film.")]
        opt = torch.optim.Adam(
            [{"params": base_params, "lr": cfg.lr},
             {"params": film_params, "lr": cfg.lr * cfg.film_lr_mult}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    opt_steps_per_epoch = max(1, len(train_dl) // max(1, cfg.grad_accum))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * opt_steps_per_epoch)
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
    trace_f = io.open(out / "trace.jsonl", "a", encoding="utf-8") if cfg.debug_trace else None
    step = start_epoch * len(train_dl)
    ga = max(1, cfg.grad_accum)
    # guardrail (2026-10-01/02, уроки run12/run14): скользящий контроль —
    # каждые guard_every шагов после guard_start считаем средний train/l1;
    # превышение порога guard_l1 подряд guard_patience раз = стоп.
    # run12 умирал к ~3000, run14 полз позже — одно окно 1000-1500 мало.
    guard_acc, guard_n, guard_bad = 0.0, 0, 0
    guard_cum, guard_cum_n = 0.0, 0
    guard_on = cfg.guard_l1 > 0 and step < 10**12
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        tot, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for batch in train_dl:
            xs, ys = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
            cc = batch[3].cuda(non_blocking=True) if cfg.cond else None
            # ВАЖНО: лосс считается ВНЕ autocast. MS-SSIM и FFT-члены в fp16
            # (внутри autocast) дают inf-градиенты на малых дисперсиях →
            # scaler схлопывается, обучение расходится (найдено 2026-10-09).
            _auxs = None
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                if cfg.aux_loss > 0:
                    pred, _auxs = model(xs, cc, return_aux=True)
                elif cfg.cond:
                    pred = model(xs, cc)
                else:
                    pred = model(xs)
            if cfg.aux_loss > 0:
                L = loss_fn(pred.float(), ys, aux=_auxs[0].float())
            else:
                L = loss_fn(pred.float(), ys)
            scaler.scale(L["total"] / ga).backward()
            tot += float(L["total"].detach())
            n += 1
            writer.add_scalar("train/loss", float(L["total"].detach()), step)
            writer.add_scalar("train/l1", float(L["l1"]), step)
            writer.add_scalar("train/edge", float(L["edge"]), step)
            writer.add_scalar("train/tone", float(L["tone"]), step)
            if "aux" in L:
                writer.add_scalar("train/aux", float(L["aux"]), step)
            step += 1
            if guard_on and step >= cfg.guard_start:
                guard_acc += float(L["l1"].detach())
                guard_n += 1
                if guard_n >= cfg.guard_every:
                    # Кумулятивное среднее (без сброса): осциллирующая смерть
                    # (run23: 0.53→0.18→0.46) оконными правилами не ловится —
                    # хорошее окно сбрасывало patience. Кумулята не врёт.
                    _gmean = guard_acc / guard_n
                    _cum_n = guard_cum_n + guard_n
                    _cum = (guard_cum + guard_acc) / _cum_n
                    guard_cum, guard_cum_n = guard_cum + guard_acc, _cum_n
                    guard_acc, guard_n = 0.0, 0
                    print(f"GUARD: window={_gmean:.4f} cumul={_cum:.4f} "
                          f"(n={_cum_n}) thr={cfg.guard_l1}", flush=True)
                    _fired = (_cum > cfg.guard_l1 and _cum_n >= 1500) \
                        or _gmean > cfg.guard_l1 * 4
                    if _fired:
                        print("GUARD-TRIGGERED — stopping", flush=True)
                        f.write(json.dumps({"guard": "triggered",
                                            "mean_l1": round(_cum, 4),
                                            "step": step}) + "\n")
                        f.flush()
                        save_ckpt(out / "guard.pt", model, opt, epoch, best, ema)
                        f.close()
                        writer.close()
                        release_lock(out)
                        return {"best_psnr": best, "guard": "triggered"}
            if n % ga == 0:
                scaler.unscale_(opt)
                if cfg.debug_trace and step <= 200:
                    _nf = [nm for nm, pp in model.named_parameters()
                           if pp.grad is not None and not torch.isfinite(pp.grad).all()]
                    if _nf or step % 10 == 0:
                        _gn = sum(float(pp.grad.float().pow(2).sum())
                                  for _, pp in model.named_parameters()
                                  if pp.grad is not None) ** 0.5
                        _st = batch[2]
                        _st = list(_st) if isinstance(_st, (list, tuple)) else [_st]
                        print(f"SCAN step={step} scale={scaler.get_scale():.1f} "
                              f"loss={float(L['total']):.4f} gnorm={_gn:.3f} "
                              f"nonfin={len(_nf)} {_nf[:5]} stems={_st}", flush=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Шаг планировщика только если оптимизатор реально шагнул:
                # при inf/nan-градиентах scaler пропускает step, и sched.step()
                # без этого условия убегает вперёд (отсюда и UserWarning).
                _prev_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= _prev_scale:
                    sched.step()
                apply_ema(model, ema, cfg.ema_decay)
                opt.zero_grad(set_to_none=True)
            if cfg.debug_trace and step % 50 == 0 and trace_f is not None:
                _tr: dict = {"step": step, "lr": opt.param_groups[0]["lr"],
                             "scale": float(scaler.get_scale())}
                for _k, _v in L.items():
                    _tr["loss_" + _k] = _num(_v.detach())
                _gn = {"spectral": 0.0, "fuse_out": 0.0, "rest": 0.0}
                for _n, _p in model.named_parameters():
                    if _p.grad is None:
                        continue
                    _g = float(_p.grad.detach().float().pow(2).sum().item())
                    if ".spectral." in _n:
                        _gn["spectral"] += _g
                    elif ".fuse." in _n or _n.startswith("outc"):
                        _gn["fuse_out"] += _g
                    else:
                        _gn["rest"] += _g
                import math as _math
                _tr["gn"] = {k: round(_math.sqrt(v), 4) if v == v and v < float("inf") else "big"
                             for k, v in _gn.items()}
                _pf = pred.detach().float()
                _tr["pred"] = [_num(_pf.min()), _num(_pf.max()), _num(_pf.mean())]
                _tr["tgt"] = [_num(ys.min()), _num(ys.max())]
                _st = batch[2]
                _tr["stems"] = list(_st) if isinstance(_st, (list, tuple)) else [_st]
                trace_f.write(json.dumps(_tr) + "\n")
                trace_f.flush()
            if step % 500 == 0:  # аварийный чекпоинт (не используется для авто-resume)
                save_ckpt(out / "step.pt", model, opt, epoch, best, ema)
        if n % ga != 0:  # trailing partial accumulation
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            _prev_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= _prev_scale:
                sched.step()
            apply_ema(model, ema, cfg.ema_decay)
            opt.zero_grad(set_to_none=True)
        # чекпоинт ВЕЖЕ валидации: при повторном дедлоке веса эпохи уже на диске
        save_ckpt(last_ckpt, model, opt, epoch, best, ema)
        rec = {"epoch": epoch, "train_loss": round(tot / max(n, 1), 4)}
        if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            backup = load_ema(model, {k: v.cuda() for k, v in ema.items()})
            # tone-PSNR (Найквист): честная метрика на 2 px/ячейку; пока только
            # трекинг (отбор best — по пиксельному val_psnr, как раньше)
            tone_map: dict[str, float] = {}
            try:
                _man = Path(cfg.data_val or cfg.data) / "val" / "manifest.jsonl"
                if _man.exists():
                    for _line in open(_man, encoding="utf-8"):
                        _r = json.loads(_line)
                        tone_map[_r["id"]] = 2.0 * _r["lpi"] / _r["scan_dpi"]
            except OSError:
                pass
            ps, ss, nv = 0.0, 0.0, 0
            tps, tnv = 0.0, 0
            with torch.no_grad():
                for batch in val_dl:
                    if nv >= cfg.val_max:
                        break
                    xs, ys = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
                    cc = batch[3].cuda(non_blocking=True) if cfg.cond else None
                    with torch.amp.autocast("cuda", enabled=cfg.amp):
                        pr = model(xs, cc) if cfg.cond else model(xs)
                    pr = pr.float()
                    ps += psnr(pr, ys)
                    ss += float(ssim_fn(pr, ys))
                    nv += 1
                    _stems = batch[2]
                    _stem = _stems[0] if isinstance(_stems, (list, tuple)) else _stems
                    _f = tone_map.get(_stem, 0.0)
                    if _f > 0:
                        _th = max(8, round(pr.shape[2] * _f))
                        _tw = max(8, round(pr.shape[3] * _f))
                        _pt = torch.nn.functional.interpolate(
                            pr, size=(_th, _tw), mode="bicubic",
                            align_corners=False, antialias=True)
                        _gt = torch.nn.functional.interpolate(
                            ys.float(), size=(_th, _tw), mode="bicubic",
                            align_corners=False, antialias=True)
                        tps += psnr(_pt, _gt)
                        tnv += 1
                    if nv % 100 == 0:
                        print(f"val {nv}/{cfg.val_max} psnr={ps/nv:.2f}", flush=True)
                    if nv == 1:
                        grid_in = xs[:1, :3] if xs.shape[1] > 3 else xs[:1]
                        grid = torch.cat([grid_in, pr[:1], ys[:1]], dim=0)
                        writer.add_images("val/scan_pred_gt", grid, epoch)
            rec["val_psnr"] = round(ps / nv, 2)
            rec["val_ssim"] = round(ss / nv, 4)
            writer.add_scalar("val/psnr", rec["val_psnr"], epoch)
            writer.add_scalar("val/ssim", rec["val_ssim"], epoch)
            if tnv > 0:
                rec["val_tone_psnr"] = round(tps / tnv, 2)
                writer.add_scalar("val/tone_psnr", rec["val_tone_psnr"], epoch)
            score = rec["val_psnr"]
            restore_model(model, backup)
            if score > best:
                best = score
                save_ckpt(out / "best.pt", model, opt, epoch, best, ema)
        save_ckpt(last_ckpt, model, opt, epoch, best, ema)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(rec, flush=True)
    # SWA-хвост (2026-10-01): среднее best+last ставит веса в центр плоской
    # области — защита от острого минимума/обрыва (run12: сдвиг 0.023 = смерть)
    if cfg.swa_tail > 0:
        try:
            _b = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["model"]
            _l = torch.load(last_ckpt, map_location="cpu", weights_only=False)["model"]
            _keys = [k for k in _l if k in _b]
            _swa = {k: (0.5 * _b[k].float() + 0.5 * _l[k].float()).to(_l[k].dtype)
                    for k in _keys}
            save_ckpt(out / "swa.pt", model, opt, cfg.epochs - 1, best, ema, override=_swa)
            model.load_state_dict(_swa)
            model.eval()
            ps, nv = 0.0, 0
            with torch.no_grad():
                for batch in val_dl:
                    if nv >= cfg.val_max:
                        break
                    xs, ys = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
                    cc = batch[3].cuda(non_blocking=True) if cfg.cond else None
                    with torch.amp.autocast("cuda", enabled=cfg.amp):
                        pr = model(xs, cc) if cfg.cond else model(xs)
                    ps += psnr(pr.float(), ys)
                    nv += 1
            swa_psnr = round(ps / max(nv, 1), 2)
            print({"swa_psnr": swa_psnr, "best_psnr": best}, flush=True)
            f = io.open(log, "a", encoding="utf-8")
            f.write(json.dumps({"swa": True, "swa_psnr": swa_psnr, "best_psnr": best}) + "\n")
            f.flush()
            f.close()
        except (OSError, KeyError, RuntimeError) as e:
            print(f"swa skipped: {e}", flush=True)
    f.close()
    writer.close()
    release_lock(out)
    return {"best_psnr": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data/synth")
    ap.add_argument("--data-extra", default="")
    ap.add_argument("--data-val", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="ml/train/_run1")
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--arch-base", type=int, default=64)
    ap.add_argument("--arch-levels", type=int, default=3)
    ap.add_argument("--arch-bottleneck", type=int, default=2)
    ap.add_argument("--arch-enc-blocks", type=int, default=1)
    ap.add_argument("--arch-dec-blocks", type=int, default=1)
    ap.add_argument("--arch-spectral-l0", type=int, default=1)
    ap.add_argument("--arch-plain", type=int, default=0,
                    help="чисто свёрточный UNet без FFC (spectral=(F,F,F))")
    ap.add_argument("--arch-spectral-mode", type=str, default="plain",
                    choices=["plain", "res", "norm", "local"])
    ap.add_argument("--use-lpf", type=int, default=0)
    ap.add_argument("--fourier-w", type=float, default=0.1)
    ap.add_argument("--lpips-w", type=float, default=0.1)
    ap.add_argument("--edge-w", type=float, default=0.0)
    ap.add_argument("--edge-alpha", type=float, default=2.0)
    ap.add_argument("--tone-w", type=float, default=0.0)
    ap.add_argument("--extra-dct", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--hard-boost", type=int, default=0)
    ap.add_argument("--root-weights", type=str, default="")
    ap.add_argument("--warm-start", type=str, default="")
    ap.add_argument("--cond", type=int, default=0, help="FiLM-conditioning на A0 (фаза 2)")
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--film-lr-mult", type=float, default=5.0)
    ap.add_argument("--l1-type", type=str, default="l1", choices=["l1", "charbonnier"])
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--grad-w", type=float, default=0.0,
                    help="вес Sobel-градиентного лосса (привязка границ к GT)")
    ap.add_argument("--aux-loss", type=float, default=0.0,
                    help="deep supervision: вес aux-головы 1/2 разрешения")
    ap.add_argument("--ff-norm", type=int, default=0,
                    help="LayerNorm на выходе spectral-ветвей (фикс pile-up)")
    ap.add_argument("--ff-zero-init", type=int, default=0,
                    help="zero-init последних conv spectral-ветви (отдельно от нормы)")
    ap.add_argument("--guard-l1", type=float, default=0.2,
                    help="guardrail: автостоп если средний train/l1 выше (0=выкл)")
    ap.add_argument("--guard-start", type=int, default=1000)
    ap.add_argument("--guard-every", type=int, default=500)
    ap.add_argument("--guard-patience", type=int, default=2)
    ap.add_argument("--debug-trace", type=int, default=0)
    ap.add_argument("--swa-tail", type=int, default=0,
                    help="SWA: усреднить best+last в swa.pt + финальный val (0=выкл)")
    args = ap.parse_args()
    cfg = TrainConfig(
        data=args.data,
        data_extra=args.data_extra,
        data_val=args.data_val,
        seed=args.seed,
        epochs=args.epochs,
        batch=args.batch,
        amp=bool(args.amp),
        num_workers=args.num_workers,
        lr=args.lr,
        out=args.out,
        val_every=args.val_every,
        arch_base=args.arch_base,
        arch_levels=args.arch_levels,
        arch_bottleneck=args.arch_bottleneck,
        arch_enc_blocks=args.arch_enc_blocks,
        arch_dec_blocks=args.arch_dec_blocks,
        arch_spectral_l0=bool(args.arch_spectral_l0),
        arch_plain=bool(args.arch_plain),
        arch_spectral_mode=args.arch_spectral_mode,
        use_lpf=bool(args.use_lpf),
        fourier_w=args.fourier_w,
        lpips_w=args.lpips_w,
        edge_w=args.edge_w,
        edge_alpha=args.edge_alpha,
        tone_w=args.tone_w,
        extra_dct=bool(args.extra_dct),
        grad_accum=max(1, args.grad_accum),
        ema_decay=args.ema_decay,
        hard_boost=bool(args.hard_boost),
        root_weights=args.root_weights,
        warm_start=args.warm_start,
        cond=bool(args.cond),
        cond_drop=args.cond_drop,
        film_lr_mult=args.film_lr_mult,
        l1_type=args.l1_type,
        focal_gamma=args.focal_gamma,
        grad_w=args.grad_w,
        aux_loss=args.aux_loss,
        ff_norm=bool(args.ff_norm),
        ff_zero_init=bool(args.ff_zero_init),
        guard_l1=args.guard_l1,
        guard_start=args.guard_start,
        guard_every=args.guard_every,
        guard_patience=args.guard_patience,
        debug_trace=bool(args.debug_trace),
        swa_tail=args.swa_tail,
    )
    print(train(cfg), flush=True)


if __name__ == "__main__":
    main()
