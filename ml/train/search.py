from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import optuna
import torch
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader, Subset

from ml.model import CompositeLoss, FourierUNet, LossConfig
from ml.train.data import DescreenDataset

ROOT = Path(__file__).resolve().parents[2]

TRAIN_SUBSET = 2000
VAL_SUBSET = 200
PROXY_EPOCHS = 3


def build_model(trial: optuna.Trial) -> tuple[FourierUNet, LossConfig, float]:
    base = trial.suggest_categorical("base", [48, 64, 96])
    levels = trial.suggest_categorical("levels", [2, 3])
    bottleneck = trial.suggest_int("bottleneck", 1, 3)
    spectral_l0 = trial.suggest_categorical("spectral_l0", [True, False])
    n = levels - 1
    spectral = tuple([spectral_l0] + [True] * (levels - 1))
    model = FourierUNet(
        base=base,
        levels=levels,
        enc_blocks=tuple([1] * n),
        bottleneck_blocks=bottleneck,
        dec_blocks=tuple([1] * n),
        spectral=spectral,
    )
    loss = LossConfig(
        w_fourier=trial.suggest_categorical("fourier_w", [0.05, 0.1, 0.2]),
        w_lpips=trial.suggest_categorical("lpips_w", [0.0, 0.05, 0.1]),
    )
    lr = trial.suggest_categorical("lr", [1e-4, 2e-4])
    return model, loss, lr


def objective(trial: optuna.Trial) -> float:
    import random

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)
    model, lcfg, lr = build_model(trial)
    trial.set_user_attr("params_M", round(model.param_count() / 1e6, 2))
    model.cuda()
    loss_fn = CompositeLoss(lcfg).cuda()
    train_ds = Subset(DescreenDataset(ROOT / "Data" / "synth", "train"), range(TRAIN_SUBSET))
    val_ds = Subset(DescreenDataset(ROOT / "Data" / "synth", "val"), range(VAL_SUBSET))
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROXY_EPOCHS * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3).cuda()
    for epoch in range(PROXY_EPOCHS):
        model.train()
        for xs, ys in train_dl:
            xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                pred = model(xs)
            L = loss_fn(pred.float(), ys)
            scaler.scale(L["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        ps, nv = 0.0, 0
        with torch.no_grad():
            for xs, ys in val_dl:
                xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
                with torch.amp.autocast("cuda"):
                    pr = model(xs).float()
                mse = torch.mean((pr - ys) ** 2).item()
                ps += 10.0 * math.log10(1.0 / max(mse, 1e-12))
                nv += 1
        score = ps / nv
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=36)
    ap.add_argument("--db", default="ml/train/search.db")
    args = ap.parse_args()
    storage = f"sqlite:///{(ROOT / args.db).as_posix()}"
    study = optuna.create_study(
        study_name="fourier-arch",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=7),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6),
    )
    study.optimize(objective, n_trials=args.trials)
    print("best:", round(study.best_value, 2), study.best_params)


if __name__ == "__main__":
    main()
