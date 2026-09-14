from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader, Subset

from ml.model import BlindHead, CompositeLoss, FourierUNet, LossConfig, MWNet, load_label_map
from ml.train.data import DescreenDataset

ROOT = Path(__file__).resolve().parents[2]

TRAIN_SUBSET = 2000
VAL_SUBSET = 200
PROXY_EPOCHS = 3
BLIND_CH = {"fourier": 96 * 4, "wavelet": 96 * 4}


def build(trial: optuna.Trial):
    arch = trial.suggest_categorical("arch", ["fourier", "wavelet"])
    bottleneck = trial.suggest_int("bottleneck", 2, 3)
    blind = trial.suggest_categorical("blind", [True, False])
    blind_w = trial.suggest_categorical("blind_w", [0.1, 0.3]) if blind else 0.0
    if arch == "fourier":
        model = FourierUNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=bottleneck,
                            dec_blocks=(1, 1), spectral=(False, True, True))
    else:
        model = MWNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=bottleneck,
                      dec_blocks=(1, 1), spectral=(False, True, True))
    head = BlindHead(BLIND_CH[arch]) if blind else None
    return model, head, blind_w


def objective(trial: optuna.Trial) -> float:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)
    model, head, blind_w = build(trial)
    trial.set_user_attr("params_M", round(model.param_count() / 1e6, 2))
    model.cuda()
    if head is not None:
        head.cuda()
    loss_fn = CompositeLoss(LossConfig(w_fourier=0.05, w_lpips=0.0)).cuda()
    ce = nn.CrossEntropyLoss()
    labels = load_label_map(ROOT / "Data" / "synth", "train")
    train_ds = Subset(DescreenDataset(ROOT / "Data" / "synth", "train"), range(TRAIN_SUBSET))
    val_ds = Subset(DescreenDataset(ROOT / "Data" / "synth", "val"), range(VAL_SUBSET))
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=True)
    params = list(model.parameters()) + (list(head.parameters()) if head else [])
    opt = torch.optim.Adam(params, lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROXY_EPOCHS * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    for epoch in range(PROXY_EPOCHS):
        model.train()
        for xs, ys, ids in train_dl:
            xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
            lb = torch.tensor([labels[i] for i in ids], device="cuda")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                if head is not None:
                    pred, feats = model(xs, return_features=True)
                else:
                    pred = model(xs)
            L = loss_fn(pred.float(), ys)
            total = L["total"]
            if head is not None:
                total = total + blind_w * ce(head(feats.float()), lb)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        ps, nv = 0.0, 0
        with torch.no_grad():
            for xs, ys, _ in val_dl:
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
    ap.add_argument("--trials", type=int, default=14)
    ap.add_argument("--db", default="ml/train/search.db")
    args = ap.parse_args()
    storage = f"sqlite:///{(ROOT / args.db).as_posix()}"
    study = optuna.create_study(
        study_name="freq-wavelet",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=11),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
    )
    if len(study.trials) == 0:
        study.enqueue_trial({"arch": "fourier", "bottleneck": 3, "blind": False, "blind_w": 0.1})
    study.optimize(objective, n_trials=args.trials)
    print("best:", round(study.best_value, 2), study.best_params)


if __name__ == "__main__":
    main()
