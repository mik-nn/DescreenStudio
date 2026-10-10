"""Optuna вокруг зафиксированного run3-скелета (рычаг 2, 2026-09-21).

Отличие от search.py (который искал архитектуру): здесь скелет ЗАМОРОЖЕН
(base 96, levels 3, blocks 1-3-1, spectral_l0=False — оптимум прошлого поиска),
ищем только учебный режим: lr и веса лосса (fourier/lpips/edge/tone).

Протокол-прокси: warm start от run3/best.pt, дообучение на смешанном подмножестве
(synth + synth_run4), оценка на ОБОИХ вал-сетах. Score = минимум из двух PSNR
(guard: нельзя просесть на старом распределении ради P1).

Урок run5/6: прокси врут на архитектуру, но для весов лосса/lr при warm start
переносимость приемлемая — улучшение дообучения обычно держится.

Запуск (после run7, GPU свободен):
    python ml/train/search_run7.py --trials 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, Subset

from ml.model import CompositeLoss, FourierUNet, LossConfig
from ml.train.data import DescreenDataset

ROOT = Path(__file__).resolve().parents[2]

TRAIN_SUBSET = 4000   # по 2000 из каждого корня, стратификация — чередование
VAL_N = 300
PROXY_STEPS = 1000    # ~1/4 эпохи смеси 20k при batch 4


def build_model() -> FourierUNet:
    model = FourierUNet(
        in_channels=3, base=96, levels=3,
        enc_blocks=(1, 1), bottleneck_blocks=3, dec_blocks=(1, 1),
        spectral=(False, True, True),
    )
    ck = torch.load(ROOT / "ml/train/_run3/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    return model


@torch.no_grad()
def val_psnr(model, ds_root: str, n: int, device: str = "cuda") -> float:
    ds = DescreenDataset(ROOT / ds_root, "val")
    ps = []
    for i in range(min(n, len(ds))):
        x, y, _ = ds[i]
        with torch.amp.autocast("cuda", enabled=False):
            pred = model(x.unsqueeze(0).float().to(device))
        mse = torch.mean((pred[0].cpu() - y) ** 2).item()
        ps.append(10.0 * float(np.log10(1.0 / max(mse, 1e-12))))
    return float(np.mean(ps))


def objective(trial: optuna.Trial) -> float:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    lr = trial.suggest_categorical("lr", [5e-6, 1e-5, 2e-5, 4e-5])
    lcfg = LossConfig(
        w_fourier=trial.suggest_categorical("fourier_w", [0.02, 0.05, 0.1, 0.2]),
        w_lpips=trial.suggest_categorical("lpips_w", [0.0, 0.02, 0.05]),
        w_edge=trial.suggest_categorical("edge_w", [0.0, 0.05, 0.1]),
        w_tone=trial.suggest_categorical("tone_w", [0.0, 0.02]),
    )

    model = build_model().cuda()
    loss_fn = CompositeLoss(lcfg).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # смешанное подмножество: чередуем корни, чтобы батчи были 50/50
    roots = ["Data/synth", "Data/synth_run4"]
    dss = [DescreenDataset(ROOT / r, "train") for r in roots]
    per = TRAIN_SUBSET // len(roots)
    idx = []
    for k, ds in enumerate(dss):
        idx += [(k, i) for i in range(per)]
    random.shuffle(idx)

    model.train()
    step = 0
    device = "cuda"
    while step < PROXY_STEPS:
        for k, i in idx:
            x, y, _ = dss[k][i]
            x = x.unsqueeze(0).float().to(device)
            y = y.unsqueeze(0).float().to(device)
            with torch.amp.autocast("cuda", enabled=True):
                pred = model(x)
            # Лосс вне autocast (как train.py): LPIPS/MS-SSIM в fp16 дают NaN.
            loss = loss_fn(pred.float(), y)["total"]
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step >= PROXY_STEPS:
                break

    model.eval()
    psnr_a = val_psnr(model, "Data/synth", VAL_N, device)
    psnr_b = val_psnr(model, "Data/synth_run4", VAL_N, device)
    score = min(psnr_a, psnr_b)
    trial.set_user_attr("psnr_synth", round(psnr_a, 2))
    trial.set_user_attr("psnr_synth_run4", round(psnr_b, 2))
    print(f"trial {trial.number}: lr={lr} loss={lcfg} synth={psnr_a:.2f} "
          f"p1={psnr_b:.2f} score={score:.2f}", flush=True)
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--study", default="run7_losslr")
    args = ap.parse_args()

    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{ROOT}/ml/train/search_run7.db",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials)
    print("best:", study.best_params, "score:", study.best_value)
    out = ROOT / "ml/train/_out/search_run7_best.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": study.best_params, "score": study.best_value,
                               "trials": len(study.trials)}, indent=1), encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
