#!/usr/bin/env python
"""Экспорт значений триалов Optuna в TensorBoard (кривая поиска).

Читает sqlite-стадию search_run7 и пишет scalars в ml/train/_optuna/tb/
(logdir TensorBoard = ml/train, подкаталог = отдельный «run»):
  optuna/trial_value — значение каждого COMPLETE триала (step = номер триала)
  optuna/best_value  — лучшее на момент триала

Идемпотентен: перезапись того же event-файла через SummaryWriter
(добавляет новые точки; старые не дублируются при том же step).

Использование:
    python ml/benchmark/export_optuna_tb.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import optuna  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

DB = ROOT / "ml" / "train" / "search_run7.db"
OUT = ROOT / "ml" / "train" / "_optuna" / "tb"


def main() -> None:
    study = optuna.load_study(study_name="run7_losslr", storage=f"sqlite:///{DB}")
    OUT.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(OUT))
    best = float("-inf")
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        v = float(t.value)
        best = max(best, v)
        writer.add_scalar("optuna/trial_value", v, t.number)
        writer.add_scalar("optuna/best_value", best, t.number)
    writer.close()
    n = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"exported {n} complete trials -> {OUT}")


if __name__ == "__main__":
    main()
