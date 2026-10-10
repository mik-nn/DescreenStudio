"""FiLM-conditioning на замер растра A0 (фаза 2, трек run9).

Идея: сеть не узнаёт редкие растры (Offset1: net_raw добавляет энергию
в кольце 110->148, conf=0.59). Подаём замер A0 (r, угол, conf) как вектор
в bottleneck через FiLM (scale/shift поканально).

Совместимость:
- zero-init последнего слоя MLP -> при нулевом cond преобразование
  тождественное (gamma=1, beta=0): warm start от run7 валиден, старые
  веса грузятся строго, новые ключи film.* — с нуля;
- cond=None эквивалентен нулевому cond (поведение run7 бит-в-бит);
- экспорт: дополнительный вход [B, cond_dim] (Candle-порт — отдельно).
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: нормировка r (максимум обучающего конверта с запасом: fine 190/300=0.633)
COND_R_MAX = 0.65
COND_DIM = 4  # [r_norm, conf, cos(ang), sin(ang)]


def build_cond_vec(r: float, angle_deg: float, conf: float) -> list[float]:
    """Вектор кондиционирования из замера A0 (numpy-сторона, детерминировано)."""
    import math

    t = math.radians(angle_deg or 0.0)
    return [
        min(max(r or 0.0, 0.0) / COND_R_MAX, 1.5),
        min(max(conf or 0.0, 0.0), 1.0),
        math.cos(t),
        math.sin(t),
    ]


class FiLMCondition(nn.Module):
    """MLP cond -> (gamma, beta) для модуляции bottleneck-каналов."""

    def __init__(self, cond_dim: int = COND_DIM, channels: int = 384, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )
        # identity-init: нулевые gamma/beta -> x*(1+0)+0 = x
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.mlp(cond.to(x.dtype))
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma)[..., None, None] + beta[..., None, None]
