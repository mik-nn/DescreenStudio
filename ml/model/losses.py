from __future__ import annotations

from dataclasses import dataclass

import lpips
import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM


@dataclass(frozen=True)
class LossConfig:
    w_l1: float = 1.0
    w_msssim: float = 0.5
    w_fourier: float = 0.1
    w_lpips: float = 0.1
    w_edge: float = 0.0  # edge-weighted L1 (run4: residual screen lives on edges)
    edge_alpha: float = 2.0  # weight = 1 + alpha * |grad_target| / mean(|grad|)
    w_tone: float = 0.0  # per-channel mean preservation (run4: fixes +0.10 brightening)


def _grad_mag(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 2:] - x[:, :, :, :-2]
    dy = x[:, :, 2:, :] - x[:, :, :-2, :]
    g = torch.sqrt(dx[:, :, 1:-1, :] ** 2 + dy[:, :, :, 1:-1] ** 2 + 1e-12)
    pad = torch.nn.functional.pad(g, (1, 1, 1, 1), mode="replicate")
    return pad.mean(dim=1, keepdim=True)


class CompositeLoss(nn.Module):
    def __init__(self, cfg: LossConfig | None = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.l1 = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = lpips.LPIPS(net="alex")
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        loss_l1 = self.l1(pred, target)
        loss_ms = 1.0 - self.ms_ssim(pred, target)
        fp = torch.abs(torch.fft.rfft2(pred, norm="ortho"))
        ft = torch.abs(torch.fft.rfft2(target, norm="ortho"))
        loss_fourier = torch.mean(torch.abs(fp - ft) / (ft.mean() + 1e-6))
        if self.cfg.w_lpips > 0:
            loss_lpips = self.lpips(2.0 * pred - 1.0, 2.0 * target - 1.0).mean()
        else:
            loss_lpips = torch.zeros((), device=pred.device)
        if self.cfg.w_edge > 0:
            g = _grad_mag(target.detach())
            w = 1.0 + self.cfg.edge_alpha * g / (g.mean() + 1e-6)
            loss_edge = (w * torch.abs(pred - target)).mean() / w.mean()
        else:
            loss_edge = torch.zeros((), device=pred.device)
        if self.cfg.w_tone > 0:
            loss_tone = torch.mean(
                torch.abs(pred.mean(dim=(0, 2, 3)) - target.mean(dim=(0, 2, 3)))
            )
        else:
            loss_tone = torch.zeros((), device=pred.device)
        total = (
            self.cfg.w_l1 * loss_l1
            + self.cfg.w_msssim * loss_ms
            + self.cfg.w_fourier * loss_fourier
            + self.cfg.w_lpips * loss_lpips
            + self.cfg.w_edge * loss_edge
            + self.cfg.w_tone * loss_tone
        )
        return {
            "total": total,
            "l1": loss_l1.detach(),
            "ms_ssim": loss_ms.detach(),
            "fourier": loss_fourier.detach(),
            "lpips": loss_lpips.detach(),
            "edge": loss_edge.detach(),
            "tone": loss_tone.detach(),
        }
