"""Loss functions for multi-step residual forecasting."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def composite_sequence_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    lambda_pred: float,
    lambda_diff: float,
    lambda_smooth: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the combined prediction, first-difference, and smoothness loss."""
    loss_pred = F.mse_loss(y_pred, y_true)

    if y_pred.size(1) >= 2:
        diff_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        diff_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        loss_diff = F.mse_loss(diff_pred, diff_true)
    else:
        loss_diff = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

    if y_pred.size(1) >= 3:
        second_diff = y_pred[:, 2:, :] - 2.0 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]
        loss_smooth = torch.mean(second_diff ** 2)
    else:
        loss_smooth = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

    total_loss = lambda_pred * loss_pred + lambda_diff * loss_diff + lambda_smooth * loss_smooth
    parts = {
        "loss": float(total_loss.detach().cpu()),
        "pred": float(loss_pred.detach().cpu()),
        "diff": float(loss_diff.detach().cpu()),
        "smooth": float(loss_smooth.detach().cpu()),
    }
    return total_loss, parts


def step_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one-step MSE for the NARX predictor."""
    loss = F.mse_loss(y_pred, y_true)
    zero = 0.0
    parts = {
        "loss": float(loss.detach().cpu()),
        "pred": float(loss.detach().cpu()),
        "diff": zero,
        "smooth": zero,
    }
    return loss, parts
