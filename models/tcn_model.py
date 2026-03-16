"""Temporal convolutional network forecaster."""

from __future__ import annotations

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Trim right-side padding to preserve causality."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Causal residual block used by the TCN encoder."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TemporalConvNet(nn.Module):
    """Stack of dilated causal convolutions."""

    def __init__(self, num_inputs: int, channels: list[int], kernel_size: int, dropout: float) -> None:
        super().__init__()
        layers = []
        for idx, out_channels in enumerate(channels):
            in_channels = num_inputs if idx == 0 else channels[idx - 1]
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** idx,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNForecaster(nn.Module):
    """Encode history with a causal TCN and decode using known future SGP4 covariates."""

    def __init__(
        self,
        history_dim: int,
        future_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        pred_len: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        channels = [hidden_dim] * num_layers
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.history_proj = nn.Linear(history_dim, hidden_dim)
        self.encoder = TemporalConvNet(hidden_dim, channels, kernel_size=kernel_size, dropout=dropout)
        self.future_proj = nn.Linear(future_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, history_x: torch.Tensor, future_cov: torch.Tensor) -> torch.Tensor:
        hist = self.history_proj(history_x).transpose(1, 2)
        encoded = self.encoder(hist)
        context = encoded[:, :, -1].unsqueeze(1).expand(-1, future_cov.size(1), -1)
        future_emb = self.future_proj(future_cov)
        output = self.head(torch.cat([context, future_emb], dim=-1))
        return output
