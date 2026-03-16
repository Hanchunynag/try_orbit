"""Paper-style NARX forecaster."""

from __future__ import annotations

import torch
import torch.nn as nn


class Snake(nn.Module):
    """Snake activation from the paper's Table 1 search space."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.as_tensor(self.alpha, device=x.device, dtype=x.dtype)
        return x + torch.sin(alpha * x).pow(2) / torch.clamp(alpha, min=torch.finfo(x.dtype).eps)


def _build_activation(name: str, snake_alpha: float) -> nn.Module:
    if name == "linear":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "snake":
        return Snake(alpha=snake_alpha)
    raise ValueError(f"Unsupported NARX activation: {name}")


class NARXForecaster(nn.Module):
    """Predict the next residual state from tapped-delay input and feedback histories."""

    def __init__(
        self,
        exogenous_dim: int,
        feedback_dim: int,
        output_dim: int,
        input_lags: int,
        feedback_lags: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str,
        dropout: float,
        snake_alpha: float,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("NARXForecaster requires at least one hidden layer.")

        input_dim = exogenous_dim * input_lags + feedback_dim * feedback_lags
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_build_activation(activation, snake_alpha))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, exogenous_history: torch.Tensor, feedback_history: torch.Tensor) -> torch.Tensor:
        batch_size = exogenous_history.size(0)
        features = torch.cat(
            [
                exogenous_history.reshape(batch_size, -1),
                feedback_history.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        return self.network(features)
