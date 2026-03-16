"""Paper-style NARX forecaster."""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
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
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("NARXForecaster requires at least one hidden layer.")

        input_dim = exogenous_dim * input_lags + feedback_dim * feedback_lags
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_build_activation(activation))
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
