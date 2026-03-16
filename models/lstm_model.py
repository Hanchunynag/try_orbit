"""LSTM encoder-decoder baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoderDecoder(nn.Module):
    """Sequence model that conditions decoding on known future static covariates."""

    def __init__(
        self,
        history_dim: int,
        future_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        pred_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.encoder = nn.LSTM(
            input_size=history_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder_cell = nn.LSTMCell(input_size=output_dim + future_dim, hidden_size=hidden_dim)
        self.hidden_init = nn.Linear(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, history_x: torch.Tensor, future_cov: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(history_x)
        h_t = self.hidden_init(hidden[-1])
        c_t = cell[-1]
        batch_size = history_x.size(0)
        prev_output = torch.zeros(batch_size, self.output_dim, device=history_x.device, dtype=history_x.dtype)
        outputs = []
        for step in range(future_cov.size(1)):
            decoder_input = torch.cat([prev_output, future_cov[:, step, :]], dim=-1)
            h_t, c_t = self.decoder_cell(decoder_input, (h_t, c_t))
            prev_output = self.output_head(h_t)
            outputs.append(prev_output)
        return torch.stack(outputs, dim=1)
