"""Sliding-window datasets for sequence modeling."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceWindowDataset(Dataset):
    """Create sequence-to-sequence windows with history and known future covariates."""

    def __init__(
        self,
        history_features: np.ndarray,
        future_covariates: np.ndarray,
        targets: np.ndarray,
        input_len: int,
        pred_len: int,
        stride: int = 1,
    ) -> None:
        if len(history_features) != len(future_covariates) or len(history_features) != len(targets):
            raise ValueError("All input sequences must have the same length.")
        max_start = len(history_features) - input_len - pred_len + 1
        if max_start <= 0:
            raise ValueError("Sequence is too short for the chosen input_len/pred_len.")
        self.history_features = history_features.astype(np.float32)
        self.future_covariates = future_covariates.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.input_len = input_len
        self.pred_len = pred_len
        self.starts = np.arange(0, max_start, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        hist_slice = slice(start, start + self.input_len)
        pred_slice = slice(start + self.input_len, start + self.input_len + self.pred_len)
        x_hist = torch.from_numpy(self.history_features[hist_slice])
        x_future = torch.from_numpy(self.future_covariates[pred_slice])
        y = torch.from_numpy(self.targets[pred_slice])
        return x_hist, x_future, y
