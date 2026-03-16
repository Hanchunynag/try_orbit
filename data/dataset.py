"""Datasets for sequence modeling."""

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


class NARXWindowDataset(Dataset):
    """Create tapped-delay windows for configurable-horizon NARX training."""

    def __init__(
        self,
        exogenous_inputs: np.ndarray,
        feedback_series: np.ndarray,
        targets: np.ndarray,
        input_lags: int,
        feedback_lags: int,
        prediction_steps: int = 1,
        stride: int = 1,
    ) -> None:
        if len(exogenous_inputs) != len(feedback_series) or len(exogenous_inputs) != len(targets):
            raise ValueError("All NARX input sequences must have the same length.")
        if input_lags <= 0 or feedback_lags <= 0:
            raise ValueError("NARX delays must be positive.")
        if prediction_steps <= 0:
            raise ValueError("Prediction steps must be positive.")
        if stride <= 0:
            raise ValueError("Stride must be positive.")

        self.exogenous_inputs = exogenous_inputs.astype(np.float32)
        self.feedback_series = feedback_series.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.input_lags = input_lags
        self.feedback_lags = feedback_lags
        self.prediction_steps = prediction_steps
        self.max_lag = max(input_lags, feedback_lags)
        max_start = len(self.targets) - self.prediction_steps + 1
        if max_start <= self.max_lag:
            raise ValueError("Sequence is too short for the chosen NARX delays and prediction steps.")
        self.starts = np.arange(self.max_lag, max_start, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step = int(self.starts[index])
        exog_slice = slice(step - self.input_lags, step)
        feedback_slice = slice(step - self.feedback_lags, step)
        target_index = step + self.prediction_steps - 1
        x_exog = torch.from_numpy(self.exogenous_inputs[exog_slice])
        x_feedback = torch.from_numpy(self.feedback_series[feedback_slice])
        y = torch.from_numpy(self.targets[target_index])
        return x_exog, x_feedback, y
