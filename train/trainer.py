"""Model construction, training, and rollout utilities."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.lstm_model import LSTMEncoderDecoder
from models.tcn_model import TCNForecaster
from train.losses import composite_sequence_loss


@dataclass
class TrainResult:
    """Artifacts returned by the trainer."""

    model: torch.nn.Module
    device: str
    history: dict[str, list[float]]
    best_model_path: str


def build_model(config, history_dim: int, future_dim: int, output_dim: int) -> torch.nn.Module:
    """Instantiate the selected PyTorch model."""
    if config.model_name == "tcn":
        return TCNForecaster(
            history_dim=history_dim,
            future_dim=future_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            pred_len=config.pred_len,
            kernel_size=config.tcn_kernel_size,
            dropout=config.dropout,
        )
    if config.model_name == "lstm":
        return LSTMEncoderDecoder(
            history_dim=history_dim,
            future_dim=future_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            pred_len=config.pred_len,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model_name: {config.model_name}")


def build_dataloaders(dataset, config) -> tuple[DataLoader, DataLoader]:
    """Split the window dataset into sequential train/validation subsets."""
    num_samples = len(dataset)
    num_train = max(int(num_samples * (1.0 - config.val_ratio)), 1)
    if num_train >= num_samples:
        num_train = max(num_samples - 1, 1)
    train_indices = list(range(0, num_train))
    val_indices = list(range(num_train, num_samples))
    if not val_indices:
        val_indices = [train_indices[-1]]
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def run_epoch(model, loader, optimizer, device, config, train_mode: bool) -> dict[str, float]:
    """Run a single training or validation epoch."""
    model.train(mode=train_mode)
    meter = {"loss": 0.0, "pred": 0.0, "diff": 0.0, "smooth": 0.0, "count": 0}
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for x_hist, x_future, y in loader:
            x_hist = x_hist.to(device)
            x_future = x_future.to(device)
            y = y.to(device)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            y_pred = model(x_hist, x_future)
            loss, parts = composite_sequence_loss(
                y_pred=y_pred,
                y_true=y,
                lambda_pred=config.lambda_pred,
                lambda_diff=config.lambda_diff,
                lambda_smooth=config.lambda_smooth,
            )
            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_size = x_hist.size(0)
            meter["count"] += batch_size
            for key in ("loss", "pred", "diff", "smooth"):
                meter[key] += parts[key] * batch_size
    count = max(meter.pop("count"), 1)
    return {key: value / count for key, value in meter.items()}


def train_model(dataset, config, models_dir: str, logger) -> TrainResult:
    """Train the selected model and save the best checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_dataloaders(dataset, config)
    sample_hist, sample_future, sample_target = dataset[0]
    model = build_model(
        config=config,
        history_dim=sample_hist.shape[-1],
        future_dim=sample_future.shape[-1],
        output_dim=sample_target.shape[-1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_pred": [],
        "val_pred": [],
        "train_diff": [],
        "val_diff": [],
        "train_smooth": [],
        "val_smooth": [],
    }
    best_model_path = os.path.join(models_dir, "best_model.pt")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, config, train_mode=True)
        val_stats = run_epoch(model, val_loader, optimizer, device, config, train_mode=False)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_pred"].append(train_stats["pred"])
        history["val_pred"].append(val_stats["pred"])
        history["train_diff"].append(train_stats["diff"])
        history["val_diff"].append(val_stats["diff"])
        history["train_smooth"].append(train_stats["smooth"])
        history["val_smooth"].append(val_stats["smooth"])
        logger.info(
            "Epoch %d/%d | train=%.6f | val=%.6f",
            epoch,
            config.epochs,
            train_stats["loss"],
            val_stats["loss"],
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    model.load_state_dict(best_state)
    return TrainResult(model=model, device=device, history=history, best_model_path=best_model_path)


def autoregressive_rollout(
    model: torch.nn.Module,
    device: str,
    static_features_full: np.ndarray,
    residual_history_full: np.ndarray,
    forecast_start_index: int,
    pred_len: int,
    scalers: dict,
    logger,
) -> np.ndarray:
    """Roll the model forward using only SGP4 features and past predicted residuals."""
    if forecast_start_index <= 0:
        raise ValueError("forecast_start_index must be positive.")

    model.eval()
    predicted = residual_history_full.copy()
    total_len = len(predicted)
    input_len = scalers["input_len"]

    with torch.no_grad():
        cursor = forecast_start_index
        while cursor < total_len:
            hist_start = cursor - input_len
            if hist_start < 0:
                raise ValueError("Not enough history before forecast start for the chosen input_len.")
            hist_features = np.concatenate(
                [static_features_full[hist_start:cursor], predicted[hist_start:cursor]],
                axis=1,
            )
            future_end = min(cursor + pred_len, total_len)
            future_cov = static_features_full[cursor:future_end]
            x_hist = scalers["history"].transform(hist_features)[None, ...].astype(np.float32)
            x_future = scalers["future"].transform(future_cov)[None, ...].astype(np.float32)
            pred_norm = model(
                torch.from_numpy(x_hist).to(device),
                torch.from_numpy(x_future).to(device),
            ).cpu().numpy()[0]
            pred_block = scalers["target"].inverse_transform(pred_norm)
            predicted[cursor:future_end] = pred_block[: future_end - cursor]
            cursor = future_end
            logger.info("Forecast rollout progress: %d / %d", cursor, total_len)
    return predicted
