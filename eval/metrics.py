"""Evaluation metrics for the residual-learning experiment."""

from __future__ import annotations

import numpy as np


COMPONENT_LABELS_POS = ("R", "T", "N")
COMPONENT_LABELS_VEL = ("vR", "vT", "vN")


def rmse(values: np.ndarray) -> float:
    """Root mean square of an error array."""
    return float(np.sqrt(np.mean(np.square(values))))


def mae(values: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(values)))


def max_abs(values: np.ndarray) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(values)))


def component_metrics(truth: np.ndarray, estimate: np.ndarray, labels: tuple[str, ...]) -> dict:
    """Compute per-component metrics for a vector time series."""
    error = estimate - truth
    metrics = {}
    for idx, label in enumerate(labels):
        comp = error[:, idx]
        metrics[label] = {
            "rmse": rmse(comp),
            "mae": mae(comp),
            "max_abs": max_abs(comp),
        }
    metrics["3d_rmse"] = rmse(np.linalg.norm(error, axis=1))
    metrics["final_3d_error"] = float(np.linalg.norm(error[-1]))
    return metrics


def improvement_percentage(baseline_value: float, corrected_value: float) -> float:
    """Percentage improvement, positive when corrected is better."""
    if abs(baseline_value) < 1e-12:
        return 0.0
    return float(100.0 * (baseline_value - corrected_value) / baseline_value)


def cumulative_vector_rmse(error_vectors: np.ndarray) -> np.ndarray:
    """Cumulative 3D RMSE over time."""
    squared = np.sum(error_vectors ** 2, axis=1)
    cumulative = np.cumsum(squared)
    counts = np.arange(1, len(error_vectors) + 1)
    return np.sqrt(cumulative / counts)


def checkpoint_error_summary(
    times_sec: np.ndarray,
    baseline_error_rtn: np.ndarray,
    corrected_error_rtn: np.ndarray,
    checkpoints_sec: list[float],
) -> dict:
    """Report instantaneous error norms near requested checkpoint times."""
    summary = {}
    baseline_norm = np.linalg.norm(baseline_error_rtn, axis=1)
    corrected_norm = np.linalg.norm(corrected_error_rtn, axis=1)
    for checkpoint in checkpoints_sec:
        idx = int(np.argmin(np.abs(times_sec - checkpoint)))
        summary[f"{checkpoint:.0f}s"] = {
            "sample_time_sec": float(times_sec[idx]),
            "baseline_norm_m": float(baseline_norm[idx]),
            "corrected_norm_m": float(corrected_norm[idx]),
            "improvement_percent": improvement_percentage(baseline_norm[idx], corrected_norm[idx]),
        }
    return summary
