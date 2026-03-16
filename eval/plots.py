"""Plotting utilities."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_3d_orbits(r_sgp4_m: np.ndarray, r_hpop_m: np.ndarray, path: str) -> None:
    """Plot the 3D nominal and synthetic-truth trajectories."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(r_sgp4_m[:, 0] / 1e3, r_sgp4_m[:, 1] / 1e3, r_sgp4_m[:, 2] / 1e3, label="SGP4", linewidth=1.0)
    ax.plot(r_hpop_m[:, 0] / 1e3, r_hpop_m[:, 1] / 1e3, r_hpop_m[:, 2] / 1e3, label="HPOP-like", linewidth=1.0)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend()
    ax.set_title("SGP4 vs HPOP-like Orbit")
    _save(fig, path)


def plot_rtn_residuals(times_sec: np.ndarray, residual_rtn_m: np.ndarray, path: str) -> None:
    """Plot RTN residual truth over the full arc."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ("R", "T", "N")
    for idx, ax in enumerate(axes):
        ax.plot(times_sec, residual_rtn_m[:, idx], linewidth=0.8)
        ax.set_ylabel(f"{labels[idx]} [m]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Clean RTN Residuals")
    _save(fig, path)


def plot_noisy_vs_clean_train(times_sec: np.ndarray, clean_rtn_m: np.ndarray, noisy_rtn_m: np.ndarray, path: str) -> None:
    """Plot clean vs noisy training residuals."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ("R", "T", "N")
    for idx, ax in enumerate(axes):
        ax.plot(times_sec, clean_rtn_m[:, idx], label="clean", linewidth=1.0)
        ax.plot(times_sec, noisy_rtn_m[:, idx], label="noisy", linewidth=0.7, alpha=0.7)
        ax.set_ylabel(f"{labels[idx]} [m]")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Training Residuals: Clean vs Noisy")
    _save(fig, path)


def plot_training_history(history: dict[str, list[float]], path: str) -> None:
    """Plot train and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Training / Validation Loss")
    _save(fig, path)


def plot_forecast_residuals(times_sec: np.ndarray, truth_rtn_m: np.ndarray, pred_rtn_m: np.ndarray, path: str) -> None:
    """Plot true vs predicted residuals on the forecast arc."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ("R", "T", "N")
    for idx, ax in enumerate(axes):
        ax.plot(times_sec, truth_rtn_m[:, idx], label="truth", linewidth=1.0)
        ax.plot(times_sec, pred_rtn_m[:, idx], label="pred", linewidth=0.8, alpha=0.85)
        ax.set_ylabel(f"{labels[idx]} [m]")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Forecast Residuals: Truth vs Prediction")
    _save(fig, path)


def plot_error_comparison(times_sec: np.ndarray, baseline_norm_m: np.ndarray, corrected_norm_m: np.ndarray, path: str) -> None:
    """Plot baseline and corrected position-error norms over time."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(times_sec, baseline_norm_m, label="SGP4 vs HPOP-like")
    ax.plot(times_sec, corrected_norm_m, label="Corrected vs HPOP-like")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("3D position error [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Baseline vs Corrected Position Error")
    _save(fig, path)


def plot_rmse_over_time(times_sec: np.ndarray, baseline_cum_rmse_m: np.ndarray, corrected_cum_rmse_m: np.ndarray, path: str) -> None:
    """Plot cumulative RMSE over the forecast arc."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(times_sec, baseline_cum_rmse_m, label="baseline cumulative RMSE")
    ax.plot(times_sec, corrected_cum_rmse_m, label="corrected cumulative RMSE")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Cumulative 3D RMSE [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Cumulative RMSE vs Time")
    _save(fig, path)


def plot_final_error_bar(baseline_final_m: float, corrected_final_m: float, path: str) -> None:
    """Plot the final error comparison as a bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["baseline", "corrected"], [baseline_final_m, corrected_final_m], color=["tab:red", "tab:green"])
    ax.set_ylabel("Final 3D position error [m]")
    ax.set_title("Final Error Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, path)
