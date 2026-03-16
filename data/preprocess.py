"""Preprocessing utilities for residual learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from orbit.dynamics import EARTH
from orbit.rtn import project_batch_eci_to_rtn, project_batch_rtn_to_eci


def build_time_grid(
    train_duration_sec: float,
    total_duration_sec: float,
    dt_train_sec: float,
    dt_full_sec: float,
) -> np.ndarray:
    """Build a dense train segment and a configurable full-arc forecast segment."""
    if train_duration_sec <= 0 or total_duration_sec <= 0:
        raise ValueError("Durations must be positive.")
    if total_duration_sec < train_duration_sec:
        raise ValueError("Total duration must be greater than or equal to train duration.")
    if dt_train_sec <= 0 or dt_full_sec <= 0:
        raise ValueError("Sampling intervals must be positive.")

    train_times = np.arange(0.0, train_duration_sec, dt_train_sec, dtype=np.float64)
    future_times = np.arange(train_duration_sec, total_duration_sec + 0.5 * dt_full_sec, dt_full_sec, dtype=np.float64)
    if len(train_times) > 0 and len(future_times) > 0 and np.isclose(train_times[-1], future_times[0]):
        future_times = future_times[1:]
    if len(train_times) == 0:
        return future_times
    if len(future_times) == 0:
        return train_times
    return np.concatenate([train_times, future_times])


def estimate_orbital_period_sec(r_eci_m: np.ndarray, v_eci_mps: np.ndarray) -> float:
    """Estimate orbital period from a Cartesian state using the vis-viva equation."""
    r_norm = np.linalg.norm(r_eci_m)
    v_sq = float(np.dot(v_eci_mps, v_eci_mps))
    inv_a = 2.0 / r_norm - v_sq / EARTH.mu_m3_s2
    if inv_a <= 0:
        return 5400.0
    semi_major_axis = 1.0 / inv_a
    return float(2.0 * np.pi * np.sqrt(semi_major_axis ** 3 / EARTH.mu_m3_s2))


def build_static_features(
    times_sec: np.ndarray,
    r_sgp4_eci_m: np.ndarray,
    v_sgp4_eci_mps: np.ndarray,
    orbital_period_sec: float,
) -> np.ndarray:
    """Create per-epoch features derived only from the nominal SGP4 trajectory."""
    r_norm = np.linalg.norm(r_sgp4_eci_m, axis=1)
    v_norm = np.linalg.norm(v_sgp4_eci_mps, axis=1)
    radial_velocity = np.sum(r_sgp4_eci_m * v_sgp4_eci_mps, axis=1) / np.maximum(r_norm, 1.0)
    angular_momentum = np.cross(r_sgp4_eci_m, v_sgp4_eci_mps)
    h_norm = np.linalg.norm(angular_momentum, axis=1)
    tangential_velocity = h_norm / np.maximum(r_norm, 1.0)
    phase = 2.0 * np.pi * times_sec / max(orbital_period_sec, 1.0)
    return np.column_stack(
        [
            r_norm,
            v_norm,
            radial_velocity,
            tangential_velocity,
            h_norm,
            times_sec,
            times_sec / max(times_sec[-1], 1.0),
            np.sin(phase),
            np.cos(phase),
        ]
    )


def build_narx_exogenous_features(
    r_sgp4_eci_m: np.ndarray,
    v_sgp4_eci_mps: np.ndarray,
    include_velocity: bool,
) -> np.ndarray:
    """Build paper-style exogenous inputs from the nominal SGP4 trajectory."""
    if include_velocity:
        return np.concatenate([r_sgp4_eci_m, v_sgp4_eci_mps], axis=1)
    return np.asarray(r_sgp4_eci_m, dtype=np.float64)


def prediction_steps_from_seconds(prediction_length_sec: float, sample_dt_sec: float) -> int:
    """Convert a prediction horizon in seconds into an integer number of samples."""
    if prediction_length_sec <= 0:
        raise ValueError("prediction_length_sec must be positive.")
    if sample_dt_sec <= 0:
        raise ValueError("sample_dt_sec must be positive.")
    steps_float = prediction_length_sec / sample_dt_sec
    steps = int(round(steps_float))
    if steps <= 0:
        raise ValueError("Prediction length must correspond to at least one sample.")
    if not np.isclose(steps_float, steps, atol=1e-9, rtol=1e-9):
        raise ValueError(
            f"Prediction length {prediction_length_sec} s is not an integer multiple of the sampling interval {sample_dt_sec} s."
        )
    return steps


def combine_target(pos_rtn: np.ndarray, vel_rtn: Optional[np.ndarray], predict_velocity: bool) -> np.ndarray:
    """Stack position and optional velocity residuals into the final target tensor."""
    if predict_velocity:
        if vel_rtn is None:
            raise ValueError("Velocity residuals are required when predict_velocity=True.")
        return np.concatenate([pos_rtn, vel_rtn], axis=1)
    return pos_rtn


def split_target(target: np.ndarray, predict_velocity: bool) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Split the residual target into position and optional velocity parts."""
    if predict_velocity:
        return target[:, :3], target[:, 3:]
    return target[:, :3], None


@dataclass
class ObservationBundle:
    """Synthetic observations used for training."""

    clean_target: np.ndarray
    noisy_target: np.ndarray
    noisy_pos_rtn: np.ndarray
    noisy_vel_rtn: Optional[np.ndarray]


def make_noisy_training_observations(
    config,
    rng: np.random.Generator,
    train_mask: np.ndarray,
    c_eci_to_rtn: np.ndarray,
    r_sgp4_eci_m: np.ndarray,
    v_sgp4_eci_mps: np.ndarray,
    r_hpop_eci_m: np.ndarray,
    v_hpop_eci_mps: np.ndarray,
    clean_pos_rtn: np.ndarray,
    clean_vel_rtn: np.ndarray,
) -> ObservationBundle:
    """Create noisy synthetic observations on the training segment only."""
    num_train = int(np.sum(train_mask))
    sigmas_pos = np.array([config.noise_sigma_r_m, config.noise_sigma_t_m, config.noise_sigma_n_m], dtype=np.float64)
    sigmas_vel = np.array([config.noise_sigma_vr_mps, config.noise_sigma_vt_mps, config.noise_sigma_vn_mps], dtype=np.float64)
    noise_pos_rtn = rng.normal(0.0, sigmas_pos, size=(num_train, 3))
    noise_vel_rtn = rng.normal(0.0, sigmas_vel, size=(num_train, 3)) if config.predict_velocity else None

    clean_target = combine_target(clean_pos_rtn[train_mask], clean_vel_rtn[train_mask], config.predict_velocity)
    if config.observation_mode == "residual":
        noisy_pos_rtn = clean_pos_rtn[train_mask] + noise_pos_rtn
        noisy_vel_rtn = clean_vel_rtn[train_mask] + noise_vel_rtn if config.predict_velocity else None
        noisy_target = combine_target(noisy_pos_rtn, noisy_vel_rtn, config.predict_velocity)
        return ObservationBundle(clean_target, noisy_target, noisy_pos_rtn, noisy_vel_rtn)

    noise_pos_eci = project_batch_rtn_to_eci(c_eci_to_rtn[train_mask], noise_pos_rtn)
    noisy_r_hpop = r_hpop_eci_m[train_mask] + noise_pos_eci
    noisy_delta_r_eci = noisy_r_hpop - r_sgp4_eci_m[train_mask]
    noisy_pos_rtn = project_batch_eci_to_rtn(c_eci_to_rtn[train_mask], noisy_delta_r_eci)

    if config.predict_velocity:
        noise_vel_eci = project_batch_rtn_to_eci(c_eci_to_rtn[train_mask], noise_vel_rtn)
        noisy_v_hpop = v_hpop_eci_mps[train_mask] + noise_vel_eci
        noisy_delta_v_eci = noisy_v_hpop - v_sgp4_eci_mps[train_mask]
        noisy_vel_rtn = project_batch_eci_to_rtn(c_eci_to_rtn[train_mask], noisy_delta_v_eci)
    else:
        noisy_vel_rtn = None

    noisy_target = combine_target(noisy_pos_rtn, noisy_vel_rtn, config.predict_velocity)
    return ObservationBundle(clean_target, noisy_target, noisy_pos_rtn, noisy_vel_rtn)


def build_history_features(static_features: np.ndarray, residual_history: np.ndarray) -> np.ndarray:
    """Concatenate static SGP4-derived features and residual history features."""
    return np.concatenate([static_features, residual_history], axis=1)


class StandardScaler:
    """Simple NumPy-based standard scaler."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "StandardScaler":
        if data.ndim != 2:
            raise ValueError("Scaler fit expects a 2D array.")
        self.mean_ = data.mean(axis=0)
        self.std_ = data.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return data * self.std_ + self.mean_

    def to_dict(self) -> dict:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}


def fit_scalers(
    history_features_train: np.ndarray,
    future_covariates_train: np.ndarray,
    target_train: np.ndarray,
) -> dict[str, StandardScaler]:
    """Fit all scalers using training-segment statistics only."""
    history_scaler = StandardScaler().fit(history_features_train)
    future_scaler = StandardScaler().fit(future_covariates_train)
    target_scaler = StandardScaler().fit(target_train)
    return {"history": history_scaler, "future": future_scaler, "target": target_scaler}


def fit_narx_scalers(
    exogenous_inputs_train: np.ndarray,
    target_train: np.ndarray,
) -> dict[str, StandardScaler]:
    """Fit the paper-style NARX scalers using training-segment statistics only."""
    exogenous_scaler = StandardScaler().fit(exogenous_inputs_train)
    target_scaler = StandardScaler().fit(target_train)
    return {"narx_input": exogenous_scaler, "target": target_scaler}
