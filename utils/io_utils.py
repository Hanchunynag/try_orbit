"""I/O helpers for outputs and serialized artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd


def ensure_output_dirs(base_dir: str) -> Dict[str, str]:
    """Create the standard output directory tree."""
    subdirs = {
        "root": base_dir,
        "figures": os.path.join(base_dir, "figures"),
        "models": os.path.join(base_dir, "models"),
        "metrics": os.path.join(base_dir, "metrics"),
        "data": os.path.join(base_dir, "data"),
        "cache": os.path.join(base_dir, "cache"),
    }
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)
    return subdirs


def to_serializable(data: Any) -> Any:
    """Convert common scientific objects into JSON-serializable structures."""
    if is_dataclass(data):
        return {k: to_serializable(v) for k, v in asdict(data).items()}
    if isinstance(data, dict):
        return {str(k): to_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_serializable(v) for v in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.floating, np.integer)):
        return data.item()
    return data


def save_json(data: Any, path: str) -> None:
    """Save JSON data with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(to_serializable(data), fp, indent=2, ensure_ascii=False)


def save_npz(path: str, **arrays: np.ndarray) -> None:
    """Save compressed NumPy arrays."""
    np.savez_compressed(path, **arrays)


def save_dataframe(df: pd.DataFrame, path: str, float_format: str = "%.9f") -> None:
    """Save a DataFrame as CSV."""
    df.to_csv(path, index=False, float_format=float_format)


def save_state_dataframe(
    times_sec: np.ndarray,
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    path: str,
    float_format: str = "%.9f",
) -> None:
    """Save state vectors into a CSV file."""
    df = pd.DataFrame(
        {
            "t_sec": times_sec,
            "x_m": positions_m[:, 0],
            "y_m": positions_m[:, 1],
            "z_m": positions_m[:, 2],
            "vx_mps": velocities_mps[:, 0],
            "vy_mps": velocities_mps[:, 1],
            "vz_mps": velocities_mps[:, 2],
        }
    )
    save_dataframe(df, path, float_format=float_format)


def save_residual_dataframe(
    times_sec: np.ndarray,
    residual_pos: np.ndarray,
    residual_vel: np.ndarray | None,
    path: str,
    float_format: str = "%.9f",
) -> None:
    """Save RTN residual vectors into CSV."""
    data = {
        "t_sec": times_sec,
        "delta_R_m": residual_pos[:, 0],
        "delta_T_m": residual_pos[:, 1],
        "delta_N_m": residual_pos[:, 2],
    }
    if residual_vel is not None:
        data.update(
            {
                "delta_vR_mps": residual_vel[:, 0],
                "delta_vT_mps": residual_vel[:, 1],
                "delta_vN_mps": residual_vel[:, 2],
            }
        )
    save_dataframe(pd.DataFrame(data), path, float_format=float_format)


def save_metrics_csv(metrics: Mapping[str, Any], path: str) -> None:
    """Flatten a JSON-like metrics dictionary into a simple CSV."""
    rows = []

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, sub_value in value.items():
                _walk(f"{prefix}.{key}" if prefix else str(key), sub_value)
            return
        rows.append({"metric": prefix, "value": to_serializable(value)})

    _walk("", metrics)
    pd.DataFrame(rows).to_csv(path, index=False)


def allocate_storage(
    shape: tuple[int, ...],
    dtype: np.dtype,
    base_path: str,
    threshold_mb: float,
) -> np.ndarray:
    """Create an ndarray or a memmap depending on expected size."""
    bytes_required = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
    megabytes = bytes_required / (1024.0 * 1024.0)
    if megabytes > threshold_mb:
        return np.memmap(base_path, dtype=dtype, mode="w+", shape=shape)
    return np.zeros(shape, dtype=dtype)
