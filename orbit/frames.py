"""Orekit frame and data-context helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _candidate_orekit_data_paths(explicit_path: Optional[str]) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    env_path = os.environ.get("OREKIT_DATA_PATH")
    if env_path:
        candidates.append(Path(env_path))
    cwd = Path.cwd()
    home = Path.home()
    common_names = [
        "orekit-data",
        "orekit-data.zip",
        "orekit-data-main",
        "orekit-data-main.zip",
    ]
    for base in (cwd, home):
        for name in common_names:
            candidates.append(base / name)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def locate_orekit_data(explicit_path: Optional[str] = None) -> Path:
    """Locate an orekit-data directory or zip file."""
    for candidate in _candidate_orekit_data_paths(explicit_path):
        if candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in _candidate_orekit_data_paths(explicit_path))
    raise FileNotFoundError(
        "Could not locate orekit-data. Provide --orekit_data_path or set OREKIT_DATA_PATH. "
        f"Searched:\n{searched}"
    )


def initialize_orekit(orekit_data_path: Optional[str], logger=None) -> dict:
    """Initialize the Orekit JVM and register orekit-data in the default data context."""
    try:
        import jpype
        import orekit_jpype as orekit
    except ImportError as exc:
        raise ImportError(
            "orekit-jpype and jpype1 are required. Install them from requirements.txt."
        ) from exc

    if not jpype.isJVMStarted():
        orekit.initVM()

    from java.io import File
    from org.orekit.data import DataContext, DirectoryCrawler, ZipJarCrawler

    data_path = locate_orekit_data(orekit_data_path)
    manager = DataContext.getDefault().getDataProvidersManager()
    manager.clearProviders()
    java_file = File(str(data_path))
    if java_file.isDirectory():
        manager.addProvider(DirectoryCrawler(java_file))
    else:
        manager.addProvider(ZipJarCrawler(java_file))
    if logger is not None:
        logger.info("Orekit initialized with data path: %s", str(data_path))
    return {
        "orekit": orekit,
        "data_path": str(data_path),
    }


def datetime_to_absolutedate(dt_utc: datetime):
    """Convert a Python UTC datetime to Orekit AbsoluteDate."""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt_utc.astimezone(timezone.utc)
    from org.orekit.time import AbsoluteDate, TimeScalesFactory

    seconds = dt_utc.second + dt_utc.microsecond * 1e-6
    return AbsoluteDate(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour,
        dt_utc.minute,
        seconds,
        TimeScalesFactory.getUTC(),
    )


def absolute_dates_from_offsets(start_date, offsets_sec: np.ndarray) -> list:
    """Create AbsoluteDate objects by shifting the reference date."""
    return [start_date.shiftedBy(float(offset)) for offset in offsets_sec]


def pv_to_numpy(pv_coordinates) -> tuple[np.ndarray, np.ndarray]:
    """Convert Orekit PVCoordinates to NumPy arrays in SI units."""
    pos = pv_coordinates.getPosition()
    vel = pv_coordinates.getVelocity()
    return (
        np.array([pos.getX(), pos.getY(), pos.getZ()], dtype=np.float64),
        np.array([vel.getX(), vel.getY(), vel.getZ()], dtype=np.float64),
    )
