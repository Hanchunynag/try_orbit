"""SGP4 propagation implemented through Orekit's TLEPropagator."""

from __future__ import annotations

import numpy as np

from utils.io_utils import allocate_storage
from orbit.frames import absolute_dates_from_offsets, pv_to_numpy


def build_orekit_tle_propagator(line1: str, line2: str):
    """Create an Orekit TLEPropagator from raw TLE lines."""
    from org.orekit.propagation.analytical.tle import TLE, TLEPropagator

    tle = TLE(line1, line2)
    propagator = TLEPropagator.selectExtrapolator(tle)
    return tle, propagator


def get_tle_epoch_state_gcrf(line1: str, line2: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the SGP4 nominal Cartesian state at the TLE epoch in GCRF."""
    from org.orekit.frames import FramesFactory

    tle, propagator = build_orekit_tle_propagator(line1, line2)
    gcrf = FramesFactory.getGCRF()
    pv = propagator.getPVCoordinates(tle.getDate(), gcrf)
    return pv_to_numpy(pv)


def propagate_sgp4_gcrf(
    line1: str,
    line2: str,
    sample_times_since_tle_sec: np.ndarray,
    chunk_size: int,
    cache_dir: str,
    memmap_threshold_mb: float,
    logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the nominal TLE orbit with Orekit SGP4 in GCRF."""
    from org.orekit.frames import FramesFactory

    tle, propagator = build_orekit_tle_propagator(line1, line2)
    gcrf = FramesFactory.getGCRF()
    num_samples = len(sample_times_since_tle_sec)
    positions = allocate_storage(
        (num_samples, 3),
        np.float64,
        base_path=f"{cache_dir}/sgp4_positions.dat",
        threshold_mb=memmap_threshold_mb,
    )
    velocities = allocate_storage(
        (num_samples, 3),
        np.float64,
        base_path=f"{cache_dir}/sgp4_velocities.dat",
        threshold_mb=memmap_threshold_mb,
    )

    all_dates = absolute_dates_from_offsets(tle.getDate(), sample_times_since_tle_sec)
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        for idx in range(start, end):
            pv = propagator.getPVCoordinates(all_dates[idx], gcrf)
            position, velocity = pv_to_numpy(pv)
            positions[idx] = position
            velocities[idx] = velocity
        logger.info("Orekit SGP4 propagation progress: %d / %d samples", end, num_samples)
    return positions, velocities
