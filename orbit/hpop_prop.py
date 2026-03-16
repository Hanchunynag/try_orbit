"""Orekit numerical truth propagation with higher-fidelity force models."""

from __future__ import annotations

import numpy as np

from orbit.frames import absolute_dates_from_offsets, pv_to_numpy
from utils.io_utils import allocate_storage


def _solar_activity_level_enum(level_name: str):
    from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation

    normalized = level_name.upper()
    if normalized == "AVERAGE":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE
    if normalized == "WEAK":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.WEAK
    if normalized == "STRONG":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.STRONG
    raise ValueError(f"Unsupported orekit_solar_activity_level: {level_name}")


def build_orekit_truth_propagator(initial_position_m: np.ndarray, initial_velocity_mps: np.ndarray, initial_date, config):
    """Build Orekit NumericalPropagator with configurable force models."""
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, Relativity, ThirdBodyAttraction
    from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient, SolarRadiationPressure
    from org.orekit.frames import FramesFactory
    from org.orekit.models.earth.atmosphere import NRLMSISE00
    from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
    from org.orekit.orbits import CartesianOrbit, OrbitType
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.utils import Constants, IERSConventions, PVCoordinates

    gcrf = FramesFactory.getGCRF()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    mu = Constants.WGS84_EARTH_MU

    pv = PVCoordinates(
        Vector3D(float(initial_position_m[0]), float(initial_position_m[1]), float(initial_position_m[2])),
        Vector3D(float(initial_velocity_mps[0]), float(initial_velocity_mps[1]), float(initial_velocity_mps[2])),
    )
    orbit = CartesianOrbit(pv, gcrf, initial_date, mu)
    state = SpacecraftState(orbit, float(config.orekit_mass_kg))

    integrator = DormandPrince853Integrator(
        float(config.orekit_min_step_sec),
        float(config.orekit_max_step_sec),
        float(config.orekit_abs_tolerance),
        float(config.orekit_rel_tolerance),
    )
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setMu(mu)
    propagator.setInitialState(state)

    gravity_provider = GravityFieldFactory.getNormalizedProvider(
        int(config.orekit_gravity_degree),
        int(config.orekit_gravity_order),
    )
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, gravity_provider))

    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()
    if config.orekit_enable_third_body:
        propagator.addForceModel(ThirdBodyAttraction(sun))
        propagator.addForceModel(ThirdBodyAttraction(moon))
    if config.orekit_enable_relativity:
        propagator.addForceModel(Relativity(mu))

    earth_shape = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )

    if config.orekit_enable_drag:
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            _solar_activity_level_enum(config.orekit_solar_activity_level),
        )
        atmosphere = NRLMSISE00(msafe, sun, earth_shape)
        spacecraft_drag = IsotropicDrag(float(config.orekit_drag_area_m2), float(config.orekit_drag_cd))
        propagator.addForceModel(DragForce(atmosphere, spacecraft_drag))

    if config.orekit_enable_srp:
        spacecraft_srp = IsotropicRadiationSingleCoefficient(
            float(config.orekit_srp_area_m2),
            float(config.orekit_srp_cr),
        )
        propagator.addForceModel(SolarRadiationPressure(sun, earth_shape, spacecraft_srp))

    return propagator


def propagate_orekit_truth(
    initial_position_m: np.ndarray,
    initial_velocity_mps: np.ndarray,
    sample_times_since_tle_sec: np.ndarray,
    initial_date,
    config,
    cache_dir: str,
    logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate synthetic truth with Orekit NumericalPropagator and sample it in GCRF."""
    from org.orekit.frames import FramesFactory

    num_samples = len(sample_times_since_tle_sec)
    positions = allocate_storage(
        (num_samples, 3),
        np.float64,
        base_path=f"{cache_dir}/truth_positions.dat",
        threshold_mb=config.memmap_threshold_mb,
    )
    velocities = allocate_storage(
        (num_samples, 3),
        np.float64,
        base_path=f"{cache_dir}/truth_velocities.dat",
        threshold_mb=config.memmap_threshold_mb,
    )
    if num_samples == 0:
        return positions, velocities

    propagator = build_orekit_truth_propagator(initial_position_m, initial_velocity_mps, initial_date, config)
    ephemeris_generator = propagator.getEphemerisGenerator()
    final_date = initial_date.shiftedBy(float(sample_times_since_tle_sec[-1]))
    propagator.propagate(final_date)
    ephemeris = ephemeris_generator.getGeneratedEphemeris()
    gcrf = FramesFactory.getGCRF()
    all_dates = absolute_dates_from_offsets(initial_date, sample_times_since_tle_sec)

    for start in range(0, num_samples, config.hpop_chunk_size):
        end = min(start + config.hpop_chunk_size, num_samples)
        for idx in range(start, end):
            pv = ephemeris.getPVCoordinates(all_dates[idx], gcrf)
            position, velocity = pv_to_numpy(pv)
            positions[idx] = position
            velocities[idx] = velocity
        logger.info("Orekit numerical truth propagation progress: %d / %d samples", end, num_samples)
    return positions, velocities
