"""HPOP-like Cowell dynamics model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EarthConstants:
    """Geophysical constants used by the synthetic truth propagator."""

    mu_m3_s2: float = 3.986004418e14
    radius_m: float = 6378136.3
    j2: float = 1.08262668e-3
    earth_rotation_rad_s: float = 7.2921150e-5


EARTH = EarthConstants()


def norm_with_floor(vec: np.ndarray, floor: float = 1e-12) -> float:
    """Compute a safe vector norm."""
    return float(max(np.linalg.norm(vec), floor))


def accel_two_body(r_eci_m: np.ndarray, mu_m3_s2: float = EARTH.mu_m3_s2) -> np.ndarray:
    """Two-body central gravity acceleration."""
    r_norm = norm_with_floor(r_eci_m)
    return -mu_m3_s2 * r_eci_m / (r_norm ** 3)


def accel_j2(
    r_eci_m: np.ndarray,
    mu_m3_s2: float = EARTH.mu_m3_s2,
    radius_m: float = EARTH.radius_m,
    j2: float = EARTH.j2,
) -> np.ndarray:
    """Earth J2 perturbation acceleration in an Earth-centered inertial frame."""
    x, y, z = r_eci_m
    r2 = np.dot(r_eci_m, r_eci_m)
    r = np.sqrt(max(r2, 1e-12))
    z2 = z * z
    factor = 1.5 * j2 * mu_m3_s2 * radius_m ** 2 / (r ** 5)
    xy_term = 5.0 * z2 / r2 - 1.0
    z_term = 5.0 * z2 / r2 - 3.0
    return factor * np.array([x * xy_term, y * xy_term, z * z_term], dtype=np.float64)


def atmospheric_density_exponential(
    altitude_m: float,
    rho_ref_kgpm3: float,
    h_ref_m: float,
    scale_height_m: float,
) -> float:
    """Simple exponential atmosphere density approximation."""
    altitude_m = max(0.0, altitude_m)
    exponent = -(altitude_m - h_ref_m) / max(scale_height_m, 1.0)
    return float(rho_ref_kgpm3 * np.exp(exponent))


def accel_drag(
    r_eci_m: np.ndarray,
    v_eci_mps: np.ndarray,
    cd: float,
    area_m2: float,
    mass_kg: float,
    rho_ref_kgpm3: float,
    h_ref_m: float,
    scale_height_m: float,
    earth_rotation_rad_s: float = EARTH.earth_rotation_rad_s,
    radius_m: float = EARTH.radius_m,
) -> np.ndarray:
    """Approximate drag acceleration using a co-rotating exponential atmosphere."""
    altitude_m = norm_with_floor(r_eci_m) - radius_m
    rho = atmospheric_density_exponential(altitude_m, rho_ref_kgpm3, h_ref_m, scale_height_m)
    omega = np.array([0.0, 0.0, earth_rotation_rad_s], dtype=np.float64)
    v_rel = v_eci_mps - np.cross(omega, r_eci_m)
    speed_rel = norm_with_floor(v_rel)
    ballistic = cd * area_m2 / max(mass_kg, 1e-9)
    return -0.5 * ballistic * rho * speed_rel * v_rel


def cowell_rhs(t_sec: float, state: np.ndarray, config) -> np.ndarray:
    """Differential equation for the HPOP-like numerical propagator."""
    del t_sec  # Time-dependent force models are not used in this approximation.
    r_eci = state[:3]
    v_eci = state[3:]
    acc = accel_two_body(r_eci) + accel_j2(r_eci)
    if config.force_drag:
        acc = acc + accel_drag(
            r_eci_m=r_eci,
            v_eci_mps=v_eci,
            cd=config.drag_cd,
            area_m2=config.drag_area_m2,
            mass_kg=config.drag_mass_kg,
            rho_ref_kgpm3=config.drag_rho_ref_kgpm3,
            h_ref_m=config.drag_h_ref_m,
            scale_height_m=config.drag_scale_height_m,
        )
    return np.concatenate([v_eci, acc])
