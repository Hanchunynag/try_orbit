"""RTN frame utilities defined from the SGP4 nominal trajectory."""

from __future__ import annotations

import numpy as np


def _normalize_rows(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a batch of row vectors with safety checks."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < eps):
        raise ValueError("Cannot construct RTN frame because a vector norm is too small.")
    return vectors / norms


def build_eci_to_rtn_matrices(r_ref_eci_m: np.ndarray, v_ref_eci_mps: np.ndarray) -> np.ndarray:
    """Build ECI->RTN rotation matrices from the SGP4 reference states.

    The output matrix has RTN basis vectors as rows, therefore:
    ``delta_r_rtn = C_eci2rtn @ delta_r_eci``.
    """
    e_r = _normalize_rows(r_ref_eci_m)
    h = np.cross(r_ref_eci_m, v_ref_eci_mps)
    e_n = _normalize_rows(h)
    e_t = _normalize_rows(np.cross(e_n, e_r))
    return np.stack([e_r, e_t, e_n], axis=1)


def project_batch_eci_to_rtn(c_eci_to_rtn: np.ndarray, delta_eci: np.ndarray) -> np.ndarray:
    """Project ECI vectors into RTN coordinates."""
    return np.einsum("nij,nj->ni", c_eci_to_rtn, delta_eci)


def project_batch_rtn_to_eci(c_eci_to_rtn: np.ndarray, delta_rtn: np.ndarray) -> np.ndarray:
    """Project RTN vectors back into ECI coordinates."""
    return np.einsum("nji,nj->ni", c_eci_to_rtn, delta_rtn)
