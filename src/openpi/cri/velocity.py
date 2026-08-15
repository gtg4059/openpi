"""Joint velocity estimation from position trajectories for offline CRI annotation."""

from __future__ import annotations

import numpy as np

from openpi.cri.constants import DROID_CONTROL_DT


def joint_velocity_from_positions(
    q: np.ndarray,
    *,
    dt: float = DROID_CONTROL_DT,
) -> np.ndarray:
    """Finite-difference joint velocities; first timestep is forced to zero.

    Args:
        q: Joint positions ``(T, J)`` or ``(J,)`` in radians.
        dt: Timestep in seconds (DROID default ``1/15``).

    Returns:
        Velocities ``(T, J)`` float64, with ``qd[0] = 0`` and
        ``qd[t] = (q[t] - q[t-1]) / dt`` for ``t >= 1``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.ndim == 1:
        q_arr = q_arr[None, :]
    if q_arr.ndim != 2:
        raise ValueError(f"q must have shape (T, J) or (J,), got {q_arr.shape}")

    qd = np.zeros_like(q_arr, dtype=np.float64)
    if q_arr.shape[0] > 1:
        qd[1:] = (q_arr[1:] - q_arr[:-1]) / dt
    return qd
