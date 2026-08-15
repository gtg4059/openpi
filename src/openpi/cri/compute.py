"""High-level ``CRI(q, qd)`` API for OpenPI data / inference pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from openpi.cri.constants import DEFAULT_ZERO_VEL_EPS
from openpi.cri.constants import NUM_CRI_POINTS
from openpi.cri.postprocess import apply_cri_zero_vel_filter
from openpi.cri.postprocess import clamp_cri

if TYPE_CHECKING:
    from openpi.cri.solver import CriSolver


class _SolverCache:
    instance: CriSolver | None = None


def get_default_solver(**kwargs) -> CriSolver:
    """Lazily construct a process-wide ``CriSolver`` (CUDA required)."""
    if _SolverCache.instance is None:
        from openpi.cri.solver import CriSolver

        _SolverCache.instance = CriSolver(**kwargs)
    return _SolverCache.instance


def reset_default_solver() -> None:
    """Drop the cached default solver (tests / reconfiguration)."""
    _SolverCache.instance = None


def compute_cri(
    q: np.ndarray,
    qd: np.ndarray,
    *,
    solver: CriSolver | None = None,
    zero_vel_eps: float = DEFAULT_ZERO_VEL_EPS,
    use_default_solver: bool = True,
) -> np.ndarray:
    """Compute Collision Risk Index from joint position / velocity.

    Pipeline (matches IsaacLab reach CRI path):

    1. Run Safetics CUDA solver ``RunSolver_CUDA_CRI_AtMotionState(q, qd)``
    2. Clamp to ``[0, 2]``
    3. If ``||qd|| <= eps``, force that row's CRI to 0 (first step / at rest)

    Args:
        q: Joint positions ``(B, J)`` or ``(J,)`` in radians.
        qd: Joint velocities ``(B, J)`` or ``(J,)`` in rad/s.
        solver: Optional ``CriSolver``. If None and ``use_default_solver``, a
            process-wide solver is created (requires CUDA + analysis dir).
        zero_vel_eps: Speed threshold for the zero-velocity filter.
        use_default_solver: When ``solver`` is None, create/use the default solver.

    Returns:
        ``float32`` array of shape ``(B, 8)``.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    qd_arr = np.asarray(qd, dtype=np.float64)
    if q_arr.ndim == 1:
        q_arr = q_arr[None, :]
    if qd_arr.ndim == 1:
        qd_arr = qd_arr[None, :]
    if q_arr.shape != qd_arr.shape:
        raise ValueError(f"q and qd shape mismatch: {q_arr.shape} vs {qd_arr.shape}")

    active = solver
    if active is None:
        if not use_default_solver:
            raise ValueError("solver is required when use_default_solver=False")
        active = get_default_solver()

    cri = active.compute(q_arr, qd_arr, return_torch=False)
    cri = np.asarray(cri, dtype=np.float32)
    if cri.ndim == 1:
        cri = cri[None, :]
    if cri.shape[-1] != NUM_CRI_POINTS:
        raise ValueError(f"expected CRI width {NUM_CRI_POINTS}, got {cri.shape}")

    # Re-apply filter with the caller's eps (solver may use its own default).
    cri = clamp_cri(cri)
    return apply_cri_zero_vel_filter(cri, qd_arr, eps=zero_vel_eps)
