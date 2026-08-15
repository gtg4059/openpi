"""CRI post-processing shared by solver and offline paths."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from openpi.cri.constants import CRI_CLAMP_MAX
from openpi.cri.constants import CRI_CLAMP_MIN
from openpi.cri.constants import DEFAULT_ZERO_VEL_EPS

ArrayT = TypeVar("ArrayT", np.ndarray, "torch.Tensor")  # noqa: F821


def clamp_cri(cri: ArrayT, *, min_val: float = CRI_CLAMP_MIN, max_val: float = CRI_CLAMP_MAX) -> ArrayT:
    """Clamp CRI values to ``[min_val, max_val]`` (IsaacLab default ``[0, 2]``)."""
    if isinstance(cri, np.ndarray):
        return np.clip(cri, min_val, max_val)
    import torch

    if not isinstance(cri, torch.Tensor):
        raise TypeError(f"cri must be np.ndarray or torch.Tensor, got {type(cri)}")
    return torch.clamp(cri, min=min_val, max=max_val)


def apply_cri_zero_vel_filter(
    cri: ArrayT,
    qd: ArrayT,
    *,
    eps: float = DEFAULT_ZERO_VEL_EPS,
) -> ArrayT:
    """Force CRI=0 for rows whose joint-speed norm is ``<= eps``.

    Matches IsaacLab ``ArticulationData._apply_cri_zero_vel_filter`` so the first
    step after reset (and any at-rest frame) cannot keep a stale nonzero CRI.
    """
    if isinstance(cri, np.ndarray) and isinstance(qd, np.ndarray):
        out = np.array(cri, copy=True, dtype=np.float32)
        speed = np.linalg.norm(np.asarray(qd, dtype=np.float64), axis=-1)
        out[speed <= eps] = 0.0
        return out

    import torch

    if not isinstance(cri, torch.Tensor) or not isinstance(qd, torch.Tensor):
        raise TypeError("cri and qd must both be np.ndarray or both be torch.Tensor")
    out = cri.to(dtype=torch.float32).clone()
    speed = torch.linalg.norm(qd.to(dtype=torch.float64), dim=-1)
    at_rest = speed <= eps
    if torch.any(at_rest):
        out[at_rest] = 0.0
    return out
