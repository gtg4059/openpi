"""Unit tests for CRI post-process and compute API (solver mocked; no CUDA required)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from openpi.cri.compute import compute_cri
from openpi.cri.constants import NUM_CRI_POINTS
from openpi.cri.postprocess import apply_cri_zero_vel_filter
from openpi.cri.postprocess import clamp_cri


def test_clamp_cri_numpy():
    cri = np.array([[-0.5, 0.5, 3.0]], dtype=np.float32)
    out = clamp_cri(cri)
    np.testing.assert_allclose(out, [[0.0, 0.5, 2.0]])


def test_zero_vel_filter_forces_cri_zero():
    cri = np.ones((2, NUM_CRI_POINTS), dtype=np.float32)
    qd = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # at rest → CRI=0
            [0.05, -0.08, 0.06, -0.04, 0.03, 0.02, 0.01],  # moving → keep
        ],
        dtype=np.float64,
    )
    out = apply_cri_zero_vel_filter(cri, qd, eps=1e-6)
    np.testing.assert_allclose(out[0], np.zeros(NUM_CRI_POINTS))
    np.testing.assert_allclose(out[1], np.ones(NUM_CRI_POINTS))


def test_zero_vel_filter_eps_boundary():
    cri = np.full((1, NUM_CRI_POINTS), 1.5, dtype=np.float32)
    # Exactly at eps → treated as at rest.
    qd = np.array([[1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    out = apply_cri_zero_vel_filter(cri, qd, eps=1e-6)
    np.testing.assert_allclose(out, np.zeros((1, NUM_CRI_POINTS)))


class _FakeSolver:
    def __init__(self, cri: np.ndarray):
        self._cri = np.asarray(cri, dtype=np.float32)

    def compute(self, q: Any, qd: Any, *, return_torch: bool = False) -> np.ndarray:
        del return_torch
        batch = np.asarray(q).reshape(-1, np.asarray(q).shape[-1]).shape[0]
        if self._cri.ndim == 1:
            return np.broadcast_to(self._cri, (batch, NUM_CRI_POINTS)).copy()
        return self._cri[:batch].copy()


def test_compute_cri_first_step_zero_velocity():
    """First step with qd≈0 must return CRI=0 even if the solver returns nonzero."""
    fake = _FakeSolver(np.full(NUM_CRI_POINTS, 0.8, dtype=np.float32))
    q = np.array([-0.5, -1.2, 1.4, -1.5, 1.57, 0.0, 0.1], dtype=np.float64)
    qd = np.zeros(7, dtype=np.float64)
    cri = compute_cri(q, qd, solver=fake)
    assert cri.shape == (1, NUM_CRI_POINTS)
    np.testing.assert_allclose(cri, 0.0)


def test_compute_cri_moving_keeps_solver_output():
    fake = _FakeSolver(np.linspace(0.1, 0.8, NUM_CRI_POINTS, dtype=np.float32))
    q = np.zeros((1, 7), dtype=np.float64)
    qd = np.array([[0.05, -0.08, 0.06, -0.04, 0.03, 0.02, 0.01]], dtype=np.float64)
    cri = compute_cri(q, qd, solver=fake)
    np.testing.assert_allclose(cri[0], np.linspace(0.1, 0.8, NUM_CRI_POINTS), atol=1e-6)


def test_compute_cri_clamps_solver_output():
    fake = _FakeSolver(np.array([3.0] * NUM_CRI_POINTS, dtype=np.float32))
    q = np.zeros((1, 7), dtype=np.float64)
    qd = np.ones((1, 7), dtype=np.float64) * 0.1
    cri = compute_cri(q, qd, solver=fake)
    np.testing.assert_allclose(cri, 2.0)


def test_compute_cri_requires_solver_when_disabled():
    with pytest.raises(ValueError, match="solver is required"):
        compute_cri(np.zeros(7), np.zeros(7), use_default_solver=False)
