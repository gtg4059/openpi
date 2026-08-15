"""Tests for finite-difference joint velocity (no CUDA)."""

from __future__ import annotations

import numpy as np

from openpi.cri.velocity import joint_velocity_from_positions


def test_first_step_zero():
    q = np.array([[0.0, 0.0], [0.1, -0.2], [0.3, -0.1]], dtype=np.float64)
    qd = joint_velocity_from_positions(q, dt=0.1)
    np.testing.assert_allclose(qd[0], 0.0)
    np.testing.assert_allclose(qd[1], [1.0, -2.0])
    np.testing.assert_allclose(qd[2], [2.0, 1.0])


def test_single_timestep():
    q = np.zeros(7, dtype=np.float64)
    qd = joint_velocity_from_positions(q, dt=1 / 15)
    assert qd.shape == (1, 7)
    np.testing.assert_allclose(qd, 0.0)


def test_rejects_nonpositive_dt():
    import pytest

    with pytest.raises(ValueError, match="dt must be positive"):
        joint_velocity_from_positions(np.zeros((2, 7)), dt=0.0)
