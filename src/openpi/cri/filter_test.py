"""Unit tests for CRI-F CBF-QP delta + 1-step observation delay."""

from __future__ import annotations

from typing import Any

import numpy as np

from openpi.cri.constants import NUM_CRI_POINTS
from openpi.cri.filter import command_delta
from openpi.cri.filter import compute_episode_cri_f
from openpi.cri.filter import shift_cri_filter_obs


def test_command_delta_zero_when_passthrough():
    qd = np.ones((2, 7), dtype=np.float32)
    d = command_delta(qd, qd)
    np.testing.assert_allclose(d, [0.0, 0.0])


def test_command_delta_norm():
    qd = np.zeros((1, 7), dtype=np.float32)
    cmd = np.zeros((1, 7), dtype=np.float32)
    cmd[0, 0] = 3.0
    cmd[0, 1] = 4.0
    d = command_delta(cmd, qd)
    np.testing.assert_allclose(d, [5.0], atol=1e-6)


def test_shift_first_step_zero():
    cri = np.arange(3 * NUM_CRI_POINTS, dtype=np.float32).reshape(3, NUM_CRI_POINTS) + 1.0
    extra = np.array([[0.5], [0.6], [0.7]], dtype=np.float32)
    cri_obs, extra_obs = shift_cri_filter_obs(cri, extra)
    np.testing.assert_allclose(cri_obs[0], 0.0)
    np.testing.assert_allclose(extra_obs[0], 0.0)
    np.testing.assert_allclose(cri_obs[1], cri[0])
    np.testing.assert_allclose(extra_obs[1], 0.5)
    np.testing.assert_allclose(cri_obs[2], cri[1])
    np.testing.assert_allclose(extra_obs[2], 0.6)


class _FakeFilterSolver:
    batch_size = 4

    def run_cri_filter(self, q: Any, qd: Any) -> dict[str, Any]:
        batch = np.asarray(q).reshape(-1, np.asarray(q).shape[-1]).shape[0]
        qd_rl = np.asarray(qd, dtype=np.float32).reshape(batch, -1)
        cri = np.ones((batch, NUM_CRI_POINTS), dtype=np.float32)
        cri[:, 0] = 1.2
        qd_cmd = qd_rl.copy()
        qd_cmd[:, 0] += 0.3
        return {
            "cri_pre": cri,
            "qd_cmd": qd_cmd,
            "delta": command_delta(qd_cmd, qd_rl),
            "cri_limit": 0.96,
            "cbf_alpha": 0.02,
            "approach_limit": 0.96 * 0.98,
            "enabled": True,
        }


def test_compute_episode_cri_f_delay_and_delta():
    q = np.zeros((5, 7), dtype=np.float64)
    q[1:] = np.linspace(0.01, 0.05, 4)[:, None]
    qd, cri, delta, qd_cmd = compute_episode_cri_f(q, solver=_FakeFilterSolver(), dt=1 / 15)
    assert qd.shape == (5, 7)
    assert cri.shape == (5, NUM_CRI_POINTS)
    assert delta.shape == (5,)
    assert qd_cmd.shape == (5, 7)
    np.testing.assert_allclose(qd[0], 0.0)
    np.testing.assert_allclose(cri[0], 0.0)
    np.testing.assert_allclose(delta[0], 0.0)
    np.testing.assert_allclose(cri[1, 0], 1.2)
    np.testing.assert_allclose(cri[1, 1:], 1.0)
    np.testing.assert_allclose(delta[1], 0.3, atol=1e-5)
