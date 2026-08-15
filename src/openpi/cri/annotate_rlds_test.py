"""Tests for RLDS CRI annotation helpers (solver mocked; no CUDA)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from openpi.cri.annotate_rlds import annotate_episode_dict
from openpi.cri.annotate_rlds import build_cri_features
from openpi.cri.annotate_rlds import compute_episode_cri
from openpi.cri.constants import NUM_CRI_POINTS


class _FakeSolver:
    batch_size = 4

    def compute(self, q: Any, qd: Any, *, return_torch: bool = False) -> np.ndarray:
        del return_torch, qd
        batch = np.asarray(q).reshape(-1, np.asarray(q).shape[-1]).shape[0]
        # Encode mean |qd| into first channel via side effect from compute_cri filter path:
        # return ones; zero-vel filter in compute_cri will zero at-rest rows.
        return np.ones((batch, NUM_CRI_POINTS), dtype=np.float32)


def test_compute_episode_cri_first_step_zero():
    q = np.zeros((5, 7), dtype=np.float64)
    q[1:] = np.linspace(0.01, 0.05, 4)[:, None]
    qd, cri = compute_episode_cri(q, solver=_FakeSolver(), dt=1 / 15)  # type: ignore[arg-type]
    assert qd.shape == (5, 7)
    assert cri.shape == (5, NUM_CRI_POINTS)
    np.testing.assert_allclose(qd[0], 0.0)
    np.testing.assert_allclose(cri[0], 0.0)
    assert np.all(cri[1:] > 0)


def test_annotate_episode_dict_adds_fields():
    steps = []
    for t in range(3):
        steps.append(
            {
                "action": np.zeros(7, dtype=np.float64),
                "observation": {
                    "joint_position": np.full(7, 0.01 * t, dtype=np.float64),
                    "gripper_position": np.zeros(1, dtype=np.float64),
                },
                "discount": np.float32(1.0),
                "is_first": t == 0,
                "is_last": t == 2,
                "is_terminal": t == 2,
                "language_instruction": b"pick",
                "language_instruction_2": b"",
                "language_instruction_3": b"",
                "reward": np.float32(0.0),
                "action_dict": {
                    "joint_position": np.zeros(7, dtype=np.float64),
                    "joint_velocity": np.zeros(7, dtype=np.float64),
                    "gripper_position": np.zeros(1, dtype=np.float64),
                    "gripper_velocity": np.zeros(1, dtype=np.float64),
                    "cartesian_position": np.zeros(6, dtype=np.float64),
                    "cartesian_velocity": np.zeros(6, dtype=np.float64),
                },
            }
        )
    episode = {"episode_metadata": {"file_path": b"x", "recording_folderpath": b"y"}, "steps": steps}
    out = annotate_episode_dict(episode, solver=_FakeSolver(), dt=1 / 15)  # type: ignore[arg-type]
    assert "cri" in out["steps"][0]["observation"]
    assert "joint_velocity" in out["steps"][0]["observation"]
    np.testing.assert_allclose(out["steps"][0]["observation"]["cri"], 0.0)


def test_build_cri_features_requires_tfds():
    tfds = pytest.importorskip("tensorflow_datasets")
    from pathlib import Path

    source = Path(__file__).resolve().parents[3] / "data" / "droid_100" / "1.0.0"
    if not source.is_dir():
        pytest.skip("droid_100 not present")
    builder = tfds.builder_from_directory(str(source))
    feats = build_cri_features(builder.info.features)
    assert "cri" in feats["steps"]["observation"]
    assert "joint_velocity" in feats["steps"]["observation"]
