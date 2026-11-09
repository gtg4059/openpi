"""Unit tests for CRI sidecar addon store."""

from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.cri.addon import CriAddonReader
from openpi.cri.addon import CriAddonWriter
from openpi.cri.addon import META_NAME
from openpi.cri.addon import episode_key
from openpi.cri.constants import CRI_ADDON_FORMAT_V3
from openpi.cri.constants import DEFAULT_NUM_JOINTS
from openpi.cri.constants import NUM_CRI_OBS_DIM
from openpi.cri.constants import NUM_CRI_POINTS


def test_episode_key_bytes_and_str():
    assert episode_key(b"/rec", b"/file") == "/rec--/file"
    assert episode_key("/rec", "/file") == "/rec--/file"


def test_addon_write_read_resume(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    addon = tmp_path / "addon"

    cri1 = np.arange(2 * NUM_CRI_POINTS, dtype=np.float32).reshape(2, NUM_CRI_POINTS)
    qd1 = np.arange(2 * DEFAULT_NUM_JOINTS, dtype=np.float32).reshape(2, DEFAULT_NUM_JOINTS)
    d1 = np.array([0.0, 0.5], dtype=np.float32)
    cri2 = np.ones((3, NUM_CRI_POINTS), dtype=np.float32)
    qd2 = np.zeros((3, DEFAULT_NUM_JOINTS), dtype=np.float32)
    d2 = np.zeros((3,), dtype=np.float32)

    w = CriAddonWriter(addon, source_dir=source, dt=1 / 15)
    w.append("a--ep1", cri1, qd1, d1)
    w.close(finished=False, num_source_episodes=2)

    w2 = CriAddonWriter(addon, source_dir=source, dt=1 / 15)
    assert w2.has("a--ep1")
    assert not w2.has("a--ep2")
    w2.append("a--ep2", cri2, qd2, d2)
    w2.close(finished=True, num_source_episodes=2)

    reader = CriAddonReader(addon)
    assert len(reader.entries) == 2
    np.testing.assert_allclose(reader.get_cri("a--ep1"), cri1)
    np.testing.assert_allclose(reader.get_delta("a--ep1").reshape(-1), d1)
    np.testing.assert_allclose(reader.get_qd("a--ep2"), qd2)
    obs = reader.get_cri_obs("a--ep1")
    assert obs.shape == (2, NUM_CRI_OBS_DIM)
    np.testing.assert_allclose(obs, cri1)
    assert reader.meta["finished"] is True
    assert reader.meta["format"] == CRI_ADDON_FORMAT_V3
    assert reader.meta["filter"] == "cbf_qp"
    assert reader.meta["filter_enabled"] is False
    assert reader.meta["cri_limit"] == 0.96


def test_reader_rejects_v2(tmp_path):
    addon = tmp_path / "v2"
    addon.mkdir()
    (addon / META_NAME).write_text(json.dumps({"format": "openpi_cri_addon_v2", "total_steps": 0}) + "\n")
    (addon / "episodes.jsonl").write_text("")
    with pytest.raises(ValueError, match="openpi_cri_addon_v3"):
        CriAddonReader(addon)


def test_annotate_refuses_v2_resume(tmp_path):
    from openpi.cri.annotate_rlds_addon import _refuse_incompatible_addon

    addon = tmp_path / "legacy"
    addon.mkdir()
    (addon / META_NAME).write_text(json.dumps({"format": "openpi_cri_addon_v2"}) + "\n")
    with pytest.raises(ValueError, match="openpi_cri_addon_v3"):
        _refuse_incompatible_addon(addon)
