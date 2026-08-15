"""Unit tests for CRI sidecar addon store."""

from __future__ import annotations

import numpy as np

from openpi.cri.addon import CriAddonReader
from openpi.cri.addon import CriAddonWriter
from openpi.cri.addon import episode_key
from openpi.cri.constants import DEFAULT_NUM_JOINTS
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
    cri2 = np.ones((3, NUM_CRI_POINTS), dtype=np.float32)
    qd2 = np.zeros((3, DEFAULT_NUM_JOINTS), dtype=np.float32)

    w = CriAddonWriter(addon, source_dir=source, dt=1 / 15)
    w.append("a--ep1", cri1, qd1)
    w.close(finished=False, num_source_episodes=2)

    w2 = CriAddonWriter(addon, source_dir=source, dt=1 / 15)
    assert w2.has("a--ep1")
    assert not w2.has("a--ep2")
    w2.append("a--ep2", cri2, qd2)
    w2.close(finished=True, num_source_episodes=2)

    reader = CriAddonReader(addon)
    assert len(reader.entries) == 2
    np.testing.assert_allclose(reader.get_cri("a--ep1"), cri1)
    np.testing.assert_allclose(reader.get_qd("a--ep2"), qd2)
    assert reader.meta["finished"] is True
