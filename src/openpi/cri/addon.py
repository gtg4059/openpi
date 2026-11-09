"""CRI sidecar addon for large RLDS datasets (no full-dataset copy).

Stores delayed CRI-F observations: ``cri`` (float32, 9), ``delta`` (float32, 1),
and ``joint_velocity`` (float32, 7) keyed by ``recording_folderpath--file_path``.

Policy obs is delayed ``cri_pre`` only (IsaacLab CRI-F). ``delta`` is
``||qd_cmd - qd_RL||`` for logging, not a time-scale.

Layout under ``addon_dir``::

    meta.json
    episodes.jsonl      # append-only; one JSON object per episode
    cri.f32             # raw float32, row-major (total_steps, 9)
    delta.f32           # raw float32, row-major (total_steps, 1)
    joint_velocity.f32  # raw float32, row-major (total_steps, 7)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from openpi.cri.constants import CBF_ALPHA
from openpi.cri.constants import CRI_ADDON_FORMAT_V3
from openpi.cri.constants import CRI_FILTER_LIMIT
from openpi.cri.constants import DEFAULT_NUM_JOINTS
from openpi.cri.constants import NUM_CRI_OBS_DIM
from openpi.cri.constants import NUM_CRI_POINTS

logger = logging.getLogger(__name__)

META_NAME = "meta.json"
EPISODES_NAME = "episodes.jsonl"
CRI_BIN_NAME = "cri.f32"
DELTA_BIN_NAME = "delta.f32"
QD_BIN_NAME = "joint_velocity.f32"


def episode_key(recording_folderpath: str | bytes, file_path: str | bytes) -> str:
    """Stable key matching DROID idle-filter / loader conventions."""
    rec = recording_folderpath.decode("utf-8") if isinstance(recording_folderpath, (bytes, bytearray)) else str(
        recording_folderpath
    )
    fp = file_path.decode("utf-8") if isinstance(file_path, (bytes, bytearray)) else str(file_path)
    return f"{rec}--{fp}"


@dataclass(frozen=True)
class EpisodeEntry:
    key: str
    offset: int
    length: int


class CriAddonWriter:
    """Append-only writer; safe to resume by skipping keys already in ``episodes.jsonl``."""

    def __init__(
        self,
        addon_dir: Path,
        *,
        source_dir: Path,
        dt: float,
        cri_limit: float = CRI_FILTER_LIMIT,
        cbf_alpha: float = CBF_ALPHA,
        filter_enabled: bool = False,
    ) -> None:
        self.addon_dir = Path(addon_dir).expanduser().resolve()
        self.addon_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.dt = float(dt)
        self.cri_limit = float(cri_limit)
        self.cbf_alpha = float(cbf_alpha)
        self.filter_enabled = bool(filter_enabled)

        self._episodes_path = self.addon_dir / EPISODES_NAME
        self._cri_path = self.addon_dir / CRI_BIN_NAME
        self._delta_path = self.addon_dir / DELTA_BIN_NAME
        self._qd_path = self.addon_dir / QD_BIN_NAME

        self.existing: dict[str, EpisodeEntry] = {}
        self._next_offset = 0
        if self._episodes_path.is_file():
            for entry in iter_episode_entries(self._episodes_path):
                self.existing[entry.key] = entry
                self._next_offset = max(self._next_offset, entry.offset + entry.length)

        self._cri_f = self._cri_path.open("ab")
        self._delta_f = self._delta_path.open("ab")
        self._qd_f = self._qd_path.open("ab")
        self._ep_f = self._episodes_path.open("a", encoding="utf-8")
        self._written_this_session = 0

    @property
    def written_this_session(self) -> int:
        return self._written_this_session

    def has(self, key: str) -> bool:
        return key in self.existing

    def append(self, key: str, cri: np.ndarray, qd: np.ndarray, delta: np.ndarray) -> None:
        if key in self.existing:
            raise ValueError(f"episode already in addon: {key}")
        cri_arr = np.asarray(cri, dtype=np.float32).reshape(-1, NUM_CRI_POINTS)
        qd_arr = np.asarray(qd, dtype=np.float32).reshape(-1, DEFAULT_NUM_JOINTS)
        delta_arr = np.asarray(delta, dtype=np.float32).reshape(-1, 1)
        if not (cri_arr.shape[0] == qd_arr.shape[0] == delta_arr.shape[0]):
            raise ValueError(f"length mismatch cri={cri_arr.shape} delta={delta_arr.shape} qd={qd_arr.shape}")
        length = int(cri_arr.shape[0])
        offset = int(self._next_offset)

        self._cri_f.write(cri_arr.tobytes(order="C"))
        self._delta_f.write(delta_arr.tobytes(order="C"))
        self._qd_f.write(qd_arr.tobytes(order="C"))
        self._cri_f.flush()
        self._delta_f.flush()
        self._qd_f.flush()

        entry = EpisodeEntry(key=key, offset=offset, length=length)
        self._ep_f.write(json.dumps({"key": key, "offset": offset, "length": length}, ensure_ascii=False) + "\n")
        self._ep_f.flush()

        self.existing[key] = entry
        self._next_offset = offset + length
        self._written_this_session += 1

    def close(self, *, finished: bool = False, num_source_episodes: int | None = None) -> None:
        self._cri_f.close()
        self._delta_f.close()
        self._qd_f.close()
        self._ep_f.close()
        meta = {
            "format": CRI_ADDON_FORMAT_V3,
            "source_dir": str(self.source_dir),
            "dt": self.dt,
            "filter": "cbf_qp",
            "cri_limit": self.cri_limit,
            "cbf_alpha": self.cbf_alpha,
            "filter_enabled": self.filter_enabled,
            "obs_delay": 1,
            "num_cri_points": NUM_CRI_POINTS,
            "num_cri_obs_dim": NUM_CRI_OBS_DIM,
            "num_joints": DEFAULT_NUM_JOINTS,
            "dtype": "float32",
            "num_episodes": len(self.existing),
            "total_steps": self._next_offset,
            "finished": bool(finished),
            "num_source_episodes": num_source_episodes,
        }
        (self.addon_dir / META_NAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "Addon writer closed: dir=%s episodes=%d steps=%d session_new=%d finished=%s",
            self.addon_dir,
            len(self.existing),
            self._next_offset,
            self._written_this_session,
            finished,
        )


class CriAddonReader:
    """Memory-mapped reader for training-time join."""

    def __init__(self, addon_dir: Path) -> None:
        self.addon_dir = Path(addon_dir).expanduser().resolve()
        meta_path = self.addon_dir / META_NAME
        if not meta_path.is_file():
            raise FileNotFoundError(f"CRI addon meta not found: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fmt = self.meta.get("format")
        if fmt != CRI_ADDON_FORMAT_V3:
            raise ValueError(
                f"CRI addon {self.addon_dir} has format={fmt!r}; expected {CRI_ADDON_FORMAT_V3}. "
                "Re-run annotate_rlds_addon to rebuild the CBF-QP sidecar."
            )
        delta_path = self.addon_dir / DELTA_BIN_NAME
        if not delta_path.is_file():
            raise FileNotFoundError(f"CRI-F addon missing {DELTA_BIN_NAME}: {delta_path}")
        self.entries = {e.key: e for e in iter_episode_entries(self.addon_dir / EPISODES_NAME)}
        total = int(self.meta.get("total_steps") or 0)
        if total <= 0:
            total = sum(e.length for e in self.entries.values())
        self._cri = np.memmap(self.addon_dir / CRI_BIN_NAME, dtype=np.float32, mode="r", shape=(total, NUM_CRI_POINTS))
        self._delta = np.memmap(delta_path, dtype=np.float32, mode="r", shape=(total, 1))
        self._qd = np.memmap(
            self.addon_dir / QD_BIN_NAME, dtype=np.float32, mode="r", shape=(total, DEFAULT_NUM_JOINTS)
        )
        logger.info("Loaded CRI addon %s (%d episodes, %d steps)", self.addon_dir, len(self.entries), total)

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def get_cri(self, key: str) -> np.ndarray:
        entry = self.entries[key]
        return np.asarray(self._cri[entry.offset : entry.offset + entry.length])

    def get_delta(self, key: str) -> np.ndarray:
        entry = self.entries[key]
        return np.asarray(self._delta[entry.offset : entry.offset + entry.length])

    def get_cri_obs(self, key: str) -> np.ndarray:
        """Policy CRI: delayed ``cri_pre`` only → ``(T, NUM_CRI_POINTS)``."""
        return self.get_cri(key)

    def get_qd(self, key: str) -> np.ndarray:
        entry = self.entries[key]
        return np.asarray(self._qd[entry.offset : entry.offset + entry.length])

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        return self.get_cri(key), self.get_qd(key)


def iter_episode_entries(path: Path) -> Iterator[EpisodeEntry]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield EpisodeEntry(key=obj["key"], offset=int(obj["offset"]), length=int(obj["length"]))


def iter_source_joint_episodes(
    source_dir: Path, *, split: str = "train"
) -> Iterator[tuple[str, np.ndarray, dict[str, Any]]]:
    """Yield ``(episode_key, joint_position[T,7], meta)`` without decoding images.

    Still streams TFRecord bytes (images are packed in the same examples) but does not JPEG-decode.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    source_dir = Path(source_dir).expanduser().resolve()
    builder = tfds.builder_from_directory(str(source_dir))
    ds = builder.as_dataset(split=split, shuffle_files=False)

    for ep in ds:
        file_path = ep["episode_metadata"]["file_path"].numpy()
        recording_folderpath = ep["episode_metadata"]["recording_folderpath"].numpy()
        key = episode_key(recording_folderpath, file_path)

        joints: list[np.ndarray] = []
        for step in ep["steps"]:
            # Touch only joint_position — skip image fields.
            q = step["observation"]["joint_position"].numpy()
            joints.append(np.asarray(q, dtype=np.float64).reshape(-1)[:DEFAULT_NUM_JOINTS])
        if not joints:
            continue
        q_arr = np.stack(joints, axis=0)
        meta = {
            "file_path": file_path.decode("utf-8") if isinstance(file_path, (bytes, bytearray)) else str(file_path),
            "recording_folderpath": (
                recording_folderpath.decode("utf-8")
                if isinstance(recording_folderpath, (bytes, bytearray))
                else str(recording_folderpath)
            ),
            "num_steps": int(q_arr.shape[0]),
        }
        yield key, q_arr, meta
