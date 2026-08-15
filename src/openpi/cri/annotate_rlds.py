"""Build a CRI-annotated RLDS copy of a DROID dataset (default: ``data/droid_100``).

Pipeline (matches IsaacLab reach CRI path + offline finite-diff velocity):

1. Load source RLDS episodes (``observation/joint_position``).
2. Estimate ``qd`` with first step ``0``, then ``(q[t]-q[t-1])/dt``.
3. Compute ``CRI(q, qd)`` via ``CriSolver`` (bundled Panda analysis under ``openpi.cri``).
4. Write a new RLDS dataset with ``observation/cri`` and ``observation/joint_velocity``.

Example::

    uv run --group rlds python -m openpi.cri.annotate_rlds

Creates ``data/droid_100/1.0.0_CRI`` next to the untouched source ``1.0.0``.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
import tyro

from openpi.cri.compute import compute_cri
from openpi.cri.constants import DEFAULT_NUM_JOINTS
from openpi.cri.constants import DROID_CONTROL_DT
from openpi.cri.constants import NUM_CRI_POINTS
from openpi.cri.constants import PACKAGE_DIR
from openpi.cri.solver import CriSolver
from openpi.cri.velocity import joint_velocity_from_positions

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SOURCE = _REPO_ROOT / "data" / "droid_100" / "1.0.0"
# Sibling of the source version dir; original ``1.0.0`` is never modified.
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "droid_100" / "1.0.0_CRI"
_DEFAULT_DATASET_NAME = "droid_100"
_TFDS_VERSION = "1.0.0"  # folder may be ``1.0.0_CRI`` (not a valid TFDS Version string)


def _as_numpy(value: Any) -> Any:
    """Convert TF EagerTensor / nested structures to numpy / python scalars."""
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy()
    if isinstance(value, dict):
        return {k: _as_numpy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_as_numpy(v) for v in value)
    return value


def build_cri_features(source_features: Any) -> Any:
    """Clone source TFDS features and add CRI / finite-diff joint velocity."""
    import tensorflow_datasets as tfds

    step = source_features["steps"]
    obs = step["observation"]
    new_obs = tfds.features.FeaturesDict(
        {
            **{k: obs[k] for k in obs.keys()},
            "joint_velocity": tfds.features.Tensor(
                shape=(DEFAULT_NUM_JOINTS,),
                dtype=np.float64,
                doc="Finite-diff joint velocity from observation/joint_position; qd[0]=0",
            ),
            "cri": tfds.features.Tensor(
                shape=(NUM_CRI_POINTS,),
                dtype=np.float32,
                doc="Collision Risk Index per Panda collision point (clamped, zero-vel filtered)",
            ),
        }
    )
    new_step = tfds.features.FeaturesDict(
        {k: (new_obs if k == "observation" else step[k]) for k in step.keys()}
    )
    return tfds.features.FeaturesDict(
        {
            "episode_metadata": source_features["episode_metadata"],
            "steps": tfds.features.Dataset(new_step),
        }
    )


def compute_episode_cri(
    joint_positions: np.ndarray,
    *,
    solver: Any,
    dt: float = DROID_CONTROL_DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(qd, cri)`` for one episode.

    Args:
        joint_positions: ``(T, 7)`` observation joint positions.
        solver: ``CriSolver`` (or duck-typed stand-in with ``batch_size`` + ``compute``).
        dt: Control timestep seconds.

    Returns:
        ``qd`` float64 ``(T, 7)``, ``cri`` float32 ``(T, 8)``.
    """
    q = np.asarray(joint_positions, dtype=np.float64)
    if q.ndim != 2 or q.shape[-1] < DEFAULT_NUM_JOINTS:
        raise ValueError(f"expected joint_positions (T, >=7), got {q.shape}")
    q = q[:, :DEFAULT_NUM_JOINTS]
    qd = joint_velocity_from_positions(q, dt=dt)

    t_len = q.shape[0]
    batch = int(solver.batch_size)
    cri_chunks: list[np.ndarray] = []
    for start in range(0, t_len, batch):
        end = min(start + batch, t_len)
        cri_chunks.append(compute_cri(q[start:end], qd[start:end], solver=solver))
    cri = np.concatenate(cri_chunks, axis=0).astype(np.float32, copy=False)
    return qd, cri


def materialize_episode(episode_tf: Any) -> dict[str, Any]:
    """Convert a TFDS episode (with nested steps Dataset) into plain numpy dicts."""
    meta = _as_numpy(episode_tf["episode_metadata"])
    steps_ds = episode_tf["steps"]
    if hasattr(steps_ds, "as_numpy_iterator"):
        steps = [_as_numpy(s) for s in steps_ds.as_numpy_iterator()]
    else:
        steps = [_as_numpy(s) for s in steps_ds]
    return {"episode_metadata": meta, "steps": steps}


def annotate_episode_dict(
    episode: dict[str, Any],
    *,
    solver: Any,
    dt: float = DROID_CONTROL_DT,
) -> dict[str, Any]:
    """Attach ``joint_velocity`` and ``cri`` under each step's observation."""
    steps: list[dict[str, Any]] = episode["steps"]
    if not steps:
        return episode

    q = np.stack([np.asarray(s["observation"]["joint_position"], dtype=np.float64) for s in steps], axis=0)
    qd, cri = compute_episode_cri(q, solver=solver, dt=dt)

    new_steps: list[dict[str, Any]] = []
    for t, step in enumerate(steps):
        obs = dict(step["observation"])
        obs["joint_velocity"] = qd[t]
        obs["cri"] = cri[t]
        new_step = dict(step)
        new_step["observation"] = obs
        new_steps.append(new_step)

    return {"episode_metadata": episode["episode_metadata"], "steps": new_steps}


def iter_source_episodes(source_dir: Path, *, split: str = "train") -> Iterator[dict[str, Any]]:
    """Yield materialized numpy episodes from an on-disk RLDS directory."""
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(source_dir.resolve()))
    ds = builder.as_dataset(split=split)
    for episode_tf in ds:
        yield materialize_episode(episode_tf)


def _shard_paths(output_dir: Path, *, dataset_name: str, split: str, num_shards: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    width = max(5, len(str(num_shards)))
    paths: list[Path] = []
    for i in range(num_shards):
        name = f"{dataset_name}-{split}.tfrecord-{i:0{width}d}-of-{num_shards:0{width}d}"
        paths.append(output_dir / name)
    return paths


def write_annotated_rlds(
    *,
    source_dir: Path,
    output_dir: Path,
    solver: Any,
    dataset_name: str = _DEFAULT_DATASET_NAME,
    split: str = "train",
    dt: float = DROID_CONTROL_DT,
    num_shards: int | None = None,
    overwrite: bool = False,
    max_episodes: int | None = None,
    tfds_version: str = _TFDS_VERSION,
) -> Path:
    """Read source RLDS, annotate with CRI, write a new RLDS folder dataset.

    By default writes to ``data/droid_100/1.0.0_CRI`` and leaves ``1.0.0`` untouched.

    Returns:
        Path to the written directory (e.g. ``.../droid_100/1.0.0_CRI``).
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source RLDS dir not found: {source_dir}")
    if output_dir == source_dir:
        raise ValueError("output_dir must differ from source_dir (refusing to overwrite source)")

    src_builder = tfds.builder_from_directory(str(source_dir))
    features = build_cri_features(src_builder.info.features)
    n_examples = int(src_builder.info.splits[split].num_examples)
    if max_episodes is not None:
        n_examples = min(n_examples, int(max_episodes))
    if num_shards is None:
        num_shards = max(1, int(src_builder.info.splits[split].num_shards))
    # Avoid empty shards when max_episodes is small.
    num_shards = max(1, min(int(num_shards), n_examples))

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output exists: {output_dir} (pass --overwrite to replace the CRI folder only)"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = _shard_paths(output_dir, dataset_name=dataset_name, split=split, num_shards=num_shards)
    writers = [tf.io.TFRecordWriter(str(p)) for p in shard_paths]
    shard_lengths = [0] * num_shards
    written = 0

    try:
        for ep_idx, episode in enumerate(
            tqdm.tqdm(iter_source_episodes(source_dir, split=split), total=n_examples, desc="annotate CRI")
        ):
            if ep_idx >= n_examples:
                break
            annotated = annotate_episode_dict(episode, solver=solver, dt=dt)
            payload = features.serialize_example(annotated)
            shard_i = ep_idx % num_shards
            writers[shard_i].write(payload)
            shard_lengths[shard_i] += 1
            written += 1
    finally:
        for w in writers:
            w.close()

    if written == 0:
        raise RuntimeError(f"no episodes written under {output_dir}")

    # Folder name may be ``1.0.0_CRI`` (invalid TFDS Version); keep metadata version semver.
    tfds.folder_dataset.write_metadata(
        data_dir=str(output_dir),
        features=features,
        split_infos=None,
        version=tfds_version,
        description=(
            f"CRI-annotated DROID RLDS derived from {source_dir}. "
            f"observation/joint_velocity is finite-diff (dt={dt}, first step 0); "
            f"observation/cri is Safetics CRI ({NUM_CRI_POINTS} points)."
        ),
    )
    logger.info("Wrote %d episodes → %s", written, output_dir)
    return output_dir


@dataclasses.dataclass
class Args:
    """CLI for annotating DROID RLDS with CRI."""

    source_dir: Path = _DEFAULT_SOURCE
    """Source RLDS version directory (contains dataset_info.json / tfrecords)."""

    output_dir: Path = _DEFAULT_OUTPUT
    """Output directory (default: sibling ``1.0.0_CRI``; source ``1.0.0`` is left intact)."""

    dataset_name: str = _DEFAULT_DATASET_NAME
    """TFDS dataset name used in shard filenames."""

    split: str = "train"
    dt: float = DROID_CONTROL_DT
    """Finite-difference timestep (seconds). Default: DROID 15 Hz."""

    analysis_dir: Path = PACKAGE_DIR
    """Safetics analysis root (lib/, Engine/, ST_AnalysisInfo.json)."""

    batch_size: int = 256
    """CRI solver batch size (padded internally)."""

    num_shards: int | None = None
    """Output shard count (default: match source)."""

    max_episodes: int | None = None
    """Optional cap for debugging (e.g. 2)."""

    overwrite: bool = False
    """Only replaces ``output_dir`` (never the source ``1.0.0`` folder)."""

    warmup_rounds: int = 5
    mock_cri: bool = False
    """If True, skip CUDA solver and write zeros for CRI (RLDS I/O smoke test only)."""


class _MockCriSolver:
    """CPU stand-in that returns zeros (still runs zero-vel filter via compute_cri)."""

    def __init__(self, batch_size: int = 256) -> None:
        self.batch_size = int(batch_size)

    def compute(self, q: Any, qd: Any, *, return_torch: bool = False) -> np.ndarray:
        del return_torch, qd
        batch = int(np.asarray(q).reshape(-1, np.asarray(q).shape[-1]).shape[0])
        return np.zeros((batch, NUM_CRI_POINTS), dtype=np.float32)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.mock_cri:
        logger.warning("mock_cri=True: writing zero CRI (no CUDA solver)")
        solver: Any = _MockCriSolver(batch_size=args.batch_size)
    else:
        solver = CriSolver(
            analysis_dir=args.analysis_dir,
            batch_size=args.batch_size,
            num_joints=DEFAULT_NUM_JOINTS,
            warmup_rounds=args.warmup_rounds,
        )
    out = write_annotated_rlds(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        solver=solver,
        dataset_name=args.dataset_name,
        split=args.split,
        dt=args.dt,
        num_shards=args.num_shards,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        tfds_version=_TFDS_VERSION,
    )
    print(f"Done: {out}")


def entrypoint(argv: Sequence[str] | None = None) -> None:
    main(tyro.cli(Args, args=argv))


if __name__ == "__main__":
    entrypoint()
