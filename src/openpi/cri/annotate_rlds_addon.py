"""Annotate a large DROID RLDS tree with a CRI *sidecar* (no full copy).

Use this when the dataset is too large to duplicate (e.g. ``/media/.../droid/1.0.1``).
Writes only ``cri`` + ``joint_velocity`` under ``--addon-dir`` (default: under the
repo ``data/`` tree on the system disk — the DROID drive is often full).

Example::

    # Smoke (2 episodes)
    uv run --group rlds python -m openpi.cri.annotate_rlds_addon \\
        --source-dir=/media/safetics/D/droid/1.0.1 \\
        --addon-dir=/home/safetics/openpi/data/droid_1.0.1_cri_addon \\
        --max-episodes=2

    # Full run (resumable)
    uv run --group rlds python -m openpi.cri.annotate_rlds_addon \\
        --source-dir=/media/safetics/D/droid/1.0.1 \\
        --addon-dir=/home/safetics/openpi/data/droid_1.0.1_cri_addon
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import tqdm
import tyro

from openpi.cri.addon import CriAddonWriter
from openpi.cri.addon import iter_source_joint_episodes
from openpi.cri.annotate_rlds import _MockCriSolver
from openpi.cri.annotate_rlds import compute_episode_cri
from openpi.cri.constants import DEFAULT_NUM_JOINTS
from openpi.cri.constants import DROID_CONTROL_DT
from openpi.cri.constants import PACKAGE_DIR
from openpi.cri.solver import CriSolver

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SOURCE = Path("/media/safetics/D/droid/1.0.1")
_DEFAULT_ADDON = _REPO_ROOT / "data" / "droid_1.0.1_cri_addon"


@dataclasses.dataclass
class Args:
    source_dir: Path = _DEFAULT_SOURCE
    """Source RLDS version directory (untouched)."""

    addon_dir: Path = _DEFAULT_ADDON
    """Sidecar output directory (cri/qd bins + episodes.jsonl). Prefer a disk with free space."""

    split: str = "train"
    dt: float = DROID_CONTROL_DT
    analysis_dir: Path = PACKAGE_DIR
    batch_size: int = 256
    max_episodes: int | None = None
    """Optional cap for debugging."""

    overwrite: bool = False
    """Delete existing addon_dir before writing (disables resume)."""

    resume: bool = True
    """Skip episode keys already present in addon_dir/episodes.jsonl."""

    warmup_rounds: int = 5
    mock_cri: bool = False
    """If True, write zeros for CRI (I/O smoke only)."""


def write_cri_addon(
    *,
    source_dir: Path,
    addon_dir: Path,
    solver: Any,
    split: str = "train",
    dt: float = DROID_CONTROL_DT,
    max_episodes: int | None = None,
    overwrite: bool = False,
    resume: bool = True,
) -> Path:
    import tensorflow_datasets as tfds

    source_dir = source_dir.expanduser().resolve()
    addon_dir = addon_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source RLDS dir not found: {source_dir}")

    if overwrite and addon_dir.exists():
        logger.warning("Removing existing addon_dir=%s", addon_dir)
        shutil.rmtree(addon_dir)

    builder = tfds.builder_from_directory(str(source_dir))
    n_source = int(builder.info.splits[split].num_examples)
    n_target = n_source if max_episodes is None else min(n_source, int(max_episodes))

    writer = CriAddonWriter(addon_dir, source_dir=source_dir, dt=dt)
    skipped = 0
    processed = 0
    try:
        pbar = tqdm.tqdm(
            iter_source_joint_episodes(source_dir, split=split),
            total=n_target,
            desc="CRI addon",
        )
        for key, q, _meta in pbar:
            if processed >= n_target:
                break
            processed += 1
            if resume and writer.has(key):
                skipped += 1
                pbar.set_postfix(new=writer.written_this_session, skip=skipped)
                continue
            qd, cri = compute_episode_cri(q, solver=solver, dt=dt)
            writer.append(key, cri=cri, qd=qd.astype("float32", copy=False))
            pbar.set_postfix(new=writer.written_this_session, skip=skipped, steps=int(q.shape[0]))
    finally:
        total_eps = len(writer.existing)
        new_eps = writer.written_this_session
        # Full corpus only: scanned every source episode and stored every key.
        finished = max_episodes is None and processed >= n_source and total_eps >= n_source
        writer.close(finished=finished, num_source_episodes=n_source)

    logger.info(
        "Addon update done: addon=%s total_episodes=%d new=%d skipped=%d scanned=%d/%d finished=%s",
        addon_dir,
        total_eps,
        new_eps,
        skipped,
        processed,
        n_target,
        finished,
    )
    return addon_dir


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
    out = write_cri_addon(
        source_dir=args.source_dir,
        addon_dir=args.addon_dir,
        solver=solver,
        split=args.split,
        dt=args.dt,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(f"Done: {out}")


def entrypoint(argv: Sequence[str] | None = None) -> None:
    main(tyro.cli(Args, args=argv))


if __name__ == "__main__":
    entrypoint()
