"""Annotate a large DROID RLDS tree with a CRI-F *sidecar* (no full copy).

Uses IsaacLab ``run_cri_filter`` to read delayed ``cri_pre``. Dataset generation
keeps CBF-QP **off** (``filter_enabled=False``) and records the CRI hard limit
0.96. Stores ``cri_pre``, ``delta=||qd_cmd-qd_RL||``, and finite-diff
``joint_velocity``.

Example::

    # Smoke (2 episodes)
    uv run --group rlds python -m openpi.cri.annotate_rlds_addon \\
        --source-dir=/media/safetics/D/droid/1.0.1 \\
        --addon-dir=/mnt/E/droid_1.0.1_cri_f_smoke \\
        --max-episodes=2

    # Full DROID (resumable; filter off, cri_limit=0.96)
    uv run --group rlds python -m openpi.cri.annotate_rlds_addon \\
        --source-dir=/media/safetics/D/droid/1.0.1 \\
        --addon-dir=/mnt/E/droid_1.0.1_cri_addon_v3

    # PolaRiS cotrain RLDS (add delayed cri_pre next to the original 1.0.0)
    uv run --group rlds python -m openpi.cri.annotate_rlds_addon \\
        --source-dir=/mnt/E/rlds/polaris_droid_cotrain_dataset/1.0.0 \\
        --addon-dir=/mnt/E/rlds/polaris_droid_cotrain_dataset_cri_addon_v3
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
import tyro

from openpi.cri.addon import META_NAME
from openpi.cri.addon import CriAddonWriter
from openpi.cri.addon import iter_source_joint_episodes
from openpi.cri.constants import CBF_ALPHA
from openpi.cri.constants import CRI_ADDON_FORMAT_V3
from openpi.cri.constants import CRI_FILTER_LIMIT
from openpi.cri.constants import DEFAULT_NUM_JOINTS
from openpi.cri.constants import DROID_CONTROL_DT
from openpi.cri.constants import NUM_CRI_POINTS
from openpi.cri.constants import PACKAGE_DIR
from openpi.cri.filter import compute_episode_cri_f
from openpi.cri.solver import CriSolver

logger = logging.getLogger(__name__)

_DEFAULT_SOURCE = Path("/media/safetics/D/droid/1.0.1")
_DEFAULT_ADDON = Path("/mnt/E/droid_1.0.1_cri_addon_v3")


@dataclasses.dataclass
class Args:
    source_dir: Path = _DEFAULT_SOURCE
    """Source RLDS version directory (untouched)."""

    addon_dir: Path = _DEFAULT_ADDON
    """Sidecar output directory (cri/delta/qd bins + episodes.jsonl)."""

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
    cri_limit: float = CRI_FILTER_LIMIT
    """IsaacLab CRI hard limit (default 0.96)."""

    cbf_alpha: float = CBF_ALPHA
    """CBF boundary coefficient α (default 0.02)."""

    filter_enabled: bool = False
    """CBF-QP filter (``set_cri_filter_enabled``). Off for dataset annotation."""

    mock_cri: bool = False
    """If True, write zeros for CRI and delta (I/O smoke only)."""


def _refuse_incompatible_addon(addon_dir: Path) -> None:
    """Refuse to resume into a pre-v3 sidecar (Newton scale / no delta)."""
    meta_path = addon_dir / META_NAME
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != CRI_ADDON_FORMAT_V3:
        raise ValueError(
            f"Existing addon {addon_dir} has format={fmt!r}; expected {CRI_ADDON_FORMAT_V3}. "
            "Use a new --addon-dir (default droid_1.0.1_cri_addon_v3) or pass --overwrite."
        )


class _MockCriFilterSolver:
    """CPU stand-in that returns zero cri_pre and qd_cmd=qd_RL."""

    def __init__(self, batch_size: int = 256, cri_limit: float = CRI_FILTER_LIMIT) -> None:
        self.batch_size = int(batch_size)
        self._cri_limit = float(cri_limit)

    def run_cri_filter(self, q: Any, qd: Any) -> dict[str, Any]:
        batch = int(np.asarray(q).reshape(-1, np.asarray(q).shape[-1]).shape[0])
        qd_rl = np.asarray(qd, dtype=np.float32).reshape(batch, -1)
        return {
            "cri_pre": np.zeros((batch, NUM_CRI_POINTS), dtype=np.float32),
            "qd_cmd": qd_rl,
            "delta": np.zeros((batch,), dtype=np.float32),
            "cri_limit": self._cri_limit,
            "cbf_alpha": CBF_ALPHA,
            "approach_limit": self._cri_limit * (1.0 - CBF_ALPHA),
            "enabled": False,
        }


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
    cri_limit: float = CRI_FILTER_LIMIT,
    cbf_alpha: float = CBF_ALPHA,
    filter_enabled: bool = False,
) -> Path:
    import tensorflow_datasets as tfds

    source_dir = source_dir.expanduser().resolve()
    addon_dir = addon_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source RLDS dir not found: {source_dir}")

    if overwrite and addon_dir.exists():
        logger.warning("Removing existing addon_dir=%s", addon_dir)
        shutil.rmtree(addon_dir)
    else:
        _refuse_incompatible_addon(addon_dir)

    builder = tfds.builder_from_directory(str(source_dir))
    n_source = int(builder.info.splits[split].num_examples)
    n_target = n_source if max_episodes is None else min(n_source, int(max_episodes))

    writer = CriAddonWriter(
        addon_dir,
        source_dir=source_dir,
        dt=dt,
        cri_limit=cri_limit,
        cbf_alpha=cbf_alpha,
        filter_enabled=filter_enabled,
    )
    skipped = 0
    processed = 0
    try:
        pbar = tqdm.tqdm(
            iter_source_joint_episodes(source_dir, split=split),
            total=n_target,
            desc="CRI-F addon",
        )
        for key, q, _meta in pbar:
            if processed >= n_target:
                break
            processed += 1
            if resume and writer.has(key):
                skipped += 1
                pbar.set_postfix(new=writer.written_this_session, skip=skipped)
                continue
            qd, cri, delta, _qd_cmd = compute_episode_cri_f(
                q, solver=solver, dt=dt, cri_limit=cri_limit, cbf_alpha=cbf_alpha
            )
            writer.append(key, cri=cri, qd=qd.astype("float32", copy=False), delta=delta)
            pbar.set_postfix(new=writer.written_this_session, skip=skipped, steps=int(q.shape[0]))
    finally:
        total_eps = len(writer.existing)
        new_eps = writer.written_this_session
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
        solver: Any = _MockCriFilterSolver(batch_size=args.batch_size, cri_limit=args.cri_limit)
    else:
        solver = CriSolver(
            analysis_dir=args.analysis_dir,
            batch_size=args.batch_size,
            num_joints=DEFAULT_NUM_JOINTS,
            warmup_rounds=args.warmup_rounds,
            cri_filter=True,
            cri_limit=args.cri_limit,
            cbf_alpha=args.cbf_alpha,
            filter_enabled=args.filter_enabled,
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
        cri_limit=args.cri_limit,
        cbf_alpha=args.cbf_alpha,
        filter_enabled=args.filter_enabled,
    )
    print(f"Done: {out}")


def entrypoint(argv: Sequence[str] | None = None) -> None:
    main(tyro.cli(Args, args=argv))


if __name__ == "__main__":
    entrypoint()
