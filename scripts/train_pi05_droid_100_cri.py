"""Smoke / debug fine-tune of pi05_base on ``data/droid_100/1.0.0_CRI``.

Sibling entry point to::

    uv run scripts/train.py pi05_droid_finetune --exp-name=my_experiment --overwrite

CRI is encoded as a separate discrete LLM span (not continuous state)::

    Task: <lang>, State: <bins>, CRI: <9 bins>;
    Action:

Usage (from repo root)::

    # 1) Compute norm stats (required once; proprio state only — CRI is tokenized)
    uv run --group rlds scripts/compute_norm_stats.py \\
        --config-name pi05_droid_100_cri_finetune --max-frames 5000

    # 2) Short training run (override dataset location when needed)
    uv run --group rlds scripts/train_pi05_droid_100_cri.py \\
        --exp-name=my_experiment --overwrite \\
        --rlds-builder-dir=/media/safetics/D/droid_100/1.0.0_CRI

Or call ``scripts/train.py`` directly::

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run --group rlds scripts/train.py \\
        pi05_droid_100_cri_finetune --exp-name=my_experiment --overwrite \\
        --rlds-builder-dir=/media/safetics/D/droid_100/1.0.0_CRI
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from pathlib import Path

import tyro


@dataclasses.dataclass
class Args:
    exp_name: str = "my_experiment"
    overwrite: bool = False
    resume: bool = False
    skip_norm_stats: bool = False
    """Skip compute_norm_stats (use existing assets)."""
    max_norm_frames: int = 5000
    config_name: str = "pi05_droid_100_cri_finetune"
    # Optional RLDS location overrides (forwarded to compute_norm_stats + train.py).
    rlds_data_dir: str | None = None
    """Parent TFDS data dir (used when builder_dir is unset)."""
    rlds_builder_dir: str | None = None
    """Absolute path to an RLDS version folder (e.g. .../droid_100/1.0.0_CRI)."""


def main(args: Args) -> None:
    repo = Path(__file__).resolve().parent.parent
    cri_dir = (
        Path(args.rlds_builder_dir).expanduser()
        if args.rlds_builder_dir
        else repo / "data" / "droid_100" / "1.0.0_CRI"
    )
    if not cri_dir.is_dir():
        raise FileNotFoundError(
            f"Missing CRI RLDS at {cri_dir}. Run:\n"
            "  uv run --group rlds python -m openpi.cri.annotate_rlds\n"
            "Or pass --rlds-builder-dir=/path/to/1.0.0_CRI"
        )

    path_flags: list[str] = []
    if args.rlds_data_dir is not None:
        path_flags.append(f"--rlds-data-dir={args.rlds_data_dir}")
    if args.rlds_builder_dir is not None:
        path_flags.append(f"--rlds-builder-dir={args.rlds_builder_dir}")

    if not args.skip_norm_stats:
        print(f"[1/2] Computing norm stats for {args.config_name} (max_frames={args.max_norm_frames})...")
        subprocess.run(
            [
                "uv",
                "run",
                "--group",
                "rlds",
                "scripts/compute_norm_stats.py",
                f"--config-name={args.config_name}",
                f"--max-frames={args.max_norm_frames}",
                *path_flags,
            ],
            cwd=repo,
            check=True,
        )
    else:
        print("[1/2] Skipping norm stats (--skip-norm-stats)")

    train_cmd = [
        "uv",
        "run",
        "--group",
        "rlds",
        "scripts/train.py",
        args.config_name,
        f"--exp-name={args.exp_name}",
        *path_flags,
    ]
    if args.overwrite:
        train_cmd.append("--overwrite")
    if args.resume:
        train_cmd.append("--resume")

    print(f"[2/2] Training: {' '.join(train_cmd)}")
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    raise SystemExit(subprocess.call(train_cmd, cwd=repo, env=env))


if __name__ == "__main__":
    main(tyro.cli(Args))
