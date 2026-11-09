"""Full-weight fine-tune of pi05_base on local DROID 1.0.1 (no LoRA, no CRI).

Fixed locations::

    DROID RLDS : /media/safetics/D/droid/1.0.1
    pretrained : <E>/openpi/pretrained/pi05_base
    checkpoints: <E>/openpi/checkpoints/pi05_full_droid_finetune/<exp_name>
    assets     : <E>/openpi/assets/pi05_full_droid_finetune
    health log : <E>/openpi/logs/train_health_<exp_name>.csv

``<E>`` resolves to ``/mnt/E`` when mounted there, else legacy ``/media/safetics/E``.

Official recipe is 100k steps / bs256 on 8x H100. This helper defaults to the
single-GPU knobs in ``pi05_full_droid_finetune`` (bs4, shuffle 32k).

Usage (from repo root)::

    uv run --group rlds scripts/train_pi05_full_droid.py \\
        --exp-name=droid_full --overwrite --monitor

    # Resume
    uv run --group rlds scripts/train_pi05_full_droid.py \\
        --exp-name=droid_full --resume --skip-norm-stats --monitor
"""

from __future__ import annotations

import dataclasses
import os
import signal
import subprocess
from pathlib import Path

import tyro

import openpi.training.config as _train_config

DROID_RLDS_DIR = _train_config._DROID_RLDS_DIR
CHECKPOINT_BASE_DIR = _train_config._CHECKPOINT_OUTPUT_DIR
PRETRAINED_DIR = _train_config._PRETRAINED_PI05_BASE_PARAMS.parent
PRETRAINED_PARAMS = _train_config._PRETRAINED_PI05_BASE_PARAMS
PRETRAINED_GS = "gs://openpi-assets/checkpoints/pi05_base"
CACHED_PRETRAINED = Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base"
CONFIG_NAME = "pi05_full_droid_finetune"
E_ROOT = _train_config._E_ROOT
HEALTH_LOG_DIR = E_ROOT / "openpi" / "logs"

_DEFAULT_BATCH_SIZE = 4
_DEFAULT_SHUFFLE_BUFFER = 32_000


@dataclasses.dataclass
class Args:
    exp_name: str = "droid_full"
    overwrite: bool = False
    resume: bool = False
    skip_norm_stats: bool = True
    """Reuse official pi05_base DROID norm stats (default)."""
    max_norm_frames: int = 50_000
    """Only used if --skip-norm-stats is unset."""
    norm_batch_size: int = 4
    batch_size: int = _DEFAULT_BATCH_SIZE
    """Global train batch. Full FT+EMA; bs4 is the 96GiB starting point."""
    rlds_shuffle_buffer_size: int = _DEFAULT_SHUFFLE_BUFFER
    """RLDS image shuffle buffer. 250k OOM'd host (~125GiB)."""
    config_name: str = CONFIG_NAME
    mem_fraction: str = "0.85"
    monitor: bool = False
    monitor_interval_sec: float = 5.0
    num_train_steps: int | None = None


def _ensure_pretrained(*, download_if_missing: bool = True) -> None:
    marker = PRETRAINED_PARAMS / "_CHECKPOINT_METADATA"
    if marker.is_file():
        print(f"[ok] pretrained : {PRETRAINED_PARAMS}")
        return

    if not download_if_missing:
        raise FileNotFoundError(
            f"Missing pretrained pi05_base at {PRETRAINED_PARAMS}.\n"
            f"Copy or download with:\n"
            f"  mkdir -p {PRETRAINED_DIR}\n"
            f"  gsutil -m cp -r {PRETRAINED_GS}/* {PRETRAINED_DIR}/"
        )

    if not PRETRAINED_DIR.parent.is_dir():
        raise FileNotFoundError(
            f"E-drive mount missing for pretrained weights: {PRETRAINED_DIR.parent}"
        )
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    cached_marker = CACHED_PRETRAINED / "params" / "_CHECKPOINT_METADATA"
    if cached_marker.is_file():
        print(f"[copy] {CACHED_PRETRAINED} → {PRETRAINED_DIR}")
        subprocess.run(
            ["rsync", "-a", f"{CACHED_PRETRAINED}/", f"{PRETRAINED_DIR}/"],
            check=True,
        )
    else:
        print(f"[download] {PRETRAINED_GS} → {PRETRAINED_DIR} (~12 GiB)")
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{PRETRAINED_GS.rstrip('/')}/*", str(PRETRAINED_DIR)],
            check=True,
        )

    if not marker.is_file():
        raise FileNotFoundError(
            f"Prefetch finished but marker missing: {marker}. Check {PRETRAINED_DIR}"
        )
    print(f"[ok] pretrained : {PRETRAINED_PARAMS}")


def _ensure_checkpoint_symlink(repo: Path) -> None:
    link = repo / "data" / "checkpoints" / CONFIG_NAME
    target = CHECKPOINT_BASE_DIR / CONFIG_NAME
    target.mkdir(parents=True, exist_ok=True)
    desired = str(target)
    if link.is_symlink():
        if os.readlink(link) == desired:
            return
        link.unlink()
    elif link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(desired)
    print(f"[ok] symlink   : {link} → {desired}")


def _require_datasets(repo: Path) -> None:
    if not DROID_RLDS_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing DROID RLDS at {DROID_RLDS_DIR}. Expected D-drive dataset."
        )
    if not (DROID_RLDS_DIR / "dataset_info.json").is_file():
        raise FileNotFoundError(f"Incomplete RLDS (no dataset_info.json): {DROID_RLDS_DIR}")

    if not CHECKPOINT_BASE_DIR.parent.is_dir():
        raise FileNotFoundError(
            f"E-drive mount missing for checkpoints: {CHECKPOINT_BASE_DIR.parent}"
        )
    CHECKPOINT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    (E_ROOT / "openpi" / "assets").mkdir(parents=True, exist_ok=True)
    HEALTH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_checkpoint_symlink(repo)

    print(f"[ok] DROID RLDS : {DROID_RLDS_DIR}")
    print(f"[ok] checkpoints: {CHECKPOINT_BASE_DIR}/{CONFIG_NAME}/…")
    print(f"[ok] assets     : {E_ROOT / 'openpi' / 'assets'}")
    print(f"[ok] health log : {HEALTH_LOG_DIR}")


def main(args: Args) -> None:
    repo = Path(__file__).resolve().parent.parent
    _ensure_pretrained()
    _require_datasets(repo)

    if not args.skip_norm_stats:
        print(
            f"[1/2] Computing norm stats for {args.config_name} "
            f"(max_frames={args.max_norm_frames})..."
        )
        subprocess.run(
            [
                "uv",
                "run",
                "--group",
                "rlds",
                "scripts/compute_norm_stats.py",
                f"--config-name={args.config_name}",
                f"--max-frames={args.max_norm_frames}",
                f"--batch-size={args.norm_batch_size}",
            ],
            cwd=repo,
            check=True,
        )
    else:
        print("[1/2] Skipping norm stats (official pi05_base assets)")

    train_cmd = [
        "uv",
        "run",
        "--group",
        "rlds",
        "scripts/train.py",
        args.config_name,
        f"--exp-name={args.exp_name}",
        f"--batch-size={args.batch_size}",
        f"--data.rlds-shuffle-buffer-size={args.rlds_shuffle_buffer_size}",
    ]
    if args.overwrite:
        train_cmd.append("--overwrite")
    if args.resume:
        train_cmd.append("--resume")
    if args.num_train_steps is not None:
        train_cmd.append(f"--num-train-steps={args.num_train_steps}")

    print(f"[2/2] Training: {' '.join(train_cmd)}")
    print(
        f"  mem knobs: batch_size={args.batch_size}, "
        f"rlds_shuffle_buffer_size={args.rlds_shuffle_buffer_size}, "
        f"XLA_MEM_FRACTION={args.mem_fraction}, "
        f"num_train_steps={args.num_train_steps or 'config-default'}"
    )
    print("  recipe: full-weight pi05_base → DROID 1.0.1 (no LoRA, no CRI)")
    print(f"  outputs: {CHECKPOINT_BASE_DIR / CONFIG_NAME / args.exp_name}")

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", args.mem_fraction)

    monitor_proc: subprocess.Popen | None = None
    code = 1
    try:
        train_proc = subprocess.Popen(train_cmd, cwd=repo, env=env)
        if args.monitor:
            mon_cmd = [
                "uv",
                "run",
                "python",
                "scripts/monitor_train_health.py",
                f"--interval-sec={args.monitor_interval_sec}",
                f"--run-name={args.exp_name}",
                f"--pid={train_proc.pid}",
                f"--log-dir={HEALTH_LOG_DIR}",
            ]
            print(f"[monitor] health logger → {HEALTH_LOG_DIR}/train_health_{args.exp_name}.csv")
            monitor_proc = subprocess.Popen(mon_cmd, cwd=repo)
        code = train_proc.wait()
    finally:
        if monitor_proc is not None and monitor_proc.poll() is None:
            monitor_proc.send_signal(signal.SIGTERM)
            try:
                monitor_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor_proc.kill()
    raise SystemExit(code)


if __name__ == "__main__":
    main(tyro.cli(Args))
