"""Fine-tune pi05_droid on full DROID + CRI sidecar (TacVLA-style modality tokens).

Fixed dataset locations (do not copy the 1.7T RLDS)::

    DROID RLDS : /media/safetics/D/droid/1.0.1
    CRI addon  : <E>/droid_1.0.1_cri_addon
    pretrained : <E>/openpi/pretrained/pi05_droid  (~12 GiB init)

Training outputs::

    checkpoints : <E>/openpi/checkpoints/pi05_droid_cri_finetune/<exp_name>
    norm stats  : data/assets/pi05_droid_cri_finetune/droid_cri
    (repo ``data/checkpoints/...`` is a symlink to E for convenience)

``<E>`` resolves to ``/mnt/E`` when mounted there, else legacy ``/media/safetics/E``.

CRI is encoded as a separate discrete LLM span (not continuous proprio state)::

    Task: <lang>, State: <bins>, CRI: <9 bins>;
    Action:

This mirrors TacVLA (arXiv:2603.12665) at the training-recipe level:
pi05_droid + LoRA + compact non-vision tokens fused in the VLM prefix.

Usage (from repo root)::

    # One-shot: norm stats + train
    uv run --group rlds scripts/train_pi05_droid_cri.py \\
        --exp-name=droid_cri_lora --overwrite

    # Or step-by-step
    uv run --group rlds scripts/compute_norm_stats.py \\
        --config-name pi05_droid_cri_finetune --max-frames 200000

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run --group rlds scripts/train.py \\
        pi05_droid_cri_finetune --exp-name=droid_cri_lora --overwrite

    # Resume (checkpoints live on E: — clean tmp there if needed)
    rm -rf /mnt/E/openpi/checkpoints/pi05_droid_cri_finetune/droid_cri_lora/*.orbax-checkpoint-tmp-*
    uv run --group rlds scripts/train_pi05_droid_cri.py \\
        --exp-name=droid_cri_lora --resume --skip-norm-stats --monitor \\
        --batch-size=8 --rlds-shuffle-buffer-size=32000
    # Optional: override length, e.g. --num-train-steps=50000

    # After unexpected reboot:
    .venv/bin/python scripts/diagnose_reboot.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
from pathlib import Path

import tyro

# Keep path resolution in sync with openpi.training.config (D: RLDS, E: CRI + ckpts).
import openpi.training.config as _train_config

DROID_RLDS_DIR = _train_config._DROID_RLDS_DIR
CRI_ADDON_DIR = _train_config._CRI_ADDON_DIR
CHECKPOINT_BASE_DIR = _train_config._CHECKPOINT_OUTPUT_DIR
PRETRAINED_DIR = _train_config._PRETRAINED_PI05_DROID_PARAMS.parent
PRETRAINED_PARAMS = _train_config._PRETRAINED_PI05_DROID_PARAMS
PRETRAINED_GS = "gs://openpi-assets/checkpoints/pi05_droid"
CONFIG_NAME = "pi05_droid_cri_finetune"

# Sized from host OOM (~121GiB RSS at shuffle≈250k) and successful bs=8 / shuffle=50k run.
_DEFAULT_BATCH_SIZE = 8
_DEFAULT_SHUFFLE_BUFFER = 32_000


@dataclasses.dataclass
class Args:
    exp_name: str = "droid_cri_lora"
    overwrite: bool = False
    resume: bool = False
    skip_norm_stats: bool = False
    """Skip compute_norm_stats (reuse existing assets)."""
    max_norm_frames: int = 50_000
    """Frames to sample when computing state/action norm stats (keep modest to avoid host OOM)."""
    norm_batch_size: int = 4
    """Batch size for compute_norm_stats only (training batch is unchanged)."""
    batch_size: int = _DEFAULT_BATCH_SIZE
    """Global train batch. bs16 OOM'd on restore; bs8 was stable on RTX PRO 6000."""
    rlds_shuffle_buffer_size: int = _DEFAULT_SHUFFLE_BUFFER
    """RLDS image shuffle buffer. 250k OOM'd host (~125GiB); 32k leaves checkpoint headroom."""
    config_name: str = CONFIG_NAME
    mem_fraction: str = "0.85"
    """XLA_PYTHON_CLIENT_MEM_FRACTION for training."""
    monitor: bool = False
    """Sample GPU/host health to data/logs/train_health_*.csv (for reboot forensics)."""
    monitor_interval_sec: float = 5.0
    """Health sampling period when --monitor is set."""
    num_train_steps: int | None = None
    """Override TrainConfig.num_train_steps (default: config value, currently 30k)."""


def _ensure_pretrained(*, download_if_missing: bool = True) -> None:
    """Keep pi05_droid init weights on E: so training does not fill ~/.cache (~12 GiB)."""
    marker = PRETRAINED_PARAMS / "_CHECKPOINT_METADATA"
    if marker.is_file():
        print(f"[ok] pretrained : {PRETRAINED_PARAMS}")
        return

    if not download_if_missing:
        raise FileNotFoundError(
            f"Missing pretrained pi05_droid at {PRETRAINED_PARAMS}.\n"
            f"Download with:\n"
            f"  mkdir -p {PRETRAINED_DIR}\n"
            f"  gsutil -m cp -r {PRETRAINED_GS}/* {PRETRAINED_DIR}/"
        )

    if not PRETRAINED_DIR.parent.is_dir():
        raise FileNotFoundError(
            f"E-drive mount missing for pretrained weights: {PRETRAINED_DIR.parent}"
        )
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] {PRETRAINED_GS} → {PRETRAINED_DIR} (~12 GiB)")
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", f"{PRETRAINED_GS.rstrip('/')}/*", str(PRETRAINED_DIR)],
        check=True,
    )
    if not marker.is_file():
        raise FileNotFoundError(
            f"Download finished but marker missing: {marker}. Check {PRETRAINED_DIR}"
        )
    print(f"[ok] pretrained : {PRETRAINED_PARAMS}")


def _ensure_checkpoint_symlink(repo: Path) -> None:
    """Keep data/checkpoints/<config> → <E>/openpi/checkpoints/<config> in sync."""
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

    if not CRI_ADDON_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing CRI addon at {CRI_ADDON_DIR}. Expected E-drive sidecar.\n"
            "Build with:\n"
            "  uv run --group rlds python -m openpi.cri.annotate_rlds_addon \\\n"
            f"    --source-dir={DROID_RLDS_DIR} --addon-dir={CRI_ADDON_DIR}"
        )
    meta_path = CRI_ADDON_DIR / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"CRI addon meta missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("finished", False):
        raise RuntimeError(
            f"CRI addon is not marked finished: {meta_path}\n"
            f"meta={meta}"
        )
    for name in ("cri.f32", "joint_velocity.f32", "episodes.jsonl"):
        if not (CRI_ADDON_DIR / name).is_file():
            raise FileNotFoundError(f"CRI addon incomplete, missing {name} under {CRI_ADDON_DIR}")

    if not CHECKPOINT_BASE_DIR.parent.is_dir():
        raise FileNotFoundError(
            f"E-drive mount missing for checkpoints: {CHECKPOINT_BASE_DIR.parent}"
        )
    CHECKPOINT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_checkpoint_symlink(repo)

    print(f"[ok] DROID RLDS : {DROID_RLDS_DIR}")
    print(f"[ok] CRI addon  : {CRI_ADDON_DIR} (episodes={meta.get('num_episodes')})")
    print(f"[ok] checkpoints: {CHECKPOINT_BASE_DIR}/…")
    print("[ok] assets     : data/assets/…")


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
        print("[1/2] Skipping norm stats (--skip-norm-stats)")

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
    print(
        "  modality: vision + language/state + CRI tokens "
        "(TacVLA-style discrete non-vision tokens, LoRA on pi05_droid)"
    )
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
            ]
            print(
                f"[monitor] starting health logger → data/logs/train_health_{args.exp_name}.csv"
            )
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
