"""LoRA fine-tune pi05_droid on full DROID + CRI-F sidecar v3 (no time-scale s).

Sidecar must be ``openpi_cri_addon_v3``: delayed ``cri_pre`` (9,), CBF-QP off,
``cri_limit=0.96``. Built by ``python -m openpi.cri.annotate_rlds_addon``.

Fixed dataset locations (do not copy the 1.7T RLDS)::

    DROID RLDS : /media/safetics/D/droid/1.0.1
    CRI addon  : <E>/droid_1.0.1_cri_addon_v3
    pretrained : <E>/openpi/pretrained/pi05_droid  (~12 GiB init)

Training outputs::

    checkpoints : <E>/openpi/checkpoints/pi05_droid_cri_finetune/<exp_name>
    norm stats  : data/assets/pi05_droid_cri_finetune/droid_cri
    (repo ``data/checkpoints/...`` is a symlink to E for convenience)

``<E>`` resolves to ``/mnt/E`` when mounted there, else legacy ``/media/safetics/E``.

CRI is a TacVLA-style always-on prefix stream (not language bins, no on/off gate).
Language stays the official Pi05 form::

    Task: <lang>, State: <bins>;
    Action:

Usage (from repo root)::

    # TacVLA-style prefix CRI, DROID:PolaRiS = 9:1 (do not overwrite droid_cri_tacvla_lora)
    uv run --group rlds scripts/train_pi05_droid_cri.py \\
        --exp-name=droid_cri_tacvla_polaris91 --overwrite --skip-norm-stats --monitor

    # Resume
    uv run --group rlds scripts/train_pi05_droid_cri.py \\
        --exp-name=droid_cri_tacvla_polaris91 --resume --skip-norm-stats --monitor
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
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
    exp_name: str = "droid_cri_tacvla_polaris91"
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
    num_train_steps: int = 20_000
    """Total LoRA steps (checkpoints at 10k / 20k)."""
    save_interval: int = 10_000
    """Write a checkpoint every N steps (and keep those steps)."""
    keep_period: int | None = 10_000
    """Do not delete checkpoints whose step is a multiple of this."""


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


def _norm_stats_path(repo: Path) -> Path:
    return repo / "data" / "assets" / CONFIG_NAME / "droid_cri" / "norm_stats.json"


def _restore_norm_stats(repo: Path, exp_name: str) -> Path | None:
    """Copy norm_stats.json back if ``data/assets`` was wiped (resume still needs it)."""
    dest = _norm_stats_path(repo)
    if dest.is_file():
        return dest
    candidates: list[Path] = []
    for exp in (exp_name, "droid_cri_v3_lora", "droid_cri_lora", "droid_cri_f_lora"):
        root = CHECKPOINT_BASE_DIR / CONFIG_NAME / exp
        if root.is_dir():
            candidates.extend(root.glob("*/assets/droid_cri/norm_stats.json"))
    if not candidates:
        return None

    def _step(path: Path) -> int:
        try:
            return int(path.parents[2].name)
        except ValueError:
            return -1

    src = max(candidates, key=_step)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"[ok] restored norm stats from {src}")
    return dest


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
    if meta.get("format") != "openpi_cri_addon_v3":
        raise ValueError(
            f"CRI addon {CRI_ADDON_DIR} has format={meta.get('format')!r}; "
            "expected openpi_cri_addon_v3 (delayed cri_pre, no time-scale s). "
            "Rebuild with: uv run --group rlds python -m openpi.cri.annotate_rlds_addon"
        )
    if bool(meta.get("filter_enabled", True)):
        raise ValueError(
            f"CRI addon {CRI_ADDON_DIR} was built with filter_enabled=True. "
            "This LoRA run expects CBF-QP off (--no-filter-enabled)."
        )
    cri_limit = float(meta.get("cri_limit", -1.0))
    if abs(cri_limit - 0.96) > 1e-6:
        raise ValueError(
            f"CRI addon {CRI_ADDON_DIR} has cri_limit={cri_limit}; expected 0.96."
        )
    if not meta.get("finished", False):
        raise RuntimeError(
            f"CRI addon is not marked finished: {meta_path}\n"
            f"meta={meta}"
        )
    for name in ("cri.f32", "delta.f32", "joint_velocity.f32", "episodes.jsonl"):
        if not (CRI_ADDON_DIR / name).is_file():
            raise FileNotFoundError(f"CRI addon incomplete, missing {name} under {CRI_ADDON_DIR}")

    if not CHECKPOINT_BASE_DIR.parent.is_dir():
        raise FileNotFoundError(
            f"E-drive mount missing for checkpoints: {CHECKPOINT_BASE_DIR.parent}"
        )
    CHECKPOINT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_checkpoint_symlink(repo)

    print(f"[ok] DROID RLDS : {DROID_RLDS_DIR}")
    print(
        f"[ok] CRI addon  : {CRI_ADDON_DIR} "
        f"(episodes={meta.get('num_episodes')}, filter_enabled={meta.get('filter_enabled')}, "
        f"cri_limit={meta.get('cri_limit')}, obs_dim={meta.get('num_cri_obs_dim')})"
    )
    print(f"[ok] checkpoints: {CHECKPOINT_BASE_DIR}/…")
    print("[ok] assets     : data/assets/…")


def main(args: Args) -> None:
    repo = Path(__file__).resolve().parent.parent
    _ensure_pretrained()
    _require_datasets(repo)

    stats_path = _norm_stats_path(repo)
    if args.skip_norm_stats and not stats_path.is_file():
        restored = _restore_norm_stats(repo, args.exp_name)
        if restored is None:
            print("[1/2] --skip-norm-stats but norm stats missing; will recompute")
            args.skip_norm_stats = False

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
        print(f"[1/2] Skipping norm stats ({stats_path})")

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
    train_cmd.append(f"--num-train-steps={args.num_train_steps}")
    train_cmd.append(f"--save-interval={args.save_interval}")
    if args.keep_period is not None:
        train_cmd.append(f"--keep-period={args.keep_period}")

    print(f"[2/2] Training: {' '.join(train_cmd)}")
    print(
        f"  mem knobs: batch_size={args.batch_size}, "
        f"rlds_shuffle_buffer_size={args.rlds_shuffle_buffer_size}, "
        f"XLA_MEM_FRACTION={args.mem_fraction}, "
        f"num_train_steps={args.num_train_steps}, "
        f"save_interval={args.save_interval}, keep_period={args.keep_period}"
    )
    print(
        "  modality: vision + language/state + TacVLA CRI prefix "
        "(delayed cri_pre, always on, CBF-QP off, LoRA on pi05_droid, DROID:PolaRiS=9:1)"
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
