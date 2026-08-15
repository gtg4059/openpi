"""Sample host/GPU health while training so hard reboots leave forensic breadcrumbs.

Writes an append-only CSV under ``data/logs/`` (flushed every sample). After an
unexpected reboot, inspect the last rows with::

    uv run python scripts/diagnose_reboot.py

Usage::

    # Alongside training (separate terminal / tmux)
    uv run python scripts/monitor_train_health.py --interval-sec 5

    # Or let train_pi05_droid_cri.py start it via --monitor
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import tyro

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _REPO / "data" / "logs"


@dataclasses.dataclass
class Args:
    interval_sec: float = 5.0
    """Sampling period seconds."""
    log_dir: Path = _DEFAULT_LOG_DIR
    """Directory for health CSV logs."""
    run_name: str = ""
    """Optional tag; default = timestamp."""
    pid: int | None = None
    """If set, stop when this PID exits (e.g. train.py)."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                out[key] = int(parts[0])  # kB
    return out


def _read_loadavg() -> tuple[float, float, float]:
    with open("/proc/loadavg", encoding="utf-8") as f:
        a, b, c, *_ = f.read().split()
    return float(a), float(b), float(c)


def _nvidia_query() -> dict[str, str]:
    keys = [
        "timestamp",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.current.sm",
        "clocks.current.memory",
        "pstate",
    ]
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(keys)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        return {"gpu_error": str(e)}
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    vals = [v.strip() for v in line.split(",")]
    return dict(zip(keys, vals, strict=False))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(args: Args) -> None:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    tag = args.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.log_dir / f"train_health_{tag}.csv"
    # Stable symlink for diagnose_reboot.py
    latest = args.log_dir / "train_health_latest.csv"

    fieldnames = [
        "wall_time",
        "uptime_sec",
        "load1",
        "load5",
        "load15",
        "mem_total_kb",
        "mem_available_kb",
        "mem_free_kb",
        "swap_free_kb",
        "gpu_util",
        "gpu_mem_util",
        "gpu_mem_used_mib",
        "gpu_mem_total_mib",
        "gpu_temp_c",
        "gpu_power_w",
        "gpu_power_limit_w",
        "gpu_sm_clock",
        "gpu_mem_clock",
        "gpu_pstate",
        "gpu_error",
        "watch_pid",
        "watch_pid_alive",
    ]

    stop = False

    def _handle_sig(signum, frame):  # noqa: ARG001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    print(f"[monitor] writing {path} (interval={args.interval_sec}s)", flush=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(path.name)
        except OSError:
            pass

        boot_mono0 = time.monotonic()
        while not stop:
            mem = _read_meminfo()
            load1, load5, load15 = _read_loadavg()
            gpu = _nvidia_query()
            alive = "" if args.pid is None else str(_pid_alive(args.pid))
            row = {
                "wall_time": _utc_now(),
                "uptime_sec": f"{time.monotonic() - boot_mono0:.1f}",
                "load1": load1,
                "load5": load5,
                "load15": load15,
                "mem_total_kb": mem.get("MemTotal", ""),
                "mem_available_kb": mem.get("MemAvailable", ""),
                "mem_free_kb": mem.get("MemFree", ""),
                "swap_free_kb": mem.get("SwapFree", ""),
                "gpu_util": gpu.get("utilization.gpu", ""),
                "gpu_mem_util": gpu.get("utilization.memory", ""),
                "gpu_mem_used_mib": gpu.get("memory.used", ""),
                "gpu_mem_total_mib": gpu.get("memory.total", ""),
                "gpu_temp_c": gpu.get("temperature.gpu", ""),
                "gpu_power_w": gpu.get("power.draw", ""),
                "gpu_power_limit_w": gpu.get("power.limit", ""),
                "gpu_sm_clock": gpu.get("clocks.current.sm", ""),
                "gpu_mem_clock": gpu.get("clocks.current.memory", ""),
                "gpu_pstate": gpu.get("pstate", ""),
                "gpu_error": gpu.get("gpu_error", ""),
                "watch_pid": args.pid if args.pid is not None else "",
                "watch_pid_alive": alive,
            }
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

            if args.pid is not None and alive == "False":
                print(f"[monitor] watched pid {args.pid} exited; stopping", flush=True)
                break

            # Sleep in small slices so SIGINT is responsive.
            end = time.time() + args.interval_sec
            while not stop and time.time() < end:
                time.sleep(min(0.5, end - time.time()))

    print(f"[monitor] stopped; log at {path}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
