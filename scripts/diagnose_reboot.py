"""Post-reboot forensics for training crashes / hard resets.

Checks:
  - boot history (clean shutdown vs abrupt)
  - previous-boot journal (OOM, NVRM/Xid, panic, thermal)
  - incomplete Orbax checkpoints
  - last rows of ``data/logs/train_health_latest.csv`` (if monitor was running)

Usage::

    .venv/bin/python scripts/diagnose_reboot.py
    # or: uv run python scripts/diagnose_reboot.py
"""

from __future__ import annotations

import csv
import dataclasses
import subprocess
from pathlib import Path

import tyro

_REPO = Path(__file__).resolve().parent.parent


@dataclasses.dataclass
class Args:
    health_log: Path = _REPO / "data" / "logs" / "train_health_latest.csv"
    checkpoint_dir: Path = _REPO / "data" / "checkpoints" / "pi05_droid_cri_finetune"
    last_health_rows: int = 15


def _run(cmd: list[str], timeout: float = 15.0) -> str:
    """Run cmd; on timeout kill the whole process group (journalctl can hang)."""
    try:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert proc.stdout is not None
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            import os
            import signal as _signal

            try:
                os.killpg(proc.pid, _signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=2)
            return f"(timeout after {timeout}s: {' '.join(cmd)})"
        if proc.returncode not in (0, None) and not out:
            return f"(failed rc={proc.returncode}: {' '.join(cmd)})"
        return out or ""
    except FileNotFoundError as e:
        return f"(failed: {e})"


def _section(title: str) -> None:
    print("\n" + "=" * 60, flush=True)
    print(title, flush=True)
    print("=" * 60, flush=True)


def _filter_lines(text: str, needles: tuple[str, ...]) -> str:
    hits = [ln for ln in text.splitlines() if any(n.lower() in ln.lower() for n in needles)]
    return "\n".join(hits[-60:]) if hits else "(none)"


def main(args: Args) -> None:
    _section("1) Boot history")
    print(_run(["who", "-b"]).rstrip(), flush=True)
    out = _run(["last", "-x", "reboot", "shutdown"], timeout=5.0)
    print("\n".join(out.splitlines()[:8]), flush=True)

    _section("2) Was previous boot a clean shutdown?")
    # Only read the TAIL of previous boot (full --grep over multi-day boots hangs).
    last_lines = _run(
        ["journalctl", "-b", "-1", "--no-pager", "-o", "short-iso", "-n", "200"],
        timeout=10.0,
    )
    shutdown_needles = (
        "Reached target Shutdown",
        "Powering Off",
        "Starting Power-Off",
        "Rebooting",
        "Shutting down",
    )
    crash_needles = (
        "Watchdog",
        "Kernel panic",
        "Oops:",
        "BUG:",
        "Hard LOCKUP",
        "Out of memory",
        "Killed process",
        "NVRM:",
        "Xid",
        "thermal",
        "Hardware Error",
        "mce",
    )
    shutdown_hits = _filter_lines(last_lines, shutdown_needles)
    crash_hits = _filter_lines(last_lines, crash_needles)
    clean = shutdown_hits != "(none)"

    if clean:
        print("Previous boot looks like a CLEAN shutdown/reboot path was entered.", flush=True)
    else:
        print(
            "No clean-shutdown markers in the last 200 journal lines of previous boot.\n"
            "→ Likely HARD RESET or power loss (journal cut mid-flight).",
            flush=True,
        )

    print("\n--- shutdown-related lines ---", flush=True)
    print(shutdown_hits, flush=True)
    print("\n--- crash/OOM/GPU markers (in last 200 lines) ---", flush=True)
    print(crash_hits, flush=True)
    print("\n--- last 20 journal lines of previous boot ---", flush=True)
    print("\n".join(last_lines.splitlines()[-20:]), flush=True)

    _section("3) Previous-boot kernel highlights (OOM / GPU / thermal)")
    kern = _run(
        ["journalctl", "-b", "-1", "-k", "--no-pager", "-o", "short-iso", "-n", "300"],
        timeout=10.0,
    )
    kern_hits = _filter_lines(
        kern,
        (
            "oom",
            "Out of memory",
            "Killed process",
            "NVRM",
            "Xid",
            "thermal",
            "mce",
            "Hardware Error",
            "lockup",
            "panic",
            "Oops",
        ),
    )
    print(kern_hits, flush=True)

    _section("4) Incomplete Orbax checkpoints")
    if not args.checkpoint_dir.is_dir():
        print(f"(missing {args.checkpoint_dir})")
    else:
        tmps = sorted(
            p
            for p in args.checkpoint_dir.rglob("*")
            if "orbax-checkpoint-tmp" in p.name
        )
        if not tmps:
            print("(none)")
        else:
            for p in tmps[-20:]:
                try:
                    if p.is_dir():
                        sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    else:
                        sz = p.stat().st_size
                except OSError:
                    sz = -1
                print(f"{p}  (~{sz / 1e6:.1f} MB)")

    _section("5) Train health monitor (last rows)")
    log = args.health_log
    if log.is_symlink():
        print(f"symlink: {log} -> {log.resolve()}")
    if not log.exists():
        print(
            f"(no health log at {log})\n"
            "Start monitor next time:\n"
            "  .venv/bin/python scripts/monitor_train_health.py --interval-sec 5\n"
            "or: scripts/train_pi05_droid_cri.py --monitor ..."
        )
    else:
        with log.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"rows={len(rows)} path={log.resolve()}")
        if rows:
            show = rows[-args.last_health_rows :]
            cols = [
                "wall_time",
                "mem_available_kb",
                "gpu_util",
                "gpu_mem_used_mib",
                "gpu_temp_c",
                "gpu_power_w",
                "load1",
            ]
            print(",".join(cols))
            for r in show:
                print(",".join(str(r.get(c, "")) for c in cols))
            last = rows[-1]
            try:
                avail_gb = int(last.get("mem_available_kb") or 0) / 1e6
                power = float(str(last.get("gpu_power_w") or "0").split()[0] or 0)
                temp = float(str(last.get("gpu_temp_c") or "0").split()[0] or 0)
            except ValueError:
                avail_gb = power = temp = -1.0
            print("\nHeuristic on last sample:")
            if last.get("gpu_error"):
                print(f"  ! gpu_error present: {last.get('gpu_error')[:120]}")
            if 0 <= avail_gb < 2:
                print(f"  ! host RAM nearly exhausted (MemAvailable≈{avail_gb:.2f} GB)")
            if power > 500:
                print(f"  ! GPU power high ({power:.0f} W) — PSU stress possible")
            if temp >= 85:
                print(f"  ! GPU temperature high ({temp:.0f} C)")
            if (
                not last.get("gpu_error")
                and avail_gb >= 2
                and power <= 500
                and temp < 85
            ):
                print("  (last sample does not show obvious RAM/power/thermal red flags)")

    _section("6) Suggested next checks")
    print(
        "- Confirm PSU wattage vs RTX PRO 6000 (up to ~600W GPU alone).\n"
        "- Keep monitor on during training (data/logs/train_health_*.csv).\n"
        "- Resume from last complete checkpoint (not *.orbax-checkpoint-tmp-*).\n"
        "- Optional host prep: bash scripts/setup_reboot_forensics.sh"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
