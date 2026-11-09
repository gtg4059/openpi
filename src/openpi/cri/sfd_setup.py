"""IsaacLab articulation 등에서 sfd_coreservice 최소 bootstrap.

articulation/__init__.py 맨 위 (ArticulationData import 전) 에 configure_cudacri 호출.

lib/<major.minor>[-cu128]/ 번들 (예: lib/3.11, lib/3.12, lib/3.12-cu128) 에
libcrypto++.so.8, libjsoncpp.so.25 등이 포함된다.
configure_cudacri 는 Python 버전 + CUDA 태그(SFD_CUDACRI_CUDA / Isaac Sim)로 디렉터리를 고른다
(구형 평탄 lib/ 도 허용).
TensorRT: Engine/.../model_fp16.engine 우선, deserialize 실패 시 같은 폴더 model.onnx 로 런타임 빌드.
Isaac Sim: export SAFETICS_TRT_PREFER_ONNX=1 로 engine 건너뛰기 가능.
cmake --build build --target isaaclab_deploy 로 patchelf 적용 lib/{3.11,3.12,3.12-cu128}/·Engine/ 을 IsaacLab articulation 에 배포.
cmake --build build --target rlinf_deploy 로 같은 번들 + lib/3.11-cu128 + libsfd_compat.so 를 RLinf pick_place_cube_plate/cri 에 배포.

Realtime tail-spike mitigation (SafeGiver SFD_CoreService_Test + Isaac host tuning):
    SFD_LOCK_GPU_CLOCK=1     → nvidia-smi -pm 1 + -lgc MAX,MAX
    SFD_LOCK_CPU_GOVERNOR=1  → cpufreq governor=performance (auto with GPU lock unless =0)
    SFD_RECLAIM_MEMORY=1     → sync + drop_caches + clear swap when hot
                               (auto with GPU lock / SFD_REALTIME_HOST unless =0)
    SFD_DISABLE_SWAP=1       → leave swap off for the whole run (auto with GPU lock)
                               so Isaac warmup cannot refill swap; swapon on exit
    SFD_REQUIRE_SWAP_HEADROOM=1 → abort only at startup if swap cannot be cleared
    SFD_PROCESS_RT=1         → mlockall + optional SCHED_FIFO / CPU affinity
                               (auto with GPU lock / SFD_REALTIME_HOST unless =0)
    SFD_MLOCKALL=1           → lock process pages (default on with SFD_PROCESS_RT)
    SFD_SCHED_FIFO=1         → SCHED_FIFO realtime priority (needs CAP_SYS_NICE/root)
    SFD_SCHED_FIFO_PRIO=N    → FIFO priority (default 40)
    SFD_CPU_AFFINITY=0,2,4   → pin process to listed CPUs
    종료 시 GPU -rgc + CPU governor 원복 + swapon (+ SCHED_OTHER restore)
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
from pathlib import Path

_gpu_clock_locked: bool = False
_gpu_clock_device: int = 0
_gpu_clock_used_sudo: bool = False

_cpu_governor_locked: bool = False
_cpu_governor_saved: str | None = None
_cpu_governor_used_sudo: bool = False

_realtime_cleanup_registered: bool = False
_realtime_signals_registered: bool = False
_swap_disabled_by_us: bool = False
_sched_fifo_applied: bool = False
_sched_fifo_saved: tuple[int, int] | None = None  # (policy, priority)
_mlockall_applied: bool = False


def _process_rt_enabled() -> bool:
    """Process RT hardening (mlock / FIFO / affinity). Default on with GPU-lock bundle."""
    if "SFD_PROCESS_RT" in os.environ:
        return _env_truthy("SFD_PROCESS_RT")
    return _realtime_bundle_enabled()


def try_mlockall() -> bool:
    """Lock current + future pages into RAM (avoids mid-loop page-fault spikes)."""
    global _mlockall_applied
    if not _env_truthy("SFD_MLOCKALL", "1" if _process_rt_enabled() else "0"):
        return False
    if _mlockall_applied:
        return True
    try:
        import ctypes
        import ctypes.util

        libc_name = ctypes.util.find_library("c")
        if not libc_name:
            print("[spike] mlockall skipped: libc not found", flush=True)
            return False
        libc = ctypes.CDLL(libc_name, use_errno=True)
        # MCL_CURRENT | MCL_FUTURE
        rc = libc.mlockall(1 | 2)
        if rc != 0:
            err = ctypes.get_errno()
            print(
                f"[spike] mlockall failed errno={err} "
                f"(need CAP_IPC_LOCK / ulimit -l unlimited?). Continuing.",
                flush=True,
            )
            return False
        _mlockall_applied = True
        print("[spike] mlockall(MCL_CURRENT|MCL_FUTURE) applied", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[spike] mlockall skipped: {exc}", flush=True)
        return False


def try_set_cpu_affinity() -> bool:
    """Pin process to ``SFD_CPU_AFFINITY`` (comma-separated CPU ids)."""
    raw = os.environ.get("SFD_CPU_AFFINITY", "").strip()
    if not raw:
        return False
    try:
        cpus = {int(x.strip()) for x in raw.split(",") if x.strip() != ""}
        if not cpus:
            return False
        os.sched_setaffinity(0, cpus)
        print(f"[spike] CPU affinity set to {sorted(cpus)}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[spike] CPU affinity failed ({raw}): {exc}", flush=True)
        return False


def try_sched_fifo() -> bool:
    """Raise this process to SCHED_FIFO (needs CAP_SYS_NICE / root). Opt-in."""
    global _sched_fifo_applied, _sched_fifo_saved
    if not _env_truthy("SFD_SCHED_FIFO"):
        return False
    if _sched_fifo_applied:
        return True
    if not hasattr(os, "SCHED_FIFO") or not hasattr(os, "sched_setscheduler"):
        print("[spike] SCHED_FIFO not supported on this platform", flush=True)
        return False
    try:
        old_policy = os.sched_getscheduler(0)
        old_param = os.sched_getparam(0)
        _sched_fifo_saved = (old_policy, int(old_param.sched_priority))
        prio = int(os.environ.get("SFD_SCHED_FIFO_PRIO", "40"))
        max_p = os.sched_get_priority_max(os.SCHED_FIFO)
        min_p = os.sched_get_priority_min(os.SCHED_FIFO)
        prio = max(min_p, min(max_p, prio))
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
        _sched_fifo_applied = True
        _register_realtime_cleanup()
        print(
            f"[spike] SCHED_FIFO priority={prio} "
            f"(was policy={old_policy} prio={old_param.sched_priority})",
            flush=True,
        )
        return True
    except PermissionError:
        print(
            "[spike] SCHED_FIFO failed: need CAP_SYS_NICE/root. "
            "Try: sudo setcap cap_sys_nice+ep $(which python)  OR  run with sudo.",
            flush=True,
        )
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[spike] SCHED_FIFO failed: {exc}", flush=True)
        return False


def restore_sched_policy() -> None:
    """Restore pre-FIFO scheduler if we changed it."""
    global _sched_fifo_applied, _sched_fifo_saved
    if not _sched_fifo_applied or _sched_fifo_saved is None:
        return
    policy, prio = _sched_fifo_saved
    try:
        os.sched_setscheduler(0, policy, os.sched_param(prio))
        print(f"[spike] scheduler restored policy={policy} prio={prio}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[spike] scheduler restore failed: {exc}", flush=True)
    _sched_fifo_applied = False
    _sched_fifo_saved = None


def apply_process_rt_hardening() -> None:
    """Process-level RT hardening against rare mid-run latency spikes.

    Complements host GPU/CPU locks. Call after tensors/solvers are warmed so
    ``mlockall`` covers the working set.
    """
    if not _process_rt_enabled() and not _env_truthy("SFD_MLOCKALL") and not _env_truthy("SFD_SCHED_FIFO"):
        # Still honor explicit affinity.
        try_set_cpu_affinity()
        return
    try_set_cpu_affinity()
    try_mlockall()
    try_sched_fifo()


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


def apply_spike_mitigation_env_early() -> None:
    """Set before first PyTorch CUDA allocation in this process (tail latency)."""
    if os.environ.get("SFD_SPIKE_MITIGATION", "1") in ("0", "false", "FALSE"):
        return
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.9",
    )
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    os.environ.setdefault("SAFETICS_USE_TENSOR_CALC_LOOP", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")


def _needs_nvidia_priv(args: list[str]) -> bool:
    return any(flag in args for flag in ("-lgc", "-rgc", "-pm", "-pl"))


def _run_privileged(
    argv: list[str],
    *,
    allow_interactive_sudo: bool = False,
    sudo_flag: str = "_gpu_clock_used_sudo",
) -> subprocess.CompletedProcess[str]:
    """Run argv; on failure retry with sudo -n, then optional interactive sudo."""
    global _gpu_clock_used_sudo, _cpu_governor_used_sudo

    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return result

    sudo = shutil.which("sudo")
    if sudo is None:
        return result

    sudo_n = subprocess.run([sudo, "-n", *argv], capture_output=True, text=True, check=False)
    if sudo_n.returncode == 0:
        if sudo_flag == "_cpu_governor_used_sudo":
            _cpu_governor_used_sudo = True
        else:
            _gpu_clock_used_sudo = True
        return sudo_n
    if sudo_n.stderr.strip():
        result = sudo_n

    if (
        allow_interactive_sudo
        and _env_truthy("SFD_LOCK_GPU_SUDO", "1")
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    ):
        print(f"[spike] needs privileges for {' '.join(argv[:2])}...; prompting sudo...", flush=True)
        sudo_i = subprocess.run([sudo, *argv], check=False)
        if sudo_i.returncode == 0:
            if sudo_flag == "_cpu_governor_used_sudo":
                _cpu_governor_used_sudo = True
            else:
                _gpu_clock_used_sudo = True
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        result = subprocess.CompletedProcess(argv, sudo_i.returncode, stdout="", stderr="interactive sudo failed")
    return result


def _run_nvidia_smi(
    args: list[str], *, check: bool = False, allow_interactive_sudo: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run nvidia-smi; on permission failure retry with sudo."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise FileNotFoundError("nvidia-smi not found on PATH")

    cmd = [nvidia_smi, *args]
    if not _needs_nvidia_priv(args):
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    else:
        result = _run_privileged(cmd, allow_interactive_sudo=allow_interactive_sudo)

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def _query_gpu_clock_mhz(device_index: int, query: str) -> int | None:
    result = _run_nvidia_smi(
        [
            f"-i={device_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return None
    text = (result.stdout or "").strip().splitlines()
    if not text:
        return None
    try:
        return int(float(text[0].strip().split(",")[0]))
    except ValueError:
        return None


def _cpu_governor_paths() -> list[Path]:
    root = Path("/sys/devices/system/cpu")
    if not root.is_dir():
        return []
    return sorted(root.glob("cpu[0-9]*/cpufreq/scaling_governor"))


def _read_cpu_governor() -> str | None:
    paths = _cpu_governor_paths()
    if not paths:
        return None
    try:
        return paths[0].read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_meminfo() -> dict[str, int] | None:
    """Return selected /proc/meminfo fields in KiB."""
    keys = ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree")
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                name = line.split(":", 1)[0]
                if name in keys:
                    out[name] = int(line.split()[1])
    except OSError:
        return None
    return out if out else None


def _swap_used_pct(info: dict[str, int]) -> float:
    total = info.get("SwapTotal", 0)
    if total <= 0:
        return 0.0
    used = total - info.get("SwapFree", 0)
    return 100.0 * used / total


def _print_mem_swap_line(info: dict[str, int], *, prefix: str = "[spike]") -> None:
    avail_mib = info.get("MemAvailable", 0) // 1024
    swap_total = info.get("SwapTotal", 0)
    swap_used = swap_total - info.get("SwapFree", 0)
    pct = _swap_used_pct(info)
    print(
        f"{prefix} MemAvailable={avail_mib} MiB  "
        f"Swap used={swap_used // 1024}/{swap_total // 1024} MiB ({pct:.0f}%)",
        flush=True,
    )


def print_gpu_runtime_snapshot() -> None:
    """Print GPU / host pressure when spike mitigation or SFD_PRINT_GPU_INFO is on."""
    if not (_env_truthy("SFD_PRINT_GPU_INFO") or _env_truthy("SFD_SPIKE_MITIGATION", "1")):
        return
    try:
        name = _run_nvidia_smi(["--query-gpu=name,driver_version", "--format=csv,noheader", "-i=0"])
        clocks = _run_nvidia_smi(
            [
                "--query-gpu=clocks.current.graphics,clocks.max.graphics",
                "--format=csv,noheader,nounits",
                "-i=0",
            ]
        )
    except FileNotFoundError:
        return
    if name.returncode == 0 and name.stdout.strip():
        line = f"[spike] GPU: {name.stdout.strip()}"
        if clocks.returncode == 0 and clocks.stdout.strip():
            line += f" clocks(cur,max MHz): {clocks.stdout.strip()}"
        print(line, flush=True)
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if alloc:
        print(f"[spike] PYTORCH_CUDA_ALLOC_CONF={alloc}", flush=True)

    gov = _read_cpu_governor()
    if gov:
        print(f"[spike] CPU governor: {gov}", flush=True)
        if gov == "powersave":
            print(
                "[spike] WARN: CPU governor=powersave (Isaac warning). "
                "Set SFD_LOCK_CPU_GOVERNOR=1 or SFD_LOCK_GPU_CLOCK=1 to force performance.",
                flush=True,
            )

    info = _read_meminfo()
    if info is not None:
        _print_mem_swap_line(info, prefix="[spike]")


def _realtime_bundle_enabled() -> bool:
    return _env_truthy("SFD_LOCK_GPU_CLOCK") or _env_truthy("SFD_REALTIME_HOST")


def _reclaim_memory_enabled() -> bool:
    if "SFD_RECLAIM_MEMORY" in os.environ:
        return _env_truthy("SFD_RECLAIM_MEMORY")
    return _realtime_bundle_enabled()


def _require_swap_headroom_enabled() -> bool:
    if "SFD_REQUIRE_SWAP_HEADROOM" in os.environ:
        return _env_truthy("SFD_REQUIRE_SWAP_HEADROOM")
    return _realtime_bundle_enabled()


def _disable_swap_enabled() -> bool:
    """Keep swap offline for the whole process (prevents Isaac warmup refill)."""
    if "SFD_DISABLE_SWAP" in os.environ:
        return _env_truthy("SFD_DISABLE_SWAP")
    return _realtime_bundle_enabled()


def _cpu_governor_enabled() -> bool:
    if "SFD_LOCK_CPU_GOVERNOR" in os.environ:
        return _env_truthy("SFD_LOCK_CPU_GOVERNOR")
    return _realtime_bundle_enabled()


def restore_swap() -> None:
    """Re-enable swap if we disabled it for this process."""
    global _swap_disabled_by_us
    if not _swap_disabled_by_us:
        return
    result = _run_privileged(
        ["swapon", "-a"],
        allow_interactive_sudo=_cpu_governor_used_sudo and sys.stdin.isatty(),
        sudo_flag="_cpu_governor_used_sudo",
    )
    if result.returncode == 0:
        print("[spike] swap re-enabled (swapon -a)", flush=True)
    else:
        print(
            f"[spike] swapon restore failed rc={result.returncode} (sudo swapon -a)",
            flush=True,
        )
    _swap_disabled_by_us = False


def unlock_gpu_clocks() -> None:
    """Release graphics clock lock previously taken by :func:`try_lock_gpu_clocks`."""
    global _gpu_clock_locked
    if not _gpu_clock_locked:
        return
    try:
        result = _run_nvidia_smi([f"-i={_gpu_clock_device}", "-rgc"], allow_interactive_sudo=False)
        if result.returncode != 0 and _gpu_clock_used_sudo and sys.stdin.isatty():
            result = _run_nvidia_smi([f"-i={_gpu_clock_device}", "-rgc"], allow_interactive_sudo=True)
        if result.returncode == 0:
            print(f"[spike] GPU clock lock released (device={_gpu_clock_device})", flush=True)
        else:
            print(
                f"[spike] GPU clock unlock failed rc={result.returncode} "
                f"(sudo nvidia-smi -i {_gpu_clock_device} -rgc)",
                flush=True,
            )
    except FileNotFoundError:
        pass
    _gpu_clock_locked = False


def restore_cpu_governor() -> None:
    """Restore CPU governor saved by :func:`try_lock_cpu_governor`."""
    global _cpu_governor_locked, _cpu_governor_saved
    if not _cpu_governor_locked:
        return
    target = _cpu_governor_saved or "powersave"
    current = _read_cpu_governor()
    if current == target:
        print(f"[spike] CPU governor restored '{target}'", flush=True)
        _cpu_governor_locked = False
        _cpu_governor_saved = None
        return

    cpupower = shutil.which("cpupower")
    if cpupower is not None:
        result = _run_privileged(
            [cpupower, "frequency-set", "-g", target],
            allow_interactive_sudo=_cpu_governor_used_sudo and sys.stdin.isatty(),
            sudo_flag="_cpu_governor_used_sudo",
        )
    else:
        paths = _cpu_governor_paths()
        joined = " ".join(str(p) for p in paths)
        result = _run_privileged(
            ["bash", "-c", f"echo {target} | tee {joined} >/dev/null"],
            allow_interactive_sudo=_cpu_governor_used_sudo and sys.stdin.isatty(),
            sudo_flag="_cpu_governor_used_sudo",
        )

    after = _read_cpu_governor()
    if result.returncode == 0 and after == target:
        print(f"[spike] CPU governor restored '{target}'", flush=True)
    else:
        print(
            f"[spike] CPU governor restore failed (want={target}, now={after}) "
            f"(sudo cpupower frequency-set -g {target})",
            flush=True,
        )
    _cpu_governor_locked = False
    _cpu_governor_saved = None


def restore_realtime_host() -> None:
    """Release GPU clock lock, restore CPU governor, re-enable swap (idempotent)."""
    restore_sched_policy()
    unlock_gpu_clocks()
    restore_cpu_governor()
    restore_swap()


def _register_realtime_cleanup() -> None:
    """Ensure host realtime tweaks are reverted on exit / SIGINT / SIGTERM."""
    global _realtime_cleanup_registered, _realtime_signals_registered

    if not _realtime_cleanup_registered:
        atexit.register(restore_realtime_host)
        _realtime_cleanup_registered = True

    if _realtime_signals_registered:
        return

    import signal

    def _on_signal(signum, _frame):  # noqa: ANN001
        restore_realtime_host()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass
    _realtime_signals_registered = True


def try_reclaim_host_memory(*, clear_swap: bool = True, keep_swap_off: bool | None = None) -> bool:
    """Drop page cache and optionally clear/disable swap (needs sudo)."""
    global _swap_disabled_by_us

    if not _reclaim_memory_enabled():
        return False
    if keep_swap_off is None:
        keep_swap_off = _disable_swap_enabled()

    before = _read_meminfo()
    if before is None:
        print("[spike] memory reclaim skipped: cannot read /proc/meminfo", flush=True)
        return False

    print("[spike] memory reclaim: before", flush=True)
    _print_mem_swap_line(before, prefix="[spike]  ")

    drop = _run_privileged(
        ["bash", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
        allow_interactive_sudo=True,
        sudo_flag="_cpu_governor_used_sudo",
    )
    if drop.returncode != 0:
        err = (drop.stderr or drop.stdout or "").strip()
        print(
            f"[spike] drop_caches failed rc={drop.returncode}"
            + (f" err={err}" if err else "")
            + " (sudo bash -c 'sync; echo 3 > /proc/sys/vm/drop_caches')",
            flush=True,
        )

    mid = _read_meminfo() or before
    swap_total = mid.get("SwapTotal", 0)
    swap_used_kib = swap_total - mid.get("SwapFree", 0)
    swap_pct = _swap_used_pct(mid)
    min_clear_pct = float(os.environ.get("SFD_SWAP_CLEAR_MIN_PCT", "25"))
    margin_kib = int(os.environ.get("SFD_SWAP_CLEAR_MARGIN_MIB", "512")) * 1024

    if keep_swap_off and _swap_disabled_by_us and swap_total == 0:
        print("[spike] swap already disabled for this run", flush=True)
        after = _read_meminfo() or mid
        print("[spike] memory reclaim: after", flush=True)
        _print_mem_swap_line(after, prefix="[spike]  ")
        return True

    did_swapoff = False
    if clear_swap and swap_total > 0 and (swap_used_kib > 0 or keep_swap_off):
        can_clear = mid.get("MemAvailable", 0) > swap_used_kib + margin_kib
        need_off = keep_swap_off or (swap_pct >= min_clear_pct)
        if need_off and (can_clear or swap_used_kib == 0):
            action = "disabling swap for run" if keep_swap_off else "clearing swap into RAM"
            print(
                f"[spike] {action} "
                f"(used={swap_used_kib // 1024} MiB, MemAvailable={mid.get('MemAvailable', 0) // 1024} MiB)...",
                flush=True,
            )
            off = _run_privileged(
                ["swapoff", "-a"],
                allow_interactive_sudo=True,
                sudo_flag="_cpu_governor_used_sudo",
            )
            if off.returncode != 0:
                err = (off.stderr or off.stdout or "").strip()
                print(
                    f"[spike] swapoff failed rc={off.returncode}"
                    + (f" err={err}" if err else "")
                    + " (sudo swapoff -a)",
                    flush=True,
                )
            else:
                did_swapoff = True
                if keep_swap_off:
                    _swap_disabled_by_us = True
                    _register_realtime_cleanup()
                    print("[spike] swap disabled until process exit (swapon on restore)", flush=True)
                else:
                    on = _run_privileged(
                        ["swapon", "-a"],
                        allow_interactive_sudo=True,
                        sudo_flag="_cpu_governor_used_sudo",
                    )
                    if on.returncode != 0:
                        print(
                            f"[spike] swapon failed rc={on.returncode} — swap may be offline until reboot/swapon",
                            flush=True,
                        )
                    else:
                        print("[spike] swap cleared and re-enabled", flush=True)
        elif need_off:
            print(
                "[spike] skip swap clear: not enough MemAvailable headroom "
                f"(need >{(swap_used_kib + margin_kib) // 1024} MiB).",
                flush=True,
            )

    after = _read_meminfo() or mid
    print("[spike] memory reclaim: after", flush=True)
    _print_mem_swap_line(after, prefix="[spike]  ")
    return did_swapoff or drop.returncode == 0


def ensure_swap_headroom(*, max_pct: float | None = None, abort: bool | None = None) -> bool:
    """Check swap pressure. Abort only at startup when requested."""
    if _swap_disabled_by_us:
        return True
    info = _read_meminfo()
    if info is None:
        return True
    if info.get("SwapTotal", 0) <= 0:
        return True
    if max_pct is None:
        max_pct = float(os.environ.get("SFD_SWAP_MAX_PCT", "50"))
    pct = _swap_used_pct(info)
    if pct <= max_pct:
        return True

    if abort is None:
        abort = _require_swap_headroom_enabled()
    msg = (
        f"[spike] swap still high after reclaim: {pct:.0f}% used "
        f"(limit SFD_SWAP_MAX_PCT={max_pct:.0f}). "
        "Close other apps or free RAM, then retry."
    )
    if abort:
        print(msg + " Aborting (SFD_REQUIRE_SWAP_HEADROOM=1).", flush=True)
        raise SystemExit(2)
    print(msg + " Continuing (warn-only).", flush=True)
    return False


def try_lock_gpu_clocks(device_index: int | None = None) -> bool:
    """Lock GPU graphics clock to max MHz (SafeGiver ``SFD_LOCK_GPU_CLOCK`` path)."""
    global _gpu_clock_locked, _gpu_clock_device

    if not _env_truthy("SFD_LOCK_GPU_CLOCK"):
        return False
    if _gpu_clock_locked:
        return True

    if device_index is None:
        device_index = int(os.environ.get("SFD_LOCK_GPU_INDEX", "0"))

    try:
        _run_nvidia_smi([f"-i={device_index}", "-pm", "1"], allow_interactive_sudo=True)
        max_mhz = _query_gpu_clock_mhz(device_index, "clocks.max.graphics")
        if max_mhz is None or max_mhz <= 0:
            print("[spike] GPU clock lock skipped: could not query max graphics clock", flush=True)
            return False

        result = _run_nvidia_smi(
            [f"-i={device_index}", "-lgc", f"{max_mhz},{max_mhz}"],
            allow_interactive_sudo=True,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            print(
                f"[spike] GPU clock lock failed (need root/sudo?). "
                f"rc={result.returncode}"
                + (f" err={err}" if err else ""),
                flush=True,
            )
            print(
                f"[spike] manual: sudo nvidia-smi -i {device_index} -pm 1 && "
                f"sudo nvidia-smi -i {device_index} -lgc {max_mhz},{max_mhz}",
                flush=True,
            )
            return False

        cur = _query_gpu_clock_mhz(device_index, "clocks.current.graphics")
        cur_s = str(cur) if cur is not None else "?"
        print(
            f"[spike] GPU clock locked to {max_mhz} MHz "
            f"(device={device_index}, current={cur_s} MHz); will release with -rgc on exit",
            flush=True,
        )
        _gpu_clock_locked = True
        _gpu_clock_device = device_index
        _register_realtime_cleanup()
        return True
    except FileNotFoundError as exc:
        print(f"[spike] GPU clock lock skipped: {exc}", flush=True)
        return False


def try_lock_cpu_governor(governor: str = "performance") -> bool:
    """Force CPU frequency governor (Isaac warns when host is on powersave)."""
    global _cpu_governor_locked, _cpu_governor_saved

    if not _cpu_governor_enabled():
        return False
    if _cpu_governor_locked:
        return True

    paths = _cpu_governor_paths()
    if not paths:
        print("[spike] CPU governor lock skipped: no cpufreq sysfs", flush=True)
        return False

    current = _read_cpu_governor()
    if current == governor:
        print(f"[spike] CPU governor already '{governor}'", flush=True)
        _cpu_governor_saved = current
        _cpu_governor_locked = True
        _register_realtime_cleanup()
        return True

    _cpu_governor_saved = current
    cpupower = shutil.which("cpupower")
    if cpupower is not None:
        result = _run_privileged(
            [cpupower, "frequency-set", "-g", governor],
            allow_interactive_sudo=True,
            sudo_flag="_cpu_governor_used_sudo",
        )
    else:
        joined = " ".join(str(p) for p in paths)
        result = _run_privileged(
            ["bash", "-c", f"echo {governor} | tee {joined} >/dev/null"],
            allow_interactive_sudo=True,
            sudo_flag="_cpu_governor_used_sudo",
        )

    after = _read_cpu_governor()
    if result.returncode != 0 or after != governor:
        err = (result.stderr or result.stdout or "").strip()
        print(
            f"[spike] CPU governor lock failed (want={governor}, now={after}, "
            f"was={current}, rc={result.returncode})"
            + (f" err={err}" if err else ""),
            flush=True,
        )
        print(f"[spike] manual: sudo cpupower frequency-set -g {governor}", flush=True)
        return False

    print(
        f"[spike] CPU governor set '{current}' → '{governor}' (restore on exit)",
        flush=True,
    )
    _cpu_governor_locked = True
    _register_realtime_cleanup()
    return True


def apply_realtime_host_tuning(*, clear_swap: bool = True, enforce_swap_headroom: bool = True) -> None:
    """Reclaim memory (swap off for run), then lock GPU clocks + CPU governor."""
    try_reclaim_host_memory(clear_swap=clear_swap, keep_swap_off=None)
    try_lock_gpu_clocks()
    try_lock_cpu_governor()
    if enforce_swap_headroom:
        ensure_swap_headroom(abort=True if _require_swap_headroom_enabled() else False)


def _python_version_tag() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _normalize_cuda_tag(raw: str | None) -> str | None:
    """Map env / version strings to a deploy tag (``cu128``) or ``None`` (default bundle)."""
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value in ("cu128", "12.8", "128", "cuda12.8", "cuda128"):
        return "cu128"
    if value in ("cu130", "13.0", "13", "cuda13", "cuda13.0", "cu13"):
        return None
    return value.lstrip("-")


def _explicit_cudacri_lib_dir() -> Path | None:
    env = os.environ.get("SFD_CUDACRI_LIB_DIR", "").strip()
    if not env:
        return None
    root = Path(env).expanduser()
    if root.is_dir():
        return root.resolve()
    return None


def _is_isaac_sim_torch(torch_lib: Path) -> bool:
    text = str(torch_lib)
    return (
        "omni.isaac.ml_archive" in text
        or "/isaac-sim/" in text
        or "/isaacsim/" in text
        or "isaacsim-ml-prebundle" in text
    )


def _detect_cuda_tag(torch_lib: Path | None = None) -> str | None:
    """Prefer ``cu128`` for Arena Docker / Isaac Sim; host CUDA 13 stays untagged."""
    tagged = _normalize_cuda_tag(os.environ.get("SFD_CUDACRI_CUDA"))
    if "SFD_CUDACRI_CUDA" in os.environ:
        return tagged

    lib = torch_lib
    if lib is None:
        try:
            import torch

            lib = Path(torch.__file__).resolve().parent / "lib"
        except Exception:
            lib = None

    # Host conda torch can also be cu128 with only libcudart.so.12 — that must
    # keep lib/3.12 (linked to that env's libtorch). Tagged cu128 is Isaac Sim 2.10.
    if lib is not None and _is_isaac_sim_torch(lib):
        return "cu128"
    return None


def _resolve_cudacri_lib_dir(root: Path) -> Path:
    """Pick ``lib/<major.minor>[-cu128]/``; fall back to untagged then a flat ``lib/``."""
    explicit = _explicit_cudacri_lib_dir()
    if explicit is not None:
        return explicit

    version = _python_version_tag()
    lib_root = root / "lib"
    tag = _detect_cuda_tag()
    candidates: list[Path] = []
    if tag:
        candidates.append(lib_root / f"{version}-{tag}")
    candidates.append(lib_root / version)

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate

    flat = lib_root
    if flat.is_dir() and any(flat.glob("sfd_coreservice*.so")):
        return flat

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"CUDACRI lib for Python {version} not found (tried {searched}). "
        f"Build with -DSFD_PYBIND_PYTHON_VERSION={version}"
        + (f" -DSFD_PYBIND_CUDA_TAG={tag}" if tag else "")
        + " and target cudacri_deploy"
    )


def _torch_cuda_runtime_dir(torch_lib: Path) -> Path | None:
    """Isaac Sim prebundle ships libcudart next to torch, not inside torch/lib."""
    for candidate in (
        torch_lib.parent.parent / "nvidia" / "cuda_runtime" / "lib",
        torch_lib.parent / "nvidia" / "cuda_runtime" / "lib",
    ):
        root = candidate.resolve()
        if (root / "libcudart.so.12").is_file() or (root / "libcudart.so.13").is_file():
            return root
    return None


def _tensorrt_lib_dirs() -> list[Path]:
    """TensorRT 10 dirs: env override, then Docker apt path, then optional pip/host caches."""
    found: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path | None) -> None:
        if path is None:
            return
        root = path.expanduser().resolve()
        key = str(root)
        if key in seen or not root.is_dir():
            return
        if not (root / "libnvinfer.so.10").is_file():
            return
        seen.add(key)
        found.append(root)

    for env_key in ("SFD_TENSORRT_LIB_DIR", "TENSORRT_LIB_DIR"):
        env = os.environ.get(env_key)
        if env:
            _add(Path(env))

    # Isaac Lab Arena docker/setup/install_tensorrt.sh installs here.
    for candidate in (
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/usr/lib"),
    ):
        _add(candidate)

    try:
        import tensorrt_libs  # type: ignore[import-not-found]

        _add(Path(tensorrt_libs.__file__).resolve().parent)
    except Exception:
        pass

    return found


def _preload_runtime_libs(paths: list[Path]) -> None:
    """``RTLD_GLOBAL`` preload. Changing LD_LIBRARY_PATH in-process is not enough for dlopen."""
    import ctypes

    for path in paths:
        if not path.is_file():
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


def configure_cudacri(cudacri_dir: str | Path) -> Path:
    import torch

    apply_spike_mitigation_env_early()
    apply_realtime_host_tuning()

    root = Path(cudacri_dir).resolve()
    lib_dir = _resolve_cudacri_lib_dir(root)

    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    extra_dirs = [lib_dir, torch_lib]
    cuda_rt = _torch_cuda_runtime_dir(torch_lib)
    if cuda_rt is not None:
        extra_dirs.append(cuda_rt)
    extra_dirs.extend(_tensorrt_lib_dirs())

    path_entries = [str(p) for p in extra_dirs if p.is_dir()]
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in (*path_entries, previous) if p)

    preload: list[Path] = []
    for name in (
        "libcudart.so.12",
        "libcudart.so.13",
        "libc10.so",
        "libc10_cuda.so",
        "libtorch_cpu.so",
        "libtorch_cuda.so",
        "libtorch.so",
        "libtorch_python.so",
    ):
        preload.append(torch_lib / name)
        if cuda_rt is not None:
            preload.append(cuda_rt / name)
    for trt_dir in _tensorrt_lib_dirs():
        preload.append(trt_dir / "libnvinfer.so.10")
        preload.append(trt_dir / "libnvonnxparser.so.10")
        preload.append(trt_dir / "libnvinfer_plugin.so.10")
    _preload_runtime_libs(preload)
    return lib_dir
