"""
CRI 실시간 제어(50Hz) wall time 모니터링 — Python / pybind 라이브러리 공용.

SFD_CoreService_Test 의 RealTimeBudgetReport / RealTimePassGrade 와 동일 기준:
  - hard budget: 1000 / control_hz ms (기본 20ms @ 50Hz)
  - soft budget: hard * (1 - margin_pct/100) (기본 18ms @ 10% margin)
  - hard budget 초과 샘플 = 아웃라이어 (20ms 위반)

사용 예 (IsaacLab / sfd_coreservice):

    from cri_realtime_monitor import (
        CriRealtimeMonitor,
        configure_cri_filter,
        run_cri_at_motion_state,
        run_cri_filter,
    )

    configure_cri_filter(service, cri_limit=0.96, alpha=0.02)
    monitor = CriRealtimeMonitor.from_env()
    for i, (q, qd) in enumerate(ticks):
        # 전방만: run_cri_at_motion_state
        # CRI_F 필터: run_cri_filter(service, q, qd_rl)
        out, wall_ms = run_cri_filter(service, q, qd, monitor=monitor, index=i)
    monitor.print_report("control loop")
    monitor.raise_if_not_pass()
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


def control_period_ms(control_hz: int = 50) -> float:
    return 1000.0 / float(control_hz) if control_hz > 0 else 20.0


@dataclass(frozen=True)
class RealTimeBudgetConfig:
    control_hz: int = 50
    margin_pct: float = 10.0

    @property
    def hard_budget_ms(self) -> float:
        return control_period_ms(self.control_hz)

    @property
    def soft_budget_ms(self) -> float:
        return self.hard_budget_ms * (1.0 - self.margin_pct / 100.0)

    @classmethod
    def from_env(cls) -> RealTimeBudgetConfig:
        return cls(
            control_hz=int(os.environ.get("SFD_CONTROL_HZ", "50")),
            margin_pct=float(os.environ.get("SFD_RT_MARGIN_PCT", "10")),
        )


@dataclass
class TimingStats:
    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    stdev_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


@dataclass
class RealTimeTickSample:
    index: int
    wall_ms: float
    iter: int = -1
    step: int = -1
    label: str = ""


@dataclass
class RealTimeBudgetReport:
    label: str
    config: RealTimeBudgetConfig
    stats: TimingStats
    within_hard_budget: int = 0
    within_soft_budget: int = 0
    total_samples: int = 0
    hard_outliers: list[RealTimeTickSample] = field(default_factory=list)
    statistical_outlier_threshold_ms: float = 0.0
    statistical_outliers: list[RealTimeTickSample] = field(default_factory=list)

    @property
    def hard_budget_ms(self) -> float:
        return self.config.hard_budget_ms

    @property
    def soft_budget_ms(self) -> float:
        return self.config.soft_budget_ms

    @property
    def hard_pass_rate_pct(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return 100.0 * self.within_hard_budget / self.total_samples

    @property
    def soft_pass_rate_pct(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return 100.0 * self.within_soft_budget / self.total_samples

    @property
    def grade(self) -> str:
        return realtime_pass_grade(self)

    @property
    def effective_hz(self) -> float:
        if self.stats.mean_ms <= 1e-9:
            return 0.0
        return 1000.0 / self.stats.mean_ms


def compute_timing_stats(samples_ms: Sequence[float]) -> TimingStats:
    stats = TimingStats()
    if not samples_ms:
        return stats
    ordered = sorted(float(x) for x in samples_ms)
    n = len(ordered)
    stats.count = n
    stats.min_ms = ordered[0]
    stats.max_ms = ordered[-1]
    stats.mean_ms = statistics.fmean(ordered)
    stats.stdev_ms = statistics.pstdev(ordered) if n > 1 else 0.0

    def percentile(p: float) -> float:
        idx = min(n - 1, int((n - 1) * p))
        return ordered[idx]

    stats.p50_ms = percentile(0.50)
    stats.p95_ms = percentile(0.95)
    stats.p99_ms = percentile(0.99)
    return stats


def realtime_pass_grade(report: RealTimeBudgetReport) -> str:
    """SFD_CoreService_Test::RealTimePassGrade 와 동일."""
    if report.total_samples <= 0:
        return "UNKNOWN"
    hard_pass = report.hard_pass_rate_pct
    soft_pass = report.soft_pass_rate_pct
    p99 = report.stats.p99_ms
    soft_budget = report.soft_budget_ms
    hard_budget = report.hard_budget_ms

    if p99 <= soft_budget and hard_pass >= 99.9:
        return "PASS"
    if p99 <= hard_budget and hard_pass >= 99.0:
        return "MARGINAL"
    if report.stats.mean_ms <= hard_budget and hard_pass >= 95.0:
        return "MARGINAL"
    if soft_pass >= 99.0:
        return "MARGINAL"
    return "FAIL"


def detect_hard_budget_outliers(
    samples: Sequence[RealTimeTickSample],
    hard_budget_ms: float,
) -> list[RealTimeTickSample]:
    """20ms(hard budget) 초과 샘플 — 실시간 제어 위반."""
    return [s for s in samples if s.wall_ms > hard_budget_ms]


def detect_statistical_outliers(
    samples: Sequence[RealTimeTickSample],
    stats: TimingStats,
) -> tuple[float, list[RealTimeTickSample]]:
    """max(p99, mean+3*stdev) 초과 — C++ PrintRunningOutliers 와 동일 임계."""
    threshold = max(stats.p99_ms, stats.mean_ms + 3.0 * stats.stdev_ms)
    outliers = [s for s in samples if s.wall_ms >= threshold]
    outliers.sort(key=lambda s: s.wall_ms, reverse=True)
    return threshold, outliers


def build_realtime_budget_report(
    label: str,
    samples: Sequence[RealTimeTickSample],
    config: RealTimeBudgetConfig | None = None,
) -> RealTimeBudgetReport:
    cfg = config or RealTimeBudgetConfig.from_env()
    wall_ms = [s.wall_ms for s in samples]
    stats = compute_timing_stats(wall_ms)
    hard_outliers = detect_hard_budget_outliers(samples, cfg.hard_budget_ms)
    stat_threshold, stat_outliers = detect_statistical_outliers(samples, stats)

    within_hard = sum(1 for w in wall_ms if w <= cfg.hard_budget_ms)
    within_soft = sum(1 for w in wall_ms if w <= cfg.soft_budget_ms)

    return RealTimeBudgetReport(
        label=label,
        config=cfg,
        stats=stats,
        within_hard_budget=within_hard,
        within_soft_budget=within_soft,
        total_samples=len(wall_ms),
        hard_outliers=sorted(hard_outliers, key=lambda s: s.wall_ms, reverse=True),
        statistical_outlier_threshold_ms=stat_threshold,
        statistical_outliers=stat_outliers,
    )


def format_realtime_budget_report(report: RealTimeBudgetReport) -> str:
    cfg = report.config
    lines = [
        f"[realtime][{report.label}] target={cfg.control_hz}Hz "
        f"budget={report.hard_budget_ms:.3f}ms "
        f"soft={report.soft_budget_ms:.3f}ms (margin={cfg.margin_pct:g}%) "
        f"grade={report.grade}",
        f"  samples={report.total_samples} "
        f"mean={report.stats.mean_ms:.3f} "
        f"p50={report.stats.p50_ms:.3f} "
        f"p95={report.stats.p95_ms:.3f} "
        f"p99={report.stats.p99_ms:.3f} "
        f"max={report.stats.max_ms:.3f} ms",
        f"  pass_hard={report.hard_pass_rate_pct:.2f}% "
        f"pass_soft={report.soft_pass_rate_pct:.2f}% "
        f"effective_hz~={report.effective_hz:.1f}",
        f"  hard_budget_violations(>{report.hard_budget_ms:.3f}ms)="
        f"{len(report.hard_outliers)}",
    ]
    return "\n".join(lines)


def format_outlier_lines(
    outliers: Sequence[RealTimeTickSample],
    *,
    budget_ms: float,
    max_lines: int = 32,
    title: str = "hard budget violations",
) -> str:
    if not outliers:
        return f"[outlier] {title}: count=0 (all samples <= {budget_ms:.3f}ms)"
    lines = [
        f"[outlier] {title}: count={len(outliers)} (budget={budget_ms:.3f}ms)",
    ]
    for sample in outliers[:max_lines]:
        extra = []
        if sample.iter >= 0:
            extra.append(f"iter={sample.iter}")
        if sample.step >= 0:
            extra.append(f"step={sample.step}")
        if sample.label:
            extra.append(sample.label)
        suffix = " ".join(extra)
        delta = sample.wall_ms - budget_ms
        lines.append(
            f"  index={sample.index} wall_ms={sample.wall_ms:.3f} "
            f"(+{delta:.3f} ms over budget) {suffix}".rstrip()
        )
    if len(outliers) > max_lines:
        lines.append(f"  ... {len(outliers) - max_lines} more")
    return "\n".join(lines)


class CriRealtimeMonitor:
    """RunSolver_CUDA_CRI_AtMotionState 호출 wall time 누적·아웃라이어 탐지."""

    def __init__(self, config: RealTimeBudgetConfig | None = None) -> None:
        self.config = config or RealTimeBudgetConfig.from_env()
        self._samples: list[RealTimeTickSample] = []
        self._next_index = 0

    @classmethod
    def from_env(cls) -> CriRealtimeMonitor:
        return cls(RealTimeBudgetConfig.from_env())

    def clear(self) -> None:
        self._samples.clear()
        self._next_index = 0

    def record(
        self,
        wall_ms: float,
        *,
        iter: int = -1,
        step: int = -1,
        label: str = "",
    ) -> RealTimeTickSample:
        sample = RealTimeTickSample(
            index=self._next_index,
            wall_ms=float(wall_ms),
            iter=iter,
            step=step,
            label=label,
        )
        self._next_index += 1
        self._samples.append(sample)
        return sample

    @property
    def samples(self) -> list[RealTimeTickSample]:
        return list(self._samples)

    @property
    def wall_ms_values(self) -> list[float]:
        return [s.wall_ms for s in self._samples]

    def build_report(self, label: str) -> RealTimeBudgetReport:
        return build_realtime_budget_report(label, self._samples, self.config)

    def print_report(self, label: str) -> RealTimeBudgetReport:
        report = self.build_report(label)
        print(format_realtime_budget_report(report))
        print(
            format_outlier_lines(
                report.hard_outliers,
                budget_ms=report.hard_budget_ms,
                title=f"{label} hard budget violations",
            )
        )
        if report.statistical_outliers and len(report.statistical_outliers) != len(
            report.hard_outliers
        ):
            print(
                format_outlier_lines(
                    report.statistical_outliers,
                    budget_ms=report.statistical_outlier_threshold_ms,
                    max_lines=16,
                    title=(
                        f"{label} statistical outliers "
                        f"(threshold={report.statistical_outlier_threshold_ms:.3f}ms)"
                    ),
                )
            )
        return report

    def raise_if_not_pass(
        self,
        label: str = "CRI realtime",
        *,
        min_grade: str = "PASS",
    ) -> RealTimeBudgetReport:
        report = self.build_report(label)
        order = {"PASS": 3, "MARGINAL": 2, "FAIL": 1, "UNKNOWN": 0}
        if order.get(report.grade, 0) < order.get(min_grade, 3):
            worst = report.hard_outliers[0] if report.hard_outliers else None
            detail = ""
            if worst is not None:
                detail = (
                    f" worst index={worst.index} wall_ms={worst.wall_ms:.3f}ms "
                    f"(budget={report.hard_budget_ms:.3f}ms)"
                )
            raise RuntimeError(
                f"{label}: grade={report.grade} (required>={min_grade}). "
                f"hard_violations={len(report.hard_outliers)}/{report.total_samples}, "
                f"pass_hard={report.hard_pass_rate_pct:.2f}%, "
                f"max={report.stats.max_ms:.3f}ms.{detail}"
            )
        return report

    def check_last_within_budget(self, wall_ms: float) -> bool:
        return wall_ms <= self.config.hard_budget_ms


def _call_service(solver_service: Any, snake: str, camel: str, *args: Any, **kwargs: Any) -> Any:
    fn = getattr(solver_service, snake, None)
    if callable(fn):
        return fn(*args, **kwargs)
    fn = getattr(solver_service, camel, None)
    if callable(fn):
        return fn(*args, **kwargs)
    raise AttributeError(f"{type(solver_service).__name__} has neither {snake} nor {camel}")


def configure_cri_filter(
    solver_service: Any,
    cri_limit: float = 0.96,
    alpha: float = 0.02,
    enabled: bool | None = True,
) -> dict[str, float | bool]:
    """Isaac CRI_F: LoadAnalysis 이후 1회. 하드 한도 0.96 + 근방 계수 α=0.02."""
    _call_service(solver_service, "set_cri_filter_limit", "RunSolver_CUDA_SetCriFilterLimit", cri_limit)
    _call_service(solver_service, "set_cbf_alpha", "RunSolver_CUDA_SetCbfAlpha", alpha)
    if enabled is not None:
        _call_service(solver_service, "set_cri_filter_enabled", "RunSolver_CUDA_SetCriFilterEnabled", enabled)
    approach = _call_service(
        solver_service, "cri_filter_approach_limit", "cudaCriFilterApproachLimit"
    )
    return {
        "cri_limit": float(cri_limit),
        "cbf_alpha": float(alpha),
        "approach_limit": float(approach),
        "enabled": True if enabled is None else bool(enabled),
    }


def run_cri_filter(
    solver_service: Any,
    q_batch: Any,
    qd_rl: Any,
    *,
    enabled: bool | None = None,
    monitor: CriRealtimeMonitor | None = None,
    index: int | None = None,
    iter: int = -1,
    step: int = -1,
    label: str = "",
    silent_native_io: Any | None = None,
) -> tuple[Any, float]:
    """
    CRI_F: RunSolver_CUDA_CRI_Filter(q, qd_RL). GPU sync 포함 wall time.
    반환 (result, wall_ms). result는 dict(cri, qd_cmd, cbf_alpha, ...) 또는 C++ 구조체.
    """
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {}
    if enabled is not None:
        kwargs["enabled"] = enabled
    if silent_native_io is not None:
        with silent_native_io():
            result = _call_service(
                solver_service, "run_cri_filter", "RunSolver_CUDA_CRI_Filter", q_batch, qd_rl, **kwargs
            )
    else:
        result = _call_service(
            solver_service, "run_cri_filter", "RunSolver_CUDA_CRI_Filter", q_batch, qd_rl, **kwargs
        )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if monitor is not None:
        monitor.record(wall_ms, iter=iter, step=step, label=label)
        _ = index
    cri = result["cri"] if isinstance(result, dict) else getattr(result, "CRI", None)
    if cri is not None and bool(getattr(cri, "is_cuda", False)):
        if isinstance(result, dict):
            result = dict(result)
            result["cri"] = cri.clone()
            if result.get("cri_pre") is not None and bool(getattr(result["cri_pre"], "is_cuda", False)):
                result["cri_pre"] = result["cri_pre"].clone()
            if result.get("qd_cmd") is not None and bool(getattr(result["qd_cmd"], "is_cuda", False)):
                result["qd_cmd"] = result["qd_cmd"].clone()
    return result, wall_ms


def run_cri_at_motion_state(
    solver_service: Any,
    q_batch: Any,
    qd_batch: Any,
    *,
    monitor: CriRealtimeMonitor | None = None,
    index: int | None = None,
    iter: int = -1,
    step: int = -1,
    label: str = "",
    silent_native_io: Any | None = None,
) -> tuple[Any, float]:
    """
    GPU sync 포함 wall time 측정 후 CRI 반환.
    monitor 가 있으면 자동 record.
    """
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if silent_native_io is not None:
        with silent_native_io():
            cri_gpu = solver_service.RunSolver_CUDA_CRI_AtMotionState(q_batch, qd_batch)
    else:
        cri_gpu = solver_service.RunSolver_CUDA_CRI_AtMotionState(q_batch, qd_batch)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if monitor is not None:
        idx = index if index is not None else monitor._next_index
        monitor.record(wall_ms, iter=iter, step=step, label=label)
    # SafetySolver CalculateLoop_T 재사용 버퍼(_loopTBuffer)와 storage 공유 시
    # 다음 CRI 호출 zero_() 로 학습/디버그 print 가 0으로 보일 수 있다.
    if cri_gpu is not None and bool(getattr(cri_gpu, "is_cuda", False)):
        cri_gpu = cri_gpu.clone()
    return cri_gpu, wall_ms


def bench_trajectory_cri(
    solver_service: Any,
    q_batches: Sequence[Any],
    qd_batches: Sequence[Any],
    *,
    times: Sequence[float] | None = None,
    repeat: int = 500,
    warmup: int = 50,
    monitor: CriRealtimeMonitor | None = None,
    silent_native_io: Any | None = None,
    print_cri_on_first_iter: bool = False,
) -> CriRealtimeMonitor:
    """전체 궤적 repeat 회 반복 CRI 측정 + 아웃라이어 탐지."""
    if len(q_batches) != len(qd_batches):
        raise ValueError("q_batches and qd_batches length mismatch")
    num_steps = len(q_batches)
    mon = monitor or CriRealtimeMonitor.from_env()

    def run_once(iter_idx: int, record: bool) -> None:
        for step in range(num_steps):
            cri, wall_ms = run_cri_at_motion_state(
                solver_service,
                q_batches[step],
                qd_batches[step],
                monitor=mon if record else None,
                iter=iter_idx,
                step=step,
                label=f"t={times[step]:g}" if times is not None else "",
                silent_native_io=silent_native_io,
            )
            if print_cri_on_first_iter and iter_idx == 0 and step == 0:
                print(f"[cri] iter0 step0 sample:\n{cri}")
            if cri is None or getattr(cri, "numel", lambda: 0)() == 0:
                raise RuntimeError(f"CRI empty at iter={iter_idx} step={step}")

    for w in range(warmup):
        run_once(-1 - w, record=False)

    for it in range(repeat):
        run_once(it, record=True)

    return mon
