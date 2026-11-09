"""Standalone Safetics CRI solver bootstrap (no Isaac Sim / Omniverse)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from openpi.cri.constants import CBF_ALPHA
from openpi.cri.constants import CRI_FILTER_LIMIT
from openpi.cri.constants import DEFAULT_NUM_JOINTS
from openpi.cri.constants import DEFAULT_ZERO_VEL_EPS
from openpi.cri.constants import NUM_CRI_POINTS
from openpi.cri.cri_realtime_monitor import configure_cri_filter
from openpi.cri.filter import command_delta
from openpi.cri.postprocess import apply_cri_zero_vel_filter
from openpi.cri.postprocess import clamp_cri

logger = logging.getLogger(__name__)

# Bundled under this package (lib/, Engine/, ST_AnalysisInfo.json, Robot_Model/).
_PACKAGE_ANALYSIS_DIR = Path(__file__).resolve().parent
_FALLBACK_ISAACLAB_ARTICULATION = Path(
    "/home/safetics/IsaacLab/source/isaaclab/isaaclab/assets/articulation"
)


def resolve_analysis_dir(analysis_dir: str | Path | None = None) -> Path:
    """Resolve the CUDACRI analysis directory containing ``lib/`` and ``ST_AnalysisInfo.json``.

    Order: explicit arg → ``OPENPI_CRI_ANALYSIS_DIR`` → ``SFD_CRI_ANALYSIS_DIR`` →
    ``openpi/cri`` package dir → IsaacLab articulation path (if present).
    """
    if analysis_dir is not None:
        root = Path(analysis_dir).expanduser().resolve()
    else:
        env = os.environ.get("OPENPI_CRI_ANALYSIS_DIR") or os.environ.get("SFD_CRI_ANALYSIS_DIR")
        if env:
            root = Path(env).expanduser().resolve()
        elif (_PACKAGE_ANALYSIS_DIR / "lib").is_dir() and (_PACKAGE_ANALYSIS_DIR / "ST_AnalysisInfo.json").is_file():
            root = _PACKAGE_ANALYSIS_DIR
        elif _FALLBACK_ISAACLAB_ARTICULATION.is_dir():
            root = _FALLBACK_ISAACLAB_ARTICULATION.resolve()
        else:
            raise FileNotFoundError(
                "CRI analysis dir not found. Expected lib/ + ST_AnalysisInfo.json under "
                f"{_PACKAGE_ANALYSIS_DIR}, or set OPENPI_CRI_ANALYSIS_DIR."
            )
    lib_dir = root / "lib"
    if not lib_dir.is_dir():
        raise FileNotFoundError(f"CUDACRI lib not found: {lib_dir}")
    if not (root / "ST_AnalysisInfo.json").is_file():
        raise FileNotFoundError(f"ST_AnalysisInfo.json not found under {root}")
    return root


def _resolve_bundled_lib_dir(analysis_dir: Path) -> Path:
    """Prefer ``lib/<major.minor>/`` (IsaacLab ``configure_cudacri``), else flat ``lib/``."""
    root = analysis_dir / "lib"
    versioned = root / f"{sys.version_info.major}.{sys.version_info.minor}"
    if versioned.is_dir():
        return versioned
    return root


def _bootstrap_sfd_coreservice(analysis_dir: Path):
    """Import ``sfd_coreservice`` without pulling IsaacLab / Omniverse packages."""
    lib_dir = _resolve_bundled_lib_dir(analysis_dir)
    for path in (analysis_dir, lib_dir, analysis_dir / "lib"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        p for p in (str(lib_dir), str(torch_lib), os.environ.get("LD_LIBRARY_PATH", "")) if p
    )

    # Prefer lightweight path bootstrap; fall back to IsaacLab sfd_setup when present.
    sfd_setup = analysis_dir / "sfd_setup.py"
    if sfd_setup.is_file():
        try:
            from sfd_setup import configure_cudacri  # type: ignore[import-not-found]

            configure_cudacri(analysis_dir)
        except Exception as exc:
            logger.debug("sfd_setup.configure_cudacri skipped: %s", exc)

    import sfd_coreservice  # type: ignore[import-not-found]

    return sfd_coreservice


def _as_batch_f64(x: np.ndarray | torch.Tensor, *, name: str, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.asarray(x, dtype=np.float64))
    elif isinstance(x, torch.Tensor):
        t = x.detach().to(dtype=torch.float64)
    else:
        raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(x)}")
    if t.ndim == 1:
        t = t[None, :]
    if t.ndim != 2:
        raise ValueError(f"{name} must have shape (B, J) or (J,), got {tuple(t.shape)}")
    return t.to(device=device, dtype=torch.float64).contiguous()


class CriSolver:
    """Thin wrapper around ``sfd_coreservice.CoreService`` for OpenPI.

    Usage::

        solver = CriSolver()  # loads analysis from OPENPI_CRI_ANALYSIS_DIR / default
        cri = solver.compute(q, qd)  # (B, NUM_CRI_POINTS) float32, zero-vel filtered + clamped
    """

    def __init__(
        self,
        analysis_dir: str | Path | None = None,
        *,
        batch_size: int = 1,
        device: str | torch.device | None = None,
        num_joints: int = DEFAULT_NUM_JOINTS,
        zero_vel_eps: float = DEFAULT_ZERO_VEL_EPS,
        warmup_rounds: int | None = None,
        cri_filter: bool = False,
        cri_limit: float = CRI_FILTER_LIMIT,
        cbf_alpha: float = CBF_ALPHA,
        filter_enabled: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CriSolver requires CUDA (sfd_coreservice GPU path).")
        self.analysis_dir = resolve_analysis_dir(analysis_dir)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        self.device = torch.device(device or "cuda:0")
        self.num_joints = int(num_joints)
        self.zero_vel_eps = float(zero_vel_eps)
        self.num_cri_points = NUM_CRI_POINTS
        self._cri_limit = float(cri_limit) if cri_limit > 0.0 else CRI_FILTER_LIMIT
        self._cbf_alpha = float(cbf_alpha) if cbf_alpha > 0.0 else CBF_ALPHA
        self._filter_enabled = bool(filter_enabled)
        self._approach_limit = self._cri_limit * (1.0 - self._cbf_alpha)

        sfd = _bootstrap_sfd_coreservice(self.analysis_dir)
        self._solver = sfd.CoreService(str(self.analysis_dir), self.batch_size)
        self._solver.RunSolver_CUDA_LoadAnalysisForCRI(str(self.analysis_dir))

        if cri_filter:
            self._require_cri_filter_api()
            cfg = configure_cri_filter(
                self._solver,
                cri_limit=self._cri_limit,
                alpha=self._cbf_alpha,
                enabled=self._filter_enabled,
            )
            self._cri_limit = float(cfg["cri_limit"])
            self._cbf_alpha = float(cfg["cbf_alpha"])
            self._approach_limit = float(cfg["approach_limit"])
            self._filter_enabled = bool(cfg["enabled"])

        rounds = (
            int(os.environ.get("SFD_ALLOC_WARMUP_ROUNDS", "15"))
            if warmup_rounds is None
            else int(warmup_rounds)
        )
        if rounds > 0:
            q0 = torch.zeros(self.batch_size, self.num_joints, dtype=torch.float64, device=self.device)
            qd0 = torch.zeros_like(q0)
            for _ in range(rounds):
                if cri_filter:
                    self._solver.run_cri_filter(q0.contiguous(), qd0.contiguous())
                else:
                    self._solver.RunSolver_CUDA_CRI_AtMotionState(q0.contiguous(), qd0.contiguous())
            torch.cuda.synchronize(self.device)
            logger.info(
                "CRI solver warm-up complete: %d rounds filter=%s (%s)",
                rounds,
                cri_filter,
                self.analysis_dir,
            )

    def _require_cri_filter_api(self) -> None:
        if not hasattr(self._solver, "run_cri_filter"):
            raise RuntimeError(
                "sfd_coreservice.CoreService.run_cri_filter is missing. "
                "Deploy a CoreService build that exports the CRI filter API."
            )

    def set_cri_filter_limit(self, cri_limit: float) -> None:
        """IsaacLab ``set_cri_filter_limit``."""
        self._require_cri_filter_api()
        self._cri_limit = float(cri_limit) if cri_limit > 0.0 else CRI_FILTER_LIMIT
        if hasattr(self._solver, "set_cri_filter_limit"):
            self._solver.set_cri_filter_limit(self._cri_limit)

    def set_cbf_alpha(self, alpha: float) -> None:
        """IsaacLab ``set_cbf_alpha``."""
        self._require_cri_filter_api()
        self._cbf_alpha = float(alpha) if alpha > 0.0 else CBF_ALPHA
        if hasattr(self._solver, "set_cbf_alpha"):
            self._solver.set_cbf_alpha(self._cbf_alpha)
        elif hasattr(self._solver, "RunSolver_CUDA_SetCbfAlpha"):
            self._solver.RunSolver_CUDA_SetCbfAlpha(self._cbf_alpha)

    def set_cri_filter_enabled(self, enabled: bool) -> None:
        """IsaacLab ``set_cri_filter_enabled`` (CBF-QP on/off)."""
        self._require_cri_filter_api()
        self._filter_enabled = bool(enabled)
        if hasattr(self._solver, "set_cri_filter_enabled"):
            self._solver.set_cri_filter_enabled(self._filter_enabled)

    def _pad_batch(
        self, q: np.ndarray | torch.Tensor, qd: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        q_t = _as_batch_f64(q, name="q", device=self.device)
        qd_t = _as_batch_f64(qd, name="qd", device=self.device)
        if q_t.shape != qd_t.shape:
            raise ValueError(f"q and qd shape mismatch: {tuple(q_t.shape)} vs {tuple(qd_t.shape)}")
        batch, joints = q_t.shape
        if batch > self.batch_size:
            raise ValueError(
                f"batch {batch} exceeds solver batch_size={self.batch_size}; "
                "construct CriSolver(batch_size=...) with a larger batch."
            )
        if joints < self.num_joints:
            raise ValueError(f"expected at least {self.num_joints} joints, got {joints}")
        if batch < self.batch_size:
            q_in = torch.zeros(self.batch_size, joints, dtype=torch.float64, device=self.device)
            qd_in = torch.zeros_like(q_in)
            q_in[:batch] = q_t
            qd_in[:batch] = qd_t
        else:
            q_in = q_t
            qd_in = qd_t
        return q_in, qd_in, batch

    def _fit_cri_width(self, cri: torch.Tensor, batch: int) -> torch.Tensor:
        cri = cri[:batch]
        if cri.shape[-1] == self.num_cri_points:
            return cri
        if cri.shape[-1] > self.num_cri_points:
            head = cri[..., : self.num_cri_points]
            if torch.max(torch.abs(head)) < 1e-12:
                return cri[..., -self.num_cri_points :]
            return head
        pad = torch.zeros(batch, self.num_cri_points - cri.shape[-1], device=cri.device, dtype=cri.dtype)
        return torch.cat([cri, pad], dim=-1)

    def _as_cpu_f32(self, value: Any, batch: int) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
        elif isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            try:
                arr = np.asarray(value)
            except (TypeError, ValueError):
                return None
        if arr.size == 0:
            return None
        if arr.ndim == 0:
            return arr
        return arr[:batch]

    def compute(
        self,
        q: np.ndarray | torch.Tensor,
        qd: np.ndarray | torch.Tensor,
        *,
        return_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Compute ``CRI(q, qd)`` with zero-vel filter and clamp.

        Args:
            q: Joint positions, shape ``(B, J)`` or ``(J,)``.
            qd: Joint velocities, shape ``(B, J)`` or ``(J,)``.
            return_torch: If True, return a CUDA float32 tensor; else numpy float32.

        Returns:
            CRI array of shape ``(B, NUM_CRI_POINTS)``.
        """
        q_in, qd_in, batch = self._pad_batch(q, qd)
        qd_t = qd_in[:batch]

        cri_gpu = self._solver.RunSolver_CUDA_CRI_AtMotionState(q_in.contiguous(), qd_in.contiguous())
        torch.cuda.synchronize(self.device)
        if cri_gpu is None:
            raise RuntimeError("CRI solver returned None")

        cri = self._fit_cri_width(cri_gpu, batch)
        cri = clamp_cri(cri.float())
        cri = apply_cri_zero_vel_filter(cri, qd_t, eps=self.zero_vel_eps)

        if return_torch:
            return cri
        return cri.detach().cpu().numpy().astype(np.float32, copy=False)

    def run_cri_filter(
        self,
        q: np.ndarray | torch.Tensor,
        qd: np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """One IsaacLab ``run_cri_filter(q, qd_RL)`` call for a batch.

        Returns numpy ``cri_pre`` ``(B, NUM_CRI_POINTS)``, ``qd_cmd``, ``delta``,
        ``cri_limit``, ``cbf_alpha``, ``approach_limit``, and ``enabled``.
        """
        self._require_cri_filter_api()
        q_in, qd_in, batch = self._pad_batch(q, qd)
        try:
            result = self._solver.run_cri_filter(q_in.contiguous(), qd_in.contiguous())
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            logger.warning("run_cri_filter native call failed; passthrough qd_RL. %s", exc)
            qd_rl = qd_in[:batch].detach().cpu().numpy().astype(np.float32, copy=False)
            return {
                "cri_pre": np.zeros((batch, self.num_cri_points), dtype=np.float32),
                "qd_cmd": qd_rl,
                "delta": np.zeros((batch,), dtype=np.float32),
                "cri_limit": self._cri_limit,
                "cbf_alpha": self._cbf_alpha,
                "approach_limit": self._approach_limit,
                "enabled": self._filter_enabled,
            }
        if result is None:
            raise RuntimeError("run_cri_filter returned None")
        cri_pre = result.get("cri_pre") if isinstance(result, dict) else None
        if cri_pre is None:
            logger.warning("run_cri_filter returned empty cri_pre; passthrough qd_RL")
            qd_rl = qd_in[:batch].detach().cpu().numpy().astype(np.float32, copy=False)
            return {
                "cri_pre": np.zeros((batch, self.num_cri_points), dtype=np.float32),
                "qd_cmd": qd_rl,
                "delta": np.zeros((batch,), dtype=np.float32),
                "cri_limit": self._cri_limit,
                "cbf_alpha": self._cbf_alpha,
                "approach_limit": self._approach_limit,
                "enabled": self._filter_enabled,
            }
        if isinstance(cri_pre, np.ndarray):
            cri_t = torch.from_numpy(np.asarray(cri_pre)).to(device=self.device)
        else:
            cri_t = cri_pre
        cri_t = self._fit_cri_width(cri_t, batch)
        cri_np = clamp_cri(cri_t.float()).detach().cpu().numpy().astype(np.float32, copy=False)

        def _as_float(key: str, default: float) -> float:
            value = result.get(key, default)
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().reshape(-1)[0].item())
            return float(value)

        enabled = result.get("enabled", self._filter_enabled)
        if isinstance(enabled, torch.Tensor):
            enabled = bool(enabled.detach().cpu().reshape(-1)[0].item())
        else:
            enabled = bool(enabled)

        qd_rl = qd_in[:batch].detach().cpu().numpy().astype(np.float32, copy=False)
        qd_cmd = self._as_cpu_f32(result.get("qd_cmd"), batch)
        if qd_cmd is None or np.asarray(qd_cmd).size == 0:
            qd_cmd = qd_rl
        else:
            qd_cmd = np.asarray(qd_cmd, dtype=np.float32)
            if qd_cmd.ndim == 1:
                qd_cmd = qd_cmd[None, :]
            qd_cmd = qd_cmd[:batch]
        delta = command_delta(qd_cmd, qd_rl)

        return {
            "cri_pre": cri_np,
            "qd_cmd": qd_cmd,
            "delta": delta,
            "cri_limit": _as_float("cri_limit", self._cri_limit),
            "cbf_alpha": _as_float("cbf_alpha", self._cbf_alpha),
            "approach_limit": _as_float("approach_limit", self._approach_limit),
            "enabled": enabled,
        }
