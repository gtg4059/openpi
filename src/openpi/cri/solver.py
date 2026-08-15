"""Standalone Safetics CRI solver bootstrap (no Isaac Sim / Omniverse)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

import numpy as np
import torch

from openpi.cri.constants import DEFAULT_NUM_JOINTS
from openpi.cri.constants import DEFAULT_ZERO_VEL_EPS
from openpi.cri.constants import NUM_CRI_POINTS
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


def _bootstrap_sfd_coreservice(analysis_dir: Path):
    """Import ``sfd_coreservice`` without pulling IsaacLab / Omniverse packages."""
    lib_dir = analysis_dir / "lib"
    for path in (analysis_dir, lib_dir):
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

        sfd = _bootstrap_sfd_coreservice(self.analysis_dir)
        self._solver = sfd.CoreService(str(self.analysis_dir), self.batch_size)
        self._solver.RunSolver_CUDA_LoadAnalysisForCRI(str(self.analysis_dir))

        rounds = (
            int(os.environ.get("SFD_ALLOC_WARMUP_ROUNDS", "15"))
            if warmup_rounds is None
            else int(warmup_rounds)
        )
        if rounds > 0:
            q0 = torch.zeros(self.batch_size, self.num_joints, dtype=torch.float64, device=self.device)
            qd0 = torch.zeros_like(q0)
            for _ in range(rounds):
                self._solver.RunSolver_CUDA_CRI_AtMotionState(q0.contiguous(), qd0.contiguous())
            torch.cuda.synchronize(self.device)
            logger.info("CRI solver warm-up complete: %d rounds (%s)", rounds, self.analysis_dir)

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

        # Solver is fixed-batch; pad unused rows with zeros.
        if batch < self.batch_size:
            q_in = torch.zeros(self.batch_size, joints, dtype=torch.float64, device=self.device)
            qd_in = torch.zeros_like(q_in)
            q_in[:batch] = q_t
            qd_in[:batch] = qd_t
        else:
            q_in = q_t
            qd_in = qd_t

        cri_gpu = self._solver.RunSolver_CUDA_CRI_AtMotionState(q_in.contiguous(), qd_in.contiguous())
        torch.cuda.synchronize(self.device)
        if cri_gpu is None:
            raise RuntimeError("CRI solver returned None")

        cri = cri_gpu[:batch]
        if cri.shape[-1] != self.num_cri_points:
            # Engine may return (B, 9) = 8 colli-points + aggregate. Prefer keeping the
            # trailing aggregate when truncating would drop the only nonzero channel.
            if cri.shape[-1] > self.num_cri_points:
                head = cri[..., : self.num_cri_points]
                if torch.max(torch.abs(head)) < 1e-12:
                    cri = cri[..., -self.num_cri_points :]
                else:
                    cri = head
            else:
                pad = torch.zeros(batch, self.num_cri_points - cri.shape[-1], device=cri.device, dtype=cri.dtype)
                cri = torch.cat([cri, pad], dim=-1)

        cri = clamp_cri(cri.float())
        cri = apply_cri_zero_vel_filter(cri, qd_t, eps=self.zero_vel_eps)

        if return_torch:
            return cri
        return cri.detach().cpu().numpy().astype(np.float32, copy=False)
