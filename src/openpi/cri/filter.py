"""CRI-F observation helpers matching IsaacLab ``articulation_data`` CBF filter mode.

IsaacLab CRI-F (``apply_cri_filter`` / ``cri_filter_pre``):

- Reset / first obs: CRI=0, ``qd_cmd=0`` (no solver).
- Each later tick: one ``run_cri_filter(q, qd_RL)``; the *next* policy obs is that ``cri_pre``.
- Command is library CBF-QP ``qd_cmd`` (``u*``). Time-scale ``s`` is not used.
- ``delta = ||qd_cmd - qd_RL||``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openpi.cri.constants import CBF_ALPHA
from openpi.cri.constants import CRI_FILTER_LIMIT
from openpi.cri.constants import DEFAULT_NUM_JOINTS
from openpi.cri.constants import DROID_CONTROL_DT
from openpi.cri.constants import NUM_CRI_POINTS
from openpi.cri.velocity import joint_velocity_from_positions


def command_delta(qd_cmd: np.ndarray | None, qd_rl: np.ndarray) -> np.ndarray:
    """Per-row ``||qd_cmd - qd_RL||`` matching IsaacLab ``apply_cri_filter``."""
    qd = np.asarray(qd_rl, dtype=np.float32)
    if qd.ndim == 1:
        qd = qd[None, :]
    if qd_cmd is None:
        return np.zeros((qd.shape[0],), dtype=np.float32)
    cmd = np.asarray(qd_cmd, dtype=np.float32)
    if cmd.ndim == 1:
        cmd = cmd[None, :]
    if cmd.shape != qd.shape:
        cmd = np.broadcast_to(cmd, qd.shape)
    return np.linalg.norm(cmd - qd, axis=-1).astype(np.float32, copy=False)


def shift_cri_filter_obs(
    cri_pre: np.ndarray,
    extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Policy-obs delay: t=0 is CRI=0 (and extra=0); later ticks read the previous solve."""
    cri = np.asarray(cri_pre, dtype=np.float32)
    if cri.ndim == 1:
        cri = cri[None, :]
    t_len = cri.shape[0]
    cri_obs = np.zeros_like(cri, dtype=np.float32)
    extra_obs = None
    if extra is not None:
        extra_arr = np.asarray(extra, dtype=np.float32).reshape(t_len, -1)
        extra_obs = np.zeros_like(extra_arr, dtype=np.float32)
    if t_len > 1:
        cri_obs[1:] = cri[:-1]
        if extra_obs is not None:
            extra_obs[1:] = extra_arr[:-1]
    return cri_obs, extra_obs


def compute_episode_cri_f(
    joint_positions: np.ndarray,
    *,
    solver: Any,
    dt: float = DROID_CONTROL_DT,
    cri_limit: float = CRI_FILTER_LIMIT,
    cbf_alpha: float = CBF_ALPHA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(qd_rl, cri_obs, delta_obs, qd_cmd)`` via ``run_cri_filter``.

    ``cri_obs`` / ``delta_obs`` are 1-step delayed (IsaacLab CRI-F policy CRI).
    ``qd_cmd`` is the instantaneous CBF-QP command (same length as ``qd_rl``).
    """
    del cbf_alpha  # configured on the solver; kept for call-site parity with Isaac cfg.
    q = np.asarray(joint_positions, dtype=np.float64)
    if q.ndim != 2 or q.shape[-1] < DEFAULT_NUM_JOINTS:
        raise ValueError(f"expected joint_positions (T, >=7), got {q.shape}")
    q = q[:, :DEFAULT_NUM_JOINTS]
    qd = joint_velocity_from_positions(q, dt=dt)

    if not hasattr(solver, "run_cri_filter"):
        raise RuntimeError(
            "sfd_coreservice.CoreService.run_cri_filter is missing. "
            "Deploy a CoreService build that exports the CRI filter API."
        )

    t_len = q.shape[0]
    batch = int(solver.batch_size)
    cri_chunks: list[np.ndarray] = []
    delta_chunks: list[np.ndarray] = []
    cmd_chunks: list[np.ndarray] = []
    for start in range(0, t_len, batch):
        end = min(start + batch, t_len)
        result = solver.run_cri_filter(q[start:end], qd[start:end])
        cri_pre = np.asarray(result["cri_pre"], dtype=np.float32)
        if cri_pre.ndim == 1:
            cri_pre = cri_pre[None, :]
        if cri_pre.shape[-1] != NUM_CRI_POINTS:
            raise ValueError(f"expected CRI width {NUM_CRI_POINTS}, got {cri_pre.shape}")
        qd_rl = qd[start:end].astype(np.float32, copy=False)
        qd_cmd = result.get("qd_cmd")
        if qd_cmd is None:
            qd_cmd = qd_rl
        else:
            qd_cmd = np.asarray(qd_cmd, dtype=np.float32)
            if qd_cmd.ndim == 1:
                qd_cmd = qd_cmd[None, :]
        delta = result.get("delta")
        if delta is None:
            delta = command_delta(qd_cmd, qd_rl)
        else:
            delta = np.asarray(delta, dtype=np.float32).reshape(-1)
        cri_chunks.append(cri_pre)
        delta_chunks.append(delta)
        cmd_chunks.append(qd_cmd)

    cri_inst = np.concatenate(cri_chunks, axis=0)
    delta_inst = np.concatenate(delta_chunks, axis=0)
    qd_cmd = np.concatenate(cmd_chunks, axis=0)
    cri_obs, delta_obs = shift_cri_filter_obs(cri_inst, delta_inst)
    assert delta_obs is not None
    return qd, cri_obs, delta_obs.reshape(-1), qd_cmd
