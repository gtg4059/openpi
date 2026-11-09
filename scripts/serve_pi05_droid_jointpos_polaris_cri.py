"""Serve ``pi05_droid_jointpos_polaris_cri`` after ``polaris_cri_lora_20k`` finishes.

The DROID laptop client does not send CRI. The server fills delayed ``cri_pre``
from joint positions (always on, no gate) before the prefix encoder.

Usage (from repo root, after training GPU is free)::

    # Wait for step 20000, then serve on :8000
    uv run scripts/serve_pi05_droid_jointpos_polaris_cri.py --wait

    # Serve a finished checkpoint immediately
    uv run scripts/serve_pi05_droid_jointpos_polaris_cri.py --step=20000
"""

from __future__ import annotations

import logging
import socket
import time
from pathlib import Path

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


_CONFIG_NAME = "pi05_droid_jointpos_polaris_cri"
_DEFAULT_EXP = "polaris_cri_lora_20k"
_DEFAULT_STEP = 20_000


def _checkpoint_root(exp_name: str) -> Path:
    cfg = _config.get_config(_CONFIG_NAME)
    return Path(cfg.checkpoint_base_dir) / _CONFIG_NAME / exp_name


def _is_ready(step_dir: Path) -> bool:
    return (step_dir / "params").exists() or (step_dir / "params" / "_METADATA").exists()


def resolve_checkpoint_dir(exp_name: str, step: int | None, *, wait: bool, poll_s: float) -> Path:
    root = _checkpoint_root(exp_name)
    target = root / str(step if step is not None else _DEFAULT_STEP)
    if _is_ready(target):
        return target
    if not wait:
        ready = sorted(
            (p for p in root.glob("*") if p.is_dir() and p.name.isdigit() and _is_ready(p)),
            key=lambda p: int(p.name),
        )
        if ready:
            logging.info("Requested %s not ready; using latest finished step %s", target, ready[-1])
            return ready[-1]
        raise FileNotFoundError(
            f"No finished checkpoint under {root}. Train first, or pass --wait for step "
            f"{step if step is not None else _DEFAULT_STEP}."
        )

    logging.info("Waiting for %s ...", target)
    while not _is_ready(target):
        time.sleep(poll_s)
    logging.info("Found %s", target)
    return target


def main(
    exp_name: str = _DEFAULT_EXP,
    step: int | None = _DEFAULT_STEP,
    *,
    wait: bool = False,
    poll_s: float = 30.0,
    default_prompt: str | None = None,
    port: int = 8000,
    record: bool = False,
) -> None:
    ckpt = resolve_checkpoint_dir(exp_name, step, wait=wait, poll_s=poll_s)
    logging.info("Serving %s from %s on port %d", _CONFIG_NAME, ckpt, port)
    policy = _policy_config.create_trained_policy(
        _config.get_config(_CONFIG_NAME), ckpt, default_prompt=default_prompt
    )
    if record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
