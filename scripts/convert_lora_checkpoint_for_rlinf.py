#!/usr/bin/env python3
"""Convert an OpenPI JAX LoRA Orbax checkpoint into RLinf-loadable PyTorch weights.

RLinf's OpenPI loader expects a directory with ``*.safetensors`` (typically
``model.safetensors``). OpenPI LoRA finetunes are saved as JAX/Orbax trees with
separate ``lora_a`` / ``lora_b`` adapters. The stock JAX→PyTorch converter used
to drop those adapters silently; this script:

1. Merges LoRA into dense base weights (openpi-specific attn/MLP quirks).
2. Writes ``model.safetensors`` + ``config.json`` (+ ``assets/`` when present).
3. Defaults to float32 storage (bf16 merged LoRA drifts ~10x vs JAX).

Example (this repo's CRI LoRA run)::

    uv run scripts/convert_lora_checkpoint_for_rlinf.py \\
        --checkpoint-dir /mnt/E/openpi/checkpoints/pi05_droid_cri_finetune/droid_cri_lora/19999 \\
        --config-name pi05_droid_cri_finetune \\
        --output-path /mnt/E/openpi/checkpoints/pi05_droid_cri_finetune/droid_cri_lora/19999_rlinf

Then point RLinf at the output directory::

    rollout.model.model_path: /path/to/10000_rlinf
    actor.model.model_path: /path/to/10000_rlinf

Notes:
- Merged weights are full dense π₀.₅ weights; RLinf does not need ``is_lora: True``
  just to load them. Use ``is_lora`` only if you want LoRA *during* RL updates.
- Action dim / horizon / CRI prompt layout must still match the RLinf env config
  you train on; this script only converts weight format.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from typing import Literal

import tyro

# Allow importing the converter module from examples/.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EXAMPLES = _REPO_ROOT / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import convert_jax_model_to_pytorch as _converter  # noqa: E402
import openpi.models.pi0_config  # noqa: E402
import openpi.training.config as _config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    checkpoint_dir: str,
    config_name: str = "pi05_droid_cri_finetune",
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16"] = "float32",
    *,
    inspect_only: bool = False,
) -> None:
    """Merge LoRA + convert JAX Orbax checkpoint for RLinf ``model_path``.

    Args:
        checkpoint_dir: OpenPI step dir containing ``params/`` (e.g. ``.../10000``).
        config_name: TrainConfig name used for the LoRA run (must be a Pi0Config).
        output_path: Destination directory. Defaults to ``<checkpoint_dir>_rlinf``.
        precision: Storage dtype. Keep float32 for LoRA merges unless you accept drift.
        inspect_only: Only print Orbax param keys.
    """
    train_cfg = _config.get_config(config_name)
    model_config = train_cfg.model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")

    if not _converter._has_lora(model_config):
        logger.warning(
            "Config %s has no LoRA variants; conversion still proceeds but merge is a no-op.",
            config_name,
        )
    elif precision != "float32":
        logger.warning(
            "LoRA merge with precision=%s increases numerical drift vs JAX; prefer float32.",
            precision,
        )

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if not (pathlib.Path(checkpoint_dir) / "params").exists():
        raise FileNotFoundError(f"Expected Orbax params at {checkpoint_dir}/params")

    if inspect_only:
        _converter.load_jax_model_and_print_keys(checkpoint_dir)
        return

    if output_path is None:
        output_path = f"{checkpoint_dir.rstrip('/')}_rlinf"
    output_path = os.path.abspath(output_path)

    logger.info("Converting %s → %s (config=%s, precision=%s)", checkpoint_dir, output_path, config_name, precision)
    _converter.convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)

    # Small sidecar for RLinf wiring (action dims / openpi config name).
    sidecar = {
        "rlinf_model_path": output_path,
        "openpi_config_name": config_name,
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "num_action_chunks_hint": model_config.action_horizon,
        "pi05": bool(model_config.pi05),
        "lora_merged": _converter._has_lora(model_config),
        "notes": [
            "Set rollout.model.model_path and actor.model.model_path to rlinf_model_path.",
            "Align actor.model.action_dim / num_action_chunks with this config (or your env).",
            "Merged checkpoint is dense; is_lora is only needed if RL itself trains LoRA.",
        ],
    }
    sidecar_path = pathlib.Path(output_path) / "rlinf_load_hint.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    logger.info("Wrote %s", sidecar_path)
    logger.info("Done. Point RLinf model_path at: %s", output_path)


if __name__ == "__main__":
    tyro.cli(main)
