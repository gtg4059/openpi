"""CRI constants matching IsaacLab Safetics / Panda analysis bundled under ``openpi.cri``."""

from __future__ import annotations

import os
from pathlib import Path

# Package root: contains lib/, Engine/, ST_AnalysisInfo.json, Robot_Model/.
PACKAGE_DIR: Path = Path(__file__).resolve().parent

# Safetics CUDA CRI output width for the bundled Panda analysis:
# 8 RobotColliPoint channels + 1 aggregate channel (engine returns shape (B, 9)).
# Per-point channels may be ~0 without scene obstacles; the last channel still carries signal.
NUM_CRI_POINTS: int = 9

# IsaacLab articulation_data._store_cri_output_buffers
CRI_CLAMP_MIN: float = 0.0
CRI_CLAMP_MAX: float = 2.0

# IsaacLab SFD_CRI_ZERO_VEL_EPS (articulation_data._apply_cri_zero_vel_filter)
DEFAULT_ZERO_VEL_EPS: float = float(os.environ.get("SFD_CRI_ZERO_VEL_EPS", "1e-6"))

# Franka Panda arm DOF used by DROID + bundled Panda ST_AnalysisInfo.json.
DEFAULT_NUM_JOINTS: int = 7

# DROID teleop / control rate (examples/droid/main.py).
DROID_CONTROL_FREQUENCY_HZ: float = 15.0
DROID_CONTROL_DT: float = 1.0 / DROID_CONTROL_FREQUENCY_HZ
