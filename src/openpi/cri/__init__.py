"""CRI (Collision Risk Index) computation for OpenPI safety modality.

Computes ``CRI(q, qd)`` via the Safetics CUDA solver (IsaacLab ``sfd_coreservice``)
before multimodal fusion. Near-zero joint velocity forces CRI to 0 (first step / at rest).

Offline RLDS annotation writes IsaacLab CRI-F delayed ``cri_pre``
(``openpi.cri.annotate_rlds`` / ``annotate_rlds_addon``).
"""

from openpi.cri.compute import compute_cri
from openpi.cri.constants import CBF_ALPHA
from openpi.cri.constants import CRI_CLAMP_MAX
from openpi.cri.constants import CRI_CLAMP_MIN
from openpi.cri.constants import CRI_FILTER_LIMIT
from openpi.cri.constants import DEFAULT_NUM_JOINTS
from openpi.cri.constants import DEFAULT_ZERO_VEL_EPS
from openpi.cri.constants import DROID_CONTROL_DT
from openpi.cri.constants import NUM_CRI_OBS_DIM
from openpi.cri.constants import NUM_CRI_POINTS
from openpi.cri.constants import PACKAGE_DIR
from openpi.cri.filter import compute_episode_cri_f
from openpi.cri.postprocess import apply_cri_zero_vel_filter
from openpi.cri.postprocess import clamp_cri
from openpi.cri.solver import CriSolver
from openpi.cri.solver import resolve_analysis_dir
from openpi.cri.velocity import joint_velocity_from_positions

__all__ = [
    "CBF_ALPHA",
    "CRI_CLAMP_MAX",
    "CRI_CLAMP_MIN",
    "CRI_FILTER_LIMIT",
    "DEFAULT_NUM_JOINTS",
    "DEFAULT_ZERO_VEL_EPS",
    "DROID_CONTROL_DT",
    "NUM_CRI_OBS_DIM",
    "NUM_CRI_POINTS",
    "PACKAGE_DIR",
    "CriSolver",
    "apply_cri_zero_vel_filter",
    "clamp_cri",
    "compute_cri",
    "compute_episode_cri_f",
    "joint_velocity_from_positions",
    "resolve_analysis_dir",
]
