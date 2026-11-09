import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # If True, forward observation/cri as a separate always-on prefix stream
    # (not language-binned, not concatenated into continuous proprio state).
    use_cri: bool = False

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if self.use_cri:
            if "observation/cri" not in data:
                raise KeyError("use_cri=True but observation/cri is missing from the batch")
            inputs["cri"] = np.asarray(data["observation/cri"], dtype=np.float32).reshape(-1)

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][..., :8])}


def _copy_if_missing(data: dict, dest: str, src: str) -> None:
    if dest not in data and src in data:
        data[dest] = data[src]


@dataclasses.dataclass(frozen=True)
class AliasDroidRobotObs(transforms.DataTransformFn):
    """Accept both robot-client keys and RLDS keys used by DROID training configs."""

    def __call__(self, data: dict) -> dict:
        out = dict(data)
        _copy_if_missing(out, "observation/image", "observation/exterior_image_1_left")
        _copy_if_missing(out, "observation/exterior_image_1_left", "observation/image")
        _copy_if_missing(out, "observation/wrist_image", "observation/wrist_image_left")
        _copy_if_missing(out, "observation/wrist_image_left", "observation/wrist_image")
        return out


@dataclasses.dataclass
class _LiveCriState:
    prev_q: np.ndarray | None = None
    prev_cri: np.ndarray | None = None
    prompt: str | None = None
    solver: object | None = None


@dataclasses.dataclass(frozen=True)
class AttachLiveCri(transforms.DataTransformFn):
    """Fill ``observation/cri`` with delayed CRI-F ``cri_pre`` when the client omits it.

    Matches IsaacLab / RLinf CRI-F and the RLDS sidecar: first query is zeros,
    later queries use the previous ``run_cri_filter`` ``cri_pre``. CBF-QP plant
    rewrite stays off (``filter_enabled=False``); this is not raw ``compute_cri``.
    """

    dt: float = 1.0 / 15.0
    num_joints: int = 7
    _state: _LiveCriState = dataclasses.field(default_factory=_LiveCriState)

    def __call__(self, data: dict) -> dict:
        if "observation/cri" in data:
            return data
        if "observation/joint_position" not in data:
            raise KeyError("AttachLiveCri needs observation/joint_position when observation/cri is missing")

        prompt = data.get("prompt")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        if prompt is not None and prompt != self._state.prompt:
            self._state.prev_q = None
            self._state.prev_cri = None
            self._state.prompt = str(prompt)

        q = np.asarray(data["observation/joint_position"], dtype=np.float64).reshape(-1)[: self.num_joints]
        if "observation/joint_velocity" in data:
            qd = np.asarray(data["observation/joint_velocity"], dtype=np.float64).reshape(-1)[: self.num_joints]
        elif self._state.prev_q is None:
            qd = np.zeros(self.num_joints, dtype=np.float64)
        else:
            qd = (q - self._state.prev_q) / self.dt

        cri_obs = (
            np.zeros(9, dtype=np.float32)
            if self._state.prev_cri is None
            else self._state.prev_cri
        )
        cri_now = self._compute_cri(q, qd)
        self._state.prev_q = q
        self._state.prev_cri = cri_now
        return {**data, "observation/cri": cri_obs}

    def _compute_cri(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        from openpi.cri.solver import CriSolver

        if self._state.solver is None:
            # Same contract as annotate_rlds_addon / RLinf fridge env:
            # CRI-F API on, CBF-QP command rewrite off.
            self._state.solver = CriSolver(cri_filter=True, filter_enabled=False)
        result = self._state.solver.run_cri_filter(q, qd)
        cri = np.asarray(result["cri_pre"], dtype=np.float32).reshape(-1)
        return cri[:9]
