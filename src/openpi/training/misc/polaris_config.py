"""PolaRiS baseline policy configs."""

from pathlib import Path
from typing import TypeAlias

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.droid_policy as droid_policy
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType

_DROID_RLDS_DIR = Path("/media/safetics/D/droid/1.0.1")
_E_ROOT = Path("/mnt/E") if Path("/mnt/E").is_dir() else Path("/media/safetics/E")
_POLARIS_RLDS_DIR = _E_ROOT / "rlds" / "polaris_droid_cotrain_dataset" / "1.0.0"
_DROID_CRI_ADDON = _E_ROOT / "droid_1.0.1_cri_addon_v3"
_POLARIS_CRI_ADDON = _E_ROOT / "rlds" / "polaris_droid_cotrain_dataset_cri_addon_v3"


def _local_droid_dataset(*, weight: float, cri: bool = False) -> droid_rlds_dataset.RLDSDataset:
    return droid_rlds_dataset.RLDSDataset(
        name="droid",
        version="1.0.1",
        weight=weight,
        filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        builder_dir=str(_DROID_RLDS_DIR),
        cri_addon_dir=str(_DROID_CRI_ADDON) if cri else None,
    )


def _local_polaris_dataset(*, weight: float, cri: bool = False) -> droid_rlds_dataset.RLDSDataset:
    return droid_rlds_dataset.RLDSDataset(
        name="polaris_droid_cotrain_dataset",
        version="1.0.0",
        weight=weight,
        filter_dict_path="gs://openpi-assets/droid/polaris_droid_cotrain_dataset_sample_ranges_v1_0_0.json",
        builder_dir=str(_POLARIS_RLDS_DIR),
        cri_addon_dir=str(_POLARIS_CRI_ADDON) if cri else None,
    )


def get_polaris_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import RLDSDroidDataConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig
    from openpi.training import config as _cfg

    return [
        #
        # PolaRiS DROID jointpos policies
        #
        TrainConfig(
            name="pi05_droid_jointpos_polaris",
            model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
            data=RLDSDroidDataConfig(
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/assets",
                    asset_id="droid",
                ),
                datasets=(
                    _local_droid_dataset(weight=0.9),
                    _local_polaris_dataset(weight=0.1),
                ),
                rlds_data_dir=str(_DROID_RLDS_DIR.parent.parent),
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            num_train_steps=1_000,
            batch_size=128,
            log_interval=100,
            save_interval=1000,
            keep_period=1000,
            num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        ),
        TrainConfig(
            name="pi0_fast_droid_jointpos_polaris",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=10,
                max_token_len=180,
            ),
            data=RLDSDroidDataConfig(
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/polaris/pi0_fast_droid_jointpos_polaris/assets",
                    asset_id="droid",
                ),
                datasets=(
                    _local_droid_dataset(weight=0.9),
                    _local_polaris_dataset(weight=0.1),
                ),
                rlds_data_dir=str(_DROID_RLDS_DIR.parent.parent),
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/polaris/pi0_fast_droid_jointpos_polaris/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            num_train_steps=1_000,
            batch_size=128,
            log_interval=100,
            save_interval=1000,
            keep_period=1000,
            num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        ),
        TrainConfig(
            name="pi0_droid_jointpos_polaris",
            model=pi0_config.Pi0Config(
                # action_dim=8, # leave as 32 default...
                action_horizon=10,
                max_token_len=100,
            ),
            data=RLDSDroidDataConfig(
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris/assets",
                    asset_id="droid",
                ),
                datasets=(
                    _local_droid_dataset(weight=0.9),
                    _local_polaris_dataset(weight=0.1),
                ),
                rlds_data_dir=str(_DROID_RLDS_DIR.parent.parent),
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            num_train_steps=1_000,
            batch_size=128,
            log_interval=100,
            save_interval=1000,
            keep_period=1000,
            num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        ),
        TrainConfig(
            name="pi0_droid_jointpos_100k_polaris",
            model=pi0_config.Pi0Config(
                # action_dim=8, # leave as 32 default...
                action_horizon=10,
                max_token_len=100,
            ),
            data=RLDSDroidDataConfig(
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_100k_polaris/assets",
                    asset_id="droid",
                ),
                datasets=(
                    _local_droid_dataset(weight=0.9),
                    _local_polaris_dataset(weight=0.1),
                ),
                rlds_data_dir=str(_DROID_RLDS_DIR.parent.parent),
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_100k_polaris/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            num_train_steps=1_000,
            batch_size=128,
            log_interval=100,
            save_interval=1000,
            keep_period=1000,
            num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        ),
        TrainConfig(
            # PolaRiS mix + delayed CRI-F sidecars, LoRA on pi05_droid_jointpos_polaris.
            # CRI is a TacVLA-style always-on prefix stream (not language bins, no gate).
            # Language matches official polaris (Task/State only, max_token_len=200).
            # DROID addon: /mnt/E/droid_1.0.1_cri_addon_v3
            # PolaRiS addon: /mnt/E/rlds/polaris_droid_cotrain_dataset_cri_addon_v3
            name="pi05_droid_jointpos_polaris_cri",
            model=pi0_config.Pi0Config(
                action_horizon=15,
                pi05=True,
                use_cri_prefix=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=RLDSDroidDataConfig(
                repo_id="droid",
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/assets",
                    asset_id="droid",
                ),
                datasets=(
                    _local_droid_dataset(weight=0.9, cri=True),
                    _local_polaris_dataset(weight=0.1, cri=True),
                ),
                rlds_data_dir=str(_DROID_RLDS_DIR.parent.parent),
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
                include_cri=True,
                rlds_shuffle_buffer_size=32_000,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=20_000,
                decay_lr=5e-5,
            ),
            num_train_steps=20_000,
            batch_size=8,
            log_interval=100,
            save_interval=10_000,
            keep_period=10_000,
            num_workers=0,
            freeze_filter=pi0_config.Pi0Config(
                action_horizon=15,
                pi05=True,
                use_cri_prefix=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
            checkpoint_base_dir=str(_cfg._CHECKPOINT_OUTPUT_DIR),
            assets_base_dir=str(_cfg._TRAIN_OUTPUT_DIR / "assets"),
        ),
        TrainConfig(
            # Official polaris weights, CRI prefix adapter only.
            # Zero-init out_proj/pos so step-0 tokens are 0; freeze everything except cri_encoder.
            # Mix and horizon match NVIDIA PolaRiS (90/10, action_horizon=15).
            name="pi05_droid_jointpos_polaris_cri_adapter",
            model=pi0_config.Pi0Config(
                action_horizon=15,
                pi05=True,
                use_cri_prefix=True,
                cri_encoder_zero_init=True,
                train_cri_encoder_only=True,
            ),
            data=RLDSDroidDataConfig(
                repo_id="droid",
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/assets",
                    asset_id="droid",
                ),
                datasets=(
                    _local_droid_dataset(weight=0.9, cri=True),
                    _local_polaris_dataset(weight=0.1, cri=True),
                ),
                rlds_data_dir=str(_DROID_RLDS_DIR.parent.parent),
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
                include_cri=True,
                rlds_shuffle_buffer_size=32_000,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=50_000,
                decay_lr=5e-5,
            ),
            num_train_steps=50_000,
            batch_size=8,
            log_interval=100,
            save_interval=10_000,
            keep_period=10_000,
            num_workers=0,
            freeze_filter=pi0_config.Pi0Config(
                action_horizon=15,
                pi05=True,
                use_cri_prefix=True,
                cri_encoder_zero_init=True,
                train_cri_encoder_only=True,
            ).get_freeze_filter(),
            ema_decay=None,
            checkpoint_base_dir=str(_cfg._CHECKPOINT_OUTPUT_DIR),
            assets_base_dir=str(_cfg._TRAIN_OUTPUT_DIR / "assets"),
        ),
        # openpi doesn't support finetuning of binning policies, so this is an inference-only config
        TrainConfig(
            name="paligemma_binning_droid_jointpos",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                max_token_len=600,
                fast_model_tokenizer=_tokenizer.BinningTokenizer,
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                    outputs=[
                        _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                        droid_policy.DroidOutputs(),
                    ],
                ),
            ),
        ),
    ]
