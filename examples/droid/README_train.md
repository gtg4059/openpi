# Training on DROID

Here we describe how to fine-tune the pi0.5 model on the *full* DROID dataset. This is an approximate open-source reproduction of the pi05-DROID training pipeline.
(small differences in data loading and the used action space) -- For a tutorial on how to fine-tune your model with a smaller, custom dataset collected on the DROID platform, see below.

In contrast to the rest of openpi, which uses LeRobot for data loading, we need to use RLDS as the data format for full DROID training (since at the moment LeRobot isn't scalable enough 
for larger datasets like DROID -- they are working on improving it though). Below, we provide instructions for updating your openpi environment for RLDS data loading and where to download the DROID dataset.

## Install

We need a few additional dependencies for RLDS data loading. The RLDS group depends on `tensorflow-cpu==2.15.0`, which only ships wheels for **Python 3.11**. Make sure you create your virtual environment with Python 3.11 before syncing:

```bash
uv python install 3.11
uv venv --python 3.11
uv sync --group rlds
```

## Download DROID dataset

You can download the DROID dataset with the following command (after installing the `gsutil` google cloud CLI):
```
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 <your_download_path>/droid/1.0.1
```

Note that downloading version 1.0.1 is important (not v1.0.0): it contains the complete set of language annotations (~75k episodes) while v1.0.0 only has annotations for 30k episodes. If for some reason you would like to use another version, modify the line `version="1.0.1"` in the `DroidRldsDataset` object [here](src/openpi/training/droid_rlds_dataset.py).

You will need 1.8TB of disk storage to download the DROID RLDS dataset.

## Run

First, change the `rlds_data_dir` path in your `TrainConfig` to the directory that you downloaded the `droid` dataset into (see [src/openpi/training/config.py](src/openpi/training/config.py)).

Then, compute normalization statistics (this will take ~10 minutes):
```bash
uv run --group rlds scripts/compute_norm_stats.py --config-name pi05_full_droid_finetune --max-frames 10_000_000
```

Run training:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py pi05_full_droid_finetune --exp-name=my_experiment --overwrite
```

### Full DROID 1.0.1 + CRI sidecar (TacVLA-style tokens)

Fixed layout used by this workspace:

- DROID RLDS: `/media/safetics/D/droid/1.0.1`
- CRI addon: `/mnt/E/droid_1.0.1_cri_addon` (legacy: `/media/safetics/E/...`)
- Pretrained init (`pi05_droid`, ~12 GiB): `/mnt/E/openpi/pretrained/pi05_droid`
- Checkpoints: `/mnt/E/openpi/checkpoints/pi05_droid_cri_finetune/<exp_name>`

CRI is fused like TacVLA's compact non-vision tokens: discrete LLM span
`Task: ..., State: ..., CRI: <9 bins>;` on top of `pi05_droid` + LoRA
(config `pi05_droid_cri_finetune`).

Keep the init checkpoint on E: so training does not download into `~/.cache`
(root disk is tight). The helper downloads it automatically if missing.
Code resolves E as `/mnt/E` if present, else `/media/safetics/E`.

```bash
# Optional: prefetch pretrained weights yourself
mkdir -p /mnt/E/openpi/pretrained/pi05_droid
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid/* \
  /mnt/E/openpi/pretrained/pi05_droid/

# Build sidecar once (if needed; resumable)
uv run --group rlds python -m openpi.cri.annotate_rlds_addon \
  --source-dir=/media/safetics/D/droid/1.0.1 \
  --addon-dir=/mnt/E/droid_1.0.1_cri_addon

# Start training (norm stats + LoRA fine-tune)
uv run --group rlds scripts/train_pi05_droid_cri.py \
  --exp-name=droid_cri_lora --overwrite
```

### Debug: pi0.5-base on `droid_100` + CRI (RLDS)

Before full-scale DROID+CRI training, smoke-test the pipeline on the annotated subset
`data/droid_100/1.0.0_CRI` (created by `python -m openpi.cri.annotate_rlds`).

This is the sibling of the custom LeRobot entrypoint
`uv run scripts/train.py pi05_droid_finetune --exp-name=my_experiment --overwrite`,
but uses RLDS + `pi05_base`. CRI is **not** concatenated into continuous state; it is
encoded as a separate discrete language span:
`Task: ..., State: ..., CRI: <9 bins>;\nAction: `.

```bash
# One-shot helper (norm stats + short train)
uv run --group rlds scripts/train_pi05_droid_100_cri.py --exp-name=my_experiment --overwrite

# Or step-by-step (same pattern as full DROID above):
uv run --group rlds scripts/compute_norm_stats.py \
  --config-name pi05_droid_100_cri_finetune --max-frames 5000

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py \
  pi05_droid_100_cri_finetune --exp-name=my_experiment --overwrite
```

Override the RLDS location on the command line (no config edit needed), e.g. when data lives under `/media/safetics/D`:

```bash
# Version folder used by ``tfds.builder_from_directory`` (preferred for CRI smoke config)
--rlds-builder-dir=/media/safetics/D/droid_100/1.0.0_CRI

# Or TFDS parent data dir (used when builder_dir is unset)
--rlds-data-dir=/media/safetics/D
```

These flags work on `scripts/train.py`, `scripts/compute_norm_stats.py`, and `scripts/train_pi05_droid_100_cri.py`.

Config name: `pi05_droid_100_cri_finetune` (small batch/steps; `wandb` off by default).

**Note**: The original pi0.5-DROID model was trained with joint velocity actions.
Joint velocity actions are not compatible with simulated evaluation environments (much harder to simulate). 
Thus, we do not recommend training with joint velocity actions and instead use joint position actions here.


## Compute Requirements

Our DROID training config requires approximately 2 days on 8x H100 GPUs for convergence (100k iterations, bs256, approx. 1 epoch).
If you start from PaliGemma instead of pi0 initialization, plan with ~5 days on 8x H100s (240k iterations, i.e. 3 epochs).

We have experimented with LoRA for cheaper finetuning, but haven't found the policies to perform well so far.


## Data Filtering

Like any diverse real-robot dataset, the DROID dataset isn't perfectly "clean" and we have found data filtering to significantly improve policy performance. Concretely, the DROID dataset contains many *idle* timesteps in which the robot does not move (in part due to the VR teleoperation interface that was used during data collection, we will not go into too much detail here). Appropriate filtering of these idle transitions can improve policy performance.

By default, our openpi training recipe implements the same idle filter used to train all pi-DROID models. We implement it by pre-computing which dataset indices to sample during training. You can check [compute_droid_nonidle_ranges.py](examples/droid/compute_droid_nonidle_ranges.py) for how we compute these indices. Roughly speaking, we filter any time steps for which the next chunk of actions would be largely idle. During training, our code automatically pulls our pre-computed list of indices from cloud storage and applies them. If you want to modify the idle filter / create your custom sampling logic, you can modify our script to generate a new index list and provide it via the `filter_dict_path="<path_to_filter_dict>"` argument in [src/openpi/training/config.py](src/openpi/training/config.py).

**Note**: our list of filtering indices is only valid for the `droid/1.0.1` dataset mentioned in the download section above, and will not provide valid filtering for any other version of the DROID dataset, so make sure you download the dataset above! If you have a custom DROID version, you can rerun the [compute_droid_nonidle_ranges.py](examples/droid/compute_droid_nonidle_ranges.py) script to generate a new list of sampling indices.

## RoboArena

Consider submitting your DROID policies to the [RoboArena benchmark](https://robo-arena.github.io/), which allows you to evaluate your policies on diverse tasks & scenes, **in the real world**! :)

If you have questions about RoboArena, please email [karl.pertsch@gmail.com](mailto:karl.pertsch@gmail.com).


# Fine-Tuning on Custom DROID Datasets

Here we describe how to fine-tune a model on a custom (smaller) dataset collected on the DROID platform. Like for other datasets, we will first convert the custom DROID dataset to LeRobot and then fine-tune a model (pi05-droid) on it.

Note: We use LeRobot here, since we assume the custom DROID fine-tuning dataset to be relatively small (<10s of hours). For larger datasets (like the full DROID dataset) we recommend using RLDS for it's better efficiency (see the example above).


## Step 1: Converting your custom DROID dataset to LeRobot

We will use a small subset of the real DROID dataset for this example. This is a subset of just 30 demonstrations -- we assume that you will use your own dataset instead, but here is the command to download our subset (1.6GB):
```
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 <your_target_path>
```

We will also download the language annotations for the DROID dataset so we can pair our demonstrations with language instructions. Again, for your own data you can manually enter your language instructions and don't need to download our annotations. To download the DROID language annotations (12MB), run:
```
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json <your_target_dir>
```

For your own dataset, make sure that each episode's directory contains a folder called `recordings/MP4` -- if not, you need to first run the MP4 video extraction (from SVO files) using the script [here](https://github.com/droid-dataset/droid/blob/main/scripts/convert/svo_to_mp4.py).

Now, we will use the `convert_droid_to_lerobot.py` script to create a LeRobot version of this dataset (takes <5min for the 30 demonstrations):
```
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir <your_target_path>
```

## Step 2: Run fine-tuning with your custom dataset

Now we can run fine-tuning with our converted custom dataset. We provide an example config for fine-tuning `pi05_droid` on the custom dataset we created. 
You can modify the config easily to work with other base models, or use your custom DROID dataset in `config.py` (seach for `pi05_droid_finetune`).

To launch training:
```
uv run scripts/train.py pi05_droid_finetune --exp-name=my_experiment --overwrite
```

Once trained, you can follow the instructions in [`examples/droid/README.md`](examples/droid/README.md) to serve the policy and run it on the robot.

