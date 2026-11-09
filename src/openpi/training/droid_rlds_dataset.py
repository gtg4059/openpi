"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from collections.abc import Sequence
import dataclasses
from enum import Enum
from enum import auto
import json
import logging
from pathlib import Path

import tqdm

import openpi.shared.download as download


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None
    # If set, load with ``tfds.builder_from_directory`` (e.g. ``data/droid_100/1.0.0_CRI``).
    # When set, ``name`` / ``version`` are only used for logging.
    builder_dir: str | None = None
    # Optional CRI sidecar from ``python -m openpi.cri.annotate_rlds_addon`` (no full RLDS copy).
    # When set with ``include_cri=True``, joins ``observation/cri`` by episode key.
    cri_addon_dir: str | None = None


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        include_cri: bool = False,
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # Ensure dataset weights sum to 1.0
        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"

        def prepare_single_dataset(dataset_cfg: RLDSDataset):
            # ds_name, version = dataset_name.split(":")
            ds_name, version = dataset_cfg.name, dataset_cfg.version
            if dataset_cfg.builder_dir is not None:
                builder_path = Path(dataset_cfg.builder_dir).expanduser().resolve()
                if not builder_path.is_dir():
                    raise FileNotFoundError(f"RLDS builder_dir not found: {builder_path}")
                builder = tfds.builder_from_directory(str(builder_path))
                logging.info(f"Loading RLDS from builder_dir={builder_path}")
            else:
                builder = tfds.builder(ds_name, data_dir=data_dir, version=version)
            dataset = dl.DLataset.from_rlds(
                builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )

            # Filter out any unsuccessful trajectories -- we use the file name to check this
            dataset = dataset.filter(
                lambda traj: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                )
            )

            # Repeat dataset so we never run out of data.
            dataset = dataset.repeat()

            # Load the filter dictionary if provided.
            # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
            # (e.g.,
            # {
            #     "<episode key>": [[0, 100], [200, 300]]
            # }
            # means keep frames 0-99 and 200-299).

            filter_dict_path = dataset_cfg.filter_dict_path
            if filter_dict_path is not None:
                cached_filter_dict_path = download.maybe_download(filter_dict_path)
                with Path(cached_filter_dict_path).open("r") as f:
                    filter_dict = json.load(f)
                logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

                keys_tensor = []
                values_tensor = []

                for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                    for start, end in ranges:
                        for t in range(start, end):
                            frame_key = f"{episode_key}--{t}"
                            keys_tensor.append(frame_key)
                            values_tensor.append(True)
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
                )
                logging.info("Filter hash table initialized")
            else:
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
                )

            cri_addon_reader = None
            if include_cri and dataset_cfg.cri_addon_dir is not None:
                from openpi.cri.addon import CriAddonReader
                from openpi.cri.constants import NUM_CRI_OBS_DIM

                cri_addon_reader = CriAddonReader(Path(dataset_cfg.cri_addon_dir))
                logging.info(f"Using CRI addon at {dataset_cfg.cri_addon_dir} ({len(cri_addon_reader.entries)} episodes)")
                cri_key_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(
                        list(cri_addon_reader.entries.keys()),
                        [True] * len(cri_addon_reader.entries),
                    ),
                    default_value=False,
                )
                dataset = dataset.filter(
                    lambda traj: cri_key_table.lookup(
                        traj["traj_metadata"]["episode_metadata"]["recording_folderpath"][0]
                        + "--"
                        + traj["traj_metadata"]["episode_metadata"]["file_path"][0]
                    )
                )

            def restructure(traj):
                """Reformat observation and action keys, sample language instruction."""
                # Important: we use joint *position* action space -- easier to simulate!
                actions = tf.concat(
                    (
                        (
                            traj["action_dict"]["joint_position"]
                            if action_space == DroidActionSpace.JOINT_POSITION
                            else traj["action_dict"]["joint_velocity"]
                        ),
                        traj["action_dict"]["gripper_position"],
                    ),
                    axis=-1,
                )
                # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
                # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
                exterior_img = tf.cond(
                    tf.random.uniform(shape=[]) > 0.5,
                    lambda: traj["observation"]["exterior_image_1_left"],
                    lambda: traj["observation"]["exterior_image_2_left"],
                )
                wrist_img = traj["observation"]["wrist_image_left"]
                # Randomly sample one of the three language instructions
                instruction = tf.random.shuffle(
                    [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
                )[0]

                traj_len = tf.shape(traj["action"])[0]
                indices = tf.as_string(tf.range(traj_len))

                # Data filtering:
                # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
                # and each step's time step index. This will index into the filter hash table, and if it returns true,
                # then the frame passes the filter.
                step_id = (
                    traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                    + "--"
                    + traj["traj_metadata"]["episode_metadata"]["file_path"]
                    + "--"
                    + indices
                )
                # PolaRiS RLDS stores ``/media/arhan/Secondary1/...``; the official
                # idle-filter JSON uses ``/media/arhan/Secondary/...``. Without this
                # rewrite every PolaRiS frame fails the filter (0% of that mix).
                filter_step_id = tf.strings.regex_replace(step_id, "/Secondary1/", "/Secondary/")
                passes_filter = self.filter_table.lookup(filter_step_id)

                observation = {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                }
                if include_cri:
                    if cri_addon_reader is not None:
                        # Join sidecar by episode key (recording_folderpath--file_path).
                        rec0 = traj["traj_metadata"]["episode_metadata"]["recording_folderpath"][0]
                        fp0 = traj["traj_metadata"]["episode_metadata"]["file_path"][0]

                        def _lookup_cri(rec, fp, length):
                            from openpi.cri.addon import episode_key as _episode_key

                            key = _episode_key(rec.numpy(), fp.numpy())
                            if key not in cri_addon_reader:
                                raise KeyError(f"CRI addon missing episode key: {key}")
                            cri_np = cri_addon_reader.get_cri_obs(key)
                            length_i = int(length.numpy())
                            if cri_np.shape[0] < length_i:
                                raise ValueError(
                                    f"CRI addon length {cri_np.shape[0]} < traj_len {length_i} for {key}"
                                )
                            return cri_np[:length_i]

                        cri = tf.py_function(_lookup_cri, [rec0, fp0, traj_len], Tout=tf.float32)
                        cri.set_shape([None, NUM_CRI_OBS_DIM])
                        observation["cri"] = cri
                    else:
                        observation["cri"] = traj["observation"]["cri"]

                return {
                    "actions": actions,
                    "observation": observation,
                    "prompt": instruction,
                    "step_id": step_id,
                    "passes_filter": passes_filter,
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def chunk_actions(traj):
                """Splits episode into action chunks."""
                traj_len = tf.shape(traj["actions"])[0]

                # For each step in the trajectory, construct indices for the next n actions
                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )

                # Cap to length of the sequence --> final chunks will repeat the last action
                # This makes sense, since we are using absolute joint + gripper position actions
                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

                # Gather the actions for each chunk
                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            # Flatten: map from trajectory dataset to dataset of individual action chunks
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            # Filter data that doesn't pass the filter
            def filter_from_dict(frame):
                return frame["passes_filter"]

            dataset = dataset.filter(filter_from_dict)

            # Remove "passes_filter" key from output
            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            dataset = dataset.map(remove_passes_filter)
            # Keep JPEG bytes through mix/shuffle. Decoding 32k RGB frames before
            # shuffle is what made the first-batch wait ~50 minutes.
            return dataset

        logging.info(f"Preparing {len(datasets)} datasets...")
        logging.info("-" * 50)
        for dataset in datasets:
            loc = dataset.builder_dir or f"{dataset.name}:{dataset.version}"
            addon = f", cri_addon={dataset.cri_addon_dir}" if dataset.cri_addon_dir else ""
            logging.info(f"    {loc} with weight {dataset.weight:.2f} (include_cri={include_cri}{addon})")
        logging.info("-" * 50)
        all_datasets = [prepare_single_dataset(dataset) for dataset in datasets]
        weights = [dataset.weight for dataset in datasets]

        final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)
        # Shuffle encoded JPEGs (small). A 250k *decoded* buffer OOMs host RAM.
        if shuffle:
            logging.info("Filling RLDS shuffle buffer (%d encoded frames)...", shuffle_buffer_size)
            final_dataset = final_dataset.shuffle(shuffle_buffer_size)

        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            return traj

        final_dataset = final_dataset.frame_map(decode_images, num_parallel_calls)
        final_dataset = final_dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        final_dataset = final_dataset.with_ram_budget(1)

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000
