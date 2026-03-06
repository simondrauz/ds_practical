import argparse
import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np
from nuscenes.eval.prediction import splits
from tqdm import trange
from trajdata import AgentType
from trajdata.caching import EnvCache
from trajdata.data_structures import SceneMetadata
from trajdata.dataset_specific.nusc import NuscDataset, nusc_utils

from user_config import DEFAULT_USER, get_available_users, get_user_paths

OUTPUT_DIR: Final[Path] = Path(__file__).resolve().parent


###########################################################################
# Parse command-line arguments for user-specific paths
parser = argparse.ArgumentParser(description="Preprocess nuScenes challenge splits")
parser.add_argument(
    "--user",
    type=str,
    choices=get_available_users(),
    default=DEFAULT_USER,
    help=f"User profile to use for paths (default: {DEFAULT_USER})",
)
args = parser.parse_args()

# Get paths from centralized config
user_paths = get_user_paths(args.user)
TRAJDATA_CACHE_DIR: Final[str] = user_paths["trajdata_cache"]
NUSC_RAW_DATA_DIR: Final[str] = user_paths["nusc_raw"]

print(f"Using paths for user: {args.user}")
print(f"  TRAJDATA_CACHE_DIR: {TRAJDATA_CACHE_DIR}")
print(f"  NUSC_RAW_DATA_DIR: {NUSC_RAW_DATA_DIR}")
###########################################################################
# Example usage:
#   python preprocess_challenge_splits.py --user simon
#   python preprocess_challenge_splits.py --user zoe
#
# Example torchrun command (both scripts now support --user argument):
#   torchrun --nproc_per_node=1 train_unified.py --user simon --eval_every=1 --vis_every=1 --batch_size=256 --eval_batch_size=256 --preprocess_workers=16 --log_dir=results/trajectory_prediction/nuScenes/models --log_tag=nusc_adaptive_tpp --train_epochs=20 --conf=results/trajectory_prediction/nuScenes/models/nusc_mm_sec4_tpp-13_Sep_2022_11_06_01/config.json --train_data=nusc_trainval-train --eval_data=nusc_trainval-train_val --history_sec=2.0 --prediction_sec=6.0

# Load training and evaluation environments and scenes
attention_radius = defaultdict(
    lambda: 20.0
)  # Default range is 20m unless otherwise specified.
attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

map_params = {"px_per_m": 2, "map_size_px": 100, "offset_frac_xy": (-0.75, 0.0)}


nusc_dataset = NuscDataset(
    "nusc_mini", NUSC_RAW_DATA_DIR, parallelizable=False, has_maps=True
)
nusc_dataset.load_dataset_obj()


for split in ["train", "train_val", "val"]:
    prediction_challenge_tokens = set(
        splits.get_prediction_challenge_split(split, dataroot=NUSC_RAW_DATA_DIR)
    )

    within_challenge_split = list()

    for idx in trange(len(nusc_dataset.dataset_obj.scene)):
        scene_info = SceneMetadata(None, None, None, idx)
        scene = nusc_dataset.get_scene(scene_info)

        for frame_idx, frame in enumerate(
            nusc_utils.frame_iterator(nusc_dataset.dataset_obj, scene)
        ):
            for agent_info in nusc_utils.agent_iterator(
                nusc_dataset.dataset_obj, frame
            ):
                instance_token: str = agent_info["instance_token"]
                sample_token: str = agent_info["sample_token"]

                if f"{instance_token}_{sample_token}" in prediction_challenge_tokens:
                    scene_info_path = EnvCache.scene_metadata_path(
                        Path(""), nusc_dataset.name, scene.name, scene.dt
                    )

                    within_challenge_split.append(
                        (
                            str(scene_info_path),
                            1,
                            [
                                (
                                    instance_token,
                                    np.array([frame_idx, frame_idx], dtype=int),
                                )
                            ],
                        )
                    )

    print(split, len(within_challenge_split))
    with open(OUTPUT_DIR / f"predchal_{split}_index.pkl", "wb") as f:
        pkl.dump(within_challenge_split, f)
