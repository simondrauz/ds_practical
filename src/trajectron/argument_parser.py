# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Add config/experimental_setup/nuScenes to path to import user_config
sys.path.insert(
    0, str(ROOT / "config" / "experimental_setup" / "nuScenes")
)

try:
    from user_config import DEFAULT_USER, get_available_users
except ImportError:
    # Fallback when user_config.py is unavailable: use current OS user.
    DEFAULT_USER = getpass.getuser() or "default"

    def get_available_users():
        return [DEFAULT_USER]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--user",
    help="user profile to use for path configuration",
    type=str,
    choices=get_available_users(),
    default=DEFAULT_USER,
)

parser.add_argument(
    "--conf",
    help="path to json config file for hyperparameters",
    type=str,
    default="config/nuScenes_mini.json",
)

parser.add_argument(
    "--debug", help="disable all disk writing processes.", action="store_true"
)

parser.add_argument(
    "--preprocess_workers",
    help="number of processes to spawn for dataset loading/preprocessing",
    type=int,
    default=0,
)


# Model Parameters
parser.add_argument(
    "--dec_final_dim", help="the size of the penultimate layer", type=int, default=32
)

parser.add_argument(
    "--sigma_eps_init",
    help="initial value of sigma_eps to use in the model's ALPaCA layer",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--alpha_init",
    help="initial value of alpha to use in the model's ALPaCA layer",
    type=float,
    default=0.001,
)

parser.add_argument(
    "--fixed_sigma",
    help="whether to use a fixed value of sigma_eps in the model's ALPaCA layer (not learning it)",
    action="store_true",
)

parser.add_argument(
    "--fixed_alpha",
    help="whether to use a fixed value of alpha in the model's ALPaCA layer (not learning it)",
    action="store_true",
)

parser.add_argument(
    "--S_init_diag_add",
    help="value to add to L0's diagonal (in the model's ALPaCA layer) to ensure L0 is positive definite",
    type=float,
    default=1e-5,
)

parser.add_argument(
    "--adaptive",
    help="whether to use a Basyesian last layer to make the model adapt to new data",
    type=bool,
    default=False,
)

parser.add_argument(
    "--single_mode_multi_sample",
    help="whether to use the multi-step sampling scheme described in section 4",
    type=bool,
    default=False,
)

parser.add_argument(
    "--single_mode_multi_sample_num",
    help="how many samples to use in the multi-step sampling scheme described in section 4",
    type=int,
    default=50,
)

parser.add_argument(
    "--only_k0",
    help="train an ablation of the model that does not perform meta-training",
    type=bool,
    default=False,
)

parser.add_argument(
    "--dynamic_edges",
    help="whether to use dynamic edges or not, options are 'no' and 'yes'",
    type=str,
    default="yes",
)

parser.add_argument(
    "--edge_state_combine_method",
    help="the method to use for combining edges of the same type",
    type=str,
    default="sum",
)

parser.add_argument(
    "--edge_influence_combine_method",
    help="the method to use for combining edge influences",
    type=str,
    default="attention",
)

parser.add_argument(
    "--edge_addition_filter",
    nargs="+",
    help="what scaling to use for edges as they're created",
    type=float,
    default=[0.25, 0.5, 0.75, 1.0],
)  # We don't automatically pad left with 0.0, if you want a sharp
# and short edge addition, then you need to have a 0.0 at the
# beginning, e.g. [0.0, 1.0].

parser.add_argument(
    "--edge_removal_filter",
    nargs="+",
    help="what scaling to use for edges as they're removed",
    type=float,
    default=[1.0, 0.0],
)  # We don't automatically pad right with 0.0, if you want a sharp drop off like
# the default, then you need to have a 0.0 at the end.

parser.add_argument(
    "--override_attention_radius",
    action="append",
    help='Specify one attention radius to override. E.g. "PEDESTRIAN VEHICLE 10.0"',
    default=[],
)

parser.add_argument(
    "--incl_robot_node",
    help="whether to include a robot node in the graph or simply model all agents",
    action="store_true",
)

parser.add_argument(
    "--map_encoding", help="Whether to use map encoding or not", action="store_true"
)

parser.add_argument(
    "--augment_input_noise",
    help="Standard deviation of Gaussian noise to add the inputs during training, not performed if 0.0",
    type=float,
    default=0.0,
)

parser.add_argument(
    "--node_freq_mult_train",
    help="Whether to use frequency multiplying of nodes during training",
    action="store_true",
)

parser.add_argument(
    "--node_freq_mult_eval",
    help="Whether to use frequency multiplying of nodes during evaluation",
    action="store_true",
)

parser.add_argument(
    "--scene_freq_mult_train",
    help="Whether to use frequency multiplying of nodes during training",
    action="store_true",
)

parser.add_argument(
    "--scene_freq_mult_eval",
    help="Whether to use frequency multiplying of nodes during evaluation",
    action="store_true",
)

parser.add_argument(
    "--scene_freq_mult_viz",
    help="Whether to use frequency multiplying of nodes during evaluation",
    action="store_true",
)

parser.add_argument(
    "--no_edge_encoding",
    help="Whether to use neighbors edge encoding",
    action="store_true",
)

# Data Parameters
parser.add_argument(
    "--log_dir",
    help="what dir to save training information (i.e., saved models, logs, etc)",
    type=str,
    default="../results/trajectory_prediction/logs",
)

parser.add_argument(
    "--trajdata_cache_dir",
    help="location of the unified dataloader cache (will be set from --user if not specified)",
    type=str,
    default=None,
)

parser.add_argument(
    "--train_data",
    help="name of data to use for training",
    type=str,
    default="nusc_mini-mini_train",
)

parser.add_argument(
    "--eval_data",
    help="name of data to use for evaluation",
    type=str,
    default="nusc_mini-mini_val",
)

parser.add_argument(
    "--eval_only_predict",
    nargs="+",
    help=(
        "agent types to evaluate/predict. Defaults to the training only_predict "
        "filter when omitted."
    ),
    type=str,
    default=None,
)

parser.add_argument(
    "--restrict_to_predchal",
    help="restrict nuScenes trainval runs to the prediction-challenge subset",
    action="store_true",
)

parser.add_argument(
    "--data_loc_dict",
    help="JSON dict of dataset locations (will be set from --user if not specified)",
    type=str,
    default=None,
)

parser.add_argument(
    "--history_sec", help="required agent history (in seconds)", type=float
)

parser.add_argument(
    "--prediction_sec", help="prediction horizon (in seconds)", type=float
)

parser.add_argument("--log_tag", help="tag for the log folder", type=str, default="")

parser.add_argument(
    "--device", help="what device to perform training on", type=str, default="cuda:0"
)

# Training Parameters
parser.add_argument(
    "--learning_rate",
    help="initial learning rate, default is whatever the config file has",
    type=float,
    default=None,
)

parser.add_argument(
    "--map_enc_learning_rate",
    help="map encoder learning rate, default is whatever the config file has",
    type=float,
    default=None,
)

parser.add_argument(
    "--lr_step",
    help="number of epochs after which to step down the LR by 0.1, the default (0) is no step downs",
    type=int,
    default=0,
)

parser.add_argument(
    "--grad_clip",
    help="the maximum magnitude of gradients (enforced by clipping)",
    type=float,
    default=None,
)

parser.add_argument(
    "--train_epochs", help="number of iterations to train for", type=int, default=1
)

parser.add_argument("--batch_size", help="training batch size", type=int, default=256)

parser.add_argument(
    "--eval_batch_size", help="evaluation batch size", type=int, default=256
)

parser.add_argument(
    "--max_train_batches",
    help=(
        "validation-only cap on training batches per epoch; processes all batches "
        "when omitted"
    ),
    type=int,
    default=None,
)

parser.add_argument(
    "--max_eval_batches",
    help=(
        "validation-only cap on evaluation batches per evaluation pass; processes "
        "all batches when omitted"
    ),
    type=int,
    default=None,
)

parser.add_argument(
    "--K",
    help="how many CVAE discrete latent modes to have in the model",
    type=int,
    default=25,
)

parser.add_argument(
    "--k_eval", help="how many samples to take during evaluation", type=int, default=25
)

parser.add_argument(
    "--seed", help="manual seed to use, default is 123", type=int, default=123
)

parser.add_argument(
    "--eval_every",
    help="how often to evaluate during training, never if None",
    type=int,
    default=1,
)

parser.add_argument(
    "--vis_every",
    help="how often to visualize during training, never if None",
    type=int,
    default=1,
)

parser.add_argument(
    "--save_every",
    help="how often to save during training, never if None",
    type=int,
    default=1,
)


def _get_explicit_cli_dests(parsed_parser: argparse.ArgumentParser) -> set:
    """Returns argparse destination names explicitly set via CLI flags."""
    explicit_dests = set()
    for token in sys.argv[1:]:
        if not token.startswith("--"):
            continue
        option = token.split("=", maxsplit=1)[0]
        action = parsed_parser._option_string_actions.get(option)
        if action is not None:
            explicit_dests.add(action.dest)

    return explicit_dests


args = parser.parse_args()
provided_cli_dests = _get_explicit_cli_dests(parser)

# Set user-specific paths if not provided explicitly
if args.trajdata_cache_dir is None or args.data_loc_dict is None:
    try:
        from user_config import get_user_paths

        user_paths = get_user_paths(args.user)

        if args.trajdata_cache_dir is None:
            args.trajdata_cache_dir = user_paths["trajdata_cache"]

        if args.data_loc_dict is None:
            # Determine dataset name from train_data
            if "nusc" in args.train_data:
                dataset_name = (
                    "nusc_trainval" if "trainval" in args.train_data else "nusc_mini"
                )
            else:
                dataset_name = "nusc_mini"  # default fallback
            nusc_raw_path = user_paths["nusc_raw"]
            args.data_loc_dict = f'{{"{dataset_name}": "{nusc_raw_path}"}}'
    except (ImportError, KeyError):
        # Fallback to repository-relative defaults if user_config is unavailable.
        default_cache = ROOT / "data" / "processed" / "trajdata_cache"
        default_raw = ROOT / "data" / "raw"
        if args.trajdata_cache_dir is None:
            args.trajdata_cache_dir = str(default_cache)
        if args.data_loc_dict is None:
            if "nusc" in args.train_data:
                dataset_name = (
                    "nusc_trainval" if "trainval" in args.train_data else "nusc_mini"
                )
            else:
                dataset_name = "nusc_mini"
            args.data_loc_dict = json.dumps({dataset_name: str(default_raw)})
