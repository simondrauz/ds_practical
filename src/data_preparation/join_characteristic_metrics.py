"""Join characteristic trajectory metrics onto Trajectron eval CSVs.

This script mirrors the dataset construction used in `train_unified.py`'s eval
loop, computes characteristic metrics for the referenced `data_idx` values, and
left-joins those metrics onto each `eval_epoch_*.csv`.

The critical requirement for correct joins is dataset alignment: the dataset
created here must match the one used during evaluation (same split, history/
future windows, filters, cache, and dataset roots).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from trajdata import AgentType, UnifiedDataset
from trajdata.data_structures.data_index import AgentDataIndex
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from data_preparation.functions_traj_metrics.agent_centric_characteristic_metrics import (  # noqa: E402
    CharacteristicMetricConfig,
    compute_characteristic_metrics as compute_agent_characteristic_metrics,
)
from data_preparation.functions_traj_metrics.scene_centric_characteristic_metrics import (  # noqa: E402
    SceneCharacteristicMetricConfig,
    compute_scene_characteristic_metrics,
)
from shared_config.map_config import load_vector_map_settings  # noqa: E402


def _str2bool(val: str) -> bool:
    """Parses common string representations of booleans for CLI flags."""
    lowered = val.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value from: {val}")


def _attention_radius() -> Dict:
    """Returns the agent interaction distances used in training/eval."""
    radius = defaultdict(lambda: 20.0)
    radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
    radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
    radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
    radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0
    return radius


def restrict_to_predchal(dataset: UnifiedDataset, split: str, city: str = "") -> None:
    """Applies the nuScenes prediction-challenge index filter.

    This mirrors the behavior in `train_unified.py` when evaluating on
    `nusc_trainval-train_val`.
    """
    predchal_path = (
        ROOT
        / "config"
        / "experimental_setup"
        / "nuScenes"
        / f"predchal{city}_{split}_index.pkl"
    )
    if not predchal_path.exists():
        raise FileNotFoundError(f"Prediction challenge index not found at {predchal_path}")

    with open(predchal_path, "rb") as f:
        within_challenge_split = pickle.load(f)

    within_challenge_split = [
        (dataset.cache_path / scene_info_path, num_elems, elems)
        for scene_info_path, num_elems, elems in within_challenge_split
    ]

    dataset._scene_index = [orig_path for orig_path, _, _ in within_challenge_split]
    dataset._data_index = AgentDataIndex(within_challenge_split, dataset.verbose)
    dataset._data_len = len(dataset._data_index)


def load_hyperparams(conf_path: Path, overrides: argparse.Namespace) -> Dict:
    """Loads hyperparameters from config JSON and applies CLI overrides."""
    with open(conf_path, "r", encoding="utf-8") as f:
        hyperparams = json.load(f)

    if overrides.eval_data is not None:
        hyperparams["eval_data"] = overrides.eval_data
    if overrides.history_sec is not None:
        hyperparams["history_sec"] = overrides.history_sec
    if overrides.prediction_sec is not None:
        hyperparams["prediction_sec"] = overrides.prediction_sec
    if overrides.trajdata_cache_dir is not None:
        hyperparams["trajdata_cache_dir"] = overrides.trajdata_cache_dir
    if overrides.data_loc_dict is not None:
        hyperparams["data_loc_dict"] = overrides.data_loc_dict
    if overrides.preprocess_workers is not None:
        hyperparams["preprocess_workers"] = overrides.preprocess_workers
    if overrides.map_encoding is not None:
        hyperparams["map_encoding"] = overrides.map_encoding
    if overrides.incl_robot_node is not None:
        hyperparams["incl_robot_node"] = overrides.incl_robot_node

    return hyperparams


def _parse_data_dirs(hyperparams: Dict) -> Dict[str, str]:
    try:
        return json.loads(hyperparams["data_loc_dict"])
    except Exception as exc:  # pragma: no cover - explicit error message path
        raise ValueError(
            "Could not parse hyperparams['data_loc_dict']; pass a valid JSON string via --data_loc_dict."
        ) from exc


def _load_analysis_incl_vector_map(analysis_conf: Path) -> bool:
    with open(analysis_conf, "r", encoding="utf-8") as f:
        analysis_cfg = yaml.safe_load(f) or {}

    try:
        return bool(analysis_cfg["trajdata"]["incl_vector_map"])
    except Exception as exc:  # pragma: no cover - explicit error message path
        raise KeyError(
            f"Missing `trajdata.incl_vector_map` in analysis config: {analysis_conf}"
        ) from exc


def build_agent_eval_dataset(
    hyperparams: Dict,
    data_dirs: Dict[str, str],
    incl_vector_map: bool,
    raster_map_params: Dict,
) -> UnifiedDataset:
    """Builds the agent-centric dataset aligned with the eval loop.

    The parameters here intentionally match the eval dataset construction in
    `train_unified.py` (including filters and map parameters).
    """
    dataset = UnifiedDataset(
        desired_data=[hyperparams["eval_data"]],
        centric="agent",
        history_sec=(hyperparams["history_sec"], hyperparams["history_sec"]),
        future_sec=(hyperparams["prediction_sec"], hyperparams["prediction_sec"]),
        agent_interaction_distances=_attention_radius(),
        incl_robot_future=hyperparams.get("incl_robot_node", False),
        incl_raster_map=hyperparams.get("map_encoding", False),
        raster_map_params=raster_map_params,
        incl_vector_map=incl_vector_map,
        only_predict=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        num_workers=hyperparams.get("preprocess_workers", 0),
        cache_location=hyperparams["trajdata_cache_dir"],
        data_dirs=data_dirs,
        verbose=True,
    )

    if hyperparams["eval_data"] == "nusc_trainval-train_val":
        # Mirror train_unified.py's prediction-challenge filtering.
        restrict_to_predchal(dataset, "train_val")

    return dataset


def build_scene_eval_dataset(
    hyperparams: Dict, data_dirs: Dict[str, str], incl_vector_map: bool
) -> UnifiedDataset:
    """Builds a scene-centric dataset keyed by (scene_path, scene_ts).

    This uses the same temporal windows and interaction radii as the eval loop
    so that the scene context is comparable to the agent-centric samples.
    """
    dataset = UnifiedDataset(
        desired_data=[hyperparams["eval_data"]],
        centric="scene",
        history_sec=(hyperparams["history_sec"], hyperparams["history_sec"]),
        future_sec=(hyperparams["prediction_sec"], hyperparams["prediction_sec"]),
        agent_interaction_distances=_attention_radius(),
        incl_robot_future=hyperparams.get("incl_robot_node", False),
        incl_raster_map=False,
        incl_vector_map=incl_vector_map,
        only_predict=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        num_workers=hyperparams.get("preprocess_workers", 0),
        cache_location=hyperparams["trajdata_cache_dir"],
        data_dirs=data_dirs,
        verbose=True,
    )
    return dataset


def iter_eval_files(metrics_root: Path, run_dir: Path | None) -> Iterable[Path]:
    """Yields eval CSV files either from a specific run or all runs."""
    if run_dir is not None:
        yield from sorted(run_dir.glob("eval_epoch_*.csv"))
        return

    for subdir in sorted(metrics_root.glob("*")):
        if not subdir.is_dir():
            continue
        yield from sorted(subdir.glob("eval_epoch_*.csv"))


def compute_metrics_for_indices(
    dataset: UnifiedDataset,
    data_indices: List[int],
    metric_cfg: CharacteristicMetricConfig,
) -> pd.DataFrame:
    """Computes characteristic metrics only for the referenced indices.

    This keeps the script efficient by avoiding a full pass over the dataset
    when the eval CSVs reference a strict subset of `data_idx` values.
    """
    rows = []
    max_idx = len(dataset) - 1
    for idx in tqdm(data_indices, desc="Characteristic metrics"):
        if idx > max_idx:
            raise IndexError(
                f"data_idx {idx} is out of range for dataset of length {len(dataset)}"
            )
        elem = dataset[idx]
        metrics = compute_agent_characteristic_metrics(elem, metric_cfg)
        # Preserve the trajectory's agent category for downstream grouped analysis.
        metrics["agent_type"] = (
            elem.agent_type.name
            if isinstance(elem.agent_type, AgentType)
            else str(elem.agent_type)
        )
        metrics["data_idx"] = int(idx)
        rows.append(metrics)
    return pd.DataFrame(rows)


def scene_keys_for_indices(
    agent_dataset: UnifiedDataset, data_indices: List[int]
) -> pd.DataFrame:
    """Maps agent-centric data_idx values to (scene_path, scene_ts)."""
    rows = []
    for idx in data_indices:
        scene_path, _agent_id, scene_ts = agent_dataset._data_index[int(idx)]
        rows.append(
            {"data_idx": int(idx), "scene_path": scene_path, "scene_ts": int(scene_ts)}
        )
    return pd.DataFrame(rows)


def compute_scene_metrics_for_keys(
    scene_dataset: UnifiedDataset,
    needed_keys: Set[Tuple[str, int]],
    scene_cfg: SceneCharacteristicMetricConfig,
) -> pd.DataFrame:
    """Computes scene-centric metrics for a set of (scene_path, scene_ts) keys."""
    rows = []
    remaining = set(needed_keys)
    for idx in tqdm(range(len(scene_dataset)), desc="Scene characteristic metrics"):
        scene_path, scene_ts = scene_dataset._data_index[idx]
        key = (scene_path, int(scene_ts))
        if key not in remaining:
            continue
        elem = scene_dataset[idx]
        metrics = compute_scene_characteristic_metrics(elem, scene_path, scene_cfg)
        rows.append(metrics)
        remaining.remove(key)
        if len(remaining) == 0:
            break

    if len(remaining) > 0:
        missing_preview = list(sorted(remaining))[:5]
        raise KeyError(
            f"Could not find {len(remaining)} scene keys in scene dataset. "
            f"Examples: {missing_preview}"
        )

    return pd.DataFrame(rows)


def write_joined(joined: pd.DataFrame, out_path: Path, out_format: str) -> None:
    """Writes the joined DataFrame as CSV or Parquet.

    Note: `out_path` should be provided without a suffix; the correct suffix is
    appended based on `out_format`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_format == "parquet":
        joined.to_parquet(out_path.with_suffix(".parquet"), index=False)
    else:
        joined.to_csv(out_path.with_suffix(".csv"), index=False)


def parse_args() -> argparse.Namespace:
    """Defines and parses command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Loads trajdata the same way as the Trajectron eval loop, computes characteristic "
            "metrics per data_idx, and joins them onto eval_epoch_*.csv files."
        )
    )
    parser.add_argument("--conf", type=Path, required=True, help="Path to config.json used for training/eval.")
    parser.add_argument(
        "--analysis_conf",
        type=Path,
        default=ROOT / "config" / "analysis_config.yaml",
        help="Path to analysis_config.yaml used to source trajdata.incl_vector_map.",
    )
    parser.add_argument(
        "--metrics_root",
        type=Path,
        default=ROOT / "results" / "trajectory_prediction" / "trajectory_metrics",
        help="Root directory containing Trajectron eval_epoch_*.csv files.",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Optional specific run directory under metrics_root to process.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=ROOT
        / "results"
        / "trajectory_prediction"
        / "trajectory_metrics_joined",
        help="Where to write joined outputs.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        help="Output format for joined files.",
    )
    parser.add_argument(
        "--incl_vector_map",
        dest="incl_vector_map",
        action="store_true",
        default=None,
        help=(
            "Include vector maps to compute off-road metrics. "
            "Defaults to trajdata.incl_vector_map from analysis_config.yaml when omitted."
        ),
    )
    parser.add_argument(
        "--no_incl_vector_map",
        dest="incl_vector_map",
        action="store_false",
        help="Disable vector map loading even if enabled in analysis_config.yaml.",
    )

    # Hyperparameter overrides to ensure dataset matches eval loop.
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--history_sec", type=float, default=None)
    parser.add_argument("--prediction_sec", type=float, default=None)
    parser.add_argument("--trajdata_cache_dir", type=str, default=None)
    parser.add_argument("--data_loc_dict", type=str, default=None)
    parser.add_argument("--preprocess_workers", type=int, default=None)
    parser.add_argument("--map_encoding", type=_str2bool, default=None)
    parser.add_argument("--incl_robot_node", type=_str2bool, default=None)

    # Metric configuration.
    parser.add_argument("--collision_threshold_m", type=float, default=0.75)
    parser.add_argument("--lane_threshold_m", type=float, default=3.0)
    parser.add_argument("--scene_min_bbox_area_m2", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    """Entry point: align dataset, compute metrics, and join per eval file."""
    args = parse_args()
    default_incl_vector_map = _load_analysis_incl_vector_map(args.analysis_conf)
    map_settings = load_vector_map_settings()
    incl_vector_map = (
        args.incl_vector_map
        if args.incl_vector_map is not None
        else default_incl_vector_map
    )
    hyperparams = load_hyperparams(args.conf, args)
    data_dirs = _parse_data_dirs(hyperparams)
    agent_dataset = build_agent_eval_dataset(
        hyperparams,
        data_dirs,
        incl_vector_map=incl_vector_map,
        raster_map_params=map_settings["raster_map_params"],
    )
    scene_dataset = build_scene_eval_dataset(
        hyperparams, data_dirs, incl_vector_map=incl_vector_map
    )

    metric_cfg = CharacteristicMetricConfig(
        collision_threshold_m=args.collision_threshold_m,
        lane_threshold_m=args.lane_threshold_m,
    )
    scene_cfg = SceneCharacteristicMetricConfig(
        min_bbox_area_m2=args.scene_min_bbox_area_m2
    )

    eval_files = list(iter_eval_files(args.metrics_root, args.run_dir))
    if len(eval_files) == 0:
        raise FileNotFoundError(f"No eval_epoch_*.csv files found under {args.metrics_root}")

    # Cache eval dataframes and gather the full set of referenced indices once.
    eval_dfs: Dict[Path, pd.DataFrame] = {}
    per_file_indices: Dict[Path, List[int]] = {}
    all_indices: Set[int] = set()
    for eval_file in eval_files:
        eval_df = pd.read_csv(eval_file)
        eval_dfs[eval_file] = eval_df
        unique_indices = np.unique(eval_df["data_idx"].astype(int)).tolist()
        per_file_indices[eval_file] = unique_indices
        all_indices.update(unique_indices)

    all_indices_sorted = sorted(all_indices)
    print(f"Computing agent metrics once for {len(all_indices_sorted)} unique data_idx values")
    agent_char_df = compute_metrics_for_indices(agent_dataset, all_indices_sorted, metric_cfg)

    print("Mapping data_idx to scene keys and computing scene metrics once")
    scene_keys_df = scene_keys_for_indices(agent_dataset, all_indices_sorted)
    needed_scene_keys = set(
        (row.scene_path, int(row.scene_ts)) for row in scene_keys_df.itertuples()
    )
    scene_char_df = compute_scene_metrics_for_keys(scene_dataset, needed_scene_keys, scene_cfg)
    scene_per_idx_df = scene_keys_df.merge(
        scene_char_df, on=["scene_path", "scene_ts"], how="left"
    )

    # Index cached metric tables so we can quickly slice per epoch.
    agent_char_by_idx = agent_char_df.set_index("data_idx")
    scene_char_by_idx = scene_per_idx_df.set_index("data_idx")

    for eval_file in eval_files:
        print(f"\nProcessing {eval_file}")
        eval_df = eval_dfs[eval_file]
        unique_indices = per_file_indices[eval_file]
        agent_subset = agent_char_by_idx.loc[unique_indices].reset_index()
        scene_subset = scene_char_by_idx.loc[unique_indices].reset_index()

        joined = (
            eval_df.merge(agent_subset, on="data_idx", how="left")
            .merge(scene_subset, on="data_idx", how="left")
        )

        run_name = eval_file.parent.name
        out_path = args.output_root / run_name / eval_file.stem
        write_joined(joined, out_path, args.format)
        print(f"Wrote joined metrics to {out_path.with_suffix('.' + args.format)}")


if __name__ == "__main__":
    main()
