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
from shared_config.config_loader import (  # noqa: E402
    load_agent_type_defaults,
    load_attention_radius,
    load_vector_map_settings,
    parse_agent_type_list,
)

DEFAULT_ONLY_PREDICT, DEFAULT_NO_TYPES = load_agent_type_defaults()
TRAJECTORY_IDENTITY_COLS = ["data_idx", "scene_path", "agent_id", "scene_ts", "agent_type"]
TRAJECTORY_IDENTITY_CHECK_COLS = ["scene_path", "agent_id", "scene_ts", "agent_type"]
EVAL_CONTEXT_COLS = ["eval_data", "history_sec", "prediction_sec", "restrict_to_predchal"]


def _agent_type_name(agent_type) -> str:
    return agent_type.name if isinstance(agent_type, AgentType) else str(agent_type)


def trajectory_identity_for_index(
    agent_dataset: UnifiedDataset,
    data_idx: int,
    elem=None,
) -> Dict:
    """Return the trajdata identity represented by one agent-centric data_idx."""
    scene_path, agent_id, scene_ts = agent_dataset._data_index[int(data_idx)]
    identity = {
        "data_idx": int(data_idx),
        "scene_path": str(scene_path),
        "agent_id": str(agent_id),
        "scene_ts": int(scene_ts),
    }
    if elem is not None:
        identity["agent_type"] = _agent_type_name(elem.agent_type)
    return identity


def _normalise_identity_series(series: pd.Series, col: str) -> pd.Series:
    if col == "scene_ts":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    return series.map(lambda value: "" if pd.isna(value) else str(value))


def _bool_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return _str2bool(value)
    raise ValueError(f"Cannot interpret boolean value from: {value!r}")


def eval_context_from_hyperparams(hyperparams: Dict) -> Dict:
    """Return eval dataset settings that joined metrics must reproduce."""
    return {
        "eval_data": str(hyperparams["eval_data"]),
        "history_sec": float(hyperparams["history_sec"]),
        "prediction_sec": float(hyperparams["prediction_sec"]),
        "restrict_to_predchal": bool(hyperparams.get("restrict_to_predchal", False)),
    }


def validate_eval_context(
    eval_df: pd.DataFrame,
    hyperparams: Dict,
    *,
    eval_file: Path,
) -> None:
    """Assert eval CSV context columns match the config used for reconstruction."""
    present_cols = [col for col in EVAL_CONTEXT_COLS if col in eval_df.columns]
    if not present_cols:
        return

    missing_cols = [col for col in EVAL_CONTEXT_COLS if col not in eval_df.columns]
    if missing_cols:
        raise ValueError(
            f"{eval_file} has partial eval context columns. Missing: {missing_cols}. "
            "Regenerate eval metrics so all context columns are written together."
        )

    expected_context = eval_context_from_hyperparams(hyperparams)
    mismatches = []
    for col in EVAL_CONTEXT_COLS:
        values = eval_df[col].drop_duplicates().tolist()
        if len(values) != 1:
            raise ValueError(
                f"{eval_file} has multiple values for eval context column {col!r}: {values[:5]}"
            )
        actual = values[0]
        expected = expected_context[col]
        if col in {"history_sec", "prediction_sec"}:
            matches = abs(float(actual) - float(expected)) <= 1e-9
        elif col == "restrict_to_predchal":
            matches = _bool_value(actual) == bool(expected)
        else:
            matches = str(actual) == str(expected)
        if not matches:
            mismatches.append({"column": col, "eval_csv": actual, "reconstructed": expected})

    if mismatches:
        raise ValueError(
            "Eval context does not match the config used to reconstruct trajdata for "
            f"{eval_file}. Mismatches: {mismatches}"
        )


def validate_eval_identity(
    eval_df: pd.DataFrame,
    expected_identity_df: pd.DataFrame,
    *,
    eval_file: Path,
) -> None:
    """Assert eval CSV identity columns match the reconstructed trajdata dataset."""
    present_cols = [col for col in TRAJECTORY_IDENTITY_CHECK_COLS if col in eval_df.columns]
    if not present_cols:
        return

    missing_cols = [col for col in TRAJECTORY_IDENTITY_CHECK_COLS if col not in eval_df.columns]
    if missing_cols:
        raise ValueError(
            f"{eval_file} has partial trajectory identity columns. Missing: {missing_cols}. "
            "Regenerate eval metrics so all identity columns are written together."
        )

    required_cols = TRAJECTORY_IDENTITY_COLS
    missing_expected_cols = [
        col for col in required_cols if col not in expected_identity_df.columns
    ]
    if missing_expected_cols:
        raise KeyError(
            "Reconstructed characteristic metrics are missing identity columns: "
            f"{missing_expected_cols}"
        )

    if expected_identity_df["data_idx"].duplicated().any():
        duplicate_count = int(expected_identity_df["data_idx"].duplicated().sum())
        raise ValueError(
            "Reconstructed trajectory identity is not unique on data_idx. "
            f"Duplicate rows: {duplicate_count}"
        )

    merged = eval_df[required_cols].merge(
        expected_identity_df[required_cols],
        on="data_idx",
        how="left",
        suffixes=("_eval", "_dataset"),
        validate="many_to_one",
        indicator="_identity_merge",
    )
    missing_dataset_rows = int((merged["_identity_merge"] != "both").sum())
    if missing_dataset_rows:
        raise ValueError(
            "Eval rows could not be found in the reconstructed trajdata dataset for "
            f"{eval_file}. Missing rows: {missing_dataset_rows}"
        )

    mismatch_mask = pd.Series(False, index=merged.index)
    for col in TRAJECTORY_IDENTITY_CHECK_COLS:
        eval_values = _normalise_identity_series(merged[f"{col}_eval"], col)
        dataset_values = _normalise_identity_series(merged[f"{col}_dataset"], col)
        mismatch_mask |= eval_values != dataset_values

    mismatch_count = int(mismatch_mask.sum())
    if mismatch_count:
        sample_cols = ["data_idx"]
        for col in TRAJECTORY_IDENTITY_CHECK_COLS:
            sample_cols.extend([f"{col}_eval", f"{col}_dataset"])
        sample = merged.loc[mismatch_mask, sample_cols].head(5).to_dict(orient="records")
        raise ValueError(
            "Eval trajectory identity does not match the reconstructed trajdata dataset for "
            f"{eval_file}. Mismatched rows: {mismatch_count}. Sample: {sample}"
        )


def drop_overlapping_columns_for_join(
    right_df: pd.DataFrame,
    existing_cols: Iterable[str],
    *,
    join_cols: Iterable[str],
) -> pd.DataFrame:
    """Drop non-key columns that would otherwise create pandas merge suffixes."""
    existing = set(existing_cols)
    join_col_set = set(join_cols)
    drop_cols = [
        col for col in right_df.columns if col in existing and col not in join_col_set
    ]
    if not drop_cols:
        return right_df
    return right_df.drop(columns=drop_cols)


def _str2bool(val: str) -> bool:
    """Parses common string representations of booleans for CLI flags."""
    lowered = val.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value from: {val}")


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

    active_env_name = Path(dataset._scene_index[0]).relative_to(dataset.cache_path).parts[
        0
    ]

    remapped_split = []
    for scene_info_path, num_elems, elems in within_challenge_split:
        scene_rel_path = Path(scene_info_path)
        if scene_rel_path.parts and scene_rel_path.parts[0] != active_env_name:
            scene_rel_path = Path(active_env_name, *scene_rel_path.parts[1:])

        remapped_scene_path = dataset.cache_path / scene_rel_path
        if not remapped_scene_path.exists():
            raise FileNotFoundError(
                "Prediction challenge split references missing scene cache: "
                f"{remapped_scene_path}"
            )

        remapped_split.append((remapped_scene_path, num_elems, elems))

    dataset._scene_index = [orig_path for orig_path, _, _ in remapped_split]
    dataset._data_index = AgentDataIndex(remapped_split, dataset.verbose)
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
    only_predict = parse_agent_type_list(
        hyperparams.get("only_predict"), DEFAULT_ONLY_PREDICT, "only_predict"
    )
    no_types = parse_agent_type_list(
        hyperparams.get("no_types"), DEFAULT_NO_TYPES, "no_types"
    )

    dataset = UnifiedDataset(
        desired_data=[hyperparams["eval_data"]],
        centric="agent",
        history_sec=(hyperparams["history_sec"], hyperparams["history_sec"]),
        future_sec=(hyperparams["prediction_sec"], hyperparams["prediction_sec"]),
        agent_interaction_distances=load_attention_radius(),
        incl_robot_future=hyperparams.get("incl_robot_node", False),
        incl_raster_map=hyperparams.get("map_encoding", False),
        raster_map_params=raster_map_params,
        incl_vector_map=incl_vector_map,
        only_predict=only_predict,
        no_types=no_types,
        num_workers=hyperparams.get("preprocess_workers", 0),
        cache_location=hyperparams["trajdata_cache_dir"],
        data_dirs=data_dirs,
        verbose=True,
    )

    if (
        hyperparams["eval_data"] == "nusc_trainval-train_val"
        and hyperparams.get("restrict_to_predchal", False)
    ):
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
    only_predict = parse_agent_type_list(
        hyperparams.get("only_predict"), DEFAULT_ONLY_PREDICT, "only_predict"
    )
    no_types = parse_agent_type_list(
        hyperparams.get("no_types"), DEFAULT_NO_TYPES, "no_types"
    )

    dataset = UnifiedDataset(
        desired_data=[hyperparams["eval_data"]],
        centric="scene",
        history_sec=(hyperparams["history_sec"], hyperparams["history_sec"]),
        future_sec=(hyperparams["prediction_sec"], hyperparams["prediction_sec"]),
        agent_interaction_distances=load_attention_radius(),
        incl_robot_future=hyperparams.get("incl_robot_node", False),
        incl_raster_map=False,
        incl_vector_map=incl_vector_map,
        only_predict=only_predict,
        no_types=no_types,
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
        metrics.update(trajectory_identity_for_index(dataset, idx, elem))
        # Preserve the trajectory's agent category for downstream grouped analysis.
        metrics["agent_type"] = _agent_type_name(elem.agent_type)
        rows.append(metrics)
    return pd.DataFrame(rows)


def scene_keys_for_indices(
    agent_dataset: UnifiedDataset, data_indices: List[int]
) -> pd.DataFrame:
    """Maps agent-centric data_idx values to (scene_path, scene_ts)."""
    rows = []
    for idx in data_indices:
        identity = trajectory_identity_for_index(agent_dataset, int(idx))
        rows.append(
            {
                "data_idx": identity["data_idx"],
                "scene_path": identity["scene_path"],
                "scene_ts": identity["scene_ts"],
            }
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
        scene_path = str(scene_path)
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

    run_dir = args.run_dir
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = args.metrics_root / run_dir
    eval_files = list(iter_eval_files(args.metrics_root, run_dir))
    if len(eval_files) == 0:
        raise FileNotFoundError(f"No eval_epoch_*.csv files found under {args.metrics_root}")

    # Cache eval dataframes and gather the full set of referenced indices once.
    eval_dfs: Dict[Path, pd.DataFrame] = {}
    per_file_indices: Dict[Path, List[int]] = {}
    all_indices: Set[int] = set()
    for eval_file in eval_files:
        eval_df = pd.read_csv(eval_file)
        validate_eval_context(eval_df, hyperparams, eval_file=eval_file)
        eval_dfs[eval_file] = eval_df
        unique_indices = np.unique(eval_df["data_idx"].astype(int)).tolist()
        per_file_indices[eval_file] = unique_indices
        all_indices.update(unique_indices)

    all_indices_sorted = sorted(all_indices)
    print(f"Computing agent metrics once for {len(all_indices_sorted)} unique data_idx values")
    agent_char_df = compute_metrics_for_indices(agent_dataset, all_indices_sorted, metric_cfg)

    attention_radius = load_attention_radius()
    agent_char_df["attention_radius_m"] = agent_char_df["agent_type"].apply(
        lambda name: attention_radius[(AgentType[name.upper()], AgentType[name.upper()])]
    )
    agent_char_df["history_sec"] = float(hyperparams["history_sec"])
    agent_char_df["prediction_sec"] = float(hyperparams["prediction_sec"])

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
        validate_eval_identity(eval_df, agent_subset, eval_file=eval_file)

        agent_subset = drop_overlapping_columns_for_join(
            agent_subset,
            eval_df.columns,
            join_cols=["data_idx"],
        )
        joined = eval_df.merge(agent_subset, on="data_idx", how="left")
        scene_subset = drop_overlapping_columns_for_join(
            scene_subset,
            joined.columns,
            join_cols=["data_idx"],
        )
        joined = joined.merge(scene_subset, on="data_idx", how="left")

        run_name = eval_file.parent.name
        out_path = args.output_root / run_name / eval_file.stem
        write_joined(joined, out_path, args.format)
        print(f"Wrote joined metrics to {out_path.with_suffix('.' + args.format)}")


if __name__ == "__main__":
    main()
