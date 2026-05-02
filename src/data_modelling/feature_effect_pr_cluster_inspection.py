from __future__ import annotations

"""Load and inspect exported feature-effect cluster artifacts from one cluster-spec manifest."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_modelling.feature_effect_cluster_exports import build_scene_step_key_frame
from data_modelling.feature_effect_performance_regimes_utils import (
    VALID_CLUSTER_PROFILE_SORT_KEYS,
    VALID_CLUSTER_SPACES,
    format_effect_feature_name,
    get_effect_cols,
)

VALID_CLUSTER_ALGORITHMS = ("hdbscan", "optics")
WHOLE_GROUP_LABEL = "Whole performance group"
NOISE_LABEL = "Noise"
DEFAULT_NOISE_COLOR = "#9AA0A6"
DEFAULT_BASELINE_COLOR = "#264653"
TARGET_ORIGINAL_UNITS_COL = "target_orig"
SCENE_METRIC_PRIORITY = [
    "scene_num_agents",
    "scene_num_PEDESTRIAN",
    "scene_num_BICYCLE",
    "scene_num_MOTORCYCLE",
    "scene_num_VEHICLE",
    "scene_bbox_area",
    "scene_bbox_width",
    "scene_bbox_height",
    "scene_spatial_density",
    "scene_density_PEDESTRIAN",
    "scene_density_BICYCLE",
    "scene_density_MOTORCYCLE",
    "scene_density_VEHICLE",
]
REQUIRED_INSPECTION_CONFIG_KEYS = {
    "cluster_spec_manifest_path",
    "performance_group",
    "inspection_algorithm",
    "inspection_cluster_space",
    "cluster_ids",
    "inspection_top_k_features",
    "inspection_top_k_table",
    "distribution_matrix_max_columns",
    "sort_cluster_profiles_by",
}
INSPECTION_SUMMARY_COLUMNS = [
    "cluster_id",
    "is_noise",
    "cluster_size",
    "cluster_size_share",
    "unique_scene_step_count",
    "unique_scene_count",
]


@dataclass
class ClusterInspectionBundle:
    manifest_path: Path
    manifest: dict[str, Any]
    cluster_spec_root: Path
    model_id: str
    target_mode: str
    effect_title_label: str
    effect_value_axis_label: str
    global_ranking_path: Path
    global_ranking_df: pd.DataFrame
    ordered_effect_cols: list[str]
    cluster_assignments_path: Path
    cluster_assignments_df: pd.DataFrame
    cluster_catalog_path: Path
    cluster_catalog_df: pd.DataFrame
    cluster_feature_effect_profiles_path: Path
    cluster_feature_effect_profiles_df: pd.DataFrame
    performance_group: str
    algorithm: str
    cluster_space: str
    candidate_label_col: str
    ordered_cluster_ids: list[int]
    selected_catalog_df: pd.DataFrame
    selected_profiles_df: pd.DataFrame
    group_assignments_df: pd.DataFrame
    trajectory_feature_cols: list[str]
    scene_metric_cols: list[str]


def resolve_effect_display_context(model_id: str, target_mode: str) -> dict[str, str]:
    """Return model-aware labels for feature-effect inspection plots."""
    normalized_model_id = str(model_id).strip().lower()
    normalized_target_mode = str(target_mode).strip().lower()
    if normalized_model_id == "xgboost":
        return {
            "effect_title_label": "SHAP",
            "effect_value_axis_label": "Mean SHAP value",
            "effect_note": "Feature effects are SHAP contributions on the model prediction scale.",
        }
    if normalized_model_id == "gam":
        scale_suffix = " (log scale)" if normalized_target_mode == "log" else ""
        note_suffix = (
            " These additive effects live on the link/log scale, so they imply multiplicative changes on the "
            "original target scale."
            if normalized_target_mode == "log"
            else " These additive effects live on the GAM link scale."
        )
        return {
            "effect_title_label": f"Additive effect{scale_suffix}",
            "effect_value_axis_label": f"Mean additive effect{scale_suffix}",
            "effect_note": f"Feature effects are GAM additive term contributions.{note_suffix}",
        }
    raise NotImplementedError(f"Feature-effect inspection is not implemented yet for model_id={model_id!r}.")


def _sanitize_slug_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "na"


def _candidate_label_col(algorithm: str, cluster_space: str) -> str:
    return f"cluster_{algorithm}_{cluster_space}"


def _cluster_display_label(cluster_id: int) -> str:
    return NOISE_LABEL if int(cluster_id) == -1 else f"Cluster {int(cluster_id)}"


def _cluster_selection_slug(cluster_ids: str | list[int]) -> str:
    if cluster_ids == "all":
        return "all"
    tokens = ["noise" if int(cluster_id) == -1 else f"cluster-{int(cluster_id)}" for cluster_id in cluster_ids]
    return "__".join(tokens) if tokens else "none"


def _artifact_path_from_manifest(
    manifest_path: Path,
    manifest_data: Mapping[str, Any],
    *,
    artifact_type: str,
    fallback_filename: str,
) -> Path:
    for artifact_row in manifest_data.get("artifacts", []):
        if artifact_row.get("artifact_type") != artifact_type:
            continue
        absolute_path = artifact_row.get("absolute_path")
        if absolute_path:
            return Path(str(absolute_path)).resolve()
        relative_path = artifact_row.get("relative_path")
        if relative_path:
            return (manifest_path.parent / str(relative_path)).resolve()
    return (manifest_path.parent / "tables" / fallback_filename).resolve()


def _blend_with_white(color: str | tuple[float, float, float], amount: float = 0.9) -> tuple[float, float, float]:
    base = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(base + (1.0 - base) * amount)


def _format_metric_label(metric_col: str) -> str:
    overrides = {
        TARGET_ORIGINAL_UNITS_COL: "Target (original units)",
        "max_speed": "Max speed",
        "std_speed": "Speed variability",
        "mean_acceleration": "Mean acceleration",
        "mean_jerk": "Mean jerk",
        "heading_change": "Heading change",
        "has_collision": "Collision flag",
        "min_neighbor_distance": "Minimum neighbor distance",
        "scene_num_agents": "Scene agent count",
        "scene_bbox_area": "Scene BBox area",
        "scene_bbox_width": "Scene BBox width",
        "scene_bbox_height": "Scene BBox height",
        "scene_spatial_density": "Scene spatial density",
    }
    if metric_col in overrides:
        return overrides[metric_col]

    label = metric_col
    if label.startswith("scene_"):
        label = label[len("scene_") :]
    label = label.replace("_", " ")
    words = []
    for word in label.split():
        if word in {"PEDESTRIAN", "BICYCLE", "MOTORCYCLE", "VEHICLE"}:
            words.append(word.title())
        elif word == "bbox":
            words.append("BBox")
        elif word == "num":
            words.append("Count")
        else:
            words.append(word.capitalize())
    return " ".join(words)


def format_metric_label(metric_col: str) -> str:
    """Return a display-ready label for one plotted trajectory or scene metric."""

    return _format_metric_label(metric_col)


def _numeric_subset_values(subset_df: pd.DataFrame, metric_col: str) -> pd.Series:
    if metric_col not in subset_df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(subset_df[metric_col], errors="coerce").dropna()


def _categorical_subset_shares(subset_df: pd.DataFrame, metric_col: str, categories: list[str]) -> pd.Series:
    if metric_col not in subset_df.columns:
        return pd.Series(index=categories, dtype=float)
    return (
        subset_df[metric_col]
        .dropna()
        .astype("string")
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
    )


def _normalized_histogram(values: pd.Series, bins: np.ndarray) -> np.ndarray:
    if values.empty:
        return np.zeros(max(len(bins) - 1, 0), dtype=float)
    counts, _ = np.histogram(values.to_numpy(dtype=float), bins=bins)
    return counts.astype(float) / float(len(values))


def _resolve_metric_plot_spec(
    subset_frames: list[tuple[str, pd.DataFrame]],
    metric_col: str,
) -> dict[str, Any]:
    baseline_df = subset_frames[-1][1]
    baseline_series = baseline_df[metric_col] if metric_col in baseline_df.columns else pd.Series(dtype="object")
    plot_type = resolve_metric_plot_type(baseline_series)

    if plot_type == "continuous":
        numeric_baseline = pd.to_numeric(baseline_series, errors="coerce").dropna()
        if len(numeric_baseline) >= 2:
            bins = np.histogram_bin_edges(numeric_baseline, bins="auto")
        else:
            bins = np.linspace(0.0, 1.0, 11)
        if len(numeric_baseline):
            x_min = float(numeric_baseline.min())
            x_max = float(numeric_baseline.max())
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
        else:
            x_min, x_max = 0.0, 1.0

        hist_max_share = 0.0
        for _, subset_df in subset_frames:
            hist_heights = _normalized_histogram(_numeric_subset_values(subset_df, metric_col), bins)
            hist_max_share = max(hist_max_share, float(hist_heights.max()) if len(hist_heights) else 0.0)
        return {
            "plot_type": plot_type,
            "bins": bins,
            "x_min": x_min,
            "x_max": x_max,
            "y_max": max(hist_max_share * 1.15, 0.1),
        }

    categories = baseline_series.dropna().astype("string").value_counts().index.tolist()
    return {
        "plot_type": plot_type,
        "categories": categories,
    }


def _apply_subset_axis_style(
    ax: plt.Axes,
    *,
    subset_label: str,
    subset_style_map: Mapping[str, Mapping[str, Any]],
) -> None:
    style = subset_style_map[subset_label]
    ax.set_facecolor(style["background"])
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style["color"])
        ax.spines[side].set_linewidth(1.4 if subset_label == WHOLE_GROUP_LABEL else 1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.22, axis="y")


def _annotate_metric_sample_size(ax: plt.Axes, *, count: int) -> None:
    ax.text(
        0.98,
        0.96,
        f"n={count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#3A3A3A",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "boxstyle": "round,pad=0.2"},
    )


def build_subset_style_map(subset_labels: list[str]) -> dict[str, dict[str, Any]]:
    """Return stable colors for clusters, noise, and the whole-group baseline."""

    non_noise_labels = [
        subset_label
        for subset_label in subset_labels
        if subset_label not in {NOISE_LABEL, WHOLE_GROUP_LABEL}
    ]
    palette = sns.color_palette("Set2", n_colors=max(len(non_noise_labels), 1))

    style_map: dict[str, dict[str, Any]] = {}
    for idx, subset_label in enumerate(non_noise_labels):
        color = palette[idx % len(palette)]
        style_map[subset_label] = {
            "color": color,
            "background": _blend_with_white(color, amount=0.9),
        }

    if NOISE_LABEL in subset_labels:
        style_map[NOISE_LABEL] = {
            "color": DEFAULT_NOISE_COLOR,
            "background": _blend_with_white(DEFAULT_NOISE_COLOR, amount=0.92),
        }
    if WHOLE_GROUP_LABEL in subset_labels:
        style_map[WHOLE_GROUP_LABEL] = {
            "color": DEFAULT_BASELINE_COLOR,
            "background": _blend_with_white(DEFAULT_BASELINE_COLOR, amount=0.93),
        }
    return style_map


def resolve_metric_plot_type(baseline_series: pd.Series) -> str:
    """Classify a metric as continuous or discrete/categorical for plotting."""

    non_null = baseline_series.dropna()
    if non_null.empty:
        return "categorical"

    numeric_non_null = pd.to_numeric(non_null, errors="coerce")
    is_numeric = bool(numeric_non_null.notna().all())
    if is_numeric and int(numeric_non_null.nunique()) > 10:
        return "continuous"
    return "categorical"


def chunk_metric_columns(metric_cols: list[str], max_columns: int) -> list[list[str]]:
    """Split wide overview matrices into deterministic pages."""

    if max_columns <= 0:
        raise ValueError("distribution_matrix_max_columns must be greater than zero.")
    if not metric_cols:
        return []
    return [metric_cols[idx : idx + max_columns] for idx in range(0, len(metric_cols), max_columns)]


def resolve_cluster_inspection_config(inspection_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the dedicated inspection notebook inputs."""

    missing_keys = REQUIRED_INSPECTION_CONFIG_KEYS - set(inspection_config)
    if missing_keys:
        raise ValueError(f"INSPECTION_CONFIG is missing required keys: {sorted(missing_keys)}")

    unknown_keys = sorted(set(inspection_config) - REQUIRED_INSPECTION_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(f"INSPECTION_CONFIG contains unsupported keys: {unknown_keys}")

    manifest_path = Path(str(inspection_config["cluster_spec_manifest_path"])).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"INSPECTION_CONFIG['cluster_spec_manifest_path'] does not exist: {manifest_path}")

    algorithm = str(inspection_config["inspection_algorithm"]).strip().lower()
    if algorithm not in VALID_CLUSTER_ALGORITHMS:
        raise ValueError(
            f"INSPECTION_CONFIG['inspection_algorithm'] must be one of {list(VALID_CLUSTER_ALGORITHMS)}, got {algorithm!r}."
        )

    cluster_space = str(inspection_config["inspection_cluster_space"]).strip().lower()
    if cluster_space not in VALID_CLUSTER_SPACES:
        raise ValueError(
            f"INSPECTION_CONFIG['inspection_cluster_space'] must be one of {list(VALID_CLUSTER_SPACES)}, got {cluster_space!r}."
        )

    cluster_ids_raw = inspection_config["cluster_ids"]
    if cluster_ids_raw == "all":
        cluster_ids: str | list[int] = "all"
    elif isinstance(cluster_ids_raw, (list, tuple)):
        cluster_ids = []
        for raw_cluster_id in cluster_ids_raw:
            if isinstance(raw_cluster_id, bool):
                raise ValueError("INSPECTION_CONFIG['cluster_ids'] must not contain booleans.")
            cluster_ids.append(int(raw_cluster_id))
    else:
        raise ValueError("INSPECTION_CONFIG['cluster_ids'] must be 'all' or a list of integers.")

    sort_cluster_profiles_by = str(inspection_config["sort_cluster_profiles_by"]).strip()
    if sort_cluster_profiles_by not in VALID_CLUSTER_PROFILE_SORT_KEYS:
        raise ValueError(
            "INSPECTION_CONFIG['sort_cluster_profiles_by'] must be one of "
            f"{list(VALID_CLUSTER_PROFILE_SORT_KEYS)}, got {sort_cluster_profiles_by!r}."
        )

    distribution_matrix_max_columns = int(inspection_config["distribution_matrix_max_columns"])
    if distribution_matrix_max_columns <= 0:
        raise ValueError("INSPECTION_CONFIG['distribution_matrix_max_columns'] must be greater than zero.")

    return {
        "cluster_spec_manifest_path": manifest_path,
        "performance_group": str(inspection_config["performance_group"]).strip(),
        "inspection_algorithm": algorithm,
        "inspection_cluster_space": cluster_space,
        "cluster_ids": cluster_ids,
        "inspection_top_k_features": int(inspection_config["inspection_top_k_features"]),
        "inspection_top_k_table": int(inspection_config["inspection_top_k_table"]),
        "distribution_matrix_max_columns": distribution_matrix_max_columns,
        "sort_cluster_profiles_by": sort_cluster_profiles_by,
    }


def _ordered_cluster_ids(candidate_profiles_df: pd.DataFrame, cluster_ids: str | list[int]) -> list[int]:
    available_cluster_ids = candidate_profiles_df["cluster_id"].astype(int).tolist()
    if cluster_ids == "all":
        selected_cluster_ids = available_cluster_ids
    else:
        selected_cluster_ids = [cluster_id for cluster_id in cluster_ids if cluster_id in set(available_cluster_ids)]
        missing_cluster_ids = sorted(set(cluster_ids) - set(selected_cluster_ids))
        if missing_cluster_ids:
            raise ValueError(
                f"Requested cluster_ids are not available for the selected candidate: {missing_cluster_ids}"
            )

    selected_profiles_df = candidate_profiles_df.loc[
        candidate_profiles_df["cluster_id"].astype(int).isin(selected_cluster_ids)
    ].copy()
    if selected_profiles_df.empty:
        raise ValueError("No cluster subsets remain after applying INSPECTION_CONFIG['cluster_ids'].")

    non_noise_ids = (
        selected_profiles_df.loc[~selected_profiles_df["is_noise"].astype(bool)]
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])["cluster_id"]
        .astype(int)
        .tolist()
    )
    ordered_cluster_ids = list(non_noise_ids)
    if (selected_profiles_df["is_noise"].astype(bool)).any():
        ordered_cluster_ids.append(-1)
    return ordered_cluster_ids


def _ordered_frame_by_cluster_ids(df: pd.DataFrame, ordered_cluster_ids: list[int]) -> pd.DataFrame:
    order_lookup = {cluster_id: order_idx for order_idx, cluster_id in enumerate(ordered_cluster_ids)}
    ordered_df = df.loc[df["cluster_id"].astype(int).isin(order_lookup)].copy()
    ordered_df["_cluster_order"] = ordered_df["cluster_id"].astype(int).map(order_lookup)
    ordered_df = ordered_df.sort_values(["_cluster_order", "cluster_id"]).drop(columns="_cluster_order")
    return ordered_df.reset_index(drop=True)


def _resolve_ordered_effect_cols(
    *,
    group_assignments_df: pd.DataFrame,
    global_ranking_df: pd.DataFrame,
) -> list[str]:
    ranking_required_cols = ["feature", "global_rank"]
    missing_cols = [col for col in ranking_required_cols if col not in global_ranking_df.columns]
    if missing_cols:
        raise KeyError(f"Feature-effect global ranking is missing required columns: {missing_cols}")

    available_effect_cols = set(get_effect_cols(group_assignments_df))
    ordered_effect_cols: list[str] = []
    ranking_df = global_ranking_df.sort_values(["global_rank", "feature"], ascending=[True, True]).reset_index(drop=True)
    for feature_name in ranking_df["feature"].astype(str):
        effect_col = f"effect__{feature_name}"
        if effect_col in available_effect_cols:
            ordered_effect_cols.append(effect_col)
    if not ordered_effect_cols:
        raise ValueError("Feature-effect global ranking does not map to any effect columns in cluster assignments.")
    return ordered_effect_cols


def _resolve_trajectory_feature_cols(
    *,
    group_assignments_df: pd.DataFrame,
    ordered_effect_cols: list[str],
) -> list[str]:
    ordered_feature_cols = [
        format_effect_feature_name(effect_col)
        for effect_col in ordered_effect_cols
        if not format_effect_feature_name(effect_col).startswith("scene_")
        and format_effect_feature_name(effect_col) in group_assignments_df.columns
    ]
    # Keep the original-unit target as the leading trajectory metric so the
    # cluster-wise loss distribution stays visible alongside feature plots.
    if TARGET_ORIGINAL_UNITS_COL in group_assignments_df.columns:
        return [TARGET_ORIGINAL_UNITS_COL, *ordered_feature_cols]
    return ordered_feature_cols


def _resolve_scene_metric_cols(group_assignments_df: pd.DataFrame) -> list[str]:
    excluded_scene_columns = {"scene_id", "scene_path", "scene_ts"}
    scene_metric_cols = [
        column_name
        for column_name in group_assignments_df.columns
        if column_name.startswith("scene_") and column_name not in excluded_scene_columns
    ]
    priority_lookup = {metric_col: idx for idx, metric_col in enumerate(SCENE_METRIC_PRIORITY)}
    return sorted(
        scene_metric_cols,
        key=lambda metric_col: (
            0 if metric_col in priority_lookup else 1,
            priority_lookup.get(metric_col, len(priority_lookup)),
            metric_col,
        ),
    )


def load_cluster_inspection_selection(
    inspection_config: Mapping[str, Any],
) -> ClusterInspectionBundle:
    """Load one exported clustering candidate and the requested inspection subset."""

    resolved_config = resolve_cluster_inspection_config(inspection_config)
    manifest_path = Path(resolved_config["cluster_spec_manifest_path"])
    manifest_data = json.loads(manifest_path.read_text())
    cluster_spec_root = manifest_path.parent

    cluster_assignments_path = _artifact_path_from_manifest(
        manifest_path,
        manifest_data,
        artifact_type="cluster_assignments",
        fallback_filename="cluster_assignments.csv",
    )
    cluster_catalog_path = _artifact_path_from_manifest(
        manifest_path,
        manifest_data,
        artifact_type="cluster_catalog",
        fallback_filename="cluster_catalog.csv",
    )
    global_ranking_path = _artifact_path_from_manifest(
        manifest_path,
        manifest_data,
        artifact_type="feature_effect_global_ranking",
        fallback_filename="feature_effect_global_ranking.csv",
    )
    cluster_feature_effect_profiles_path = _artifact_path_from_manifest(
        manifest_path,
        manifest_data,
        artifact_type="cluster_feature_effect_profiles",
        fallback_filename="cluster_feature_effect_profiles.csv",
    )

    cluster_assignments_df = pd.read_csv(cluster_assignments_path)
    cluster_catalog_df = pd.read_csv(cluster_catalog_path)
    global_ranking_df = pd.read_csv(global_ranking_path)
    cluster_feature_effect_profiles_df = pd.read_csv(cluster_feature_effect_profiles_path)

    run_context = manifest_data.get("run_context", {})
    model_id = str(run_context.get("model_id", "")).strip()
    target_mode = str(run_context.get("target_mode", "")).strip()
    if not model_id or not target_mode:
        raise KeyError("Cluster manifest run_context must include non-empty 'model_id' and 'target_mode'.")
    effect_display_context = resolve_effect_display_context(model_id, target_mode)

    performance_group = str(resolved_config["performance_group"])
    algorithm = str(resolved_config["inspection_algorithm"])
    cluster_space = str(resolved_config["inspection_cluster_space"])
    candidate_label_col = _candidate_label_col(algorithm, cluster_space)
    if candidate_label_col not in cluster_assignments_df.columns:
        raise KeyError(f"Cluster assignments is missing candidate label column: {candidate_label_col}")

    group_assignments_df = cluster_assignments_df.loc[
        cluster_assignments_df["performance_group"].astype(str) == performance_group
    ].copy()
    if group_assignments_df.empty:
        available_groups = sorted(cluster_assignments_df["performance_group"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"Performance group {performance_group!r} is not available in cluster assignments. "
            f"Available groups: {available_groups}"
        )

    candidate_profiles_df = cluster_feature_effect_profiles_df.loc[
        (cluster_feature_effect_profiles_df["performance_group"].astype(str) == performance_group)
        & (cluster_feature_effect_profiles_df["algorithm"].astype(str).str.lower() == algorithm)
        & (cluster_feature_effect_profiles_df["cluster_space"].astype(str).str.lower() == cluster_space)
    ].copy()
    if candidate_profiles_df.empty:
        raise ValueError(
            "No cluster feature-effect profiles were found for "
            f"group={performance_group!r}, algorithm={algorithm!r}, cluster_space={cluster_space!r}."
        )

    candidate_catalog_df = cluster_catalog_df.loc[
        (cluster_catalog_df["performance_group"].astype(str) == performance_group)
        & (cluster_catalog_df["algorithm"].astype(str).str.lower() == algorithm)
        & (cluster_catalog_df["cluster_space"].astype(str).str.lower() == cluster_space)
    ].copy()
    if candidate_catalog_df.empty:
        raise ValueError(
            "No cluster catalog rows were found for "
            f"group={performance_group!r}, algorithm={algorithm!r}, cluster_space={cluster_space!r}."
        )

    ordered_effect_cols = _resolve_ordered_effect_cols(
        group_assignments_df=group_assignments_df,
        global_ranking_df=global_ranking_df,
    )
    ordered_cluster_ids = _ordered_cluster_ids(candidate_profiles_df, resolved_config["cluster_ids"])
    selected_profiles_df = _ordered_frame_by_cluster_ids(candidate_profiles_df, ordered_cluster_ids)
    selected_catalog_df = _ordered_frame_by_cluster_ids(candidate_catalog_df, ordered_cluster_ids)
    return ClusterInspectionBundle(
        manifest_path=manifest_path,
        manifest=dict(manifest_data),
        cluster_spec_root=cluster_spec_root,
        model_id=model_id,
        target_mode=target_mode,
        effect_title_label=effect_display_context["effect_title_label"],
        effect_value_axis_label=effect_display_context["effect_value_axis_label"],
        global_ranking_path=global_ranking_path,
        global_ranking_df=global_ranking_df,
        ordered_effect_cols=ordered_effect_cols,
        cluster_assignments_path=cluster_assignments_path,
        cluster_assignments_df=cluster_assignments_df,
        cluster_catalog_path=cluster_catalog_path,
        cluster_catalog_df=cluster_catalog_df,
        cluster_feature_effect_profiles_path=cluster_feature_effect_profiles_path,
        cluster_feature_effect_profiles_df=cluster_feature_effect_profiles_df,
        performance_group=performance_group,
        algorithm=algorithm,
        cluster_space=cluster_space,
        candidate_label_col=candidate_label_col,
        ordered_cluster_ids=ordered_cluster_ids,
        selected_catalog_df=selected_catalog_df,
        selected_profiles_df=selected_profiles_df,
        group_assignments_df=group_assignments_df,
        trajectory_feature_cols=_resolve_trajectory_feature_cols(
            group_assignments_df=group_assignments_df,
            ordered_effect_cols=ordered_effect_cols,
        ),
        scene_metric_cols=_resolve_scene_metric_cols(group_assignments_df),
    )


def build_cluster_inspection_export_layout(
    cluster_spec_root: Path,
    *,
    performance_group: str,
    algorithm: str,
    cluster_space: str,
    cluster_ids: str | list[int],
    create_dirs: bool = True,
) -> dict[str, Path]:
    """Build deterministic output directories for one inspection run."""

    inspection_root = cluster_spec_root / "inspection"
    dirname = "__".join(
        [
            f"group-{_sanitize_slug_token(performance_group)}",
            f"alg-{_sanitize_slug_token(algorithm)}",
            f"space-{_sanitize_slug_token(cluster_space)}",
            f"selection-{_sanitize_slug_token(_cluster_selection_slug(cluster_ids))}",
        ]
    )
    selection_root = inspection_root / dirname
    plots_dir = selection_root / "plots"
    if create_dirs:
        plots_dir.mkdir(parents=True, exist_ok=True)
    return {
        "inspection_root": inspection_root,
        "selection_root": selection_root,
        "plots_dir": plots_dir,
    }


def build_top_driver_table(
    profile_df: pd.DataFrame,
    *,
    effect_cols: list[str],
    top_k_table: int,
) -> pd.DataFrame:
    """Build a compact per-subset top-driver table for notebook display."""

    rows: list[dict[str, Any]] = []
    for profile_row in profile_df.to_dict(orient="records"):
        row = {
            "cluster_id": int(profile_row["cluster_id"]),
            "cluster_label": _cluster_display_label(int(profile_row["cluster_id"])),
            "is_noise": bool(profile_row["is_noise"]),
            "cluster_size": int(profile_row["cluster_size"]),
            "cluster_size_share": float(profile_row["cluster_size_share"]),
        }
        ordered_effect_cols = sorted(
            effect_cols,
            key=lambda effect_col: abs(float(profile_row[effect_col])),
            reverse=True,
        )
        for rank in range(1, top_k_table + 1):
            if rank <= len(ordered_effect_cols):
                effect_col = ordered_effect_cols[rank - 1]
                effect_value = float(profile_row[effect_col])
                row[f"top_feature_{rank}"] = format_effect_feature_name(effect_col)
                row[f"top_direction_{rank}"] = (
                    "positive" if effect_value > 0 else "negative" if effect_value < 0 else "neutral"
                )
                row[f"top_abs_effect_{rank}"] = abs(effect_value)
            else:
                row[f"top_feature_{rank}"] = pd.NA
                row[f"top_direction_{rank}"] = pd.NA
                row[f"top_abs_effect_{rank}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_subset_summary_table(selected_catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact subset summary shown at the start of inspection."""

    summary_df = selected_catalog_df[INSPECTION_SUMMARY_COLUMNS].copy()
    return summary_df.reset_index(drop=True)


def plot_candidate_umap_scatter(
    group_assignments_df: pd.DataFrame,
    *,
    candidate_label_col: str,
    performance_group: str,
    algorithm: str,
    cluster_space: str,
    plot_path: Path,
    subset_style_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Plot the stored visualization UMAP embedding for one candidate clustering."""

    plot_df = group_assignments_df.copy()
    plot_df[candidate_label_col] = pd.to_numeric(plot_df[candidate_label_col], errors="coerce").astype("Int64")
    cluster_ids = sorted({int(cluster_id) for cluster_id in plot_df[candidate_label_col].dropna().astype(int).tolist()})
    subset_labels = [_cluster_display_label(cluster_id) for cluster_id in cluster_ids] + [WHOLE_GROUP_LABEL]
    subset_style_map = subset_style_map or build_subset_style_map(subset_labels)

    non_noise_ids = [cluster_id for cluster_id in cluster_ids if cluster_id != -1]
    ordered_ids = non_noise_ids + ([-1] if -1 in cluster_ids else [])

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    for cluster_id in ordered_ids:
        subset_df = plot_df.loc[plot_df[candidate_label_col] == cluster_id]
        if subset_df.empty:
            continue
        subset_label = _cluster_display_label(cluster_id)
        subset_style = subset_style_map[subset_label]
        ax.scatter(
            subset_df["viz_umap_x"],
            subset_df["viz_umap_y"],
            s=34,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.35,
            color=subset_style["color"],
            label=subset_label,
        )
    ax.set_title(
        f"Candidate UMAP scatter\n{performance_group} ({algorithm}, space={cluster_space})",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_cluster_profile_barplots(
    profile_df: pd.DataFrame,
    *,
    effect_cols: list[str],
    top_k_features: int,
    performance_group: str,
    algorithm: str,
    cluster_space: str,
    plot_path: Path,
    effect_title_label: str,
    effect_value_axis_label: str,
    subset_style_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Plot signed feature-effect bar charts for the selected cluster subsets."""

    subset_labels = [_cluster_display_label(int(cluster_id)) for cluster_id in profile_df["cluster_id"].astype(int)]
    subset_style_map = subset_style_map or build_subset_style_map(subset_labels + [WHOLE_GROUP_LABEL])

    n_rows = len(profile_df)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10.5, max(4.8, 3.4 * n_rows)), squeeze=False)
    for axis_idx, profile_row in enumerate(profile_df.to_dict(orient="records")):
        subset_label = _cluster_display_label(int(profile_row["cluster_id"]))
        subset_style = subset_style_map[subset_label]
        ax = axes[axis_idx][0]
        _apply_subset_axis_style(ax, subset_label=subset_label, subset_style_map=subset_style_map)
        top_effect_cols = sorted(
            effect_cols,
            key=lambda effect_col: abs(float(profile_row[effect_col])),
            reverse=True,
        )[:top_k_features]
        top_effect_cols = list(reversed(top_effect_cols))
        values = [float(profile_row[effect_col]) for effect_col in top_effect_cols]
        labels = [format_effect_feature_name(effect_col) for effect_col in top_effect_cols]
        colors = ["#C44E52" if value > 0 else "#4C72B0" for value in values]
        ax.barh(labels, values, color=colors, edgecolor=subset_style["color"], linewidth=1.0, alpha=0.9)
        ax.axvline(0, color="#303030", linewidth=1.0)
        ax.set_title(
            f"{subset_label} | n={int(profile_row['cluster_size'])} | share={float(profile_row['cluster_size_share']):.2%}",
            fontsize=12,
            color=mcolors.to_hex(subset_style["color"]),
            fontweight="bold",
        )
        ax.set_xlabel(effect_value_axis_label)
        ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"{effect_title_label} cluster profiles\n{performance_group} ({algorithm}, space={cluster_space})",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_cluster_profile_heatmap(
    profile_df: pd.DataFrame,
    *,
    ordered_effect_cols: list[str],
    performance_group: str,
    algorithm: str,
    cluster_space: str,
    plot_path: Path,
    effect_title_label: str,
    effect_value_axis_label: str,
) -> None:
    """Plot a feature-effect heatmap over the selected cluster subsets."""

    available_effect_cols = [effect_col for effect_col in ordered_effect_cols if effect_col in profile_df.columns]
    if not available_effect_cols:
        raise ValueError("Selected profiles do not contain any globally ranked feature-effect columns.")
    heatmap_df = profile_df[available_effect_cols].copy()
    heatmap_df.columns = [format_effect_feature_name(effect_col) for effect_col in available_effect_cols]
    heatmap_df.index = [
        f"{_cluster_display_label(int(cluster_id))} (n={int(cluster_size)})"
        for cluster_id, cluster_size in zip(profile_df["cluster_id"], profile_df["cluster_size"])
    ]

    fig, ax = plt.subplots(
        figsize=(max(10, len(available_effect_cols) * 0.55), max(4.8, len(profile_df) * 0.95))
    )
    sns.heatmap(heatmap_df, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": effect_value_axis_label})
    ax.set_title(
        f"{effect_title_label} heatmap\n{performance_group} ({algorithm}, space={cluster_space})",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Subset")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def resolve_scene_step_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique scene-step for scene-level plotting."""

    scene_step_keys_df = build_scene_step_key_frame(df)
    if scene_step_keys_df.empty:
        return pd.DataFrame(columns=df.columns)

    working_df = df.copy()
    if "scene_path" in working_df.columns and working_df["scene_path"].notna().any():
        scene_key_series = working_df["scene_path"].astype("string")
    else:
        scene_key_series = working_df["scene_id"].astype("string")
    working_df["_scene_key"] = scene_key_series
    working_df["_scene_ts_key"] = pd.to_numeric(working_df["scene_ts"], errors="coerce").astype("Int64")
    return (
        working_df.loc[
            working_df["_scene_key"].notna() & working_df["_scene_ts_key"].notna()
        ]
        .drop_duplicates(["_scene_key", "_scene_ts_key"])
        .drop(columns=["_scene_key", "_scene_ts_key"])
        .reset_index(drop=True)
    )


def build_distribution_subset_frames(
    inspection_bundle: ClusterInspectionBundle,
    *,
    scene_level: bool,
) -> list[tuple[str, pd.DataFrame]]:
    """Return ordered subset frames for trajectory- or scene-level plots."""

    subset_frames: list[tuple[str, pd.DataFrame]] = []
    for catalog_row in inspection_bundle.selected_catalog_df.to_dict(orient="records"):
        cluster_id = int(catalog_row["cluster_id"])
        subset_df = inspection_bundle.group_assignments_df.loc[
            pd.to_numeric(inspection_bundle.group_assignments_df[inspection_bundle.candidate_label_col], errors="coerce")
            .astype("Int64")
            == cluster_id
        ].copy()
        if scene_level:
            subset_df = resolve_scene_step_frame(subset_df)
        subset_frames.append((_cluster_display_label(cluster_id), subset_df))

    # The whole performance group baseline is always appended last so the
    # downstream overview matrices can keep one stable baseline row.
    baseline_df = inspection_bundle.group_assignments_df.copy()
    if scene_level:
        baseline_df = resolve_scene_step_frame(baseline_df)
    subset_frames.append((WHOLE_GROUP_LABEL, baseline_df))
    return subset_frames


def plot_metric_distribution_panels(
    subset_frames: list[tuple[str, pd.DataFrame]],
    *,
    metric_col: str,
    plot_title: str,
    plot_path: Path,
    subset_style_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Plot one figure per metric with shared scales over the requested subsets."""

    subset_labels = [subset_label for subset_label, _ in subset_frames]
    subset_style_map = subset_style_map or build_subset_style_map(subset_labels)
    metric_spec = _resolve_metric_plot_spec(subset_frames, metric_col)
    metric_label = _format_metric_label(metric_col)

    if metric_spec["plot_type"] == "continuous":
        fig, axes = plt.subplots(
            2,
            len(subset_frames),
            figsize=(max(4.2 * len(subset_frames), 10), 7.4),
            squeeze=False,
        )
        bins = metric_spec["bins"]
        for col_idx, (subset_label, subset_df) in enumerate(subset_frames):
            subset_style = subset_style_map[subset_label]
            values = _numeric_subset_values(subset_df, metric_col)
            hist_ax = axes[0][col_idx]
            box_ax = axes[1][col_idx]
            _apply_subset_axis_style(hist_ax, subset_label=subset_label, subset_style_map=subset_style_map)
            _apply_subset_axis_style(box_ax, subset_label=subset_label, subset_style_map=subset_style_map)

            hist_heights = _normalized_histogram(values, bins)
            hist_ax.bar(
                bins[:-1],
                hist_heights,
                width=np.diff(bins),
                align="edge",
                color=subset_style["color"],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.5,
            )
            if not values.empty:
                hist_ax.axvline(float(values.mean()), color=subset_style["color"], linestyle="-", linewidth=1.6)
                hist_ax.axvline(float(values.median()), color="#303030", linestyle="--", linewidth=1.2)
            hist_ax.set_xlim(metric_spec["x_min"], metric_spec["x_max"])
            hist_ax.set_ylim(0, metric_spec["y_max"])
            hist_ax.set_title(f"{subset_label}", fontsize=11, fontweight="bold")
            hist_ax.set_xlabel("")
            hist_ax.set_ylabel("Share" if col_idx == 0 else "")
            _annotate_metric_sample_size(hist_ax, count=int(len(values)))

            if not values.empty:
                boxplot = box_ax.boxplot(
                    values.to_numpy(dtype=float),
                    vert=False,
                    patch_artist=True,
                    showmeans=True,
                    widths=0.55,
                    meanprops={"marker": "o", "markerfacecolor": subset_style["color"], "markeredgecolor": "white"},
                    boxprops={"facecolor": subset_style["color"], "alpha": 0.6, "edgecolor": subset_style["color"]},
                    medianprops={"color": "#303030", "linewidth": 1.4},
                    whiskerprops={"color": subset_style["color"], "linewidth": 1.1},
                    capprops={"color": subset_style["color"], "linewidth": 1.1},
                )
                _ = boxplot
            else:
                box_ax.text(0.5, 0.5, "No data", transform=box_ax.transAxes, ha="center", va="center", fontsize=10)
            box_ax.set_xlim(metric_spec["x_min"], metric_spec["x_max"])
            box_ax.set_xlabel(metric_label)
            box_ax.set_yticks([])
            box_ax.set_ylabel("")
        fig.suptitle(plot_title, fontsize=16, fontweight="bold", y=1.02)
    else:
        fig, axes = plt.subplots(
            1,
            len(subset_frames),
            figsize=(max(4.2 * len(subset_frames), 10), 4.8),
            squeeze=False,
        )
        categories = metric_spec["categories"]
        for col_idx, (subset_label, subset_df) in enumerate(subset_frames):
            subset_style = subset_style_map[subset_label]
            ax = axes[0][col_idx]
            _apply_subset_axis_style(ax, subset_label=subset_label, subset_style_map=subset_style_map)
            subset_shares = _categorical_subset_shares(subset_df, metric_col, categories)
            ax.bar(
                subset_shares.index.tolist(),
                subset_shares.values.tolist(),
                color=subset_style["color"],
                alpha=0.84,
                edgecolor="white",
                linewidth=0.6,
            )
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{subset_label}", fontsize=11, fontweight="bold")
            ax.set_xlabel(metric_label)
            ax.set_ylabel("Share" if col_idx == 0 else "")
            ax.tick_params(axis="x", rotation=32)
            _annotate_metric_sample_size(
                ax,
                count=int(subset_df[metric_col].dropna().shape[0]) if metric_col in subset_df.columns else 0,
            )
        fig.suptitle(plot_title, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _plot_metric_overview_matrix(
    subset_frames: list[tuple[str, pd.DataFrame]],
    *,
    metric_cols: list[str],
    plot_title: str,
    plot_path: Path,
    subset_style_map: Mapping[str, Mapping[str, Any]],
) -> None:
    n_rows = len(subset_frames)
    n_cols = len(metric_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(3.2 * n_cols, 8), max(2.1 * n_rows + 1.2, 5.5)),
        squeeze=False,
    )

    metric_specs = {metric_col: _resolve_metric_plot_spec(subset_frames, metric_col) for metric_col in metric_cols}
    for col_idx, metric_col in enumerate(metric_cols):
        metric_label = _format_metric_label(metric_col)
        metric_spec = metric_specs[metric_col]
        for row_idx, (subset_label, subset_df) in enumerate(subset_frames):
            subset_style = subset_style_map[subset_label]
            ax = axes[row_idx][col_idx]
            _apply_subset_axis_style(ax, subset_label=subset_label, subset_style_map=subset_style_map)

            if metric_spec["plot_type"] == "continuous":
                values = _numeric_subset_values(subset_df, metric_col)
                bins = metric_spec["bins"]
                hist_heights = _normalized_histogram(values, bins)
                ax.bar(
                    bins[:-1],
                    hist_heights,
                    width=np.diff(bins),
                    align="edge",
                    color=subset_style["color"],
                    alpha=0.82,
                    edgecolor="white",
                    linewidth=0.35,
                )
                ax.set_xlim(metric_spec["x_min"], metric_spec["x_max"])
                ax.set_ylim(0, metric_spec["y_max"])
                if row_idx != n_rows - 1:
                    ax.set_xticklabels([])
                ax.set_yticks([])
                sample_count = int(len(values))
            else:
                categories = metric_spec["categories"]
                shares = _categorical_subset_shares(subset_df, metric_col, categories)
                x_positions = np.arange(len(categories), dtype=float)
                ax.bar(
                    x_positions,
                    shares.values.tolist(),
                    color=subset_style["color"],
                    alpha=0.84,
                    edgecolor="white",
                    linewidth=0.35,
                )
                ax.set_ylim(0, 1.0)
                if row_idx == n_rows - 1:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=7)
                else:
                    ax.set_xticks([])
                ax.set_yticks([])
                sample_count = int(subset_df[metric_col].dropna().shape[0]) if metric_col in subset_df.columns else 0

            if row_idx == 0:
                ax.set_title(metric_label, fontsize=10.5, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(subset_label, rotation=0, ha="right", va="center", labelpad=28, fontsize=10)
            else:
                ax.set_ylabel("")
            _annotate_metric_sample_size(ax, count=sample_count)

    fig.suptitle(plot_title, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_metric_overview_matrix_pages(
    subset_frames: list[tuple[str, pd.DataFrame]],
    *,
    metric_cols: list[str],
    plot_title_prefix: str,
    output_dir: Path,
    output_stem: str,
    subset_style_map: Mapping[str, Mapping[str, Any]] | None = None,
    max_columns: int = 6,
) -> list[Path]:
    """Write one or more overview-matrix figures for a metric family."""

    subset_labels = [subset_label for subset_label, _ in subset_frames]
    subset_style_map = subset_style_map or build_subset_style_map(subset_labels)
    column_chunks = chunk_metric_columns(metric_cols, max_columns=max_columns)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_paths: list[Path] = []
    for page_idx, column_chunk in enumerate(column_chunks, start=1):
        plot_path = output_dir / f"{output_stem}__page-{page_idx:02d}.png"
        page_paths.append(plot_path)
        _plot_metric_overview_matrix(
            subset_frames,
            metric_cols=column_chunk,
            plot_title=f"{plot_title_prefix} (page {page_idx}/{len(column_chunks)})",
            plot_path=plot_path,
            subset_style_map=subset_style_map,
        )
    return page_paths
