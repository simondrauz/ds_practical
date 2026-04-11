from __future__ import annotations

"""Load and inspect exported SHAP cluster artifacts from one cluster-spec manifest."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_modelling.shap_cluster_exports import build_scene_step_key_frame, summarize_scene_steps
from data_modelling.shap_performance_regimes_utils import (
    VALID_CLUSTER_PROFILE_SORT_KEYS,
    VALID_CLUSTER_SPACES,
    format_shap_feature_name,
    get_shap_cols,
)

VALID_CLUSTER_ALGORITHMS = ("hdbscan", "optics")
REQUIRED_INSPECTION_CONFIG_KEYS = {
    "cluster_spec_manifest_path",
    "performance_group",
    "inspection_algorithm",
    "inspection_cluster_space",
    "cluster_ids",
    "inspection_top_k_features",
    "inspection_top_k_table",
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
    cluster_assignments_path: Path
    cluster_assignments_df: pd.DataFrame
    cluster_catalog_path: Path
    cluster_catalog_df: pd.DataFrame
    cluster_shap_profiles_path: Path
    cluster_shap_profiles_df: pd.DataFrame
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


def _sanitize_slug_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "na"


def _candidate_label_col(algorithm: str, cluster_space: str) -> str:
    return f"cluster_{algorithm}_{cluster_space}"


def _cluster_display_label(cluster_id: int) -> str:
    return "Noise" if int(cluster_id) == -1 else f"Cluster {int(cluster_id)}"


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

    return {
        "cluster_spec_manifest_path": manifest_path,
        "performance_group": str(inspection_config["performance_group"]).strip(),
        "inspection_algorithm": algorithm,
        "inspection_cluster_space": cluster_space,
        "cluster_ids": cluster_ids,
        "inspection_top_k_features": int(inspection_config["inspection_top_k_features"]),
        "inspection_top_k_table": int(inspection_config["inspection_top_k_table"]),
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


def _resolve_trajectory_feature_cols(cluster_assignments_df: pd.DataFrame) -> list[str]:
    shap_cols = get_shap_cols(cluster_assignments_df)
    return [
        format_shap_feature_name(shap_col)
        for shap_col in shap_cols
        if not format_shap_feature_name(shap_col).startswith("scene_")
        and format_shap_feature_name(shap_col) in cluster_assignments_df.columns
    ]


def _resolve_scene_metric_cols(cluster_assignments_df: pd.DataFrame) -> list[str]:
    excluded_scene_columns = {"scene_id", "scene_path", "scene_ts"}
    return sorted(
        [
            column_name
            for column_name in cluster_assignments_df.columns
            if column_name.startswith("scene_") and column_name not in excluded_scene_columns
        ]
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
    cluster_shap_profiles_path = _artifact_path_from_manifest(
        manifest_path,
        manifest_data,
        artifact_type="cluster_shap_profiles",
        fallback_filename="cluster_shap_profiles.csv",
    )

    cluster_assignments_df = pd.read_csv(cluster_assignments_path)
    cluster_catalog_df = pd.read_csv(cluster_catalog_path)
    cluster_shap_profiles_df = pd.read_csv(cluster_shap_profiles_path)

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

    candidate_profiles_df = cluster_shap_profiles_df.loc[
        (cluster_shap_profiles_df["performance_group"].astype(str) == performance_group)
        & (cluster_shap_profiles_df["algorithm"].astype(str).str.lower() == algorithm)
        & (cluster_shap_profiles_df["cluster_space"].astype(str).str.lower() == cluster_space)
    ].copy()
    if candidate_profiles_df.empty:
        raise ValueError(
            "No cluster SHAP profiles were found for "
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

    ordered_cluster_ids = _ordered_cluster_ids(candidate_profiles_df, resolved_config["cluster_ids"])
    selected_profiles_df = _ordered_frame_by_cluster_ids(candidate_profiles_df, ordered_cluster_ids)
    selected_catalog_df = _ordered_frame_by_cluster_ids(candidate_catalog_df, ordered_cluster_ids)
    return ClusterInspectionBundle(
        manifest_path=manifest_path,
        manifest=dict(manifest_data),
        cluster_spec_root=cluster_spec_root,
        cluster_assignments_path=cluster_assignments_path,
        cluster_assignments_df=cluster_assignments_df,
        cluster_catalog_path=cluster_catalog_path,
        cluster_catalog_df=cluster_catalog_df,
        cluster_shap_profiles_path=cluster_shap_profiles_path,
        cluster_shap_profiles_df=cluster_shap_profiles_df,
        performance_group=performance_group,
        algorithm=algorithm,
        cluster_space=cluster_space,
        candidate_label_col=candidate_label_col,
        ordered_cluster_ids=ordered_cluster_ids,
        selected_catalog_df=selected_catalog_df,
        selected_profiles_df=selected_profiles_df,
        group_assignments_df=group_assignments_df,
        trajectory_feature_cols=_resolve_trajectory_feature_cols(cluster_assignments_df),
        scene_metric_cols=_resolve_scene_metric_cols(cluster_assignments_df),
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
    shap_cols: list[str],
    top_k_table: int,
) -> pd.DataFrame:
    """Build a compact per-subset SHAP driver table for notebook display."""

    rows: list[dict[str, Any]] = []
    for profile_row in profile_df.to_dict(orient="records"):
        row = {
            "cluster_id": int(profile_row["cluster_id"]),
            "cluster_label": _cluster_display_label(int(profile_row["cluster_id"])),
            "is_noise": bool(profile_row["is_noise"]),
            "cluster_size": int(profile_row["cluster_size"]),
            "cluster_size_share": float(profile_row["cluster_size_share"]),
        }
        ordered_shap_cols = sorted(
            shap_cols,
            key=lambda shap_col: abs(float(profile_row[shap_col])),
            reverse=True,
        )
        for rank in range(1, top_k_table + 1):
            if rank <= len(ordered_shap_cols):
                shap_col = ordered_shap_cols[rank - 1]
                shap_value = float(profile_row[shap_col])
                row[f"top_feature_{rank}"] = format_shap_feature_name(shap_col)
                row[f"top_direction_{rank}"] = "positive" if shap_value > 0 else "negative" if shap_value < 0 else "neutral"
                row[f"top_abs_shap_{rank}"] = abs(shap_value)
            else:
                row[f"top_feature_{rank}"] = pd.NA
                row[f"top_direction_{rank}"] = pd.NA
                row[f"top_abs_shap_{rank}"] = np.nan
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
) -> None:
    """Plot the stored visualization UMAP embedding for one candidate clustering."""

    plot_df = group_assignments_df.copy()
    plot_df[candidate_label_col] = pd.to_numeric(plot_df[candidate_label_col], errors="coerce").astype("Int64")
    cluster_ids = sorted({int(cluster_id) for cluster_id in plot_df[candidate_label_col].dropna().astype(int).tolist()})
    non_noise_ids = [cluster_id for cluster_id in cluster_ids if cluster_id != -1]
    palette = sns.color_palette("tab10", n_colors=max(len(non_noise_ids), 1))
    color_lookup = {cluster_id: palette[idx % len(palette)] for idx, cluster_id in enumerate(non_noise_ids)}
    if -1 in cluster_ids:
        color_lookup[-1] = (0.65, 0.65, 0.65)

    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster_id in non_noise_ids + ([-1] if -1 in cluster_ids else []):
        subset_df = plot_df.loc[plot_df[candidate_label_col] == cluster_id]
        if subset_df.empty:
            continue
        ax.scatter(
            subset_df["viz_umap_x"],
            subset_df["viz_umap_y"],
            s=28,
            alpha=0.85,
            edgecolors="none",
            color=color_lookup[cluster_id],
            label=_cluster_display_label(cluster_id),
        )
    ax.set_title(
        f"Candidate UMAP scatter - {performance_group} ({algorithm}, space={cluster_space})"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_cluster_profile_barplots(
    profile_df: pd.DataFrame,
    *,
    shap_cols: list[str],
    top_k_features: int,
    performance_group: str,
    algorithm: str,
    cluster_space: str,
    plot_path: Path,
) -> None:
    """Plot signed SHAP bar charts for the selected cluster subsets."""

    n_rows = len(profile_df)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, max(4.5, 3.2 * n_rows)), squeeze=False)
    for axis_idx, profile_row in enumerate(profile_df.to_dict(orient="records")):
        ax = axes[axis_idx][0]
        top_shap_cols = sorted(
            shap_cols,
            key=lambda shap_col: abs(float(profile_row[shap_col])),
            reverse=True,
        )[:top_k_features]
        top_shap_cols = list(reversed(top_shap_cols))
        values = [float(profile_row[shap_col]) for shap_col in top_shap_cols]
        labels = [format_shap_feature_name(shap_col) for shap_col in top_shap_cols]
        colors = ["#C44E52" if value > 0 else "#4C72B0" for value in values]
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_title(
            f"{_cluster_display_label(int(profile_row['cluster_id']))} | "
            f"n={int(profile_row['cluster_size'])} | share={float(profile_row['cluster_size_share']):.2%}"
        )
        ax.set_xlabel("Mean SHAP value")
        ax.set_ylabel("Feature")
    fig.suptitle(
        f"SHAP cluster profiles - {performance_group} ({algorithm}, space={cluster_space})",
        fontsize=15,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_cluster_profile_heatmap(
    profile_df: pd.DataFrame,
    *,
    shap_cols: list[str],
    performance_group: str,
    algorithm: str,
    cluster_space: str,
    plot_path: Path,
) -> None:
    """Plot a SHAP heatmap over the selected cluster subsets."""

    ordered_shap_cols = (
        profile_df[shap_cols]
        .abs()
        .mean(axis=0)
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    heatmap_df = profile_df[ordered_shap_cols].copy()
    heatmap_df.columns = [format_shap_feature_name(shap_col) for shap_col in ordered_shap_cols]
    heatmap_df.index = [
        f"{_cluster_display_label(int(cluster_id))} (n={int(cluster_size)})"
        for cluster_id, cluster_size in zip(profile_df["cluster_id"], profile_df["cluster_size"])
    ]

    fig, ax = plt.subplots(
        figsize=(max(10, len(ordered_shap_cols) * 0.55), max(4.5, len(profile_df) * 0.9))
    )
    sns.heatmap(heatmap_df, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(
        f"SHAP heatmap - {performance_group} ({algorithm}, space={cluster_space})"
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Subset")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
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

    baseline_df = inspection_bundle.group_assignments_df.copy()
    if scene_level:
        baseline_df = resolve_scene_step_frame(baseline_df)
    subset_frames.append(("Whole performance group", baseline_df))
    return subset_frames


def plot_metric_distribution_panels(
    subset_frames: list[tuple[str, pd.DataFrame]],
    *,
    metric_col: str,
    plot_title: str,
    plot_path: Path,
) -> None:
    """Plot one figure per metric with shared scales over the requested subsets."""

    baseline_label, baseline_df = subset_frames[-1]
    baseline_series = baseline_df[metric_col] if metric_col in baseline_df.columns else pd.Series(dtype="object")
    numeric_baseline = pd.to_numeric(baseline_series, errors="coerce").dropna()
    is_continuous_numeric = len(numeric_baseline.unique()) > 10

    fig, axes = plt.subplots(
        len(subset_frames),
        1,
        figsize=(11, max(4.5, 2.8 * len(subset_frames))),
        squeeze=False,
    )

    if is_continuous_numeric:
        if len(numeric_baseline) >= 2:
            bins = np.histogram_bin_edges(numeric_baseline, bins="auto")
        else:
            bins = 10
        if len(numeric_baseline):
            x_min = float(numeric_baseline.min())
            x_max = float(numeric_baseline.max())
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
        else:
            x_min, x_max = 0.0, 1.0
        for axis_idx, (subset_label, subset_df) in enumerate(subset_frames):
            ax = axes[axis_idx][0]
            subset_values = pd.to_numeric(subset_df[metric_col], errors="coerce").dropna()
            ax.hist(subset_values, bins=bins, color="#4C72B0", alpha=0.85)
            ax.set_xlim(x_min, x_max)
            ax.set_title(f"{subset_label} (n={len(subset_values)})")
            ax.set_xlabel(metric_col)
            ax.set_ylabel("Count")
    else:
        if baseline_series.dropna().empty:
            categories = []
        else:
            categories = baseline_series.dropna().astype("string").value_counts().index.tolist()
        for axis_idx, (subset_label, subset_df) in enumerate(subset_frames):
            ax = axes[axis_idx][0]
            if metric_col not in subset_df.columns:
                subset_counts = pd.Series(index=categories, dtype=float)
            else:
                subset_counts = (
                    subset_df[metric_col]
                    .dropna()
                    .astype("string")
                    .value_counts(normalize=True)
                    .reindex(categories, fill_value=0.0)
                )
            ax.bar(subset_counts.index.tolist(), subset_counts.values.tolist(), color="#55A868", alpha=0.85)
            ax.set_ylim(0, 1)
            ax.set_title(f"{subset_label} (n={int(subset_df[metric_col].dropna().shape[0]) if metric_col in subset_df.columns else 0})")
            ax.set_xlabel(metric_col)
            ax.set_ylabel("Share")
            ax.tick_params(axis="x", rotation=30)

    fig.suptitle(plot_title, fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
