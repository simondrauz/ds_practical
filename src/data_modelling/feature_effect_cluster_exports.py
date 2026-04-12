from __future__ import annotations

"""Write lean per-cluster exports for downstream feature-effect inspection notebooks.

The full `cluster_assignments.csv` remains the authoritative export because it
already contains the joined metrics, feature columns, effect columns, UMAP
coordinates, and every candidate label column. Per-cluster member files are
kept intentionally slim so candidate-wide exports stay readable and do not
duplicate the full assignment table many times.
"""

import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from data_modelling.feature_effect_performance_regimes_utils import (
    build_cluster_feature_effect_profiles,
    get_effect_cols,
)

CLUSTER_CATALOG_FILENAME = "cluster_catalog.csv"
CLUSTER_FEATURE_EFFECT_PROFILES_FILENAME = "cluster_feature_effect_profiles.csv"
CLUSTER_MEMBER_FILENAME_TEMPLATE = (
    "cluster_members__group-{performance_group}__alg-{algorithm}__space-{cluster_space}__label-{cluster_label}.csv"
)
DEFAULT_MEMBER_EXPORT_COLUMNS = [
    "row_id",
    "data_idx",
    "outer_fold",
    "performance_group",
    "target_orig",
    "oof_pred_orig",
    "scene_id",
    "scene_path",
    "scene_ts",
    "agent_type",
]


def _sanitize_slug_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "na"


def _resolve_scene_id_series(df: pd.DataFrame) -> pd.Series:
    if "scene_id" in df.columns and df["scene_id"].notna().any():
        scene_ids = df["scene_id"].astype("string")
        return scene_ids.fillna("unknown_scene")
    if "scene_path" in df.columns and df["scene_path"].notna().any():
        return df["scene_path"].astype("string").map(lambda value: Path(str(value)).parent.name)
    return pd.Series(["unknown_scene"] * len(df), index=df.index, dtype="string")


def build_scene_step_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return unique scene-step keys for one subset of assignment rows.

    Scene metrics are attached at `(scene_path, scene_ts)` granularity in the
    joined metrics export, so scene-level summaries must dedupe by scene-step
    rather than trajectory row to avoid overweighting dense timesteps.
    """

    if "scene_ts" not in df.columns:
        return pd.DataFrame(columns=["scene_key", "scene_id", "scene_ts"])

    if "scene_path" in df.columns and df["scene_path"].notna().any():
        scene_key_series = df["scene_path"].astype("string")
    elif "scene_id" in df.columns and df["scene_id"].notna().any():
        scene_key_series = df["scene_id"].astype("string")
    else:
        return pd.DataFrame(columns=["scene_key", "scene_id", "scene_ts"])

    scene_keys_df = pd.DataFrame(
        {
            "scene_key": scene_key_series,
            "scene_id": _resolve_scene_id_series(df),
            "scene_ts": pd.to_numeric(df["scene_ts"], errors="coerce").astype("Int64"),
        }
    )
    return (
        scene_keys_df.loc[scene_keys_df["scene_key"].notna() & scene_keys_df["scene_ts"].notna()]
        .drop_duplicates()
        .sort_values(["scene_key", "scene_ts"])
        .reset_index(drop=True)
    )


def summarize_scene_steps(df: pd.DataFrame) -> dict[str, int]:
    """Summarize unique scene-step and scene counts for one subset."""

    scene_steps_df = build_scene_step_key_frame(df)
    if scene_steps_df.empty:
        return {"unique_scene_step_count": 0, "unique_scene_count": 0}
    return {
        "unique_scene_step_count": int(len(scene_steps_df)),
        "unique_scene_count": int(scene_steps_df["scene_id"].dropna().nunique()),
    }


def resolve_member_export_columns(
    clustered_df: pd.DataFrame,
    *,
    performance_metric_col: str,
) -> list[str]:
    """Return the ordered slim member-file columns supported by the assignment table."""

    ordered_columns = []
    for column_name in DEFAULT_MEMBER_EXPORT_COLUMNS:
        if column_name in clustered_df.columns:
            ordered_columns.append(column_name)
    if performance_metric_col in clustered_df.columns and performance_metric_col not in ordered_columns:
        ordered_columns.insert(4, performance_metric_col)
    return ordered_columns


def _cluster_member_filename(
    performance_group: str,
    algorithm: str,
    cluster_space: str,
    cluster_label: str,
) -> str:
    return CLUSTER_MEMBER_FILENAME_TEMPLATE.format(
        performance_group=_sanitize_slug_token(performance_group),
        algorithm=_sanitize_slug_token(algorithm),
        cluster_space=_sanitize_slug_token(cluster_space),
        cluster_label=_sanitize_slug_token(cluster_label),
    )


def write_cluster_exports(
    clustered_df: pd.DataFrame,
    cluster_scores_df: pd.DataFrame,
    *,
    export_layout: Mapping[str, Path],
    performance_metric_col: str,
    performance_group_col: str = "performance_group",
    effect_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Write candidate-wide cluster exports for downstream inspection notebooks."""

    effect_cols = effect_cols or get_effect_cols(clustered_df)
    tables_dir = Path(export_layout["tables_dir"])
    cluster_spec_root = Path(export_layout["cluster_spec_root"])
    cluster_feature_effect_profiles_path = tables_dir / CLUSTER_FEATURE_EFFECT_PROFILES_FILENAME
    cluster_catalog_path = tables_dir / CLUSTER_CATALOG_FILENAME

    cluster_feature_effect_profiles_df = build_cluster_feature_effect_profiles(
        clustered_df,
        cluster_scores_df,
        performance_group_col=performance_group_col,
        effect_cols=effect_cols,
        include_noise=True,
    )
    cluster_feature_effect_profiles_df.to_csv(cluster_feature_effect_profiles_path, index=False)

    member_export_cols = resolve_member_export_columns(
        clustered_df,
        performance_metric_col=performance_metric_col,
    )

    catalog_rows: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = [
        {
            "artifact_kind": "table",
            "artifact_type": "cluster_feature_effect_profiles",
            "relative_path": str(cluster_feature_effect_profiles_path.relative_to(cluster_spec_root)),
            "absolute_path": str(cluster_feature_effect_profiles_path.resolve()),
        }
    ]

    for profile_row in cluster_feature_effect_profiles_df.to_dict(orient="records"):
        performance_group = str(profile_row["performance_group"])
        algorithm = str(profile_row["algorithm"])
        cluster_space = str(profile_row["cluster_space"])
        candidate_label_col = str(profile_row["candidate_label_col"])
        cluster_id = int(profile_row["cluster_id"])
        cluster_label = str(profile_row["cluster_label"])
        group_df = clustered_df.loc[clustered_df[performance_group_col].astype(str) == performance_group].copy()
        member_df = group_df.loc[
            pd.to_numeric(group_df[candidate_label_col], errors="coerce").astype("Int64") == cluster_id
        ].copy()
        if member_df.empty:
            continue

        member_export_df = member_df[member_export_cols].copy()
        if "row_id" in member_export_df.columns:
            member_export_df = member_export_df.sort_values(["row_id"]).reset_index(drop=True)
        else:
            member_export_df = member_export_df.reset_index(drop=True)

        member_path = tables_dir / _cluster_member_filename(
            performance_group,
            algorithm,
            cluster_space,
            cluster_label,
        )
        member_export_df.to_csv(member_path, index=False)

        scene_summary = summarize_scene_steps(member_df)
        member_relative_path = str(member_path.relative_to(cluster_spec_root))
        catalog_rows.append(
            {
                "performance_group": performance_group,
                "algorithm": algorithm,
                "cluster_space": cluster_space,
                "candidate_label_col": candidate_label_col,
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "is_noise": bool(profile_row["is_noise"]),
                "cluster_size": int(profile_row["cluster_size"]),
                "cluster_size_share": float(profile_row["cluster_size_share"]),
                "cluster_rank_by_size": profile_row["cluster_rank_by_size"],
                "unique_scene_step_count": scene_summary["unique_scene_step_count"],
                "unique_scene_count": scene_summary["unique_scene_count"],
                "members_relative_path": member_relative_path,
            }
        )
        artifact_records.append(
            {
                "artifact_kind": "table",
                "artifact_type": "cluster_members",
                "relative_path": member_relative_path,
                "absolute_path": str(member_path.resolve()),
                "performance_group": performance_group,
                "algorithm": algorithm,
                "cluster_space": cluster_space,
                "cluster_id": cluster_id,
                "is_noise": bool(profile_row["is_noise"]),
                "cluster_size": int(profile_row["cluster_size"]),
            }
        )

    cluster_catalog_df = pd.DataFrame(
        catalog_rows,
        columns=[
            "performance_group",
            "algorithm",
            "cluster_space",
            "candidate_label_col",
            "cluster_id",
            "cluster_label",
            "is_noise",
            "cluster_size",
            "cluster_size_share",
            "cluster_rank_by_size",
            "unique_scene_step_count",
            "unique_scene_count",
            "members_relative_path",
        ],
    )
    if not cluster_catalog_df.empty:
        cluster_catalog_df = cluster_catalog_df.sort_values(
            ["performance_group", "algorithm", "cluster_space", "is_noise", "cluster_rank_by_size", "cluster_id"],
            ascending=[True, True, True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)
    cluster_catalog_df.to_csv(cluster_catalog_path, index=False)
    artifact_records.insert(
        0,
        {
            "artifact_kind": "table",
            "artifact_type": "cluster_catalog",
            "relative_path": str(cluster_catalog_path.relative_to(cluster_spec_root)),
            "absolute_path": str(cluster_catalog_path.resolve()),
        },
    )

    return {
        "cluster_feature_effect_profiles_df": cluster_feature_effect_profiles_df,
        "cluster_feature_effect_profiles_path": cluster_feature_effect_profiles_path,
        "cluster_catalog_df": cluster_catalog_df,
        "cluster_catalog_path": cluster_catalog_path,
        "artifact_records": artifact_records,
    }
