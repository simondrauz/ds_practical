from __future__ import annotations

"""Helpers for assembling and clustering run-scoped SHAP regime analysis tables."""
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

SHAP_PREFIX = "shap__"
VALID_PERFORMANCE_GROUPS = ("easy", "medium", "hard")
VALID_CLUSTER_ALGORITHMS = ("hdbscan", "optics")
VALID_CLUSTER_SPACES = ("raw", "umap")
VALID_CLUSTER_PROFILE_SORT_KEYS = ("cluster_size", "cluster_rank_by_size")

TRUSTWORTHINESS_COLUMNS = [
    "performance_group",
    "n_components",
    "trustworthiness_view",
    "trustworthiness_n_neighbors",
    "trustworthiness",
    "selected_for_clustering",
]
CLUSTER_SHAP_PROFILE_PREFIX_COLUMNS = [
    "performance_group",
    "selected_algorithm",
    "selected_cluster_space",
    "cluster_id",
    "cluster_size",
    "cluster_size_share",
    "cluster_rank_by_size",
    "dominant_feature_1",
    "dominant_abs_shap_1",
    "dominant_direction_1",
    "dominant_feature_2",
    "dominant_abs_shap_2",
    "dominant_direction_2",
    "dominant_feature_3",
    "dominant_abs_shap_3",
    "dominant_direction_3",
]
CLUSTER_SCORE_REQUIRED_COLUMNS = [
    "performance_group",
    "algorithm",
    "cluster_space",
    "candidate_label_col",
]

_CLUSTER_SPEC_REQUIRED_KEYS = {
    "groups",
    "algorithms",
    "evaluate_umap_latent_space",
    "umap_selected_n_components",
    "trustworthiness_neighbor_values",
    "cluster_umap_n_neighbors",
    "cluster_umap_min_dist",
    "viz_umap_n_neighbors",
    "viz_umap_min_dist",
    "random_state",
    "min_cluster_size",
    "min_samples",
    "optics_cluster_method",
    "optics_xi",
    "distance_metric",
}
_CLUSTER_SPEC_FORBIDDEN_KEYS = {
    "umap_candidate_dims": "Remove CLUSTER_SPEC['umap_candidate_dims']; the notebook derives candidate dimensions from the SHAP feature count.",
    "umap_n_neighbors": "Replace CLUSTER_SPEC['umap_n_neighbors'] with explicit 'cluster_umap_n_neighbors' and 'viz_umap_n_neighbors' values.",
    "umap_min_dist": "Replace CLUSTER_SPEC['umap_min_dist'] with explicit 'cluster_umap_min_dist' and 'viz_umap_min_dist' values.",
    "trustworthiness_n_neighbors": "Replace CLUSTER_SPEC['trustworthiness_n_neighbors'] with CLUSTER_SPEC['trustworthiness_neighbor_values'].",
    "min_cluster_size_fraction": "Replace CLUSTER_SPEC['min_cluster_size_fraction'] with an explicit integer CLUSTER_SPEC['min_cluster_size'].",
    "min_cluster_size_min": "Replace CLUSTER_SPEC['min_cluster_size_min'] with an explicit integer CLUSTER_SPEC['min_cluster_size'].",
}
_INSPECTION_CONFIG_REQUIRED_KEYS = {
    "inspection_algorithm",
    "inspection_cluster_space",
    "inspection_top_k_features",
    "inspection_top_k_table",
    "sort_cluster_profiles_by",
}


def resolve_raw_metric_col(manifest: dict, target_col: str) -> str:
    """Resolve the raw metric name associated with one modelling target."""
    raw_target_col = manifest.get("raw_target_col")
    if raw_target_col:
        return str(raw_target_col)
    if target_col.endswith("_log"):
        return target_col[:-4]
    return target_col


def assert_columns_present(df: pd.DataFrame, required_cols: Iterable[str], *, df_name: str) -> None:
    """Raise a descriptive error when a dataframe misses required columns."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"{df_name} is missing required columns: {missing_cols}")


def assert_unique_key(df: pd.DataFrame, key_cols: list[str], *, df_name: str) -> None:
    """Raise when a dataframe is not unique on the expected merge key."""
    duplicate_count = int(df.duplicated(subset=key_cols).sum())
    if duplicate_count:
        raise ValueError(
            f"{df_name} is not unique on the feature key. Duplicate rows found: {duplicate_count}"
        )


def _empty_trustworthiness_df() -> pd.DataFrame:
    """Return the standard empty trustworthiness table used by the notebook."""
    return pd.DataFrame(columns=TRUSTWORTHINESS_COLUMNS)


def _empty_cluster_shap_profiles_df(shap_cols: list[str]) -> pd.DataFrame:
    """Return the standard empty selected-cluster profile table."""
    return pd.DataFrame(columns=CLUSTER_SHAP_PROFILE_PREFIX_COLUMNS + list(shap_cols))


def _raise_missing_keys(config_name: str, missing_keys: set[str]) -> None:
    missing = sorted(missing_keys)
    raise ValueError(
        f"{config_name} is missing required keys: {missing}. "
        "Update the notebook input cell before rerunning."
    )


def _reject_forbidden_keys(config: Mapping[str, Any], *, config_name: str, forbidden_keys: Mapping[str, str]) -> None:
    """Fail fast when the notebook still uses removed legacy config keys."""
    for key, message in forbidden_keys.items():
        if key in config:
            raise ValueError(f"{config_name}['{key}'] is no longer supported. {message}")


def _reject_unknown_keys(config: Mapping[str, Any], *, config_name: str, allowed_keys: set[str]) -> None:
    """Keep notebook config blocks small and explicit by rejecting stray keys."""
    unknown_keys = sorted(set(config) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"{config_name} contains unsupported keys: {unknown_keys}. "
            "Keep only the documented notebook inputs."
        )


def _resolve_bool(value: Any, *, config_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{config_name} must be True or False, got {value!r}.")
    return value


def _resolve_int(value: Any, *, config_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{config_name} must be an integer, got {value!r}.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{config_name} must be an integer, got {value!r}.") from exc


def _resolve_positive_int(value: Any, *, config_name: str) -> int:
    resolved_value = _resolve_int(value, config_name=config_name)
    if resolved_value < 1:
        raise ValueError(f"{config_name} must be >= 1, got {resolved_value}.")
    return resolved_value


def _resolve_non_negative_float(value: Any, *, config_name: str) -> float:
    try:
        resolved_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{config_name} must be a float, got {value!r}.") from exc
    if resolved_value < 0:
        raise ValueError(f"{config_name} must be >= 0, got {resolved_value}.")
    return resolved_value


def _resolve_fraction(value: Any, *, config_name: str) -> float:
    try:
        resolved_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{config_name} must be a float in the open interval (0, 1), got {value!r}.") from exc
    if not 0 < resolved_value < 1:
        raise ValueError(f"{config_name} must be in the open interval (0, 1), got {resolved_value}.")
    return resolved_value


def _resolve_non_empty_string(value: Any, *, config_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{config_name} must be a non-empty string, got {value!r}.")
    return value.strip()


def _resolve_choice_sequence(
    value: Any,
    *,
    config_name: str,
    allowed_values: tuple[str, ...],
) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{config_name} must be a non-empty list, got {value!r}.")

    resolved_values = [_resolve_non_empty_string(item, config_name=config_name) for item in value]
    duplicate_values = sorted({item for item in resolved_values if resolved_values.count(item) > 1})
    if duplicate_values:
        raise ValueError(f"{config_name} contains duplicates: {duplicate_values}.")

    invalid_values = [item for item in resolved_values if item not in allowed_values]
    if invalid_values:
        raise ValueError(
            f"{config_name} contains unsupported values: {invalid_values}. "
            f"Expected values drawn from {list(allowed_values)}."
        )
    return resolved_values


def _resolve_group_specific_positive_ints(
    value: Any,
    *,
    config_name: str,
    groups: list[str],
) -> dict[str, int]:
    """Normalize one scalar-or-per-group notebook input into a full group mapping."""
    if isinstance(value, Mapping):
        missing_groups = sorted(set(groups) - set(value))
        extra_groups = sorted(set(value) - set(groups))
        if missing_groups:
            raise ValueError(
                f"{config_name} is missing per-group values for {missing_groups}. "
                "Provide one value for every configured performance group."
            )
        if extra_groups:
            raise ValueError(
                f"{config_name} contains unexpected performance groups: {extra_groups}. "
                f"Expected groups: {groups}."
            )
        return {
            performance_group: _resolve_positive_int(
                value[performance_group],
                config_name=f"{config_name}[{performance_group!r}]",
            )
            for performance_group in groups
        }

    resolved_value = _resolve_positive_int(value, config_name=config_name)
    return {performance_group: resolved_value for performance_group in groups}


def resolve_cluster_spec(
    cluster_spec: Mapping[str, Any],
    *,
    shap_cols: list[str],
) -> dict[str, Any]:
    """Validate notebook clustering inputs and derive the internal clustering config.

    The notebook intentionally exposes one small user-editable configuration block.
    This helper rejects legacy aliases, validates the documented keys, and derives
    the candidate UMAP dimensions from the loaded SHAP feature columns so every
    downstream step consumes one explicit, normalized contract.
    """
    _reject_forbidden_keys(cluster_spec, config_name="CLUSTER_SPEC", forbidden_keys=_CLUSTER_SPEC_FORBIDDEN_KEYS)
    missing_keys = _CLUSTER_SPEC_REQUIRED_KEYS - set(cluster_spec)
    if missing_keys:
        _raise_missing_keys("CLUSTER_SPEC", missing_keys)
    _reject_unknown_keys(
        cluster_spec,
        config_name="CLUSTER_SPEC",
        allowed_keys=_CLUSTER_SPEC_REQUIRED_KEYS,
    )

    groups = _resolve_choice_sequence(
        cluster_spec["groups"],
        config_name="CLUSTER_SPEC['groups']",
        allowed_values=VALID_PERFORMANCE_GROUPS,
    )
    algorithms = _resolve_choice_sequence(
        cluster_spec["algorithms"],
        config_name="CLUSTER_SPEC['algorithms']",
        allowed_values=VALID_CLUSTER_ALGORITHMS,
    )
    evaluate_umap_latent_space = _resolve_bool(
        cluster_spec["evaluate_umap_latent_space"],
        config_name="CLUSTER_SPEC['evaluate_umap_latent_space']",
    )

    if not shap_cols:
        raise ValueError("Cannot resolve CLUSTER_SPEC because no SHAP columns were detected in the analysis table.")
    umap_candidate_dims = list(range(1, len(shap_cols)))
    if evaluate_umap_latent_space and not umap_candidate_dims:
        raise ValueError(
            "CLUSTER_SPEC['evaluate_umap_latent_space']=True requires at least two SHAP feature columns. "
            "Disable reduced-space clustering or rerun the upstream export with more features."
        )

    umap_selected_n_components = _resolve_group_specific_positive_ints(
        cluster_spec["umap_selected_n_components"],
        config_name="CLUSTER_SPEC['umap_selected_n_components']",
        groups=groups,
    )
    if evaluate_umap_latent_space:
        invalid_selected_dims = {
            performance_group: selected_dim
            for performance_group, selected_dim in umap_selected_n_components.items()
            if selected_dim not in umap_candidate_dims
        }
        if invalid_selected_dims:
            raise ValueError(
                "CLUSTER_SPEC['umap_selected_n_components'] contains dimensions outside the derived candidate set. "
                f"Invalid selections: {invalid_selected_dims}. Valid dims: {umap_candidate_dims}."
            )

    raw_trustworthiness_values = cluster_spec["trustworthiness_neighbor_values"]
    if not isinstance(raw_trustworthiness_values, (list, tuple)):
        raise ValueError(
            "CLUSTER_SPEC['trustworthiness_neighbor_values'] must be a non-empty list of integers."
        )
    trustworthiness_neighbor_values = [
        _resolve_positive_int(
            value,
            config_name=f"CLUSTER_SPEC['trustworthiness_neighbor_values'][{idx}]",
        )
        for idx, value in enumerate(raw_trustworthiness_values)
    ]
    if not trustworthiness_neighbor_values:
        raise ValueError("CLUSTER_SPEC['trustworthiness_neighbor_values'] must contain at least one value.")

    optics_cluster_method = _resolve_non_empty_string(
        cluster_spec["optics_cluster_method"],
        config_name="CLUSTER_SPEC['optics_cluster_method']",
    )
    if optics_cluster_method not in {"xi", "dbscan"}:
        raise ValueError(
            "CLUSTER_SPEC['optics_cluster_method'] must be 'xi' or 'dbscan', "
            f"got {optics_cluster_method!r}."
        )

    resolved_cluster_spec = {
        "groups": groups,
        "algorithms": algorithms,
        "evaluate_umap_latent_space": evaluate_umap_latent_space,
        "umap_candidate_dims": umap_candidate_dims,
        "umap_selected_n_components": umap_selected_n_components,
        "trustworthiness_neighbor_values": trustworthiness_neighbor_values,
        "trustworthiness_mean_view": _trustworthiness_mean_view_name(trustworthiness_neighbor_values),
        "cluster_umap_n_neighbors": _resolve_positive_int(
            cluster_spec["cluster_umap_n_neighbors"],
            config_name="CLUSTER_SPEC['cluster_umap_n_neighbors']",
        ),
        "cluster_umap_min_dist": _resolve_non_negative_float(
            cluster_spec["cluster_umap_min_dist"],
            config_name="CLUSTER_SPEC['cluster_umap_min_dist']",
        ),
        "viz_umap_n_neighbors": _resolve_positive_int(
            cluster_spec["viz_umap_n_neighbors"],
            config_name="CLUSTER_SPEC['viz_umap_n_neighbors']",
        ),
        "viz_umap_min_dist": _resolve_non_negative_float(
            cluster_spec["viz_umap_min_dist"],
            config_name="CLUSTER_SPEC['viz_umap_min_dist']",
        ),
        "random_state": _resolve_int(
            cluster_spec["random_state"],
            config_name="CLUSTER_SPEC['random_state']",
        ),
        "min_cluster_size": _resolve_group_specific_positive_ints(
            cluster_spec["min_cluster_size"],
            config_name="CLUSTER_SPEC['min_cluster_size']",
            groups=groups,
        ),
        "min_samples": _resolve_group_specific_positive_ints(
            cluster_spec["min_samples"],
            config_name="CLUSTER_SPEC['min_samples']",
            groups=groups,
        ),
        "optics_cluster_method": optics_cluster_method,
        "optics_xi": _resolve_fraction(
            cluster_spec["optics_xi"],
            config_name="CLUSTER_SPEC['optics_xi']",
        ),
        "distance_metric": _resolve_non_empty_string(
            cluster_spec["distance_metric"],
            config_name="CLUSTER_SPEC['distance_metric']",
        ),
    }
    return resolved_cluster_spec


def resolve_inspection_config(
    inspection_config: Mapping[str, Any],
    *,
    cluster_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate notebook inspection inputs against the resolved clustering config."""
    missing_keys = _INSPECTION_CONFIG_REQUIRED_KEYS - set(inspection_config)
    if missing_keys:
        _raise_missing_keys("INSPECTION_CONFIG", missing_keys)
    _reject_unknown_keys(
        inspection_config,
        config_name="INSPECTION_CONFIG",
        allowed_keys=_INSPECTION_CONFIG_REQUIRED_KEYS,
    )

    inspection_algorithm = _resolve_non_empty_string(
        inspection_config["inspection_algorithm"],
        config_name="INSPECTION_CONFIG['inspection_algorithm']",
    )
    if inspection_algorithm not in cluster_spec["algorithms"]:
        raise ValueError(
            f"INSPECTION_CONFIG['inspection_algorithm']={inspection_algorithm!r} is not enabled in CLUSTER_SPEC['algorithms']={cluster_spec['algorithms']}."
        )

    inspection_cluster_space = _resolve_non_empty_string(
        inspection_config["inspection_cluster_space"],
        config_name="INSPECTION_CONFIG['inspection_cluster_space']",
    )
    if inspection_cluster_space not in VALID_CLUSTER_SPACES:
        raise ValueError(
            "INSPECTION_CONFIG['inspection_cluster_space'] must be 'raw' or 'umap', "
            f"got {inspection_cluster_space!r}."
        )
    if inspection_cluster_space == "umap" and not cluster_spec["evaluate_umap_latent_space"]:
        raise ValueError(
            "INSPECTION_CONFIG['inspection_cluster_space']='umap' requires "
            "CLUSTER_SPEC['evaluate_umap_latent_space']=True."
        )

    sort_cluster_profiles_by = _resolve_non_empty_string(
        inspection_config["sort_cluster_profiles_by"],
        config_name="INSPECTION_CONFIG['sort_cluster_profiles_by']",
    )
    if sort_cluster_profiles_by not in VALID_CLUSTER_PROFILE_SORT_KEYS:
        raise ValueError(
            "INSPECTION_CONFIG['sort_cluster_profiles_by'] must be one of "
            f"{list(VALID_CLUSTER_PROFILE_SORT_KEYS)}, got {sort_cluster_profiles_by!r}."
        )

    return {
        "inspection_algorithm": inspection_algorithm,
        "inspection_cluster_space": inspection_cluster_space,
        "inspection_top_k_features": _resolve_positive_int(
            inspection_config["inspection_top_k_features"],
            config_name="INSPECTION_CONFIG['inspection_top_k_features']",
        ),
        "inspection_top_k_table": _resolve_positive_int(
            inspection_config["inspection_top_k_table"],
            config_name="INSPECTION_CONFIG['inspection_top_k_table']",
        ),
        "sort_cluster_profiles_by": sort_cluster_profiles_by,
    }


def select_inspection_cluster_runs(
    cluster_scores_df: pd.DataFrame,
    *,
    cluster_spec: Mapping[str, Any],
    inspection_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Return the cluster runs inspected in the notebook after validating coverage."""
    assert_columns_present(
        cluster_scores_df,
        CLUSTER_SCORE_REQUIRED_COLUMNS,
        df_name="cluster scores table",
    )

    inspected_cluster_runs_df = (
        cluster_scores_df.loc[
            (cluster_scores_df["algorithm"] == inspection_config["inspection_algorithm"])
            & (cluster_scores_df["cluster_space"] == inspection_config["inspection_cluster_space"])
        ]
        .copy()
        .sort_values("performance_group")
    )
    available_groups = set(cluster_scores_df["performance_group"])
    expected_groups = [group for group in cluster_spec["groups"] if group in available_groups]
    missing_groups = [
        group for group in expected_groups
        if group not in set(inspected_cluster_runs_df["performance_group"])
    ]
    if missing_groups:
        raise ValueError(
            "Inspection selection is missing clustering results for performance groups "
            f"{missing_groups}. Update INSPECTION_CONFIG or rerun the clustering step."
        )
    return inspected_cluster_runs_df


def prepare_shap_value_export(
    *,
    model_df_oof: pd.DataFrame,
    feature_cols: list[str],
    shap_values: np.ndarray,
    base_values: np.ndarray | float | None = None,
) -> pd.DataFrame:
    """Build a run-scoped per-row SHAP export with a stable column contract.

    SHAP explainers occasionally return `(n_rows, n_features, 1)` for single-output
    models. The notebook works with a plain 2D table, so the helper normalizes that
    edge shape once here and then enforces one row per OOF modelling row.
    """
    required_cols = feature_cols + ["row_id", "outer_fold", "oof_pred_orig", "target_orig"]
    assert_columns_present(model_df_oof, required_cols, df_name="model_data_with_oof")

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3 and shap_array.shape[-1] == 1:
        shap_array = shap_array[..., 0]
    if shap_array.ndim != 2:
        raise ValueError(
            f"Expected SHAP values to be 2D after normalization, got shape={shap_array.shape}"
        )
    expected_shape = (len(model_df_oof), len(feature_cols))
    if shap_array.shape != expected_shape:
        raise ValueError(
            "SHAP values shape does not match the OOF modelling table. "
            f"expected={expected_shape}, actual={shap_array.shape}"
        )

    shap_col_names = [f"shap__{feature}" for feature in feature_cols]
    shap_export_df = model_df_oof[feature_cols + ["row_id", "outer_fold", "oof_pred_orig", "target_orig"]].copy()
    shap_export_df = pd.concat(
        [
            shap_export_df.reset_index(drop=True),
            pd.DataFrame(shap_array, columns=shap_col_names),
        ],
        axis=1,
    )

    if base_values is not None:
        base_array = np.asarray(base_values)
        if base_array.ndim == 0:
            shap_export_df["shap_base_value"] = float(base_array)
        else:
            base_array = base_array.reshape(-1)
            if len(base_array) != len(model_df_oof):
                raise ValueError(
                    "SHAP base values length does not match the OOF modelling table. "
                    f"expected={len(model_df_oof)}, actual={len(base_array)}"
                )
            shap_export_df["shap_base_value"] = base_array

    return shap_export_df


def assign_performance_groups(
    metric_values: pd.Series,
    *,
    lower_is_better: bool = True,
) -> tuple[pd.Series, float, float]:
    """Assign easy/medium/hard groups from quartile thresholds.

    The notebook defines regimes relative to the empirical performance distribution
    within the selected run: the best quartile is `easy`, the worst quartile is
    `hard`, and the middle half is `medium`.
    """
    if metric_values.isna().any():
        missing_count = int(metric_values.isna().sum())
        raise ValueError(f"Performance metric contains missing values: {missing_count}")

    q25 = float(metric_values.quantile(0.25))
    q75 = float(metric_values.quantile(0.75))

    if lower_is_better:
        labels = np.select(
            [metric_values <= q25, metric_values >= q75],
            ["easy", "hard"],
            default="medium",
        )
    else:
        labels = np.select(
            [metric_values >= q75, metric_values <= q25],
            ["easy", "hard"],
            default="medium",
        )

    return pd.Series(labels, index=metric_values.index, name="performance_group"), q25, q75


def build_group_summary_df(
    *,
    analysis_df: pd.DataFrame,
    performance_metric_col: str,
    performance_group_col: str,
    q25: float,
    q75: float,
) -> pd.DataFrame:
    """Build a compact one-row summary of the quartile grouping result."""
    return pd.DataFrame(
        [
            {
                "metric_col": performance_metric_col,
                "q25": q25,
                "q75": q75,
                "n_total": len(analysis_df),
                "n_easy": int((analysis_df[performance_group_col] == "easy").sum()),
                "n_medium": int((analysis_df[performance_group_col] == "medium").sum()),
                "n_hard": int((analysis_df[performance_group_col] == "hard").sum()),
                "n_equal_q25": int((analysis_df[performance_metric_col] == q25).sum()),
                "n_equal_q75": int((analysis_df[performance_metric_col] == q75).sum()),
            }
        ]
    )


def assemble_step1_analysis_table(
    *,
    prepared_model_df: pd.DataFrame,
    joined_metrics_df: pd.DataFrame,
    shap_values_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    performance_metric_col: str,
    lower_is_better: bool = True,
    performance_group_col: str = "performance_group",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join prepared rows, run metrics, and SHAP exports into one analysis table."""
    key_cols = list(feature_cols)
    assert_columns_present(prepared_model_df, key_cols + [target_col], df_name="prepared data")
    assert_columns_present(joined_metrics_df, key_cols + [performance_metric_col], df_name="joined metrics")

    shap_required_cols = key_cols + ["row_id", "outer_fold", "oof_pred_orig", "target_orig"]
    assert_columns_present(shap_values_df, shap_required_cols, df_name="SHAP export")
    expected_shap_cols = [f"shap__{feature}" for feature in feature_cols]
    assert_columns_present(shap_values_df, expected_shap_cols, df_name="SHAP export")

    assert_unique_key(prepared_model_df, key_cols, df_name="prepared data")
    assert_unique_key(joined_metrics_df, key_cols, df_name="joined metrics")
    assert_unique_key(shap_values_df, key_cols, df_name="SHAP export")

    joined_metric_cols = [col for col in joined_metrics_df.columns if col not in key_cols]
    analysis_df = prepared_model_df.merge(
        joined_metrics_df[key_cols + joined_metric_cols],
        on=key_cols,
        how="left",
        validate="one_to_one",
        indicator="_metrics_merge",
        sort=False,
    )
    merge_mismatch_count = int((analysis_df["_metrics_merge"] != "both").sum())
    if merge_mismatch_count:
        raise ValueError(
            "Prepared rows could not be fully aligned back to the joined metrics export. "
            f"Unmatched rows: {merge_mismatch_count}"
        )
    analysis_df = analysis_df.drop(columns=["_metrics_merge"])

    shap_merge_cols = [col for col in shap_values_df.columns if col not in key_cols]
    overlapping_cols = sorted(set(shap_merge_cols) & set(analysis_df.columns))
    if overlapping_cols:
        raise ValueError(
            "SHAP export has overlapping non-key columns with the prepared/metrics merge. "
            f"Overlaps: {overlapping_cols}"
        )

    analysis_df = analysis_df.merge(
        shap_values_df[key_cols + shap_merge_cols],
        on=key_cols,
        how="left",
        validate="one_to_one",
        indicator="_shap_merge",
        sort=False,
    )
    shap_mismatch_count = int((analysis_df["_shap_merge"] != "both").sum())
    if shap_mismatch_count:
        raise ValueError(
            "Prepared rows could not be fully aligned back to the SHAP export. "
            f"Unmatched rows: {shap_mismatch_count}"
        )
    analysis_df = analysis_df.drop(columns=["_shap_merge"])

    performance_groups, q25, q75 = assign_performance_groups(
        analysis_df[performance_metric_col],
        lower_is_better=lower_is_better,
    )
    analysis_df[performance_group_col] = performance_groups

    group_summary_df = build_group_summary_df(
        analysis_df=analysis_df,
        performance_metric_col=performance_metric_col,
        performance_group_col=performance_group_col,
        q25=q25,
        q75=q75,
    )

    return analysis_df, group_summary_df


def get_shap_cols(df: pd.DataFrame, *, prefix: str = SHAP_PREFIX) -> list[str]:
    """Return SHAP columns in dataframe order and fail when none are present."""
    shap_cols = [col for col in df.columns if col.startswith(prefix)]
    if not shap_cols:
        raise ValueError(f"No SHAP columns found with prefix {prefix!r}.")
    return shap_cols


def format_shap_feature_name(shap_col: str, *, prefix: str = SHAP_PREFIX) -> str:
    """Convert one SHAP column name back to its original feature name."""
    if shap_col.startswith(prefix):
        return shap_col[len(prefix) :]
    return shap_col


def _require_step2_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    """Import clustering dependencies lazily so step 1 stays lightweight."""
    try:
        import hdbscan
        from hdbscan.validity import validity_index
    except ImportError as exc:
        raise ImportError(
            "Clustering requires the 'hdbscan' package. Install the repo requirements into "
            "the adaptive-py310 environment before running this notebook."
        ) from exc

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "Clustering requires the 'umap-learn' package. It should be available in adaptive-py310."
        ) from exc

    try:
        from sklearn.cluster import OPTICS
        from sklearn.manifold import trustworthiness
    except ImportError as exc:
        raise ImportError(
            "Clustering requires scikit-learn with OPTICS and trustworthiness support."
        ) from exc

    return hdbscan, validity_index, umap, OPTICS, trustworthiness


def _get_group_specific_int(cluster_spec: Mapping[str, Any], *, performance_group: str, param_name: str) -> int:
    """Read a per-group integer from the resolved cluster spec."""
    return int(cluster_spec[param_name][performance_group])


def _clip_umap_candidate_dims(candidate_dims: Iterable[int], *, n_features: int, n_rows: int) -> list[int]:
    """Keep only reduced dimensions that are mathematically valid for one group."""
    max_dim = min(int(n_features) - 1, int(n_rows) - 1)
    valid_dims = sorted({int(dim) for dim in candidate_dims if 1 <= int(dim) <= max_dim})
    return valid_dims


def _effective_neighbor_count(requested_neighbors: int, n_rows: int) -> int:
    """Clip neighbor counts to the valid range for the available group rows."""
    return max(2, min(int(requested_neighbors), int(n_rows) - 1))


def _coerce_label_series(length: int) -> pd.Series:
    """Create an empty nullable integer label column for cluster assignments."""
    return pd.Series(pd.array([pd.NA] * length, dtype="Int64"))


def _compute_dbcv_score(validity_index_fn, X: np.ndarray, labels: np.ndarray) -> tuple[float, bool]:
    """Compute DBCV and record whether the score is valid for model selection."""
    non_noise_clusters = sorted({int(label) for label in labels if int(label) != -1})
    if len(non_noise_clusters) < 2:
        return float("nan"), False
    try:
        # hdbscan.validity.validity_index expects a float64 buffer; UMAP embeddings are float32 by default.
        X_for_dbcv = np.ascontiguousarray(X, dtype=np.float64)
        return float(validity_index_fn(X_for_dbcv, labels)), True
    except Exception:
        return float("nan"), False


def _resolve_trustworthiness_neighbor_values(cluster_spec: dict[str, Any]) -> list[int]:
    """Return the validated trustworthiness neighborhood values from the resolved config."""
    return [int(value) for value in cluster_spec["trustworthiness_neighbor_values"]]


def _trustworthiness_mean_view_name(trustworthiness_neighbor_values: list[int]) -> str:
    return "mean_" + "_".join(str(int(value)) for value in trustworthiness_neighbor_values)


def evaluate_umap_dimensions(
    X: np.ndarray,
    *,
    performance_group: str,
    cluster_spec: dict[str, Any],
    trustworthiness_fn,
    umap_module,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Evaluate every valid reduced dimension for one performance group.

    The same UMAP neighborhood parameters are reused across all candidate
    dimensions so the trustworthiness curves isolate the effect of dimension
    count rather than changing multiple hyperparameters at once.
    """
    candidate_dims = _clip_umap_candidate_dims(
        cluster_spec["umap_candidate_dims"],
        n_features=X.shape[1],
        n_rows=len(X),
    )
    if not candidate_dims:
        return _empty_trustworthiness_df(), {}

    n_neighbors = _effective_neighbor_count(cluster_spec["cluster_umap_n_neighbors"], len(X))
    trustworthiness_neighbor_values = _resolve_trustworthiness_neighbor_values(cluster_spec)
    mean_view_name = str(cluster_spec["trustworthiness_mean_view"])

    trust_rows: list[dict[str, Any]] = []
    embeddings: dict[int, np.ndarray] = {}
    selected_umap_dim = int(cluster_spec["umap_selected_n_components"][performance_group])

    for n_components in candidate_dims:
        umap_model = umap_module.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=cluster_spec["cluster_umap_min_dist"],
            random_state=cluster_spec["random_state"],
        )
        embedding = umap_model.fit_transform(X)
        embeddings[n_components] = embedding
        trust_scores: list[float] = []
        for trust_neighbors in trustworthiness_neighbor_values:
            effective_trust_neighbors = _effective_neighbor_count(trust_neighbors, len(X))
            trust_score = float(trustworthiness_fn(X, embedding, n_neighbors=effective_trust_neighbors))
            trust_scores.append(trust_score)
            trust_rows.append(
                {
                    "performance_group": performance_group,
                    "n_components": n_components,
                    "trustworthiness_view": f"nn_{int(trust_neighbors)}",
                    "trustworthiness_n_neighbors": int(trust_neighbors),
                    "trustworthiness": trust_score,
                    "selected_for_clustering": False,
                }
            )
        if trust_scores:
            trust_rows.append(
                {
                    "performance_group": performance_group,
                    "n_components": n_components,
                    "trustworthiness_view": mean_view_name,
                    "trustworthiness_n_neighbors": pd.NA,
                    "trustworthiness": float(np.mean(trust_scores)),
                    "selected_for_clustering": False,
                }
            )

    trust_df = pd.DataFrame(trust_rows)
    if not trust_df.empty:
        trust_df["trustworthiness_n_neighbors"] = pd.array(trust_df["trustworthiness_n_neighbors"], dtype="Int64")
    if selected_umap_dim is not None:
        if selected_umap_dim not in candidate_dims:
            raise ValueError(
                f"Selected UMAP dimension {selected_umap_dim} is invalid for performance_group={performance_group!r}. "
                f"Valid dims: {candidate_dims}"
            )
        trust_df.loc[trust_df["n_components"] == selected_umap_dim, "selected_for_clustering"] = True
    return trust_df, embeddings


def _compute_visual_umap_embedding(
    X: np.ndarray,
    *,
    cluster_spec: dict[str, Any],
    umap_module,
) -> np.ndarray:
    """Build the shared 2D visualization embedding used in notebook plots."""
    n_neighbors = _effective_neighbor_count(cluster_spec["viz_umap_n_neighbors"], len(X))
    umap_model = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=cluster_spec["viz_umap_min_dist"],
        random_state=cluster_spec["random_state"],
    )
    return umap_model.fit_transform(X)


def evaluate_umap_trustworthiness_by_group(
    analysis_df: pd.DataFrame,
    *,
    cluster_spec: dict[str, Any],
    performance_group_col: str = "performance_group",
    shap_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate UMAP trustworthiness over candidate dimensions for each performance group."""
    _, _, umap_module, _, trustworthiness_fn = _require_step2_dependencies()

    shap_cols = shap_cols or get_shap_cols(analysis_df)
    assert_columns_present(analysis_df, [performance_group_col] + shap_cols, df_name="regime analysis table")

    trustworthiness_rows: list[pd.DataFrame] = []
    groups = [group for group in cluster_spec["groups"] if group in set(analysis_df[performance_group_col])]
    for performance_group in groups:
        group_df = analysis_df.loc[analysis_df[performance_group_col] == performance_group].copy()
        X_raw = group_df[shap_cols].to_numpy(dtype=float)
        trust_df, _ = evaluate_umap_dimensions(
            X_raw,
            performance_group=performance_group,
            cluster_spec=cluster_spec,
            trustworthiness_fn=trustworthiness_fn,
            umap_module=umap_module,
        )
        if not trust_df.empty:
            trustworthiness_rows.append(trust_df)

    return (
        pd.concat(trustworthiness_rows, ignore_index=True)
        if trustworthiness_rows
        else _empty_trustworthiness_df()
    )


def _fit_hdbscan_labels(
    X: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    hdbscan_module,
) -> np.ndarray:
    """Fit one HDBSCAN run and return the cluster labels."""
    clusterer = hdbscan_module.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
        allow_single_cluster=False,
    )
    return clusterer.fit_predict(X)


def _fit_optics_labels(
    X: np.ndarray,
    *,
    min_samples: int,
    min_cluster_size: int,
    xi: float,
    metric: str,
    optics_cls,
    cluster_method: str,
) -> np.ndarray:
    """Fit one OPTICS run and return the cluster labels."""
    clusterer = optics_cls(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        xi=xi,
        metric=metric,
        cluster_method=cluster_method,
    )
    return clusterer.fit_predict(X)


def _select_best_group_run(group_scores_df: pd.DataFrame) -> int | None:
    """Select the best valid clustering run with deterministic tie-breakers."""
    valid_scores_df = group_scores_df.loc[group_scores_df["valid_for_selection"]].copy()
    if valid_scores_df.empty:
        return None

    # Prefer stronger DBCV first, then less noise, then the notebook's fixed
    # raw-before-UMAP and HDBSCAN-before-OPTICS tie-breakers for reproducibility.
    valid_scores_df["cluster_space_priority"] = valid_scores_df["cluster_space"].map({"raw": 0, "umap": 1}).fillna(99)
    valid_scores_df["algorithm_priority"] = valid_scores_df["algorithm"].map({"hdbscan": 0, "optics": 1}).fillna(99)
    valid_scores_df = valid_scores_df.sort_values(
        ["dbcv_cluster_space", "noise_fraction", "cluster_space_priority", "algorithm_priority"],
        ascending=[False, True, True, True],
    )
    return int(valid_scores_df.iloc[0]["score_row_id"])


def _rank_cluster_profiles(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rank cluster summaries from largest to smallest cluster."""
    if summary_df.empty:
        return summary_df.copy()

    ranked_df = summary_df.sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
    ranked_df["cluster_rank_by_size"] = np.arange(1, len(ranked_df) + 1, dtype=int)
    return ranked_df


def _append_dominant_feature_fields(summary_df: pd.DataFrame, *, shap_cols: list[str], top_k: int = 3) -> pd.DataFrame:
    """Attach top absolute SHAP drivers so tables are readable without scanning all columns."""
    if summary_df.empty:
        return summary_df.copy()

    enriched_df = summary_df.copy()
    label_lookup = {shap_col: format_shap_feature_name(shap_col) for shap_col in shap_cols}

    for row_idx, row in enriched_df.iterrows():
        ordered_shap_cols = sorted(shap_cols, key=lambda shap_col: abs(float(row[shap_col])), reverse=True)
        for rank in range(1, top_k + 1):
            feature_col = f"dominant_feature_{rank}"
            magnitude_col = f"dominant_abs_shap_{rank}"
            direction_col = f"dominant_direction_{rank}"
            if rank <= len(ordered_shap_cols):
                shap_col = ordered_shap_cols[rank - 1]
                shap_value = float(row[shap_col])
                direction = "positive" if shap_value > 0 else "negative" if shap_value < 0 else "neutral"
                enriched_df.loc[row_idx, feature_col] = label_lookup[shap_col]
                enriched_df.loc[row_idx, magnitude_col] = abs(shap_value)
                enriched_df.loc[row_idx, direction_col] = direction
            else:
                enriched_df.loc[row_idx, feature_col] = pd.NA
                enriched_df.loc[row_idx, magnitude_col] = np.nan
                enriched_df.loc[row_idx, direction_col] = pd.NA

    return enriched_df


def _build_selected_cluster_shap_summary(
    group_df: pd.DataFrame,
    *,
    labels: np.ndarray,
    performance_group: str,
    selected_algorithm: str,
    selected_cluster_space: str,
    shap_cols: list[str],
) -> pd.DataFrame:
    """Summarize mean signed SHAP values for every non-noise cluster in one group."""
    summary_rows: list[dict[str, Any]] = []
    cluster_ids = sorted({int(label) for label in labels if int(label) != -1})
    group_size = int(len(group_df))
    for cluster_id in cluster_ids:
        cluster_rows = group_df.loc[labels == cluster_id]
        if cluster_rows.empty:
            continue
        cluster_size = int(len(cluster_rows))
        row: dict[str, Any] = {
            "performance_group": performance_group,
            "selected_algorithm": selected_algorithm,
            "selected_cluster_space": selected_cluster_space,
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
            "cluster_size_share": float(cluster_size / group_size),
        }
        for shap_col in shap_cols:
            row[shap_col] = float(cluster_rows[shap_col].mean())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df
    summary_df = _rank_cluster_profiles(summary_df)
    summary_df = _append_dominant_feature_fields(summary_df, shap_cols=shap_cols, top_k=3)
    return summary_df


def build_cluster_shap_profiles(
    clustered_df: pd.DataFrame,
    cluster_runs_df: pd.DataFrame,
    *,
    performance_group_col: str = "performance_group",
    shap_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build mean signed SHAP profiles for the selected cluster run in each group."""
    shap_cols = shap_cols or get_shap_cols(clustered_df)
    assert_columns_present(
        clustered_df,
        [performance_group_col] + shap_cols,
        df_name="cluster assignment table",
    )
    assert_columns_present(
        cluster_runs_df,
        CLUSTER_SCORE_REQUIRED_COLUMNS,
        df_name="cluster run selection table",
    )

    summary_frames: list[pd.DataFrame] = []
    for _, cluster_run in cluster_runs_df.iterrows():
        performance_group = str(cluster_run["performance_group"])
        label_col = str(cluster_run["candidate_label_col"])
        if label_col not in clustered_df.columns:
            raise KeyError(f"Cluster assignment table is missing candidate label column: {label_col}")

        group_df = clustered_df.loc[clustered_df[performance_group_col] == performance_group].copy()
        labels = group_df[label_col].to_numpy(dtype="int64")
        summary_df = _build_selected_cluster_shap_summary(
            group_df,
            labels=labels,
            performance_group=performance_group,
            selected_algorithm=str(cluster_run["algorithm"]),
            selected_cluster_space=str(cluster_run["cluster_space"]),
            shap_cols=shap_cols,
        )
        if not summary_df.empty:
            summary_frames.append(summary_df)

    if summary_frames:
        return pd.concat(summary_frames, ignore_index=True)

    return _empty_cluster_shap_profiles_df(shap_cols)


def run_step2_clustering(
    analysis_df: pd.DataFrame,
    *,
    cluster_spec: dict[str, Any],
    performance_group_col: str = "performance_group",
    row_id_col: str = "row_id",
    shap_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Cluster SHAP rows within each performance group and return notebook-friendly artifacts."""
    hdbscan_module, validity_index_fn, umap_module, optics_cls, trustworthiness_fn = _require_step2_dependencies()

    shap_cols = shap_cols or get_shap_cols(analysis_df)
    assert_columns_present(analysis_df, [performance_group_col, row_id_col] + shap_cols, df_name="regime analysis table")

    missing_shap_count = int(analysis_df[shap_cols].isna().sum().sum())
    if missing_shap_count:
        raise ValueError(f"Clustering cannot proceed with missing SHAP values. Missing cells: {missing_shap_count}")

    clustered_df = analysis_df.copy()
    for algorithm in cluster_spec["algorithms"]:
        cluster_col = f"cluster_{algorithm}_raw"
        clustered_df[cluster_col] = _coerce_label_series(len(clustered_df))
        if cluster_spec["evaluate_umap_latent_space"]:
            cluster_col = f"cluster_{algorithm}_umap"
            clustered_df[cluster_col] = _coerce_label_series(len(clustered_df))

    clustered_df["selected_cluster"] = _coerce_label_series(len(clustered_df))
    clustered_df["selected_algorithm"] = pd.Series([pd.NA] * len(clustered_df), dtype="object")
    clustered_df["selected_cluster_space"] = pd.Series([pd.NA] * len(clustered_df), dtype="object")
    clustered_df["selected_noise"] = pd.Series([pd.NA] * len(clustered_df), dtype="boolean")
    clustered_df["viz_umap_x"] = np.nan
    clustered_df["viz_umap_y"] = np.nan

    trustworthiness_rows: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []
    score_row_id = 0

    groups = [group for group in cluster_spec["groups"] if group in set(analysis_df[performance_group_col])]
    for performance_group in groups:
        group_mask = clustered_df[performance_group_col] == performance_group
        group_df = clustered_df.loc[group_mask].copy()
        X_raw = group_df[shap_cols].to_numpy(dtype=float)
        group_size = len(group_df)

        min_cluster_size = _get_group_specific_int(
            cluster_spec,
            performance_group=performance_group,
            param_name="min_cluster_size",
        )
        min_samples = _get_group_specific_int(
            cluster_spec,
            performance_group=performance_group,
            param_name="min_samples",
        )

        if group_size < max(2, min_cluster_size, min_samples):
            raise ValueError(
                f"Performance group {performance_group!r} has too few rows for clustering. "
                f"rows={group_size}, min_cluster_size={min_cluster_size}, min_samples={min_samples}"
            )

        viz_embedding = _compute_visual_umap_embedding(
            X_raw,
            cluster_spec=cluster_spec,
            umap_module=umap_module,
        )
        clustered_df.loc[group_mask, "viz_umap_x"] = viz_embedding[:, 0]
        clustered_df.loc[group_mask, "viz_umap_y"] = viz_embedding[:, 1]

        trust_df = pd.DataFrame()
        selected_umap_embedding = None
        selected_umap_dim = _get_group_specific_int(
            cluster_spec,
            performance_group=performance_group,
            param_name="umap_selected_n_components",
        )
        if cluster_spec["evaluate_umap_latent_space"]:
            trust_df, umap_embeddings = evaluate_umap_dimensions(
                X_raw,
                performance_group=performance_group,
                cluster_spec=cluster_spec,
                trustworthiness_fn=trustworthiness_fn,
                umap_module=umap_module,
            )
            if not trust_df.empty:
                trustworthiness_rows.append(trust_df)
            if selected_umap_dim is not None:
                selected_umap_embedding = umap_embeddings[selected_umap_dim]

        spaces: dict[str, tuple[np.ndarray, int | None]] = {"raw": (X_raw, None)}
        if selected_umap_embedding is not None and selected_umap_dim is not None:
            spaces["umap"] = (selected_umap_embedding, selected_umap_dim)

        group_score_row_ids: list[int] = []
        for cluster_space, (X_space, selected_dim) in spaces.items():
            for algorithm in cluster_spec["algorithms"]:
                if algorithm == "hdbscan":
                    labels = _fit_hdbscan_labels(
                        X_space,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric=cluster_spec["distance_metric"],
                        hdbscan_module=hdbscan_module,
                    )
                    min_samples_value = min_samples
                elif algorithm == "optics":
                    labels = _fit_optics_labels(
                        X_space,
                        min_samples=min_samples,
                        min_cluster_size=min_cluster_size,
                        xi=cluster_spec["optics_xi"],
                        metric=cluster_spec["distance_metric"],
                        optics_cls=optics_cls,
                        cluster_method=cluster_spec["optics_cluster_method"],
                    )
                    min_samples_value = min_samples
                else:
                    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

                candidate_col = f"cluster_{algorithm}_{cluster_space}"
                clustered_df.loc[group_mask, candidate_col] = pd.array(labels, dtype="Int64")

                non_noise_cluster_ids = sorted({int(label) for label in labels if int(label) != -1})
                n_clusters = len(non_noise_cluster_ids)
                noise_count = int((labels == -1).sum())
                clustered_count = int((labels != -1).sum())
                dbcv_cluster_space, valid_for_selection = _compute_dbcv_score(validity_index_fn, X_space, labels)
                dbcv_raw_shap_space, valid_for_raw_shap_evaluation = _compute_dbcv_score(validity_index_fn, X_raw, labels)
                group_score_row_ids.append(score_row_id)
                score_rows.append(
                    {
                        "score_row_id": score_row_id,
                        "performance_group": performance_group,
                        "algorithm": algorithm,
                        "cluster_space": cluster_space,
                        "candidate_label_col": candidate_col,
                        "input_dim": int(X_space.shape[1]),
                        "group_size": int(group_size),
                        "min_cluster_size": int(min_cluster_size),
                        "min_samples": int(min_samples_value),
                        "optics_xi": float(cluster_spec["optics_xi"]) if algorithm == "optics" else np.nan,
                        "umap_selected_n_components": (
                            int(selected_dim) if cluster_space == "umap" and selected_dim is not None else np.nan
                        ),
                        "n_clusters": int(n_clusters),
                        "noise_count": int(noise_count),
                        "noise_fraction": float(noise_count / group_size),
                        "clustered_fraction": float(clustered_count / group_size),
                        "dbcv": dbcv_cluster_space,
                        "dbcv_cluster_space": dbcv_cluster_space,
                        "dbcv_raw_shap_space": dbcv_raw_shap_space,
                        "valid_for_selection": bool(valid_for_selection),
                        "valid_for_raw_shap_evaluation": bool(valid_for_raw_shap_evaluation),
                        "selected_for_group": False,
                    }
                )
                score_row_id += 1

        group_scores_df = pd.DataFrame([row for row in score_rows if row["score_row_id"] in group_score_row_ids])
        best_score_row_id = _select_best_group_run(group_scores_df)
        if best_score_row_id is None:
            continue

        for row in score_rows:
            if row["score_row_id"] == best_score_row_id:
                row["selected_for_group"] = True
                best_row = row
                break
        else:
            raise RuntimeError(f"Selected score_row_id={best_score_row_id} could not be found.")

        selected_labels = clustered_df.loc[group_mask, best_row["candidate_label_col"]].to_numpy(dtype="int64")
        clustered_df.loc[group_mask, "selected_cluster"] = pd.array(selected_labels, dtype="Int64")
        clustered_df.loc[group_mask, "selected_algorithm"] = best_row["algorithm"]
        clustered_df.loc[group_mask, "selected_cluster_space"] = best_row["cluster_space"]
        clustered_df.loc[group_mask, "selected_noise"] = pd.array(selected_labels == -1, dtype="boolean")

    trustworthiness_df = (
        pd.concat(trustworthiness_rows, ignore_index=True)
        if trustworthiness_rows
        else _empty_trustworthiness_df()
    )
    cluster_scores_df = pd.DataFrame(score_rows).sort_values(
        ["performance_group", "selected_for_group", "dbcv_cluster_space", "algorithm", "cluster_space"],
        ascending=[True, False, False, True, True],
    )
    selected_cluster_runs_df = cluster_scores_df.loc[cluster_scores_df["selected_for_group"]].copy()
    cluster_shap_summary_df = build_cluster_shap_profiles(
        clustered_df,
        selected_cluster_runs_df,
        performance_group_col=performance_group_col,
        shap_cols=shap_cols,
    )

    return {
        "clustered_df": clustered_df,
        "trustworthiness_df": trustworthiness_df,
        "cluster_scores_df": cluster_scores_df,
        "cluster_shap_summary_df": cluster_shap_summary_df,
    }
