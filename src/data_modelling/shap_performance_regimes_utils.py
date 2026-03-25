from __future__ import annotations

"""Helpers for assembling and clustering run-scoped SHAP regime analysis tables."""

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

SHAP_PREFIX = "shap__"


def resolve_raw_metric_col(manifest: dict, target_col: str) -> str:
    """Resolve the raw metric name associated with one modelling target."""
    raw_target_col = manifest.get("raw_target_col")
    if raw_target_col:
        return str(raw_target_col)
    if target_col.endswith("_log"):
        return target_col[:-4]
    return target_col


def assert_columns_present(df: pd.DataFrame, required_cols: Iterable[str], *, df_name: str) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"{df_name} is missing required columns: {missing_cols}")


def assert_unique_key(df: pd.DataFrame, key_cols: list[str], *, df_name: str) -> None:
    duplicate_count = int(df.duplicated(subset=key_cols).sum())
    if duplicate_count:
        raise ValueError(
            f"{df_name} is not unique on the feature key. Duplicate rows found: {duplicate_count}"
        )


def prepare_shap_value_export(
    *,
    model_df_oof: pd.DataFrame,
    feature_cols: list[str],
    shap_values: np.ndarray,
    base_values: np.ndarray | float | None = None,
) -> pd.DataFrame:
    """Build a run-scoped per-row SHAP export with a stable column contract."""
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
    """Assign easy/medium/hard groups from quartile thresholds."""
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
    shap_cols = [col for col in df.columns if col.startswith(prefix)]
    if not shap_cols:
        raise ValueError(f"No SHAP columns found with prefix {prefix!r}.")
    return shap_cols


def format_shap_feature_name(shap_col: str, *, prefix: str = SHAP_PREFIX) -> str:
    if shap_col.startswith(prefix):
        return shap_col[len(prefix) :]
    return shap_col


def _require_step2_dependencies() -> tuple[Any, Any, Any, Any, Any]:
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


def derive_min_cluster_size(
    group_size: int,
    *,
    min_fraction: float,
    min_size: int,
) -> int:
    return max(int(min_size), int(math.ceil(float(min_fraction) * int(group_size))))


def _clip_umap_candidate_dims(candidate_dims: Iterable[int], *, n_features: int, n_rows: int) -> list[int]:
    max_dim = min(int(n_features) - 1, int(n_rows) - 1)
    valid_dims = sorted({int(dim) for dim in candidate_dims if 1 <= int(dim) <= max_dim})
    return valid_dims


def _effective_neighbor_count(requested_neighbors: int, n_rows: int) -> int:
    return max(2, min(int(requested_neighbors), int(n_rows) - 1))


def _coerce_label_series(length: int) -> pd.Series:
    return pd.Series(pd.array([pd.NA] * length, dtype="Int64"))


def _compute_dbcv_score(validity_index_fn, X: np.ndarray, labels: np.ndarray) -> tuple[float, bool]:
    non_noise_clusters = sorted({int(label) for label in labels if int(label) != -1})
    if len(non_noise_clusters) < 2:
        return float("nan"), False
    try:
        return float(validity_index_fn(X, labels)), True
    except Exception:
        return float("nan"), False


def _resolve_manual_umap_dim_selection(
    selection_config: Any,
    *,
    performance_group: str,
) -> int | None:
    if selection_config is None:
        return None
    if isinstance(selection_config, dict):
        selected_value = selection_config.get(performance_group)
        return None if selected_value is None else int(selected_value)
    return int(selection_config)


def evaluate_umap_dimensions(
    X: np.ndarray,
    *,
    performance_group: str,
    cluster_spec: dict[str, Any],
    trustworthiness_fn,
    umap_module,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    candidate_dims = _clip_umap_candidate_dims(
        cluster_spec["umap_candidate_dims"],
        n_features=X.shape[1],
        n_rows=len(X),
    )
    if not candidate_dims:
        return pd.DataFrame(), {}

    n_neighbors = _effective_neighbor_count(cluster_spec["umap_n_neighbors"], len(X))
    trust_neighbors = _effective_neighbor_count(
        cluster_spec.get("trustworthiness_n_neighbors", cluster_spec["umap_n_neighbors"]),
        len(X),
    )

    trust_rows: list[dict[str, Any]] = []
    embeddings: dict[int, np.ndarray] = {}
    selected_umap_dim = _resolve_manual_umap_dim_selection(
        cluster_spec.get("umap_selected_n_components"),
        performance_group=performance_group,
    )

    for n_components in candidate_dims:
        umap_model = umap_module.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=cluster_spec["umap_min_dist"],
            random_state=cluster_spec["random_state"],
        )
        embedding = umap_model.fit_transform(X)
        embeddings[n_components] = embedding
        trust_score = float(trustworthiness_fn(X, embedding, n_neighbors=trust_neighbors))
        trust_rows.append(
            {
                "performance_group": performance_group,
                "n_components": n_components,
                "trustworthiness": trust_score,
                "selected_for_clustering": False,
            }
        )

    trust_df = pd.DataFrame(trust_rows)
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
    n_neighbors = _effective_neighbor_count(cluster_spec["umap_n_neighbors"], len(X))
    umap_model = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=cluster_spec["umap_min_dist"],
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
        else pd.DataFrame(
            columns=[
                "performance_group",
                "n_components",
                "trustworthiness",
                "selected_for_clustering",
            ]
        )
    )


def _fit_hdbscan_labels(X: np.ndarray, *, min_cluster_size: int, metric: str, hdbscan_module) -> np.ndarray:
    clusterer = hdbscan_module.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_cluster_size,
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
    clusterer = optics_cls(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        xi=xi,
        metric=metric,
        cluster_method=cluster_method,
    )
    return clusterer.fit_predict(X)


def _select_best_group_run(group_scores_df: pd.DataFrame) -> int | None:
    valid_scores_df = group_scores_df.loc[group_scores_df["valid_for_selection"]].copy()
    if valid_scores_df.empty:
        return None

    valid_scores_df["cluster_space_priority"] = valid_scores_df["cluster_space"].map({"raw": 0, "umap": 1}).fillna(99)
    valid_scores_df["algorithm_priority"] = valid_scores_df["algorithm"].map({"hdbscan": 0, "optics": 1}).fillna(99)
    valid_scores_df = valid_scores_df.sort_values(
        ["dbcv", "noise_fraction", "cluster_space_priority", "algorithm_priority"],
        ascending=[False, True, True, True],
    )
    return int(valid_scores_df.iloc[0]["score_row_id"])


def _rank_cluster_profiles(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    ranked_df = summary_df.sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
    ranked_df["cluster_rank_by_size"] = np.arange(1, len(ranked_df) + 1, dtype=int)
    return ranked_df


def _append_dominant_feature_fields(summary_df: pd.DataFrame, *, shap_cols: list[str], top_k: int = 3) -> pd.DataFrame:
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
    """Build cluster SHAP profiles for a user-selected set of cluster runs."""
    shap_cols = shap_cols or get_shap_cols(clustered_df)
    assert_columns_present(
        clustered_df,
        [performance_group_col] + shap_cols,
        df_name="cluster assignment table",
    )
    assert_columns_present(
        cluster_runs_df,
        ["performance_group", "algorithm", "cluster_space", "candidate_label_col"],
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

    return pd.DataFrame(
        columns=[
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
            *shap_cols,
        ]
    )


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
    candidate_label_cols: list[str] = []
    for algorithm in cluster_spec["algorithms"]:
        cluster_col = f"cluster_{algorithm}_raw"
        clustered_df[cluster_col] = _coerce_label_series(len(clustered_df))
        candidate_label_cols.append(cluster_col)
        if cluster_spec["evaluate_umap_latent_space"]:
            cluster_col = f"cluster_{algorithm}_umap"
            clustered_df[cluster_col] = _coerce_label_series(len(clustered_df))
            candidate_label_cols.append(cluster_col)

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

        min_cluster_size = derive_min_cluster_size(
            group_size,
            min_fraction=cluster_spec["min_cluster_size_fraction"],
            min_size=cluster_spec["min_cluster_size_min"],
        )
        if group_size < max(2, min_cluster_size):
            raise ValueError(
                f"Performance group {performance_group!r} has too few rows for clustering. "
                f"rows={group_size}, min_cluster_size={min_cluster_size}"
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
        selected_umap_dim = _resolve_manual_umap_dim_selection(
            cluster_spec.get("umap_selected_n_components"),
            performance_group=performance_group,
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
                        metric=cluster_spec["distance_metric"],
                        hdbscan_module=hdbscan_module,
                    )
                    min_samples_value = min_cluster_size
                elif algorithm == "optics":
                    labels = _fit_optics_labels(
                        X_space,
                        min_samples=min_cluster_size,
                        min_cluster_size=min_cluster_size,
                        xi=cluster_spec["optics_xi"],
                        metric=cluster_spec["distance_metric"],
                        optics_cls=optics_cls,
                        cluster_method=cluster_spec["optics_cluster_method"],
                    )
                    min_samples_value = min_cluster_size
                else:
                    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

                candidate_col = f"cluster_{algorithm}_{cluster_space}"
                clustered_df.loc[group_mask, candidate_col] = pd.array(labels, dtype="Int64")

                non_noise_cluster_ids = sorted({int(label) for label in labels if int(label) != -1})
                n_clusters = len(non_noise_cluster_ids)
                noise_count = int((labels == -1).sum())
                clustered_count = int((labels != -1).sum())
                dbcv, valid_for_selection = _compute_dbcv_score(validity_index_fn, X_space, labels)
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
                        "dbcv": dbcv,
                        "valid_for_selection": bool(valid_for_selection),
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
        else pd.DataFrame(
            columns=[
                "performance_group",
                "n_components",
                "trustworthiness",
                "meets_threshold",
                "selected",
                "selection_reason",
            ]
        )
    )
    cluster_scores_df = pd.DataFrame(score_rows).sort_values(
        ["performance_group", "selected_for_group", "dbcv", "algorithm", "cluster_space"],
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
