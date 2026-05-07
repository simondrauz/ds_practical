from __future__ import annotations

"""Helpers for assembling and clustering run-scoped feature-effect regime analysis tables."""
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .prepared_data import IDENTITY_COLS, MODEL_SETTING_COLS

EFFECT_PREFIX = "effect__"
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
CLUSTER_FEATURE_EFFECT_PROFILE_PREFIX_COLUMNS = [
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
    "umap_candidate_dims": "Remove CLUSTER_SPEC['umap_candidate_dims']; the notebook derives candidate dimensions from the loaded feature-effect columns.",
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
            f"{df_name} is not unique on key {key_cols}. Duplicate rows found: {duplicate_count}"
        )


def _available_identity_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in IDENTITY_COLS if col in df.columns]


def _unique_preserve_order(cols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered_cols: list[str] = []
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        ordered_cols.append(col)
    return ordered_cols


def _resolve_trajectory_key_cols(
    *,
    prepared_model_df: pd.DataFrame,
    joined_metrics_df: pd.DataFrame,
    feature_effects_df: pd.DataFrame,
) -> list[str]:
    """Resolve the stable row key shared by prepared, metrics, and effect exports."""
    if "data_idx" not in prepared_model_df.columns:
        raise ValueError(
            "Prepared modelling data is missing 'data_idx'. Regenerate prepared data from "
            "joined metrics so trajectory rows are aligned by eval identity, not row position."
        )
    if "data_idx" not in feature_effects_df.columns:
        raise ValueError(
            "Feature-effect export is missing 'data_idx'. Regenerate model OOF and "
            "feature-effect exports with the stable trajectory identity columns preserved."
        )

    optional_context_cols = [
        col
        for col in ["run_name", "eval_csv_name"]
        if col in prepared_model_df.columns
        and col in joined_metrics_df.columns
        and col in feature_effects_df.columns
    ]
    return optional_context_cols + ["data_idx"]


def _ensure_prepared_row_id(prepared_model_df: pd.DataFrame) -> pd.DataFrame:
    """Expose a stable row-level key for prepared modelling tables."""
    if "row_id" in prepared_model_df.columns:
        prepared_with_row_id = prepared_model_df.copy()
    else:
        prepared_with_row_id = prepared_model_df.copy()
        prepared_with_row_id.insert(0, "row_id", prepared_with_row_id.index.to_numpy())

    assert_unique_key(prepared_with_row_id, ["row_id"], df_name="prepared data")
    return prepared_with_row_id


def _empty_trustworthiness_df() -> pd.DataFrame:
    """Return the standard empty trustworthiness table used by the notebook."""
    return pd.DataFrame(columns=TRUSTWORTHINESS_COLUMNS)


def _empty_cluster_feature_effect_profiles_df(effect_cols: list[str]) -> pd.DataFrame:
    """Return the standard empty selected-cluster profile table."""
    return pd.DataFrame(columns=CLUSTER_FEATURE_EFFECT_PROFILE_PREFIX_COLUMNS + list(effect_cols))


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


def _normalize_json_value(value: Any) -> Any:
    """Convert notebook/runtime values into stable JSON-compatible primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return value


def _stable_json_dumps(value: Any) -> str:
    """Serialize one nested value deterministically for hashing and manifests."""
    return json.dumps(_normalize_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sanitize_slug_token(value: Any) -> str:
    """Keep file and directory name fragments stable and portable."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "na"


def _short_stable_hash(value: Any, *, length: int = 12) -> str:
    serialized = _stable_json_dumps(value).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:length]


def _summarize_group_mapping(group_mapping: Mapping[str, Any]) -> str:
    """Render one group->value mapping into a compact readable slug fragment."""
    return "-".join(
        f"{_sanitize_slug_token(group)}{_sanitize_slug_token(group_mapping[group])}"
        for group in sorted(group_mapping)
    )


def _build_cluster_spec_readable_slug(cluster_spec: Mapping[str, Any]) -> str:
    """Keep folder names human-readable while the hash guarantees uniqueness."""
    selected_dims = cluster_spec.get("umap_selected_n_components", {})
    return "__".join(
        [
            f"groups-{'-'.join(_sanitize_slug_token(group) for group in cluster_spec['groups'])}",
            f"algs-{'-'.join(_sanitize_slug_token(algorithm) for algorithm in cluster_spec['algorithms'])}",
            f"umap-{'on' if cluster_spec['evaluate_umap_latent_space'] else 'off'}",
            f"dims-{_summarize_group_mapping(selected_dims) if selected_dims else 'none'}",
        ]
    )


def _default_feature_effect_regime_results_root() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "interpretable_model" / "feature_effect_performance_regimes"


def resolve_feature_effect_regime_export_context(
    *,
    model_id: str,
    run_name: str,
    target_col: str,
    eval_csv_name: str,
    lower_is_better: bool,
    performance_group_col: str,
    results_root: Path | None = None,
) -> dict[str, Any]:
    """Normalize the notebook inputs that change the underlying exported data."""
    resolved_results_root = (results_root or _default_feature_effect_regime_results_root()).resolve()
    normalized_target_col = _resolve_non_empty_string(target_col, config_name="target_col")
    normalized_eval_csv_name = _resolve_non_empty_string(eval_csv_name, config_name="eval_csv_name")
    normalized_group_col = _resolve_non_empty_string(
        performance_group_col,
        config_name="performance_group_col",
    )
    data_context = {
        "target_col": normalized_target_col,
        "eval_csv_name": normalized_eval_csv_name,
        "lower_is_better": _resolve_bool(lower_is_better, config_name="lower_is_better"),
        "performance_group_col": normalized_group_col,
    }
    data_context_slug = "__".join(
        [
            f"target-{_sanitize_slug_token(data_context['target_col'])}",
            f"eval-{_sanitize_slug_token(data_context['eval_csv_name'])}",
            f"lower-is-better-{str(data_context['lower_is_better']).lower()}",
            f"group-col-{_sanitize_slug_token(data_context['performance_group_col'])}",
        ]
    )
    model_root = resolved_results_root / _resolve_non_empty_string(model_id, config_name="model_id")
    run_root = model_root / run_name
    target_root = run_root / normalized_target_col
    data_context_root = target_root / data_context_slug
    return {
        "model_id": _resolve_non_empty_string(model_id, config_name="model_id"),
        "run_name": _resolve_non_empty_string(run_name, config_name="run_name"),
        "target_col": normalized_target_col,
        "eval_csv_name": normalized_eval_csv_name,
        "lower_is_better": data_context["lower_is_better"],
        "performance_group_col": normalized_group_col,
        "results_root": resolved_results_root,
        "model_root": model_root,
        "run_root": run_root,
        "target_root": target_root,
        "data_context_slug": data_context_slug,
        "data_context_root": data_context_root,
    }


def build_feature_effect_regime_export_layout(
    *,
    export_context: Mapping[str, Any],
    cluster_spec: Mapping[str, Any],
    create_dirs: bool = True,
) -> dict[str, Any]:
    """Build the canonical cluster-spec-scoped export layout for one notebook run."""
    cluster_spec_hash = _short_stable_hash(cluster_spec)
    cluster_spec_readable_slug = _build_cluster_spec_readable_slug(cluster_spec)
    cluster_spec_dirname = f"cluster_spec__{cluster_spec_readable_slug}__{cluster_spec_hash}"
    cluster_spec_root = Path(export_context["data_context_root"]) / cluster_spec_dirname
    tables_dir = cluster_spec_root / "tables"
    plots_dir = cluster_spec_root / "plots"
    manifest_path = cluster_spec_root / "manifest.json"
    if create_dirs:
        tables_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cluster_spec_hash": cluster_spec_hash,
        "cluster_spec_readable_slug": cluster_spec_readable_slug,
        "cluster_spec_dirname": cluster_spec_dirname,
        "cluster_spec_root": cluster_spec_root,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "manifest_path": manifest_path,
    }


def build_feature_effect_regime_artifact_names(
    *,
    cluster_spec: Mapping[str, Any],
    inspection_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return stable filenames for every exported feature-effect regime artifact."""
    trustworthiness_views = [
        *(f"nn_{int(value)}" for value in cluster_spec["trustworthiness_neighbor_values"]),
        str(cluster_spec["trustworthiness_mean_view"]),
    ]
    return {
        "tables": {
            "regime_analysis": "regime_analysis.csv",
            "performance_group_summary": "performance_group_summary.csv",
            "umap_trustworthiness": "umap_trustworthiness.csv",
            "cluster_scores": "cluster_scores.csv",
            "cluster_assignments": "cluster_assignments.csv",
            "cluster_feature_effect_profiles": "cluster_feature_effect_profiles.csv",
            "cluster_catalog": "cluster_catalog.csv",
            "feature_effect_global_ranking": "feature_effect_global_ranking.csv",
        },
        "plots": {
            "raw_algorithm_comparison_grid": "algorithm_comparison_grid__space-raw.png",
            "umap_algorithm_comparison_grid": "algorithm_comparison_grid__space-umap.png",
            "umap_trustworthiness_curves": {
                trustworthiness_view: (
                    f"umap_trustworthiness_curve__view-{_sanitize_slug_token(trustworthiness_view)}.png"
                )
                for trustworthiness_view in trustworthiness_views
            },
        },
    }


def load_or_initialize_feature_effect_regime_manifest(
    manifest_path: Path,
    *,
    run_context: Mapping[str, Any] | None = None,
    data_context: Mapping[str, Any] | None = None,
    cluster_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one cluster-spec manifest or return an initialized structure."""
    manifest_data: dict[str, Any]
    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text())
    else:
        manifest_data = {}
    manifest_data["schema_version"] = 1
    manifest_data["run_context"] = _normalize_json_value(run_context or manifest_data.get("run_context", {}))
    manifest_data["data_context"] = _normalize_json_value(data_context or manifest_data.get("data_context", {}))
    manifest_data["cluster_spec"] = _normalize_json_value(cluster_spec or manifest_data.get("cluster_spec", {}))
    existing_artifacts = manifest_data.get("artifacts", [])
    manifest_data["artifacts"] = existing_artifacts if isinstance(existing_artifacts, list) else []
    return manifest_data


def merge_feature_effect_regime_artifact_records(
    manifest_data: Mapping[str, Any],
    *,
    artifact_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Upsert artifact rows by relative path while preserving older distinct exports."""
    merged_manifest = {
        "schema_version": manifest_data["schema_version"],
        "run_context": _normalize_json_value(manifest_data.get("run_context", {})),
        "data_context": _normalize_json_value(manifest_data.get("data_context", {})),
        "cluster_spec": _normalize_json_value(manifest_data.get("cluster_spec", {})),
    }
    record_lookup: dict[str, dict[str, Any]] = {}
    for artifact in manifest_data.get("artifacts", []):
        normalized_artifact = _normalize_json_value(artifact)
        relative_path = normalized_artifact.get("relative_path")
        if relative_path:
            record_lookup[str(relative_path)] = normalized_artifact
    for artifact in artifact_records:
        normalized_artifact = _normalize_json_value(artifact)
        relative_path = normalized_artifact.get("relative_path")
        if not relative_path:
            raise ValueError("Each artifact record must include a non-empty 'relative_path'.")
        record_lookup[str(relative_path)] = normalized_artifact
    merged_manifest["artifacts"] = [
        record_lookup[key]
        for key in sorted(record_lookup)
    ]
    return merged_manifest


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
    effect_cols: list[str],
) -> dict[str, Any]:
    """Validate notebook clustering inputs and derive the internal clustering config.

    The notebook intentionally exposes one small user-editable configuration block.
    This helper rejects legacy aliases, validates the documented keys, and derives
    the candidate UMAP dimensions from the loaded feature-effect columns so every
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

    if not effect_cols:
        raise ValueError(
            "Cannot resolve CLUSTER_SPEC because no feature-effect columns were detected in the analysis table."
        )
    umap_candidate_dims = list(range(1, len(effect_cols)))
    if evaluate_umap_latent_space and not umap_candidate_dims:
        raise ValueError(
            "CLUSTER_SPEC['evaluate_umap_latent_space']=True requires at least two feature-effect columns. "
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


def prepare_feature_effect_export(
    *,
    model_df_oof: pd.DataFrame,
    feature_cols: list[str],
    effect_values: np.ndarray,
    base_values: np.ndarray | float | None = None,
) -> pd.DataFrame:
    """Build a run-scoped per-row feature-effect export with a stable column contract."""
    identity_cols = _available_identity_cols(model_df_oof)
    required_cols = _unique_preserve_order(
        feature_cols + identity_cols + ["row_id", "outer_fold", "oof_pred_orig", "target_orig"]
    )
    assert_columns_present(model_df_oof, required_cols, df_name="model_data_with_oof")

    effect_array = np.asarray(effect_values)
    if effect_array.ndim == 3 and effect_array.shape[-1] == 1:
        effect_array = effect_array[..., 0]
    if effect_array.ndim != 2:
        raise ValueError(
            f"Expected feature effects to be 2D after normalization, got shape={effect_array.shape}"
        )
    expected_shape = (len(model_df_oof), len(feature_cols))
    if effect_array.shape != expected_shape:
        raise ValueError(
            "Feature effects shape does not match the OOF modelling table. "
            f"expected={expected_shape}, actual={effect_array.shape}"
        )

    effect_col_names = [f"{EFFECT_PREFIX}{feature}" for feature in feature_cols]
    export_cols = _unique_preserve_order(
        identity_cols + feature_cols + ["row_id", "outer_fold", "oof_pred_orig", "target_orig"]
    )
    effect_export_df = model_df_oof[export_cols].copy()
    effect_export_df = pd.concat(
        [
            effect_export_df.reset_index(drop=True),
            pd.DataFrame(effect_array, columns=effect_col_names),
        ],
        axis=1,
    )

    if base_values is not None:
        base_array = np.asarray(base_values)
        if base_array.ndim == 0:
            effect_export_df["effect_base_value"] = float(base_array)
        else:
            base_array = base_array.reshape(-1)
            if len(base_array) != len(model_df_oof):
                raise ValueError(
                    "Feature-effect base values length does not match the OOF modelling table. "
                    f"expected={len(model_df_oof)}, actual={len(base_array)}"
                )
            effect_export_df["effect_base_value"] = base_array

    return effect_export_df


def compute_gam_feature_effects(
    *,
    model: Any,
    X_scaled: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature GAM term contributions and intercept/base values on the link scale."""
    X_scaled_array = np.asarray(X_scaled, dtype=float)
    if X_scaled_array.ndim != 2:
        raise ValueError(f"Expected X_scaled to be 2D, got shape={X_scaled_array.shape}")
    if X_scaled_array.shape[1] != len(feature_cols):
        raise ValueError(
            "Scaled feature matrix width does not match feature_cols. "
            f"expected={len(feature_cols)}, actual={X_scaled_array.shape[1]}"
        )

    expected_term_count = len(feature_cols) + 1
    if len(model.terms) != expected_term_count:
        raise ValueError(
            "Expected one GAM term per feature plus one intercept. "
            f"expected={expected_term_count}, actual={len(model.terms)}"
        )

    model_matrix = model._modelmat(X_scaled_array)
    effect_array = np.empty((len(X_scaled_array), len(feature_cols)), dtype=float)
    for feature_idx, feature_name in enumerate(feature_cols):
        term = model.terms[feature_idx]
        if getattr(term, "isintercept", False):
            raise ValueError(f"Unexpected intercept term at feature index {feature_idx}.")
        term_feature = getattr(term, "feature", feature_idx)
        if term_feature is not None and int(term_feature) != feature_idx:
            raise ValueError(
                "GAM term ordering no longer matches feature_cols. "
                f"feature={feature_name!r}, expected_index={feature_idx}, term_feature={term_feature}"
            )
        coef_indices = model.terms.get_coef_indices(feature_idx)
        effect_array[:, feature_idx] = np.asarray(
            model_matrix[:, coef_indices].dot(model.coef_[coef_indices]),
            dtype=float,
        ).reshape(-1)

    intercept_term_idx = len(feature_cols)
    intercept_term = model.terms[intercept_term_idx]
    if not getattr(intercept_term, "isintercept", False):
        raise ValueError("Expected the final GAM term to be the intercept.")
    intercept_coef_indices = model.terms.get_coef_indices(intercept_term_idx)
    base_values = np.asarray(
        model_matrix[:, intercept_coef_indices].dot(model.coef_[intercept_coef_indices]),
        dtype=float,
    ).reshape(-1)

    linear_predictor = np.asarray(model._linear_predictor(X_scaled_array), dtype=float).reshape(-1)
    reconstructed = base_values + effect_array.sum(axis=1)
    if not np.allclose(reconstructed, linear_predictor, rtol=1e-9, atol=1e-9):
        max_abs_error = float(np.max(np.abs(reconstructed - linear_predictor)))
        raise ValueError(
            "Computed GAM feature effects do not reconstruct the link-scale predictor. "
            f"max_abs_error={max_abs_error}"
        )

    return effect_array, base_values


def build_feature_effect_importance_table(
    *,
    model_id: str,
    feature_cols: list[str],
    effect_values: np.ndarray | None = None,
    p_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the unified global feature-effect ranking table for one model family."""
    if model_id == "xgboost":
        if effect_values is None:
            raise ValueError("XGBoost feature-effect importance requires effect_values.")
        effect_array = np.asarray(effect_values, dtype=float)
        expected_shape = (effect_array.shape[0], len(feature_cols))
        if effect_array.ndim != 2 or effect_array.shape[1] != len(feature_cols):
            raise ValueError(
                "XGBoost effect_values must be a 2D array with one column per feature. "
                f"expected second dimension={len(feature_cols)}, actual_shape={effect_array.shape}"
            )
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_abs_shap": np.abs(effect_array).mean(axis=0),
                "importance_metric": "mean_abs_shap",
                "importance_value": np.abs(effect_array).mean(axis=0),
                "importance_ascending": False,
            }
        ).sort_values(["importance_value", "feature"], ascending=[False, True]).reset_index(drop=True)
    elif model_id == "gam":
        if p_values is None:
            raise ValueError("GAM feature-effect importance requires p_values.")
        p_value_array = np.asarray(p_values, dtype=float).reshape(-1)
        if len(p_value_array) != len(feature_cols):
            raise ValueError(
                "GAM p_values length does not match feature_cols. "
                f"expected={len(feature_cols)}, actual={len(p_value_array)}"
            )
        bounded_p_values = np.maximum(p_value_array, 1e-300)
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "p_value": p_value_array,
                "neg_log10_p_value": -np.log10(bounded_p_values),
                "significant_0_05": p_value_array < 0.05,
                "importance_metric": "p_value",
                "importance_value": p_value_array,
                "importance_ascending": True,
            }
        ).sort_values(["importance_value", "feature"], ascending=[True, True]).reset_index(drop=True)
    else:
        raise NotImplementedError(f"Feature-effect importance is not implemented yet for model_id={model_id!r}.")

    importance_df.insert(1, "global_rank", np.arange(1, len(importance_df) + 1, dtype=int))
    return importance_df


def assign_performance_groups(
    metric_values: pd.Series,
    *,
    lower_is_better: bool = True,
    n_groups: int = 3,
    use_log_space: bool = True,
    random_state: int = 42,
) -> tuple[pd.Series, dict[str, object]]:
    """Assign easy/medium/hard groups via k-means clustering on the performance metric.

    Clustering runs in log1p space by default because ADE/FDE distributions are
    heavily right-skewed; this is consistent with how the rest of the pipeline
    handles `ml_ade` (see `prepared_data.py` and the data-preparation notebook).
    Centroids and boundaries are back-transformed to the raw metric scale in the
    returned `group_info` dict.
    """
    if metric_values.isna().any():
        missing_count = int(metric_values.isna().sum())
        raise ValueError(f"Performance metric contains missing values: {missing_count}")

    if use_log_space:
        if (metric_values < -1).any():
            raise ValueError(
                "Cannot apply log1p transform: performance metric contains values < -1."
            )
        cluster_values = np.log1p(metric_values.to_numpy())
    else:
        cluster_values = metric_values.to_numpy()

    kmeans = KMeans(n_clusters=n_groups, random_state=random_state)
    cluster_ids = kmeans.fit_predict(cluster_values.reshape(-1, 1))

    # Sort centroid indices so rank 0 = smallest centroid (best when lower_is_better).
    sorted_centroid_indices = np.argsort(kmeans.cluster_centers_.ravel())
    if lower_is_better:
        rank_to_label = {0: "easy", 1: "medium", 2: "hard"}
    else:
        rank_to_label = {0: "hard", 1: "medium", 2: "easy"}

    # Map each row's cluster id to its sorted rank, then to a label.
    cluster_id_to_rank = {int(cid): rank for rank, cid in enumerate(sorted_centroid_indices)}
    labels = np.array([rank_to_label[cluster_id_to_rank[cid]] for cid in cluster_ids])

    sorted_centroids_log = kmeans.cluster_centers_.ravel()[sorted_centroid_indices]
    if use_log_space:
        sorted_centroids_raw = np.expm1(sorted_centroids_log)
        # Boundaries are midpoints in log space, back-transformed.
        boundary_low_raw = float(np.expm1((sorted_centroids_log[0] + sorted_centroids_log[1]) / 2))
        boundary_high_raw = float(np.expm1((sorted_centroids_log[1] + sorted_centroids_log[2]) / 2))
    else:
        sorted_centroids_raw = sorted_centroids_log
        boundary_low_raw = float((sorted_centroids_raw[0] + sorted_centroids_raw[1]) / 2)
        boundary_high_raw = float((sorted_centroids_raw[1] + sorted_centroids_raw[2]) / 2)

    if lower_is_better:
        group_info: dict[str, object] = {
            "centroid_easy": float(sorted_centroids_raw[0]),
            "centroid_medium": float(sorted_centroids_raw[1]),
            "centroid_hard": float(sorted_centroids_raw[2]),
            "boundary_easy_medium": boundary_low_raw,
            "boundary_medium_hard": boundary_high_raw,
        }
    else:
        group_info = {
            "centroid_easy": float(sorted_centroids_raw[2]),
            "centroid_medium": float(sorted_centroids_raw[1]),
            "centroid_hard": float(sorted_centroids_raw[0]),
            "boundary_easy_medium": boundary_high_raw,
            "boundary_medium_hard": boundary_low_raw,
        }
    group_info["use_log_space"] = use_log_space
    group_info["n_groups"] = n_groups

    return pd.Series(labels, index=metric_values.index, name="performance_group"), group_info


def build_group_summary_df(
    *,
    analysis_df: pd.DataFrame,
    performance_metric_col: str,
    performance_group_col: str,
    group_info: dict[str, object],
) -> pd.DataFrame:
    """Build a compact one-row summary of the k-means grouping result."""
    return pd.DataFrame(
        [
            {
                "metric_col": performance_metric_col,
                "centroid_easy": group_info["centroid_easy"],
                "centroid_medium": group_info["centroid_medium"],
                "centroid_hard": group_info["centroid_hard"],
                "boundary_easy_medium": group_info["boundary_easy_medium"],
                "boundary_medium_hard": group_info["boundary_medium_hard"],
                "use_log_space": group_info["use_log_space"],
                "n_total": len(analysis_df),
                "n_easy": int((analysis_df[performance_group_col] == "easy").sum()),
                "n_medium": int((analysis_df[performance_group_col] == "medium").sum()),
                "n_hard": int((analysis_df[performance_group_col] == "hard").sum()),
            }
        ]
    )


def assemble_step1_analysis_table(
    *,
    prepared_model_df: pd.DataFrame,
    joined_metrics_df: pd.DataFrame,
    feature_effects_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    performance_metric_col: str,
    lower_is_better: bool = True,
    performance_group_col: str = "performance_group",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join prepared rows, run metrics, and feature-effect exports into one analysis table."""
    key_cols = list(feature_cols)
    prepared_model_df = _ensure_prepared_row_id(prepared_model_df)
    assert_columns_present(prepared_model_df, key_cols + [target_col], df_name="prepared data")
    assert_columns_present(
        joined_metrics_df,
        ["data_idx", performance_metric_col],
        df_name="joined metrics",
    )

    trajectory_key_cols = _resolve_trajectory_key_cols(
        prepared_model_df=prepared_model_df,
        joined_metrics_df=joined_metrics_df,
        feature_effects_df=feature_effects_df,
    )

    effect_required_cols = _unique_preserve_order(
        key_cols + trajectory_key_cols + ["row_id", "outer_fold", "oof_pred_orig", "target_orig"]
    )
    assert_columns_present(feature_effects_df, effect_required_cols, df_name="feature-effect export")
    expected_effect_cols = [f"{EFFECT_PREFIX}{feature}" for feature in feature_cols]
    assert_columns_present(feature_effects_df, expected_effect_cols, df_name="feature-effect export")

    assert_unique_key(prepared_model_df, trajectory_key_cols, df_name="prepared data")
    assert_unique_key(joined_metrics_df, trajectory_key_cols, df_name="joined metrics")
    assert_unique_key(feature_effects_df, trajectory_key_cols, df_name="feature-effect export")

    joined_metric_cols = [
        col
        for col in joined_metrics_df.columns
        if col not in (key_cols + trajectory_key_cols)
        and col not in MODEL_SETTING_COLS
    ]
    analysis_df = prepared_model_df.merge(
        joined_metrics_df[trajectory_key_cols + joined_metric_cols],
        on=trajectory_key_cols,
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

    non_key_identity_cols = set(IDENTITY_COLS) - set(trajectory_key_cols)
    effect_merge_cols = [
        col
        for col in feature_effects_df.columns
        if col
        not in (
            set(key_cols)
            | set(trajectory_key_cols)
            | non_key_identity_cols
            | {"row_id"}
        )
    ]
    overlapping_cols = sorted(set(effect_merge_cols) & set(analysis_df.columns))
    if overlapping_cols:
        raise ValueError(
            "Feature-effect export has overlapping non-key columns with the prepared/metrics merge. "
            f"Overlaps: {overlapping_cols}"
        )

    analysis_df = analysis_df.merge(
        feature_effects_df[trajectory_key_cols + effect_merge_cols],
        on=trajectory_key_cols,
        how="left",
        validate="one_to_one",
        indicator="_feature_effect_merge",
        sort=False,
    )
    effect_mismatch_count = int((analysis_df["_feature_effect_merge"] != "both").sum())
    if effect_mismatch_count:
        raise ValueError(
            "Prepared rows could not be fully aligned back to the feature-effect export. "
            f"Unmatched rows: {effect_mismatch_count}"
        )
    analysis_df = analysis_df.drop(columns=["_feature_effect_merge"])

    performance_groups, group_info = assign_performance_groups(
        analysis_df[performance_metric_col],
        lower_is_better=lower_is_better,
        random_state=random_state,
    )
    analysis_df[performance_group_col] = performance_groups

    group_summary_df = build_group_summary_df(
        analysis_df=analysis_df,
        performance_metric_col=performance_metric_col,
        performance_group_col=performance_group_col,
        group_info=group_info,
    )

    return analysis_df, group_summary_df


def get_effect_cols(df: pd.DataFrame, *, prefix: str = EFFECT_PREFIX) -> list[str]:
    """Return effect columns in dataframe order and fail when none are present."""
    effect_cols = [col for col in df.columns if col.startswith(prefix)]
    if not effect_cols:
        raise ValueError(f"No feature-effect columns found with prefix {prefix!r}.")
    return effect_cols


def format_effect_feature_name(effect_col: str, *, prefix: str = EFFECT_PREFIX) -> str:
    """Convert one effect column name back to its original feature name."""
    if effect_col.startswith(prefix):
        return effect_col[len(prefix) :]
    return effect_col


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
    effect_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate UMAP trustworthiness over candidate dimensions for each performance group."""
    _, _, umap_module, _, trustworthiness_fn = _require_step2_dependencies()

    effect_cols = effect_cols or get_effect_cols(analysis_df)
    assert_columns_present(analysis_df, [performance_group_col] + effect_cols, df_name="regime analysis table")

    trustworthiness_rows: list[pd.DataFrame] = []
    groups = [group for group in cluster_spec["groups"] if group in set(analysis_df[performance_group_col])]
    for performance_group in groups:
        group_df = analysis_df.loc[analysis_df[performance_group_col] == performance_group].copy()
        X_raw = group_df[effect_cols].to_numpy(dtype=float)
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

    ranked_df = summary_df.copy()
    ranked_df["cluster_rank_by_size"] = pd.Series([pd.NA] * len(ranked_df), dtype="Int64")
    non_noise_mask = ~ranked_df["is_noise"].astype(bool)
    if non_noise_mask.any():
        ranked_non_noise_df = (
            ranked_df.loc[non_noise_mask]
            .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
            .reset_index()
        )
        ranked_non_noise_df["cluster_rank_by_size"] = pd.array(
            np.arange(1, len(ranked_non_noise_df) + 1, dtype=int),
            dtype="Int64",
        )
        for row in ranked_non_noise_df.itertuples():
            ranked_df.loc[row.index, "cluster_rank_by_size"] = row.cluster_rank_by_size
    return ranked_df


def _cluster_label(cluster_id: int) -> str:
    return "noise" if int(cluster_id) == -1 else f"cluster_{int(cluster_id)}"


def _build_cluster_feature_effect_summary(
    group_df: pd.DataFrame,
    *,
    labels: np.ndarray,
    performance_group: str,
    algorithm: str,
    cluster_space: str,
    candidate_label_col: str,
    effect_cols: list[str],
    include_noise: bool,
) -> pd.DataFrame:
    """Summarize mean signed feature effects for every cluster in one candidate run."""
    summary_rows: list[dict[str, Any]] = []
    cluster_ids = sorted({int(label) for label in labels if include_noise or int(label) != -1})
    group_size = int(len(group_df))
    for cluster_id in cluster_ids:
        cluster_rows = group_df.loc[labels == cluster_id]
        if cluster_rows.empty:
            continue
        is_noise = int(cluster_id) == -1
        cluster_size = int(len(cluster_rows))
        row: dict[str, Any] = {
            "performance_group": performance_group,
            "algorithm": algorithm,
            "cluster_space": cluster_space,
            "candidate_label_col": candidate_label_col,
            "cluster_id": cluster_id,
            "cluster_label": _cluster_label(cluster_id),
            "is_noise": is_noise,
            "cluster_size": cluster_size,
            "cluster_size_share": float(cluster_size / group_size),
        }
        for effect_col in effect_cols:
            row[effect_col] = float(cluster_rows[effect_col].mean())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df
    summary_df = _rank_cluster_profiles(summary_df)
    return summary_df.sort_values(
        ["is_noise", "cluster_rank_by_size", "cluster_id"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_cluster_feature_effect_profiles(
    clustered_df: pd.DataFrame,
    cluster_runs_df: pd.DataFrame,
    *,
    performance_group_col: str = "performance_group",
    effect_cols: list[str] | None = None,
    include_noise: bool = False,
) -> pd.DataFrame:
    """Build mean signed feature-effect profiles for one or more candidate cluster runs."""
    effect_cols = effect_cols or get_effect_cols(clustered_df)
    assert_columns_present(
        clustered_df,
        [performance_group_col] + effect_cols,
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
        summary_df = _build_cluster_feature_effect_summary(
            group_df,
            labels=labels,
            performance_group=performance_group,
            algorithm=str(cluster_run["algorithm"]),
            cluster_space=str(cluster_run["cluster_space"]),
            candidate_label_col=label_col,
            effect_cols=effect_cols,
            include_noise=include_noise,
        )
        if not summary_df.empty:
            summary_frames.append(summary_df)

    if summary_frames:
        return pd.concat(summary_frames, ignore_index=True).sort_values(
            ["performance_group", "algorithm", "cluster_space", "is_noise", "cluster_rank_by_size", "cluster_id"],
            ascending=[True, True, True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)

    return _empty_cluster_feature_effect_profiles_df(effect_cols)


def run_step2_clustering(
    analysis_df: pd.DataFrame,
    *,
    cluster_spec: dict[str, Any],
    performance_group_col: str = "performance_group",
    row_id_col: str = "row_id",
    effect_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Cluster feature-effect rows within each performance group and return notebook-friendly artifacts."""
    hdbscan_module, validity_index_fn, umap_module, optics_cls, trustworthiness_fn = _require_step2_dependencies()

    effect_cols = effect_cols or get_effect_cols(analysis_df)
    assert_columns_present(
        analysis_df,
        [performance_group_col, row_id_col] + effect_cols,
        df_name="regime analysis table",
    )

    missing_effect_count = int(analysis_df[effect_cols].isna().sum().sum())
    if missing_effect_count:
        raise ValueError(
            f"Clustering cannot proceed with missing feature effects. Missing cells: {missing_effect_count}"
        )

    clustered_df = analysis_df.copy()
    for algorithm in cluster_spec["algorithms"]:
        cluster_col = f"cluster_{algorithm}_raw"
        clustered_df[cluster_col] = _coerce_label_series(len(clustered_df))
        if cluster_spec["evaluate_umap_latent_space"]:
            cluster_col = f"cluster_{algorithm}_umap"
            clustered_df[cluster_col] = _coerce_label_series(len(clustered_df))

    clustered_df["viz_umap_x"] = np.nan
    clustered_df["viz_umap_y"] = np.nan

    trustworthiness_rows: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []
    score_row_id = 0

    groups = [group for group in cluster_spec["groups"] if group in set(analysis_df[performance_group_col])]
    for performance_group in groups:
        group_mask = clustered_df[performance_group_col] == performance_group
        group_df = clustered_df.loc[group_mask].copy()
        X_raw = group_df[effect_cols].to_numpy(dtype=float)
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
                dbcv_raw_effect_space, valid_for_raw_effect_evaluation = _compute_dbcv_score(validity_index_fn, X_raw, labels)
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
                        "dbcv_raw_effect_space": dbcv_raw_effect_space,
                        "valid_for_selection": bool(valid_for_selection),
                        "valid_for_raw_effect_evaluation": bool(valid_for_raw_effect_evaluation),
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
                break
        else:
            raise RuntimeError(f"Selected score_row_id={best_score_row_id} could not be found.")

    trustworthiness_df = (
        pd.concat(trustworthiness_rows, ignore_index=True)
        if trustworthiness_rows
        else _empty_trustworthiness_df()
    )
    cluster_scores_df = pd.DataFrame(score_rows).sort_values(
        ["performance_group", "selected_for_group", "dbcv_cluster_space", "algorithm", "cluster_space"],
        ascending=[True, False, False, True, True],
    )
    cluster_feature_effect_profiles_df = build_cluster_feature_effect_profiles(
        clustered_df,
        cluster_scores_df,
        performance_group_col=performance_group_col,
        effect_cols=effect_cols,
        include_noise=True,
    )

    return {
        "clustered_df": clustered_df,
        "trustworthiness_df": trustworthiness_df,
        "cluster_scores_df": cluster_scores_df,
        "cluster_feature_effect_summary_df": cluster_feature_effect_profiles_df,
        "cluster_feature_effect_profiles_df": cluster_feature_effect_profiles_df,
    }
