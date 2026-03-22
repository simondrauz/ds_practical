from __future__ import annotations

"""Helpers for loading manifest-driven modelling outputs.

Workflow overview:
1. Resolve the one manifest that describes a completed modelling run.
2. Read the manifest and validate the fields downstream notebooks depend on.
3. Load required artifacts eagerly and optional summaries only when present.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _results_root() -> Path:
    return _repo_root() / "results" / "interpretable_model"


class ExportedModelInfo(TypedDict):
    name: str
    kind: str
    target_mode: str
    selection_metric_name: str
    selection_metric_value: Any


def resolve_manifest_path(model_id: str, run_name: str, target_col: str | None = None) -> Path:
    # Step 1: locate the tables directory for the requested run.
    manifest_dir = _results_root() / model_id / run_name / "tables"
    if not manifest_dir.exists():
        raise FileNotFoundError(
            f"No tables directory found for model_id={model_id}, run_name={run_name}: {manifest_dir}"
        )

    if target_col is not None:
        # An explicit target override removes ambiguity when one run exports multiple targets.
        manifest_path = manifest_dir / f"run_manifest_{target_col}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest not found for target_col={target_col}: {manifest_path}")
        return manifest_path

    # Without a target override, only one manifest may exist or the caller must choose.
    manifest_candidates = sorted(manifest_dir.glob("run_manifest_*.json"))
    if not manifest_candidates:
        raise FileNotFoundError(f"No run_manifest_*.json files found in {manifest_dir}")
    if len(manifest_candidates) > 1:
        raise ValueError(
            f"Multiple run manifests found in {manifest_dir}. Set TARGET_COL explicitly. "
            f"Candidates: {[p.name for p in manifest_candidates]}"
        )
    return manifest_candidates[0]


@dataclass
class RunContext:
    manifest_path: Path
    manifest: dict[str, Any]
    target_col: str
    feature_cols: list[str]
    tables_dir: Path
    plots_dir: Path
    nested_resampling: dict[str, Any]
    final_model: dict[str, Any]
    model_data_path: Path
    model_df_oof: pd.DataFrame
    metrics_path: Path | None = None
    oof_metrics_df: pd.DataFrame | None = None
    full_data_tuning_summary_path: Path | None = None
    full_data_tuning_summary: dict[str, Any] | None = None


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _require_manifest_field(manifest: dict[str, Any], field_name: str) -> Any:
    if field_name not in manifest:
        raise KeyError(f"Manifest is missing required field {field_name!r}.")
    return manifest[field_name]


def _manifest_path(value: str | None, *, field_name: str) -> Path | None:
    if value is None:
        return None
    if not value:
        raise ValueError(f"Manifest field {field_name!r} must not be an empty path.")
    return Path(value)


def _require_manifest_path(manifest: dict[str, Any], field_name: str) -> Path:
    path = _manifest_path(_require_manifest_field(manifest, field_name), field_name=field_name)
    if path is None:
        raise ValueError(f"Manifest field {field_name!r} must not be null.")
    return path


def _load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def get_exported_model_info(manifest: dict[str, Any]) -> ExportedModelInfo:
    # Step 1: gather the fields used by both training and analysis notebooks.
    final_model = manifest.get("final_model", {})
    target_col = _require_manifest_field(manifest, "target_col")
    model_id = _require_manifest_field(manifest, "model_id")
    is_gam_model = model_id == "gam" or model_id.startswith("gam-")

    # Step 2: resolve notebook-facing labels from most explicit export metadata down to
    # older manifest fields so existing results remain loadable.
    exported_name = _first_non_none(
        final_model.get("exported_model_name"),
        final_model.get("selected_variant_name"),
        manifest.get("variant_name"),
        "XGBoost" if model_id == "xgboost" else model_id,
    )
    exported_kind = _first_non_none(
        final_model.get("exported_model_kind"),
        final_model.get("selected_variant_model_kind"),
        manifest.get("model_kind"),
        model_id,
    )
    exported_target_mode = _first_non_none(
        final_model.get("exported_model_target_mode"),
        final_model.get("selected_variant_target_mode"),
        manifest.get("target_mode"),
        "log" if target_col.endswith("_log") else "raw",
    )
    selection_metric_name = _first_non_none(
        final_model.get("exported_model_selection_metric_name"),
        manifest.get("selection_metric_name"),
        "lowest_cv_rmse" if is_gam_model else "best_cv_score",
    )
    selection_metric_value = _first_non_none(
        final_model.get("exported_model_selection_metric_value"),
        final_model.get("selected_cv_rmse"),
        final_model.get("best_cv_score"),
        manifest.get("selection_metric_value"),
    )

    return {
        "name": exported_name,
        "kind": exported_kind,
        "target_mode": exported_target_mode,
        "selection_metric_name": selection_metric_name,
        "selection_metric_value": selection_metric_value,
    }


def format_exported_model_label(exported_model_info: dict[str, Any]) -> str:
    return (
        f"{exported_model_info['name']} "
        f"({exported_model_info['kind']}, target_mode={exported_model_info['target_mode']})"
    )


def load_run_context(model_id: str, run_name: str, target_col: str | None = None) -> RunContext:
    # Step 1: resolve and read the manifest that defines the exported artifacts.
    manifest_path = resolve_manifest_path(model_id, run_name, target_col)
    manifest = json.loads(manifest_path.read_text())

    if _require_manifest_field(manifest, "model_id") != model_id:
        raise ValueError(f"Manifest model_id={manifest['model_id']} does not match MODEL_ID={model_id}")

    # Step 2: validate the fields the downstream notebooks assume are always present.
    nested_resampling = manifest.get("nested_resampling", {})
    final_model = manifest.get("final_model", {})
    feature_cols = _require_manifest_field(manifest, "feature_cols")
    target_col_value = _require_manifest_field(manifest, "target_col")
    tables_dir = _require_manifest_path(manifest, "tables_dir")
    plots_dir = _require_manifest_path(manifest, "plots_dir")
    model_data_path_value = nested_resampling.get("model_data_with_oof_path")
    if model_data_path_value is None:
        raise KeyError(f"Manifest is missing nested_resampling.model_data_with_oof_path: {manifest_path}")
    model_data_path = _manifest_path(model_data_path_value, field_name="nested_resampling.model_data_with_oof_path")
    metrics_path = _manifest_path(nested_resampling.get("oof_metrics_path"), field_name="nested_resampling.oof_metrics_path")
    full_data_tuning_summary_path = _manifest_path(
        final_model.get("full_data_tuning_summary_path"),
        field_name="final_model.full_data_tuning_summary_path",
    )

    if model_data_path is None:
        raise ValueError("Manifest field 'nested_resampling.model_data_with_oof_path' must not be empty.")

    # Step 3: load required and optional artifacts.
    # OOF model data is required because both analysis notebooks derive every later table/plot from it.
    model_df_oof = pd.read_csv(model_data_path)
    oof_metrics_df = _load_optional_csv(metrics_path)
    full_data_tuning_summary = _load_optional_json(full_data_tuning_summary_path)

    return RunContext(
        manifest_path=manifest_path,
        manifest=manifest,
        target_col=target_col_value,
        feature_cols=feature_cols,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        nested_resampling=nested_resampling,
        final_model=final_model,
        model_data_path=model_data_path,
        model_df_oof=model_df_oof,
        metrics_path=metrics_path,
        oof_metrics_df=oof_metrics_df,
        full_data_tuning_summary_path=full_data_tuning_summary_path,
        full_data_tuning_summary=full_data_tuning_summary,
    )
