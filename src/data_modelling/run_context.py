from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _results_root() -> Path:
    return _repo_root() / "results" / "interpretable_model"


def resolve_manifest_path(model_id: str, run_name: str, target_col: str | None = None) -> Path:
    manifest_dir = _results_root() / model_id / run_name / "tables"
    if not manifest_dir.exists():
        raise FileNotFoundError(
            f"No tables directory found for model_id={model_id}, run_name={run_name}: {manifest_dir}"
        )

    if target_col is not None:
        manifest_path = manifest_dir / f"run_manifest_{target_col}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest not found for target_col={target_col}: {manifest_path}")
        return manifest_path

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


def get_exported_model_info(manifest: dict[str, Any]) -> dict[str, Any]:
    final_model = manifest.get("final_model", {})
    target_col = manifest["target_col"]
    model_id = manifest["model_id"]
    is_gam_model = model_id == "gam" or model_id.startswith("gam-")

    exported_name = final_model.get("exported_model_name")
    if exported_name is None:
        exported_name = final_model.get("selected_variant_name")
    if exported_name is None:
        exported_name = manifest.get("variant_name")
    if exported_name is None:
        exported_name = "XGBoost" if model_id == "xgboost" else model_id

    exported_kind = final_model.get("exported_model_kind")
    if exported_kind is None:
        exported_kind = final_model.get("selected_variant_model_kind")
    if exported_kind is None:
        exported_kind = manifest.get("model_kind", model_id)

    exported_target_mode = final_model.get("exported_model_target_mode")
    if exported_target_mode is None:
        exported_target_mode = final_model.get("selected_variant_target_mode")
    if exported_target_mode is None:
        exported_target_mode = manifest.get("target_mode")
    if exported_target_mode is None:
        exported_target_mode = "log" if target_col.endswith("_log") else "raw"

    selection_metric_name = final_model.get("exported_model_selection_metric_name")
    if selection_metric_name is None:
        selection_metric_name = manifest.get("selection_metric_name")
    if selection_metric_name is None:
        selection_metric_name = "lowest_cv_rmse" if is_gam_model else "best_cv_score"

    selection_metric_value = final_model.get("exported_model_selection_metric_value")
    if selection_metric_value is None:
        selection_metric_value = final_model.get("selected_cv_rmse", final_model.get("best_cv_score"))
    if selection_metric_value is None:
        selection_metric_value = manifest.get("selection_metric_value")

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
    manifest_path = resolve_manifest_path(model_id, run_name, target_col)
    manifest = json.loads(manifest_path.read_text())

    if manifest["model_id"] != model_id:
        raise ValueError(f"Manifest model_id={manifest['model_id']} does not match MODEL_ID={model_id}")

    nested_resampling = manifest.get("nested_resampling", {})
    final_model = manifest.get("final_model", {})
    model_data_path_value = nested_resampling.get("model_data_with_oof_path")
    if model_data_path_value is None:
        raise KeyError(f"Manifest is missing nested_resampling.model_data_with_oof_path: {manifest_path}")
    model_data_path = Path(model_data_path_value)
    metrics_path_value = nested_resampling.get("oof_metrics_path")
    metrics_path = Path(metrics_path_value) if metrics_path_value else None
    tuning_summary_value = final_model.get("full_data_tuning_summary_path")
    full_data_tuning_summary_path = Path(tuning_summary_value) if tuning_summary_value else None

    oof_metrics_df = pd.read_csv(metrics_path) if metrics_path and metrics_path.exists() else None
    full_data_tuning_summary = (
        json.loads(full_data_tuning_summary_path.read_text())
        if full_data_tuning_summary_path and full_data_tuning_summary_path.exists()
        else None
    )

    return RunContext(
        manifest_path=manifest_path,
        manifest=manifest,
        target_col=manifest["target_col"],
        feature_cols=manifest["feature_cols"],
        tables_dir=Path(manifest["tables_dir"]),
        plots_dir=Path(manifest["plots_dir"]),
        nested_resampling=nested_resampling,
        final_model=final_model,
        model_data_path=model_data_path,
        model_df_oof=pd.read_csv(model_data_path),
        metrics_path=metrics_path,
        oof_metrics_df=oof_metrics_df,
        full_data_tuning_summary_path=full_data_tuning_summary_path,
        full_data_tuning_summary=full_data_tuning_summary,
    )
