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


def load_run_context(model_id: str, run_name: str, target_col: str | None = None) -> RunContext:
    manifest_path = resolve_manifest_path(model_id, run_name, target_col)
    manifest = json.loads(manifest_path.read_text())

    if manifest["model_id"] != model_id:
        raise ValueError(f"Manifest model_id={manifest['model_id']} does not match MODEL_ID={model_id}")

    nested_resampling = manifest["nested_resampling"]
    final_model = manifest["final_model"]
    model_data_path = Path(nested_resampling["model_data_with_oof_path"])
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
