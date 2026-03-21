from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common_metrics import regression_metrics, to_original_scale


def summarize_nested_cv(nested_cv_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "outer_rmse",
                "mean": nested_cv_df["outer_rmse"].mean(),
                "std": nested_cv_df["outer_rmse"].std(ddof=1),
            },
            {
                "metric": "outer_mae",
                "mean": nested_cv_df["outer_mae"].mean(),
                "std": nested_cv_df["outer_mae"].std(ddof=1),
            },
            {
                "metric": "outer_r2",
                "mean": nested_cv_df["outer_r2"].mean(),
                "std": nested_cv_df["outer_r2"].std(ddof=1),
            },
        ]
    )


def build_oof_frame(
    model_df: pd.DataFrame,
    row_ids,
    oof_pred,
    oof_fold,
    *,
    target_orig,
    pred_scale_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    pred_scale_kwargs = pred_scale_kwargs or {}
    model_df_oof = model_df.copy()
    model_df_oof["row_id"] = np.asarray(row_ids)
    model_df_oof["oof_pred"] = np.asarray(oof_pred)
    model_df_oof["outer_fold"] = np.asarray(oof_fold)
    model_df_oof["oof_pred_orig"] = to_original_scale(model_df_oof["oof_pred"].to_numpy(), **pred_scale_kwargs)
    model_df_oof["target_orig"] = np.asarray(target_orig)
    return model_df_oof


def build_oof_metrics_df(
    y_true_raw,
    y_pred_raw,
    *,
    target_col: str | None = None,
    target_mode: str | None = None,
) -> pd.DataFrame:
    y_true_orig = to_original_scale(np.asarray(y_true_raw), target_col=target_col, target_mode=target_mode)
    y_pred_orig = to_original_scale(np.asarray(y_pred_raw), target_col=target_col, target_mode=target_mode)
    return pd.DataFrame([regression_metrics(y_true_orig, y_pred_orig, split_name="OOF")])


def build_run_manifest(
    *,
    model_id: str,
    run_name: str,
    target_col: str,
    feature_cols: list[str],
    save_dir: Path,
    plots_dir: Path,
    tables_dir: Path,
    nested_resampling: dict[str, Any],
    final_model: dict[str, Any],
    analysis: dict[str, Any],
    extra_manifest_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "model_id": model_id,
        "run_name": run_name,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "artifact_root": str(save_dir),
        "plots_dir": str(plots_dir),
        "tables_dir": str(tables_dir),
        "nested_resampling": nested_resampling,
        "final_model": final_model,
        "analysis": analysis,
    }
    if extra_manifest_fields:
        manifest.update(extra_manifest_fields)
    return manifest


def write_manifest(manifest: dict[str, Any], tables_dir: Path, target_col: str) -> Path:
    manifest_path = tables_dir / f"run_manifest_{target_col}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
