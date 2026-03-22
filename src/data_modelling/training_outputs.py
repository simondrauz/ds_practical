from __future__ import annotations

"""Helpers for exporting metrics and manifests from modelling notebooks.

Workflow overview:
1. Summarize nested-resampling metrics on a fixed schema.
2. Attach aligned OOF predictions back onto the modelling frame.
3. Compute original-scale metrics for downstream diagnostics.
4. Build and persist a manifest that points analysis notebooks at the saved artifacts.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common_metrics import regression_metrics, to_original_scale


def _require_columns(df: pd.DataFrame, required_cols: list[str], *, frame_name: str) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"{frame_name} is missing required columns: {missing_cols}")


def _aligned_array(values, *, expected_len: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[0] != expected_len:
        raise ValueError(
            f"{name} must contain exactly {expected_len} rows to align with model_df. "
            f"Received {array.shape[0]}."
        )
    return array


def summarize_nested_cv(nested_cv_df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: verify the fold-level metric table contains the shared export schema.
    _require_columns(
        nested_cv_df,
        ["outer_rmse", "outer_mae", "outer_r2"],
        frame_name="nested_cv_df",
    )

    # Step 2: reduce fold-level metrics into one verification table used by manifests and notebooks.
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
    # Step 1: validate that every exported vector aligns with the filtered modelling frame.
    pred_scale_kwargs = pred_scale_kwargs or {}
    expected_len = len(model_df)
    row_ids_array = _aligned_array(row_ids, expected_len=expected_len, name="row_ids")
    oof_pred_array = _aligned_array(oof_pred, expected_len=expected_len, name="oof_pred")
    oof_fold_array = _aligned_array(oof_fold, expected_len=expected_len, name="oof_fold")
    target_orig_array = _aligned_array(target_orig, expected_len=expected_len, name="target_orig")

    # Step 2: copy the modelling frame so exported diagnostics preserve the exact feature/target rows
    # that survived preprocessing and CV splitting.
    model_df_oof = model_df.copy()
    model_df_oof["row_id"] = row_ids_array
    model_df_oof["oof_pred"] = oof_pred_array
    model_df_oof["outer_fold"] = oof_fold_array

    # Step 3: convert predictions back to the original target scale before storing them so every
    # downstream diagnostic can compare like-for-like values without re-deriving target mode.
    model_df_oof["oof_pred_orig"] = to_original_scale(model_df_oof["oof_pred"].to_numpy(), **pred_scale_kwargs)
    model_df_oof["target_orig"] = target_orig_array
    return model_df_oof


def build_oof_metrics_df(
    y_true_raw,
    y_pred_raw,
    *,
    target_col: str | None = None,
    target_mode: str | None = None,
) -> pd.DataFrame:
    # Metrics must be computed on the original scale so RMSE/MAE stay interpretable for readers
    # and comparable across raw-target and log-target training variants.
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
    # Step 1: assemble the shared manifest schema analysis notebooks already understand.
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
        # Extra fields carry model-specific details, but they should never replace the core schema.
        manifest.update(extra_manifest_fields)
    return manifest


def write_manifest(manifest: dict[str, Any], tables_dir: Path, target_col: str) -> Path:
    # Persist the manifest next to the exported tables so analysis notebooks only need the run id
    # and optional target override to reconstruct the entire artifact set.
    manifest_path = tables_dir / f"run_manifest_{target_col}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
