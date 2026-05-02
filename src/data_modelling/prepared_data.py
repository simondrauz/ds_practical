from __future__ import annotations

"""Prepared-data helpers shared by the modelling notebooks.

Workflow overview:
1. Load the prepared CSV and optionally surface inspection tables.
2. Resolve which target column the modelling workflow should use.
3. Keep only numeric feature columns so downstream estimators receive valid inputs.
4. Drop incomplete rows once the feature/target contract is fixed.
"""

from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np
import pandas as pd


DisplayFn = Callable[[Any], None]


class SingleTargetModelData(TypedDict):
    target_col: str
    feature_cols: list[str]
    model_df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    row_ids: np.ndarray


class DualTargetModelData(TypedDict):
    base_target_name: str
    raw_target_col: str
    log_target_col: str
    target_col: str
    feature_cols: list[str]
    model_df: pd.DataFrame
    X: np.ndarray
    y_raw: np.ndarray
    y_log: np.ndarray
    row_ids: np.ndarray
    n_features: int


def _maybe_display(obj, display_fn: DisplayFn | None) -> None:
    if display_fn is not None:
        display_fn(obj)


def _resolve_single_target_col(df: pd.DataFrame, target_col: str | None, default_target: str) -> str:
    if target_col is not None:
        if target_col not in df.columns:
            raise AssertionError(f"TARGET_COL={target_col} not found in dataset columns.")
        return target_col

    preferred_log_target = f"{default_target}_log"
    if preferred_log_target in df.columns:
        # Prefer the log target when both exist because XGBoost training/evaluation
        # already treats a `_log` suffix as the contract for inverse-scaling metrics.
        return preferred_log_target
    if default_target in df.columns:
        return default_target
    return df.columns[-1]


def _resolve_dual_target_sources(
    df: pd.DataFrame,
    target_col: str | None,
    default_target: str,
) -> tuple[str, str | None, str | None]:
    if target_col is not None:
        base_target_name = target_col[:-4] if target_col.endswith("_log") else target_col
    elif default_target in df.columns:
        base_target_name = default_target
    else:
        # Fall back to the last column so notebooks still work with ad-hoc exports
        # whose target was appended during preparation.
        base_target_name = df.columns[-1].removesuffix("_log")

    raw_target_source_col = base_target_name if base_target_name in df.columns else None
    log_target_source_col = f"{base_target_name}_log" if f"{base_target_name}_log" in df.columns else None
    return base_target_name, raw_target_source_col, log_target_source_col


def load_prepared_data(
    data_path: Path,
    *,
    display_fn: DisplayFn | None = None,
    include_missing_summary: bool = False,
    include_dtype_summary: bool = False,
) -> pd.DataFrame:
    # Step 1: load the exact prepared dataset that later modelling stages consume.
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    _maybe_display(df.head(), display_fn)

    # Step 2: optionally expose verification tables so readers can confirm the input contract.
    if include_missing_summary:
        missing_summary = df.isna().sum().sort_values(ascending=False)
        print("\nMissing values per column:")
        _maybe_display(missing_summary.to_frame("missing_count"), display_fn)

    if include_dtype_summary:
        dtype_summary = df.dtypes.astype(str).to_frame("dtype")
        print("\nColumn dtypes:")
        _maybe_display(dtype_summary, display_fn)

    return df


def _filter_numeric_feature_cols(df: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    # Estimators and scaling steps in these notebooks assume numeric inputs only.
    # Dropping non-numeric columns early keeps that assumption explicit and testable.
    non_numeric_features = [c for c in candidate_cols if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric_features:
        print("WARNING: Non-numeric features found and dropped:")
        print(non_numeric_features)
    return [c for c in candidate_cols if c not in non_numeric_features]


def prepare_single_target_model_data(
    df: pd.DataFrame,
    *,
    target_col: str | None = None,
    default_target: str = "ml_ade",
) -> SingleTargetModelData:
    # Step 1: resolve the target column that defines the modelling/evaluation scale.
    resolved_target_col = _resolve_single_target_col(df, target_col, default_target)

    # Step 2: retain only numeric predictors and keep the target out of the feature matrix.
    feature_cols = _filter_numeric_feature_cols(
        df,
        [c for c in df.columns if c != resolved_target_col],
    )

    # Step 3: freeze the modelling frame and row ids after dropping incomplete rows.
    # Downstream notebooks rely on `row_ids` to map OOF predictions back to original rows.
    model_df = df[feature_cols + [resolved_target_col]].dropna().copy()

    return {
        "target_col": resolved_target_col,
        "feature_cols": feature_cols,
        "model_df": model_df,
        "X": model_df[feature_cols],
        "y": model_df[resolved_target_col],
        "row_ids": model_df.index.to_numpy(),
    }


def prepare_dual_target_model_data(
    df: pd.DataFrame,
    *,
    target_col: str | None = None,
    default_target: str = "ml_ade",
) -> DualTargetModelData:
    # Step 1: resolve the base target and discover whether raw/log variants already exist.
    base_target_name, raw_target_source_col, log_target_source_col = _resolve_dual_target_sources(
        df,
        target_col,
        default_target,
    )

    # Step 2: build a purely numeric feature set so GAM variants can share one matrix.
    excluded_cols = {c for c in [raw_target_source_col, log_target_source_col] if c is not None}
    feature_cols = _filter_numeric_feature_cols(df, [c for c in df.columns if c not in excluded_cols])

    # Verify model settings are present and preserved
    MODEL_SETTING_COLS = ['attention_radius_m', 'history_sec', 'prediction_sec']
    missing_settings = [c for c in MODEL_SETTING_COLS if c not in feature_cols]
    if missing_settings:
        print(f"WARNING: Expected model settings not found in features: {missing_settings}")
    else:
        print(f"✓ Model settings preserved: {[c for c in MODEL_SETTING_COLS if c in feature_cols]}")

    # Step 3: drop incomplete rows only after the full feature/target contract is known.
    model_df = df[feature_cols].dropna().copy()

    if raw_target_source_col is None and log_target_source_col is not None:
        raw_target_col = base_target_name
        # When only the log target is stored, recover the raw target so raw-scale diagnostics
        # and cohort analyses can still be computed from one consistent modelling frame.
        model_df[raw_target_col] = np.expm1(model_df[log_target_source_col].to_numpy())
    else:
        raw_target_col = raw_target_source_col

    if raw_target_col is None:
        raise ValueError("Could not resolve a raw target column for GAM comparison workflow.")

    if log_target_source_col is None:
        if (model_df[raw_target_col] < -1).any():
            # `log1p` is only defined for values >= -1, so fail before creating a misleading
            # log target that downstream selection and inverse-scaling would treat as valid.
            raise ValueError("Cannot derive a log target because raw target contains values below -1.")
        log_target_col = f"{raw_target_col}_log"
        model_df[log_target_col] = np.log1p(model_df[raw_target_col].to_numpy())
    else:
        log_target_col = log_target_source_col

    # Step 4: expose both target views because the GAM notebook compares raw and log variants
    # on the same rows/features and selects the winning target mode later in the pipeline.
    target_col_out = log_target_col
    X = model_df[feature_cols].to_numpy()
    y_raw = model_df[raw_target_col].to_numpy()
    y_log = model_df[log_target_col].to_numpy()

    return {
        "base_target_name": base_target_name,
        "raw_target_col": raw_target_col,
        "log_target_col": log_target_col,
        "target_col": target_col_out,
        "feature_cols": feature_cols,
        "model_df": model_df,
        "X": X,
        "y_raw": y_raw,
        "y_log": y_log,
        "row_ids": model_df.index.to_numpy(),
        "n_features": X.shape[1],
    }
