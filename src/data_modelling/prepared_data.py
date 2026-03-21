from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd


DisplayFn = Callable[[Any], None]


def _maybe_display(obj, display_fn: DisplayFn | None) -> None:
    if display_fn is not None:
        display_fn(obj)


def load_prepared_data(
    data_path: Path,
    *,
    display_fn: DisplayFn | None = None,
    include_missing_summary: bool = False,
    include_dtype_summary: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    _maybe_display(df.head(), display_fn)

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
) -> dict[str, Any]:
    if target_col is not None:
        if target_col not in df.columns:
            raise AssertionError(f"TARGET_COL={target_col} not found in dataset columns.")
        resolved_target_col = target_col
    else:
        if f"{default_target}_log" in df.columns:
            resolved_target_col = f"{default_target}_log"
        elif default_target in df.columns:
            resolved_target_col = default_target
        else:
            resolved_target_col = df.columns[-1]

    feature_cols = _filter_numeric_feature_cols(
        df,
        [c for c in df.columns if c != resolved_target_col],
    )
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
) -> dict[str, Any]:
    if target_col is not None:
        base_target_name = target_col[:-4] if target_col.endswith("_log") else target_col
    elif default_target in df.columns:
        base_target_name = default_target
    else:
        base_target_name = df.columns[-1].removesuffix("_log")

    raw_target_source_col = base_target_name if base_target_name in df.columns else None
    log_target_source_col = f"{base_target_name}_log" if f"{base_target_name}_log" in df.columns else None

    excluded_cols = {c for c in [raw_target_source_col, log_target_source_col] if c is not None}
    feature_cols = _filter_numeric_feature_cols(df, [c for c in df.columns if c not in excluded_cols])

    required_cols = feature_cols.copy()
    if raw_target_source_col is not None:
        required_cols.append(raw_target_source_col)
    if log_target_source_col is not None and log_target_source_col not in required_cols:
        required_cols.append(log_target_source_col)

    model_df = df[required_cols].dropna().copy()

    if raw_target_source_col is None and log_target_source_col is not None:
        raw_target_col = base_target_name
        model_df[raw_target_col] = np.expm1(model_df[log_target_source_col].to_numpy())
    else:
        raw_target_col = raw_target_source_col

    if raw_target_col is None:
        raise ValueError("Could not resolve a raw target column for GAM comparison workflow.")

    if log_target_source_col is None:
        if (model_df[raw_target_col] < -1).any():
            raise ValueError("Cannot derive a log target because raw target contains values below -1.")
        log_target_col = f"{raw_target_col}_log"
        model_df[log_target_col] = np.log1p(model_df[raw_target_col].to_numpy())
    else:
        log_target_col = log_target_source_col

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
