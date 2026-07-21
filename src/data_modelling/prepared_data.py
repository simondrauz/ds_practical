from __future__ import annotations

"""Prepared-data helpers shared by the modelling notebooks.

Workflow overview:
1. Load the prepared CSV and optionally surface inspection tables.
2. Resolve which target column the modelling workflow should use.
3. Keep only numeric feature columns so downstream estimators receive valid inputs.
4. Drop incomplete rows once the feature/target contract is fixed.
"""

import json
from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np
import pandas as pd


DisplayFn = Callable[[Any], None]

IDENTITY_COLS = ["run_name", "eval_csv_name", "data_idx"]
MODEL_SETTING_COLS = ["attention_radius_m", "history_sec", "prediction_sec"]
INCLUDE_MODEL_SETTINGS_AS_FEATURES_KEY = "include_model_settings_as_features"


class SingleTargetModelData(TypedDict):
    target_col: str
    feature_cols: list[str]
    identity_cols: list[str]
    model_setting_cols: list[str]
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
    identity_cols: list[str]
    model_setting_cols: list[str]
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
    metadata_path = data_path.with_suffix(".metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if INCLUDE_MODEL_SETTINGS_AS_FEATURES_KEY in metadata:
            df.attrs[INCLUDE_MODEL_SETTINGS_AS_FEATURES_KEY] = metadata[
                INCLUDE_MODEL_SETTINGS_AS_FEATURES_KEY
            ]
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


def _available_identity_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in IDENTITY_COLS if col in df.columns]


def _available_model_setting_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in MODEL_SETTING_COLS if col in df.columns]


def _model_setting_feature_exclusions(
    df: pd.DataFrame,
    *,
    include_model_settings_as_features: bool | None,
) -> list[str]:
    model_setting_cols = _available_model_setting_cols(df)
    if not model_setting_cols:
        return []

    if include_model_settings_as_features is None:
        include_model_settings_as_features = df.attrs.get(
            INCLUDE_MODEL_SETTINGS_AS_FEATURES_KEY
        )

    if include_model_settings_as_features is None:
        raise ValueError(
            "Prepared data contains model-setting columns, but "
            "include_model_settings_as_features was not set. Set "
            "INCLUDE_MODEL_SETTINGS_AS_FEATURES to True or False in "
            "interpretable_model_data_preparation.ipynb and rerun the prepared-data export."
        )
    if not isinstance(include_model_settings_as_features, bool):
        raise TypeError(
            "include_model_settings_as_features must be True or False when model-setting "
            f"columns are present; got {include_model_settings_as_features!r}."
        )

    if include_model_settings_as_features:
        excluded_cols: list[str] = []
    else:
        excluded_cols = model_setting_cols

    if excluded_cols:
        print(f"Model settings excluded from features: {excluded_cols}")
    elif model_setting_cols:
        print(f"Model settings included as features: {model_setting_cols}")
    return excluded_cols


def prepare_single_target_model_data(
    df: pd.DataFrame,
    *,
    target_col: str | None = None,
    default_target: str = "ml_ade",
    include_model_settings_as_features: bool | None = None,
    retain_model_settings_as_metadata: bool = False,
) -> SingleTargetModelData:
    if (
        target_col is not None
        and target_col not in df.columns
        and target_col.endswith("_log")
    ):
        raw_target_col = target_col[:-4]
        if raw_target_col in df.columns:
            if (df[raw_target_col] < -1).any():
                raise AssertionError(
                    f"Cannot derive {target_col} because {raw_target_col} contains values < -1."
                )
            df = df.copy()
            df[target_col] = np.log1p(df[raw_target_col].to_numpy())

    # Step 1: resolve the target column that defines the modelling/evaluation scale.
    resolved_target_col = _resolve_single_target_col(df, target_col, default_target)
    target_variant_cols = {resolved_target_col}
    if resolved_target_col.endswith("_log"):
        target_variant_cols.add(resolved_target_col[:-4])
    else:
        target_variant_cols.add(f"{resolved_target_col}_log")

    # Step 2: retain only numeric predictors and keep row identity out of the feature matrix.
    identity_cols = _available_identity_cols(df)
    model_setting_cols = _available_model_setting_cols(df)
    excluded_model_setting_cols = _model_setting_feature_exclusions(
        df,
        include_model_settings_as_features=include_model_settings_as_features,
    )
    feature_cols = _filter_numeric_feature_cols(
        df,
        [
            c
            for c in df.columns
            if c not in target_variant_cols
            and c not in identity_cols
            and c not in excluded_model_setting_cols
        ],
    )
    metadata_cols = (
        [c for c in model_setting_cols if c not in feature_cols]
        if retain_model_settings_as_metadata
        else []
    )

    # Step 3: freeze the modelling frame and row ids after dropping incomplete rows.
    # Downstream notebooks rely on `row_ids` to map OOF predictions back to original rows.
    model_df = df[identity_cols + feature_cols + metadata_cols + [resolved_target_col]].dropna().copy()

    return {
        "target_col": resolved_target_col,
        "feature_cols": feature_cols,
        "identity_cols": identity_cols,
        "model_setting_cols": model_setting_cols,
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
    include_model_settings_as_features: bool | None = None,
    retain_model_settings_as_metadata: bool = False,
) -> DualTargetModelData:
    # Step 1: resolve the base target and discover whether raw/log variants already exist.
    base_target_name, raw_target_source_col, log_target_source_col = _resolve_dual_target_sources(
        df,
        target_col,
        default_target,
    )

    # Step 2: build a purely numeric feature set so GAM variants can share one matrix.
    identity_cols = _available_identity_cols(df)
    model_setting_cols = _available_model_setting_cols(df)
    excluded_cols = {c for c in [raw_target_source_col, log_target_source_col] if c is not None}
    excluded_model_setting_cols = _model_setting_feature_exclusions(
        df,
        include_model_settings_as_features=include_model_settings_as_features,
    )
    feature_cols = _filter_numeric_feature_cols(
        df,
        [
            c
            for c in df.columns
            if c not in excluded_cols
            and c not in identity_cols
            and c not in excluded_model_setting_cols
        ],
    )
    metadata_cols = (
        [c for c in model_setting_cols if c not in feature_cols]
        if retain_model_settings_as_metadata
        else []
    )

    # Step 3: drop incomplete rows only after the full feature/target contract is known.
    target_source_cols = [
        c for c in [raw_target_source_col, log_target_source_col] if c is not None
    ]
    model_df = df[identity_cols + feature_cols + metadata_cols + target_source_cols].dropna().copy()

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
        "identity_cols": identity_cols,
        "model_setting_cols": model_setting_cols,
        "model_df": model_df,
        "X": X,
        "y_raw": y_raw,
        "y_log": y_log,
        "row_ids": model_df.index.to_numpy(),
        "n_features": X.shape[1],
    }
