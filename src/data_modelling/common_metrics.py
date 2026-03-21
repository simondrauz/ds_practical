from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def is_log_target(*, target_col: str | None = None, target_mode: str | None = None) -> bool:
    """Infer whether values are stored on the log scale."""
    if target_mode is not None:
        if target_mode not in {"log", "raw"}:
            raise ValueError(f"Unsupported target_mode={target_mode!r}. Expected 'log' or 'raw'.")
        return target_mode == "log"

    if target_col is None:
        return False

    return target_col.endswith("_log")


def to_original_scale(
    values,
    *,
    target_col: str | None = None,
    target_mode: str | None = None,
) -> np.ndarray:
    values = np.asarray(values)
    if is_log_target(target_col=target_col, target_mode=target_mode):
        return np.expm1(values)
    return values


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true_orig, y_pred_orig, *, split_name: str | None = None) -> dict:
    metrics = {
        "R²": r2_score(y_true_orig, y_pred_orig),
        "MAE": mean_absolute_error(y_true_orig, y_pred_orig),
        "RMSE": rmse(y_true_orig, y_pred_orig),
    }
    if split_name is not None:
        metrics = {"Split": split_name, **metrics}
    return metrics
