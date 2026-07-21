"""Regenerate and check the numeric tables of ``Report/main.tex`` against artifacts.

The report states three tables whose values are not copied verbatim from a single
exported artifact:

* Table 2  (``tab:model_variant_selection``)   -- OOF metrics per model variant on the
  main MI+VIF feature set, including the Spearman rank correlation.
* Table 13 (``tab:appendix_nested_cv_stability``) -- outer-fold mean/sd per variant and
  feature set.
* Table 14 (``tab:appendix_regime_errors``)    -- OOF errors stratified by performance
  group for the two selected MI+VIF models.

This script recomputes them from the run artifacts under ``results/interpretable_model``
and compares the result with the values printed in the report. Run it after any change
to the underlying runs to confirm the report is still accurate::

    python scripts/verify_report_tables.py            # check, exit 1 on mismatch
    python scripts/verify_report_tables.py --show     # also print the regenerated tables

Table 13 is read straight from the exported ``nested_cv_optuna_summary_*.csv`` files;
Tables 2 and 14 are computed from the exported out-of-fold predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERPRETABLE_ROOT = REPO_ROOT / "results" / "interpretable_model"

MAIN_CONTEXT = "full_trainval_12ep_1seed_MI_correct"
VIF_ONLY_CONTEXT = "full_trainval_12ep_1seed_vif_only_no_collision"

# Variant label -> run directory relative to INTERPRETABLE_ROOT, with "{ctx}" as the
# context placeholder. The raw-target XGBoost fits live in separate quickfix exports.
VARIANT_DIRS = {
    "LinearGAM (raw)": "gam-linear/{ctx}",
    "LinearGAM (log)": "gam-linear-log/{ctx}",
    "GammaGAM": "gam-gamma/{ctx}",
    "XGBoost (raw)": "xgboost/{ctx}_xgboost_raw_target_quickfix",
    "XGBoost (log)": "xgboost/{ctx}",
}

# Performance-group boundaries on the original ml_ade scale, from k-means on
# log(1 + ml_ade); see Section 3.6 and Table 3 of the report. Table 14 is reconstructed
# by thresholding on these boundaries rather than by reading the stored k-means labels,
# so a trajectory sitting exactly on a boundary may land in a different group than in
# the group counts of Table 3. This does not affect any metric at three decimals.
EASY_MEDIUM_BOUNDARY = 0.388
MEDIUM_HARD_BOUNDARY = 1.198

# ---------------------------------------------------------------------------
# Expected values, transcribed from Report/main.tex
# ---------------------------------------------------------------------------

# Table 2: variant -> (RMSE, R2, MAE, Spearman)
EXPECTED_TABLE_2 = {
    "LinearGAM (raw)": (0.448, 0.492, 0.263, 0.779),
    "LinearGAM (log)": (0.445, 0.498, 0.250, 0.791),
    "GammaGAM": (0.445, 0.498, 0.253, 0.796),
    "XGBoost (raw)": (0.389, 0.617, 0.223, 0.827),
    "XGBoost (log)": (0.388, 0.619, 0.214, 0.835),
}

# Table 13: (context, variant) -> (rmse_mean, rmse_sd, mae_mean, mae_sd, r2_mean, r2_sd)
EXPECTED_TABLE_13 = {
    ("MI+VIF", "LinearGAM (raw)"): (0.448, 0.022, 0.263, 0.003, 0.491, 0.052),
    ("MI+VIF", "LinearGAM (log)"): (0.445, 0.017, 0.250, 0.004, 0.498, 0.035),
    ("MI+VIF", "GammaGAM"): (0.445, 0.010, 0.253, 0.002, 0.498, 0.007),
    ("MI+VIF", "XGBoost (raw)"): (0.389, 0.009, 0.223, 0.004, 0.617, 0.006),
    ("MI+VIF", "XGBoost (log)"): (0.388, 0.010, 0.214, 0.005, 0.619, 0.008),
    ("VIF only", "LinearGAM (raw)"): (0.440, 0.011, 0.268, 0.002, 0.515, 0.010),
    ("VIF only", "LinearGAM (log)"): (0.444, 0.009, 0.255, 0.003, 0.506, 0.007),
    ("VIF only", "GammaGAM"): (0.475, 0.013, 0.277, 0.004, 0.434, 0.019),
    ("VIF only", "XGBoost (raw)"): (0.386, 0.008, 0.220, 0.003, 0.626, 0.011),
    ("VIF only", "XGBoost (log)"): (0.387, 0.008, 0.213, 0.002, 0.624, 0.007),
}

# Table 14: (variant, group) -> (actual_mean, predicted_mean, MAE, RMSE, bias)
EXPECTED_TABLE_14 = {
    ("XGBoost (log)", "easy"): (0.177, 0.258, 0.109, 0.177, 0.081),
    ("XGBoost (log)", "medium"): (0.658, 0.622, 0.231, 0.319, -0.036),
    ("XGBoost (log)", "hard"): (2.028, 1.383, 0.773, 0.993, -0.645),
    ("LinearGAM (log)", "easy"): (0.177, 0.280, 0.138, 0.199, 0.103),
    ("LinearGAM (log)", "medium"): (0.658, 0.611, 0.241, 0.360, -0.048),
    ("LinearGAM (log)", "hard"): (2.028, 1.210, 0.915, 1.149, -0.818),
}

TOLERANCE = 0.0005  # values are reported to three decimals


def run_dir(variant: str, context: str) -> Path:
    return INTERPRETABLE_ROOT / VARIANT_DIRS[variant].format(ctx=context)


def load_oof(variant: str, context: str) -> tuple[pd.Series, pd.Series]:
    """Return (actual, predicted) out-of-fold values on the original ml_ade scale."""
    tables = run_dir(variant, context) / "tables"
    matches = sorted(tables.glob("model_data_with_oof_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one model_data_with_oof_*.csv in {tables}, found {len(matches)}"
        )
    df = pd.read_csv(matches[0])
    if "oof_pred_orig" in df.columns:
        # Log-target fits export predictions back-transformed to the original scale.
        return df["target_orig"], df["oof_pred_orig"]
    target_col = next(c for c in ("ml_ade", "ml_ade_log") if c in df.columns)
    return df[target_col], df["oof_pred"]


def compute_table_2() -> dict[str, tuple[float, float, float, float]]:
    out = {}
    for variant in VARIANT_DIRS:
        actual, pred = load_oof(variant, MAIN_CONTEXT)
        resid = pred - actual
        rmse = float(np.sqrt((resid**2).mean()))
        mae = float(resid.abs().mean())
        r2 = float(1 - (resid**2).sum() / ((actual - actual.mean()) ** 2).sum())
        rho = float(spearmanr(pred, actual).statistic)
        out[variant] = (rmse, r2, mae, rho)
    return out


def read_table_13() -> dict[tuple[str, str], tuple[float, ...]]:
    out = {}
    for label, context in (("MI+VIF", MAIN_CONTEXT), ("VIF only", VIF_ONLY_CONTEXT)):
        for variant in VARIANT_DIRS:
            tables = run_dir(variant, context) / "tables"
            matches = sorted(tables.glob("nested_cv_optuna_summary_*.csv"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"expected exactly one nested_cv_optuna_summary_*.csv in {tables}, "
                    f"found {len(matches)}"
                )
            summary = pd.read_csv(matches[0]).set_index("metric")
            out[(label, variant)] = tuple(
                float(summary.loc[metric, stat])
                for metric in ("outer_rmse", "outer_mae", "outer_r2")
                for stat in ("mean", "std")
            )
    return out


def compute_table_14() -> dict[tuple[str, str], tuple[float, ...]]:
    out = {}
    for variant in ("XGBoost (log)", "LinearGAM (log)"):
        actual, pred = load_oof(variant, MAIN_CONTEXT)
        groups = pd.cut(
            actual,
            [-np.inf, EASY_MEDIUM_BOUNDARY, MEDIUM_HARD_BOUNDARY, np.inf],
            labels=["easy", "medium", "hard"],
        )
        for group in ("easy", "medium", "hard"):
            mask = groups == group
            a, p = actual[mask], pred[mask]
            resid = p - a
            out[(variant, group)] = (
                float(a.mean()),
                float(p.mean()),
                float(resid.abs().mean()),
                float(np.sqrt((resid**2).mean())),
                float(resid.mean()),
            )
    return out


def check(name: str, computed: dict, expected: dict, columns: list[str]) -> list[str]:
    """Compare computed against expected values; return a list of mismatch messages."""
    failures = []
    missing = set(expected) - set(computed)
    for key in sorted(missing, key=str):
        failures.append(f"{name}: no computed value for {key}")
    for key, exp in expected.items():
        if key not in computed:
            continue
        for column, got, want in zip(columns, computed[key], exp):
            if abs(got - want) > TOLERANCE:
                failures.append(
                    f"{name}: {key} {column}: report says {want:.3f}, artifacts give {got:.3f}"
                )
    return failures


def show(title: str, values: dict, columns: list[str]) -> None:
    print(f"\n{title}")
    key_width = max(len(str(k)) for k in values)
    print("  " + " " * key_width + "  " + "".join(f"{c:>10}" for c in columns))
    for key in values:
        row = "".join(f"{v:>10.3f}" for v in values[key])
        print(f"  {str(key):<{key_width}}  {row}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show", action="store_true", help="print the regenerated tables in addition to checking"
    )
    args = parser.parse_args()

    table_2 = compute_table_2()
    table_13 = read_table_13()
    table_14 = compute_table_14()

    cols_2 = ["RMSE", "R2", "MAE", "Spearman"]
    cols_13 = ["RMSE", "RMSE sd", "MAE", "MAE sd", "R2", "R2 sd"]
    cols_14 = ["actual", "predicted", "MAE", "RMSE", "bias"]

    if args.show:
        show("Table 2  - OOF metrics per variant (MI+VIF)", table_2, cols_2)
        show("Table 13 - nested-CV outer-fold stability", table_13, cols_13)
        show("Table 14 - OOF errors by performance group", table_14, cols_14)

    failures = (
        check("Table 2", table_2, EXPECTED_TABLE_2, cols_2)
        + check("Table 13", table_13, EXPECTED_TABLE_13, cols_13)
        + check("Table 14", table_14, EXPECTED_TABLE_14, cols_14)
    )

    print()
    if failures:
        print(f"FAILED: {len(failures)} value(s) in Report/main.tex disagree with the artifacts")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    checked = sum(
        len(exp) * len(cols)
        for exp, cols in (
            (EXPECTED_TABLE_2, cols_2),
            (EXPECTED_TABLE_13, cols_13),
            (EXPECTED_TABLE_14, cols_14),
        )
    )
    print(f"OK: all {checked} values in Tables 2, 13 and 14 match the run artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
