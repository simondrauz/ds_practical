"""Conditional PDP analysis for feature-effect hypotheses.

This script evaluates whether the PDP effect of feature A changes across
quantile strata of feature B for finished XGBoost interpretable-model runs.

Example:
    conda run -n adaptive-py310 python src/data_modelling/conditional_pdp_hypothesis.py \
        --feature-a heading_change_per_sec \
        --feature-b std_speed
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans


DEFAULT_RUNS = [
    "full_trainval_12ep_1seed_MI_correct",
    "full_trainval_12ep_1seed_vif_only_no_collision",
    "sweep_large_30ep_1seed_MI_corrected",
]

FEATURE_ALIASES = {
    "heading_change_per_second": "heading_change_per_sec",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "results" / "interpretable_model" / "xgboost"
COMBINED_OUTPUT_DIR = MODEL_ROOT / "conditional_pdp_hypotheses"
PERFORMANCE_REGIME_ROOT = (
    REPO_ROOT / "results" / "interpretable_model" / "feature_effect_performance_regimes"
)
PERFORMANCE_GROUP_COL = "performance_group"


@dataclass(frozen=True)
class RunContext:
    run_name: str
    artifact_dir: Path
    plots_dir: Path
    tables_dir: Path
    target_col: str
    feature_cols: list[str]
    model_path: Path
    data_path: Path


def _slug(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _resolve_feature_name(name: str) -> str:
    return FEATURE_ALIASES.get(name, name)


def _read_manifest(run_name: str, target_col: str) -> RunContext:
    artifact_dir = MODEL_ROOT / run_name
    manifest_path = artifact_dir / "tables" / f"run_manifest_{target_col}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for {run_name}: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    target = manifest["target_col"]
    feature_cols = list(manifest["feature_cols"])

    model_path = artifact_dir / f"xgb_model_{target}.json"
    data_path = artifact_dir / "tables" / f"model_data_with_oof_{target}.csv"
    plots_dir = artifact_dir / "plots"
    tables_dir = artifact_dir / "tables"
    for path in [model_path, data_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact for {run_name}: {path}")

    return RunContext(
        run_name=run_name,
        artifact_dir=artifact_dir,
        plots_dir=plots_dir,
        tables_dir=tables_dir,
        target_col=target,
        feature_cols=feature_cols,
        model_path=model_path,
        data_path=data_path,
    )


def _validate_features(ctx: RunContext, feature_a: str, feature_b: str) -> None:
    missing = [feat for feat in [feature_a, feature_b] if feat not in ctx.feature_cols]
    if missing:
        available = ", ".join(ctx.feature_cols)
        raise ValueError(
            f"{ctx.run_name}: missing feature(s) {missing}. Available features: {available}"
        )


def _assign_performance_groups(
    metric_values: pd.Series,
    *,
    lower_is_better: bool = True,
    n_groups: int = 3,
    use_log_space: bool = True,
    random_state: int = 42,
) -> tuple[pd.Series, dict[str, object]]:
    """Match feature_effect_performance_regimes.ipynb group assignment."""
    if metric_values.isna().any():
        missing_count = int(metric_values.isna().sum())
        raise ValueError(f"Performance metric contains missing values: {missing_count}")

    if use_log_space:
        if (metric_values < -1).any():
            raise ValueError(
                "Cannot apply log1p transform: performance metric contains values < -1."
            )
        cluster_values = np.log1p(metric_values.to_numpy())
    else:
        cluster_values = metric_values.to_numpy()

    kmeans = KMeans(n_clusters=n_groups, random_state=random_state)
    cluster_ids = kmeans.fit_predict(cluster_values.reshape(-1, 1))
    sorted_centroid_indices = np.argsort(kmeans.cluster_centers_.ravel())

    rank_to_label = (
        {0: "easy", 1: "medium", 2: "hard"}
        if lower_is_better
        else {0: "hard", 1: "medium", 2: "easy"}
    )
    cluster_id_to_rank = {
        int(cluster_id): rank for rank, cluster_id in enumerate(sorted_centroid_indices)
    }
    labels = np.array([rank_to_label[cluster_id_to_rank[cid]] for cid in cluster_ids])

    sorted_centroids = kmeans.cluster_centers_.ravel()[sorted_centroid_indices]
    if use_log_space:
        sorted_centroids_raw = np.expm1(sorted_centroids)
        boundary_low_raw = float(np.expm1((sorted_centroids[0] + sorted_centroids[1]) / 2))
        boundary_high_raw = float(
            np.expm1((sorted_centroids[1] + sorted_centroids[2]) / 2)
        )
    else:
        sorted_centroids_raw = sorted_centroids
        boundary_low_raw = float((sorted_centroids_raw[0] + sorted_centroids_raw[1]) / 2)
        boundary_high_raw = float((sorted_centroids_raw[1] + sorted_centroids_raw[2]) / 2)

    if lower_is_better:
        group_info: dict[str, object] = {
            "centroid_easy": float(sorted_centroids_raw[0]),
            "centroid_medium": float(sorted_centroids_raw[1]),
            "centroid_hard": float(sorted_centroids_raw[2]),
            "boundary_easy_medium": boundary_low_raw,
            "boundary_medium_hard": boundary_high_raw,
        }
    else:
        group_info = {
            "centroid_easy": float(sorted_centroids_raw[2]),
            "centroid_medium": float(sorted_centroids_raw[1]),
            "centroid_hard": float(sorted_centroids_raw[0]),
            "boundary_easy_medium": boundary_high_raw,
            "boundary_medium_hard": boundary_low_raw,
        }
    group_info["use_log_space"] = use_log_space
    group_info["n_groups"] = n_groups
    return pd.Series(labels, index=metric_values.index, name=PERFORMANCE_GROUP_COL), group_info


def _find_regime_analysis_path(ctx: RunContext) -> Path | None:
    run_root = PERFORMANCE_REGIME_ROOT / "xgboost" / ctx.run_name / ctx.target_col
    candidates = sorted(run_root.rglob("tables/regime_analysis.csv"))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(
            f"{ctx.run_name}: found multiple regime_analysis.csv files; "
            f"using {candidates[-1]}"
        )
    return candidates[-1]


def _merge_exported_performance_groups(
    x_model: pd.DataFrame,
    ctx: RunContext,
) -> tuple[pd.Series, dict[str, object]]:
    regime_path = _find_regime_analysis_path(ctx)
    if regime_path is None:
        raise FileNotFoundError(f"{ctx.run_name}: no exported regime_analysis.csv found.")

    regime_df = pd.read_csv(regime_path)
    if PERFORMANCE_GROUP_COL not in regime_df.columns:
        raise KeyError(f"{regime_path} is missing {PERFORMANCE_GROUP_COL!r}.")

    key_candidates = [
        ["row_id"],
        ["run_name", "eval_csv_name", "data_idx"],
        ["eval_csv_name", "data_idx"],
        ["data_idx"],
    ]
    for key_cols in key_candidates:
        if all(col in x_model.columns and col in regime_df.columns for col in key_cols):
            regime_groups = regime_df[key_cols + [PERFORMANCE_GROUP_COL]].drop_duplicates(
                subset=key_cols
            )
            merged = x_model[key_cols].merge(
                regime_groups,
                on=key_cols,
                how="left",
                validate="many_to_one",
                sort=False,
            )
            if merged[PERFORMANCE_GROUP_COL].notna().all():
                return (
                    merged[PERFORMANCE_GROUP_COL],
                    {"source": "regime_analysis", "path": str(regime_path)},
                )

    raise ValueError(
        f"{ctx.run_name}: could not align exported performance groups from {regime_path}."
    )


def _resolve_performance_metric_col(df: pd.DataFrame, requested_col: str) -> str:
    if requested_col == "auto":
        for col in ["target_orig", "ml_ade", "ml_fde", "min_ade_5"]:
            if col in df.columns:
                return col
    elif requested_col in df.columns:
        return requested_col
    raise KeyError(
        f"Could not resolve performance metric column {requested_col!r}. "
        f"Available columns include: {', '.join(df.columns[:30])}"
    )


def _add_performance_groups(
    x_model: pd.DataFrame,
    ctx: RunContext,
    *,
    source: str,
    metric_col: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if source in {"auto", "regime_analysis"}:
        try:
            groups, group_info = _merge_exported_performance_groups(x_model, ctx)
            grouped = x_model.copy()
            grouped[PERFORMANCE_GROUP_COL] = groups.to_numpy()
            group_info = {**group_info, "method": "exported"}
            return grouped, group_info
        except (FileNotFoundError, ValueError, KeyError) as exc:
            if source == "regime_analysis":
                raise
            raise FileNotFoundError(
                f"{ctx.run_name}: performance-group analysis requires an exported "
                "feature_effect_performance_regimes regime_analysis.csv. "
                "This run does not appear to have one. Restrict --runs to runs with "
                "exported performance groups, or pass --performance-group-source "
                "recompute explicitly if you want to recreate groups from the metric."
            ) from exc

    resolved_metric_col = _resolve_performance_metric_col(x_model, metric_col)
    groups, group_info = _assign_performance_groups(
        x_model[resolved_metric_col],
        lower_is_better=True,
        random_state=random_state,
    )
    grouped = x_model.copy()
    grouped[PERFORMANCE_GROUP_COL] = groups
    group_info = {
        **group_info,
        "source": "recomputed",
        "method": "kmeans_log1p",
        "performance_metric_col": resolved_metric_col,
    }
    return grouped, group_info


def _make_strata(
    values: pd.Series,
    n_strata: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    strata = pd.qcut(values, q=n_strata, labels=False, duplicates="drop")
    _, edges = pd.qcut(values, q=n_strata, retbins=True, duplicates="drop")
    strata_arr = np.asarray(strata)
    labels = []
    actual_n = len(edges) - 1
    for idx in range(actual_n):
        if actual_n == 3:
            prefix = ["low", "medium", "high"][idx]
        elif idx == 0:
            prefix = "Q1 lowest"
        elif idx == actual_n - 1:
            prefix = f"Q{actual_n} highest"
        else:
            prefix = f"Q{idx + 1}"
        labels.append(prefix)
    return strata_arr, edges, labels


def _grid_from_support(
    x: pd.Series,
    strata: np.ndarray,
    n_strata: int,
    grid_resolution: int,
    percentile_low: float,
    percentile_high: float,
    grid_scope: str,
) -> np.ndarray:
    if grid_scope == "global":
        low, high = np.nanpercentile(x, [percentile_low, percentile_high])
    else:
        lows = []
        highs = []
        for idx in range(n_strata):
            x_stratum = x[strata == idx]
            if x_stratum.empty:
                continue
            lo, hi = np.nanpercentile(x_stratum, [percentile_low, percentile_high])
            lows.append(lo)
            highs.append(hi)
        low = max(lows)
        high = min(highs)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            low, high = np.nanpercentile(x, [percentile_low, percentile_high])

    if low == high:
        raise ValueError("Cannot create PDP grid because feature A has no variation.")
    return np.linspace(float(low), float(high), grid_resolution)


def _predict_conditional_pdp(
    model: xgb.XGBRegressor,
    x_stratum: pd.DataFrame,
    feature_a: str,
    grid: np.ndarray,
    max_rows: int,
    random_state: int,
) -> np.ndarray:
    if max_rows > 0 and len(x_stratum) > max_rows:
        x_eval = x_stratum.sample(n=max_rows, random_state=random_state)
    else:
        x_eval = x_stratum

    pdp = []
    for value in grid:
        x_eval_grid = x_eval.copy()
        x_eval_grid[feature_a] = value
        pred = model.predict(x_eval_grid)
        pdp.append(float(np.mean(pred)))
    return np.asarray(pdp, dtype=float)


def _center_curve(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "baseline":
        return y - y[0]
    if mode == "mean":
        return y - y.mean()
    if mode == "none":
        return y
    raise ValueError(f"Unknown centering mode: {mode}")


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan")
    return float(np.polyfit(x, y, deg=1)[0])


def _hypothesis_summary(summary_df: pd.DataFrame) -> dict[str, float | str | bool]:
    ordered = summary_df.sort_values("stratum_index")
    b_mid = ordered["feature_b_mid"].to_numpy(dtype=float)
    effect_delta = ordered["effect_delta"].to_numpy(dtype=float)
    slope = ordered["linear_slope"].to_numpy(dtype=float)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 2:
            return float("nan")
        if np.allclose(a[valid], a[valid][0]) or np.allclose(b[valid], b[valid][0]):
            return float("nan")
        return float(np.corrcoef(a[valid], b[valid])[0, 1])

    delta_diff = np.diff(effect_delta)
    slope_diff = np.diff(slope)
    return {
        "effect_delta_pearson_vs_feature_b_mid": corr(b_mid, effect_delta),
        "linear_slope_pearson_vs_feature_b_mid": corr(b_mid, slope),
        "effect_delta_monotone_non_decreasing": bool(np.all(delta_diff >= -1e-12)),
        "linear_slope_monotone_non_decreasing": bool(np.all(slope_diff >= -1e-12)),
        "lowest_stratum_effect_delta": float(effect_delta[0]),
        "highest_stratum_effect_delta": float(effect_delta[-1]),
        "highest_minus_lowest_effect_delta": float(effect_delta[-1] - effect_delta[0]),
        "lowest_stratum_linear_slope": float(slope[0]),
        "highest_stratum_linear_slope": float(slope[-1]),
        "highest_minus_lowest_linear_slope": float(slope[-1] - slope[0]),
    }


def _plot_run(
    ctx: RunContext,
    feature_a: str,
    feature_b: str,
    target_col: str,
    grid: np.ndarray,
    curves: list[dict[str, object]],
    center: str,
    performance_group: str | None,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    cmap = plt.get_cmap("coolwarm")
    denom = max(len(curves) - 1, 1)

    for idx, curve in enumerate(curves):
        color = cmap(idx / denom)
        label = (
            f"{curve['stratum_label']} [{curve['feature_b_low']:.3g}, "
            f"{curve['feature_b_high']:.3g}] (n={curve['n_obs']:,})"
        )
        ax.plot(grid, curve["effect"], color=color, linewidth=2.2, label=label)

    baseline = (
        0.0
        if center in {"baseline", "mean"}
        else float(np.mean([c["raw_pdp"][0] for c in curves]))
    )
    ax.axhline(baseline, color="black", linestyle="--", alpha=0.35, linewidth=0.9)
    ax.set_xlabel(feature_a)
    ax.set_ylabel(f"{center}-centered PDP on {target_col}")
    title_suffix = (
        f" within performance_group={performance_group}"
        if performance_group is not None
        else ""
    )
    ax.set_title(f"{ctx.run_name}\nEffect of {feature_a} by strata of {feature_b}{title_suffix}")
    ax.legend(title=feature_b, fontsize=8, title_fontsize=9, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_run(
    ctx: RunContext,
    feature_a: str,
    feature_b: str,
    n_strata: int,
    grid_resolution: int,
    grid_percentiles: tuple[float, float],
    grid_scope: str,
    center: str,
    max_rows_per_stratum: int,
    performance_group: str | None,
    performance_group_source: str,
    performance_metric_col: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    _validate_features(ctx, feature_a, feature_b)

    model = xgb.XGBRegressor()
    model.load_model(ctx.model_path)

    df = pd.read_csv(ctx.data_path)
    metadata_cols = [
        col
        for col in [
            "row_id",
            "run_name",
            "eval_csv_name",
            "data_idx",
            "target_orig",
            "ml_ade",
            "ml_fde",
            "min_ade_5",
        ]
        if col in df.columns
    ]
    x_model = df[ctx.feature_cols + metadata_cols].copy()
    for feat in [feature_a, feature_b]:
        if not pd.api.types.is_numeric_dtype(x_model[feat]):
            raise TypeError(
                f"{ctx.run_name}: {feat} must be numeric for quantile PDP analysis."
            )

    group_info: dict[str, object] = {}
    if performance_group is not None:
        x_model, group_info = _add_performance_groups(
            x_model,
            ctx,
            source=performance_group_source,
            metric_col=performance_metric_col,
            random_state=random_state,
        )
        available_groups = sorted(x_model[PERFORMANCE_GROUP_COL].dropna().unique())
        if performance_group not in available_groups:
            raise ValueError(
                f"{ctx.run_name}: performance group {performance_group!r} not found. "
                f"Available groups: {available_groups}"
            )
        x_model = x_model.loc[x_model[PERFORMANCE_GROUP_COL] == performance_group].copy()
        if len(x_model) < n_strata * 10:
            raise ValueError(
                f"{ctx.run_name}: only {len(x_model)} rows in performance group "
                f"{performance_group!r}; too few for {n_strata} strata."
            )

    x_features = x_model[ctx.feature_cols].copy()

    strata, edges, labels = _make_strata(x_features[feature_b], n_strata=n_strata)
    actual_n_strata = len(edges) - 1
    grid = _grid_from_support(
        x_features[feature_a],
        strata=strata,
        n_strata=actual_n_strata,
        grid_resolution=grid_resolution,
        percentile_low=grid_percentiles[0],
        percentile_high=grid_percentiles[1],
        grid_scope=grid_scope,
    )

    curves = []
    rows = []
    for idx in range(actual_n_strata):
        mask = strata == idx
        x_stratum = x_features.loc[mask]
        if len(x_stratum) < 10:
            continue

        raw_pdp = _predict_conditional_pdp(
            model=model,
            x_stratum=x_stratum,
            feature_a=feature_a,
            grid=grid,
            max_rows=max_rows_per_stratum,
            random_state=random_state,
        )
        effect = _center_curve(raw_pdp, center)
        feature_b_low = float(edges[idx])
        feature_b_high = float(edges[idx + 1])
        feature_b_mid = (feature_b_low + feature_b_high) / 2.0
        effect_delta = float(effect[-1] - effect[0])
        linear_slope = _linear_slope(grid, effect)
        multiplier_end_vs_start = (
            float(np.exp(raw_pdp[-1] - raw_pdp[0]))
            if ctx.target_col.endswith("_log")
            else float("nan")
        )

        row = {
            "run_name": ctx.run_name,
            "target_col": ctx.target_col,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "performance_group": performance_group or "all",
            "performance_group_source": group_info.get("source", "none"),
            "performance_group_method": group_info.get("method", "none"),
            "performance_metric_col": group_info.get("performance_metric_col", ""),
            "grid_scope": grid_scope,
            "center": center,
            "stratum_index": idx,
            "stratum_label": labels[idx],
            "feature_b_low": feature_b_low,
            "feature_b_high": feature_b_high,
            "feature_b_mid": feature_b_mid,
            "n_obs": int(len(x_stratum)),
            "feature_a_grid_min": float(grid[0]),
            "feature_a_grid_max": float(grid[-1]),
            "raw_pdp_start": float(raw_pdp[0]),
            "raw_pdp_end": float(raw_pdp[-1]),
            "effect_start": float(effect[0]),
            "effect_end": float(effect[-1]),
            "effect_delta": effect_delta,
            "linear_slope": linear_slope,
            "max_abs_effect": float(np.max(np.abs(effect))),
            "multiplier_end_vs_start": multiplier_end_vs_start,
        }
        rows.append(row)
        curves.append({**row, "raw_pdp": raw_pdp, "effect": effect})

    if not rows:
        raise ValueError(f"{ctx.run_name}: no strata had enough observations.")

    summary_df = pd.DataFrame(rows)
    hypothesis = {
        "run_name": ctx.run_name,
        "target_col": ctx.target_col,
        "feature_a": feature_a,
        "feature_b": feature_b,
        "performance_group": performance_group or "all",
        "performance_group_source": group_info.get("source", "none"),
        "performance_group_method": group_info.get("method", "none"),
        "performance_metric_col": group_info.get("performance_metric_col", ""),
        "n_group_rows": int(len(x_features)),
        "n_strata": len(summary_df),
        "grid_scope": grid_scope,
        "center": center,
        **_hypothesis_summary(summary_df),
    }
    hypothesis_df = pd.DataFrame([hypothesis])

    ctx.plots_dir.mkdir(parents=True, exist_ok=True)
    ctx.tables_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"conditional_pdp_{ctx.target_col}_{_slug(feature_a)}_by_"
        f"{_slug(feature_b)}_{len(summary_df)}q_{grid_scope}_{center}"
    )
    if performance_group is not None:
        stem = f"{stem}_group-{_slug(performance_group)}"
    plot_path = ctx.plots_dir / f"{stem}.png"
    summary_path = ctx.tables_dir / f"{stem}_summary.csv"
    hypothesis_path = ctx.tables_dir / f"{stem}_hypothesis.csv"

    _plot_run(
        ctx=ctx,
        feature_a=feature_a,
        feature_b=feature_b,
        target_col=ctx.target_col,
        grid=grid,
        curves=curves,
        center=center,
        performance_group=performance_group,
        out_path=plot_path,
    )
    summary_df.to_csv(summary_path, index=False)
    hypothesis_df.to_csv(hypothesis_path, index=False)

    return summary_df, hypothesis_df, plot_path


def _plot_combined(
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    run_names = list(summary_df["run_name"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(run_names),
        figsize=(6.2 * len(run_names), 5.0),
        squeeze=False,
        sharey=False,
    )
    for ax, run_name in zip(axes.flatten(), run_names):
        run_df = summary_df[summary_df["run_name"] == run_name].sort_values(
            "stratum_index"
        )
        line_delta = ax.plot(
            run_df["feature_b_mid"],
            run_df["effect_delta"],
            marker="o",
            linewidth=2,
            label="end-start effect",
            color="#1f77b4",
        )
        ax_slope = ax.twinx()
        line_slope = ax_slope.plot(
            run_df["feature_b_mid"],
            run_df["linear_slope"],
            marker="s",
            linewidth=2,
            label="linear slope",
            color="#ff7f0e",
        )
        feature_b = str(run_df["feature_b"].iloc[0])
        ax.set_title(run_name)
        ax.set_xlabel(f"{feature_b} stratum midpoint")
        ax.set_ylabel("End-start effect")
        ax_slope.set_ylabel("Linear slope")
        ax.grid(True, alpha=0.25)
        lines = line_delta + line_slope
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, fontsize=8, loc="best")
    fig.suptitle(
        "Conditional PDP hypothesis check: larger values indicate stronger "
        "feature-A effect",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether the PDP effect of feature A varies across quantile "
            "strata of feature B for finished XGBoost pipeline runs."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        help="Run directory names under results/interpretable_model/xgboost.",
    )
    parser.add_argument(
        "--feature-a",
        default="heading_change_per_sec",
        help="Focal feature whose effect is evaluated.",
    )
    parser.add_argument(
        "--feature-b",
        default="std_speed",
        help="Conditioning feature used to define quantile strata.",
    )
    parser.add_argument("--target-col", default="ml_ade_log")
    parser.add_argument("--n-strata", type=int, default=5)
    parser.add_argument("--grid-resolution", type=int, default=50)
    parser.add_argument(
        "--grid-percentiles",
        nargs=2,
        type=float,
        default=(5.0, 95.0),
        metavar=("LOW", "HIGH"),
        help="Percentile range for the feature-A grid.",
    )
    parser.add_argument(
        "--grid-scope",
        choices=["overlap", "global"],
        default="overlap",
        help=(
            "overlap restricts feature A to common within-stratum support; global "
            "uses the full-run percentile range."
        ),
    )
    parser.add_argument(
        "--center",
        choices=["baseline", "mean", "none"],
        default="baseline",
        help="How to center each conditional PDP curve before comparing effects.",
    )
    parser.add_argument(
        "--max-rows-per-stratum",
        type=int,
        default=0,
        help=(
            "Optional row cap per stratum for faster approximate PDPs. "
            "0 uses all rows."
        ),
    )
    parser.add_argument(
        "--performance-group",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Optional performance group to filter before computing conditional PDPs.",
    )
    parser.add_argument(
        "--performance-group-source",
        choices=["auto", "regime_analysis", "recompute"],
        default="auto",
        help=(
            "auto requires exported regime_analysis.csv. Use recompute explicitly "
            "to recreate the notebook's KMeans-on-log1p groups from the metric."
        ),
    )
    parser.add_argument(
        "--performance-metric-col",
        default="auto",
        help=(
            "Metric column for recomputed performance groups. 'auto' prefers "
            "target_orig, then ml_ade."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_a = _resolve_feature_name(args.feature_a)
    feature_b = _resolve_feature_name(args.feature_b)

    all_summaries = []
    all_hypotheses = []
    plot_paths = []
    for run_name in args.runs:
        try:
            ctx = _read_manifest(run_name, args.target_col)
            summary_df, hypothesis_df, plot_path = analyze_run(
                ctx=ctx,
                feature_a=feature_a,
                feature_b=feature_b,
                n_strata=args.n_strata,
                grid_resolution=args.grid_resolution,
                grid_percentiles=tuple(args.grid_percentiles),
                grid_scope=args.grid_scope,
                center=args.center,
                max_rows_per_stratum=args.max_rows_per_stratum,
                performance_group=args.performance_group,
                performance_group_source=args.performance_group_source,
                performance_metric_col=args.performance_metric_col,
                random_state=args.random_state,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        all_summaries.append(summary_df)
        all_hypotheses.append(hypothesis_df)
        plot_paths.append(plot_path)

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_hypothesis = pd.concat(all_hypotheses, ignore_index=True)

    COMBINED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = (
        f"conditional_pdp_{_slug(feature_a)}_by_{_slug(feature_b)}_"
        f"{args.n_strata}q_{args.grid_scope}_{args.center}"
    )
    if args.performance_group is not None:
        stem = f"{stem}_group-{_slug(args.performance_group)}"
    combined_summary_path = COMBINED_OUTPUT_DIR / f"{stem}_summary.csv"
    combined_hypothesis_path = COMBINED_OUTPUT_DIR / f"{stem}_hypothesis.csv"
    combined_plot_path = COMBINED_OUTPUT_DIR / f"{stem}_effect_summary.png"
    combined_summary.to_csv(combined_summary_path, index=False)
    combined_hypothesis.to_csv(combined_hypothesis_path, index=False)
    _plot_combined(combined_summary, combined_plot_path)

    print("Conditional PDP analysis complete.")
    print(f"Feature A: {feature_a}")
    print(f"Feature B: {feature_b}")
    if args.performance_group is not None:
        print(f"Performance group: {args.performance_group}")
    print("Run plots:")
    for path in plot_paths:
        print(f"- {path}")
    print(f"Combined summary:    {combined_summary_path}")
    print(f"Combined hypothesis: {combined_hypothesis_path}")
    print(f"Combined plot:       {combined_plot_path}")
    print()
    print("Hypothesis summary:")
    print(
        combined_hypothesis[
            [
                "run_name",
                "highest_minus_lowest_effect_delta",
                "effect_delta_pearson_vs_feature_b_mid",
                "effect_delta_monotone_non_decreasing",
                "highest_minus_lowest_linear_slope",
                "linear_slope_pearson_vs_feature_b_mid",
                "linear_slope_monotone_non_decreasing",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
