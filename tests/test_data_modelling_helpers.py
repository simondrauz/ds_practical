from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_modelling.common_metrics import is_log_target, regression_metrics, rmse, to_original_scale
from data_modelling.prepared_data import (
    load_prepared_data,
    prepare_dual_target_model_data,
    prepare_single_target_model_data,
)
from data_modelling.run_context import (
    format_exported_model_label,
    get_exported_model_info,
    load_run_context,
    resolve_manifest_path,
)
from data_modelling.training_outputs import (
    build_oof_frame,
    build_oof_metrics_df,
    build_run_manifest,
    summarize_nested_cv,
    write_manifest,
)


def test_is_log_target_prefers_explicit_target_mode():
    assert is_log_target(target_col="ml_ade", target_mode="log") is True
    assert is_log_target(target_col="ml_ade_log", target_mode="raw") is False


def test_to_original_scale_handles_mode_and_column_name():
    values = np.log1p(np.array([0.5, 1.5]))
    np.testing.assert_allclose(to_original_scale(values, target_col="ml_ade_log"), [0.5, 1.5])
    np.testing.assert_allclose(to_original_scale(values, target_mode="log"), [0.5, 1.5])
    np.testing.assert_allclose(to_original_scale([1.0, 2.0], target_col="ml_ade"), [1.0, 2.0])


def test_to_original_scale_rejects_invalid_target_mode():
    with pytest.raises(ValueError):
        to_original_scale([1.0], target_mode="unknown")


def test_regression_metrics_and_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.0])

    metrics = regression_metrics(y_true, y_pred, split_name="OOF")

    assert metrics["Split"] == "OOF"
    assert metrics["MAE"] == pytest.approx(0.5)
    assert metrics["RMSE"] == pytest.approx(rmse(y_true, y_pred))
    assert metrics["R²"] == pytest.approx(0.375)


def test_resolve_manifest_path_requires_explicit_target_when_multiple_exist(tmp_path, monkeypatch):
    repo_root = tmp_path
    manifest_dir = repo_root / "results" / "interpretable_model" / "gam" / "run_a" / "tables"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "run_manifest_ml_ade.json").write_text("{}")
    (manifest_dir / "run_manifest_ml_fde.json").write_text("{}")

    monkeypatch.setattr("data_modelling.run_context._repo_root", lambda: repo_root)

    with pytest.raises(ValueError):
        resolve_manifest_path("gam", "run_a")


def test_resolve_manifest_path_autodiscovers_single_manifest(tmp_path, monkeypatch):
    repo_root = tmp_path
    manifest_dir = repo_root / "results" / "interpretable_model" / "xgboost" / "run_a" / "tables"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "run_manifest_ml_ade_log.json"
    manifest_path.write_text("{}")

    monkeypatch.setattr("data_modelling.run_context._repo_root", lambda: repo_root)

    assert resolve_manifest_path("xgboost", "run_a") == manifest_path


def test_resolve_manifest_path_raises_for_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("data_modelling.run_context._repo_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError):
        resolve_manifest_path("gam", "missing_run")


def test_load_run_context_reads_optional_outputs(tmp_path, monkeypatch):
    repo_root = tmp_path
    tables_dir = repo_root / "results" / "interpretable_model" / "gam" / "run_a" / "tables"
    plots_dir = repo_root / "results" / "interpretable_model" / "gam" / "run_a" / "plots"
    tables_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    model_data_path = tables_dir / "model_data_with_oof_ml_ade_log.csv"
    metrics_path = tables_dir / "metrics_oof_ml_ade_log.csv"
    tuning_summary_path = tables_dir / "full_data_tuning_optuna_summary_ml_ade_log.json"

    pd.DataFrame({"speed": [1.0], "target_orig": [2.0], "oof_pred_orig": [1.9]}).to_csv(model_data_path, index=False)
    pd.DataFrame({"Split": ["OOF"], "RMSE": [0.1]}).to_csv(metrics_path, index=False)
    tuning_summary_path.write_text(json.dumps({"best_cv_score": 0.1}))

    manifest = {
        "model_id": "gam",
        "run_name": "run_a",
        "target_col": "ml_ade_log",
        "feature_cols": ["speed"],
        "plots_dir": str(plots_dir),
        "tables_dir": str(tables_dir),
        "nested_resampling": {
            "model_data_with_oof_path": str(model_data_path),
            "oof_metrics_path": str(metrics_path),
        },
        "final_model": {
            "model_path": str(tables_dir / "gam_model_ml_ade_log.pkl"),
            "full_data_tuning_summary_path": str(tuning_summary_path),
        },
        "analysis": {"poor_well_quantile": 0.2},
    }
    (tables_dir / "run_manifest_ml_ade_log.json").write_text(json.dumps(manifest))

    monkeypatch.setattr("data_modelling.run_context._repo_root", lambda: repo_root)

    ctx = load_run_context("gam", "run_a", "ml_ade_log")

    assert ctx.target_col == "ml_ade_log"
    assert ctx.feature_cols == ["speed"]
    assert list(ctx.model_df_oof.columns) == ["speed", "target_orig", "oof_pred_orig"]
    assert ctx.metrics_path == metrics_path
    assert ctx.oof_metrics_df is not None
    assert ctx.full_data_tuning_summary == {"best_cv_score": 0.1}


def test_get_exported_model_info_handles_gam_and_xgboost_fallbacks():
    gam_info = get_exported_model_info(
        {
            "model_id": "gam",
            "target_col": "ml_ade_log",
            "final_model": {
                "selected_variant_name": "LinearGAM (log)",
                "selected_variant_model_kind": "linear",
                "selected_variant_target_mode": "log",
                "selected_cv_rmse": 0.123,
            },
        }
    )
    xgb_info = get_exported_model_info(
        {
            "model_id": "xgboost",
            "target_col": "ml_ade_log",
            "final_model": {
                "best_cv_score": 0.456,
            },
        }
    )

    assert gam_info == {
        "name": "LinearGAM (log)",
        "kind": "linear",
        "target_mode": "log",
        "selection_metric_name": "lowest_cv_rmse",
        "selection_metric_value": 0.123,
    }
    assert xgb_info == {
        "name": "XGBoost",
        "kind": "xgboost",
        "target_mode": "log",
        "selection_metric_name": "best_cv_score",
        "selection_metric_value": 0.456,
    }
    assert format_exported_model_label(gam_info) == "LinearGAM (log) (linear, target_mode=log)"


def test_summarize_nested_cv_preserves_expected_metrics():
    nested_cv_df = pd.DataFrame(
        {
            "outer_rmse": [1.0, 2.0, 3.0],
            "outer_mae": [0.5, 1.0, 1.5],
            "outer_r2": [0.1, 0.2, 0.3],
        }
    )

    summary = summarize_nested_cv(nested_cv_df)

    assert summary["metric"].tolist() == ["outer_rmse", "outer_mae", "outer_r2"]
    assert summary.loc[0, "mean"] == pytest.approx(2.0)
    assert summary.loc[1, "mean"] == pytest.approx(1.0)


def test_build_oof_frame_adds_shared_output_columns():
    model_df = pd.DataFrame({"feature_a": [10, 20], "target": [0.1, 0.2]})
    frame = build_oof_frame(
        model_df,
        row_ids=np.array([5, 6]),
        oof_pred=np.log1p(np.array([0.5, 0.7])),
        oof_fold=np.array([1, 2]),
        target_orig=np.array([0.4, 0.8]),
        pred_scale_kwargs={"target_mode": "log"},
    )

    assert frame["row_id"].tolist() == [5, 6]
    assert frame["outer_fold"].tolist() == [1, 2]
    np.testing.assert_allclose(frame["oof_pred_orig"], [0.5, 0.7])
    np.testing.assert_allclose(frame["target_orig"], [0.4, 0.8])


def test_build_oof_metrics_df_returns_expected_columns():
    frame = build_oof_metrics_df(
        np.log1p(np.array([1.0, 2.0])),
        np.log1p(np.array([1.5, 2.5])),
        target_mode="log",
    )

    assert frame.columns.tolist() == ["Split", "R²", "MAE", "RMSE"]
    assert frame.loc[0, "Split"] == "OOF"
    assert frame.loc[0, "MAE"] == pytest.approx(0.5)


def test_build_run_manifest_preserves_schema_and_write_manifest(tmp_path):
    manifest = build_run_manifest(
        model_id="xgboost",
        run_name="run_a",
        target_col="ml_ade_log",
        feature_cols=["speed", "heading"],
        save_dir=tmp_path / "artifacts",
        plots_dir=tmp_path / "plots",
        tables_dir=tmp_path / "tables",
        nested_resampling={"model_data_with_oof_path": "data.csv"},
        final_model={"model_path": "model.json", "best_iteration": 10},
        analysis={"poor_well_quantile": 0.2},
        extra_manifest_fields={"raw_target_col": "ml_ade"},
    )

    assert manifest["model_id"] == "xgboost"
    assert manifest["target_col"] == "ml_ade_log"
    assert manifest["feature_cols"] == ["speed", "heading"]
    assert manifest["raw_target_col"] == "ml_ade"
    assert manifest["analysis"]["poor_well_quantile"] == 0.2

    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    manifest_path = write_manifest(manifest, tables_dir, "ml_ade_log")

    assert manifest_path.name == "run_manifest_ml_ade_log.json"
    assert json.loads(manifest_path.read_text())["final_model"]["best_iteration"] == 10


def test_load_prepared_data_includes_optional_summaries(tmp_path):
    data_path = tmp_path / "prepared.csv"
    pd.DataFrame({"a": [1.0], "b": [2.0]}).to_csv(data_path, index=False)
    displayed = []

    df = load_prepared_data(
        data_path,
        display_fn=displayed.append,
        include_missing_summary=True,
        include_dtype_summary=True,
    )

    assert df.columns.tolist() == ["a", "b"]
    assert len(displayed) == 3


def test_prepare_single_target_model_data_filters_non_numeric_and_resolves_target():
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": ["x", "y"],
            "ml_ade_log": [0.1, 0.2],
        }
    )

    prepared = prepare_single_target_model_data(df)

    assert prepared["target_col"] == "ml_ade_log"
    assert prepared["feature_cols"] == ["feature_a"]
    assert prepared["model_df"].columns.tolist() == ["feature_a", "ml_ade_log"]


def test_prepare_dual_target_model_data_derives_missing_log_target():
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": ["x", "y"],
            "ml_ade": [1.0, 3.0],
        }
    )

    prepared = prepare_dual_target_model_data(df)

    assert prepared["raw_target_col"] == "ml_ade"
    assert prepared["log_target_col"] == "ml_ade_log"
    assert prepared["target_col"] == "ml_ade_log"
    assert prepared["feature_cols"] == ["feature_a"]
    np.testing.assert_allclose(prepared["y_log"], np.log1p([1.0, 3.0]))
