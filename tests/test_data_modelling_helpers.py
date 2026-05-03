from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_modelling.feature_effect_cluster_exports as feature_effect_cluster_exports
import data_modelling.feature_effect_pr_cluster_inspection as feature_effect_pr_cluster_inspection
from data_modelling.common_metrics import is_log_target, regression_metrics, rmse, to_original_scale
import data_modelling.feature_effect_performance_regimes_utils as feature_effect_performance_regimes_utils
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


def _patch_feature_effect_regime_dependencies(monkeypatch: pytest.MonkeyPatch, umap_call_log: list[dict[str, float]]) -> None:
    class DummyUMAP:
        def __init__(self, *, n_components: int, n_neighbors: int, min_dist: float, random_state: int):
            umap_call_log.append(
                {
                    "n_components": int(n_components),
                    "n_neighbors": int(n_neighbors),
                    "min_dist": float(min_dist),
                    "random_state": int(random_state),
                }
            )
            self.n_components = int(n_components)
            self.n_neighbors = int(n_neighbors)
            self.min_dist = float(min_dist)

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            base = np.arange(len(X) * self.n_components, dtype=float).reshape(len(X), self.n_components)
            return base + float(self.n_neighbors) + self.min_dist

    class DummyHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_predict(self, X: np.ndarray) -> np.ndarray:
            midpoint = max(1, len(X) // 2)
            return np.array([0 if idx < midpoint else 1 for idx in range(len(X))], dtype=int)

    class DummyOPTICS:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_predict(self, X: np.ndarray) -> np.ndarray:
            midpoint = max(1, len(X) // 2)
            return np.array([0 if idx < midpoint else 1 for idx in range(len(X))], dtype=int)

    def dummy_trustworthiness(X: np.ndarray, embedding: np.ndarray, *, n_neighbors: int) -> float:
        return float(n_neighbors) + (float(embedding.shape[1]) / 100.0)

    def dummy_validity_index(X: np.ndarray, labels: np.ndarray) -> float:
        return 0.42

    dummy_umap_module = type("DummyUMAPModule", (), {"UMAP": DummyUMAP})
    dummy_hdbscan_module = type("DummyHDBSCANModule", (), {"HDBSCAN": DummyHDBSCAN})
    monkeypatch.setattr(
        feature_effect_performance_regimes_utils,
        "_require_step2_dependencies",
        lambda: (dummy_hdbscan_module, dummy_validity_index, dummy_umap_module, DummyOPTICS, dummy_trustworthiness),
    )


def _resolved_feature_effect_regime_cluster_spec(raw_cluster_spec: dict[str, object]) -> dict[str, object]:
    return feature_effect_performance_regimes_utils.resolve_cluster_spec(
        raw_cluster_spec,
        effect_cols=["effect__speed", "effect__heading", "effect__distance", "effect__distance_to_goal"],
    )


def _resolved_feature_effect_regime_inspection_config(
    cluster_spec: dict[str, object],
    *,
    inspection_algorithm: str = "hdbscan",
    inspection_cluster_space: str = "raw",
    inspection_top_k_features: int = 8,
    inspection_top_k_table: int = 3,
    sort_cluster_profiles_by: str = "cluster_size",
) -> dict[str, object]:
    return feature_effect_performance_regimes_utils.resolve_inspection_config(
        {
            "inspection_algorithm": inspection_algorithm,
            "inspection_cluster_space": inspection_cluster_space,
            "inspection_top_k_features": inspection_top_k_features,
            "inspection_top_k_table": inspection_top_k_table,
            "sort_cluster_profiles_by": sort_cluster_profiles_by,
        },
        cluster_spec=cluster_spec,
    )


def _sample_cluster_export_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    clustered_df = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4],
            "data_idx": [10, 11, 12, 13, 14],
            "outer_fold": [0, 0, 0, 0, 0],
            "performance_group": ["easy"] * 5,
            "ml_ade": [0.11, 0.12, 0.31, 0.32, 0.9],
            "target_orig": [0.11, 0.12, 0.31, 0.32, 0.9],
            "oof_pred_orig": [0.10, 0.14, 0.33, 0.30, 0.85],
            "scene_id": ["scene_a", "scene_a", "scene_a", "scene_b", "scene_b"],
            "scene_path": [
                "/tmp/scene_a/frame.json",
                "/tmp/scene_a/frame.json",
                "/tmp/scene_a/frame.json",
                "/tmp/scene_b/frame.json",
                "/tmp/scene_b/frame.json",
            ],
            "scene_ts": [0, 0, 1, 0, 0],
            "agent_type": ["pedestrian"] * 5,
            "speed": [1.0, 1.1, 2.2, 2.3, 0.4],
            "acceleration": [0.2, 0.25, 0.4, 0.35, 0.05],
            "scene_density": [3, 3, 4, 2, 2],
            "scene_weather": ["sunny", "sunny", "rain", "rain", "rain"],
            "effect__speed": [0.8, 0.6, -0.4, -0.5, 0.1],
            "effect__acceleration": [0.2, 0.1, -0.3, -0.2, 0.05],
            "cluster_hdbscan_raw": [0, 0, 1, 1, -1],
        }
    )
    cluster_scores_df = pd.DataFrame(
        [
            {
                "score_row_id": 0,
                "performance_group": "easy",
                "algorithm": "hdbscan",
                "cluster_space": "raw",
                "candidate_label_col": "cluster_hdbscan_raw",
                "input_dim": 2,
                "group_size": 5,
                "min_cluster_size": 2,
                "min_samples": 2,
                "optics_xi": np.nan,
                "umap_selected_n_components": np.nan,
                "n_clusters": 2,
                "noise_count": 1,
                "noise_fraction": 0.2,
                "clustered_fraction": 0.8,
                "dbcv": 0.42,
                "dbcv_cluster_space": 0.42,
                "dbcv_raw_effect_space": 0.42,
                "valid_for_selection": True,
                "valid_for_raw_effect_evaluation": True,
                "selected_for_group": True,
            }
        ]
    )
    return clustered_df, cluster_scores_df


def _sample_feature_effect_global_ranking_df(*, reverse_order: bool = False) -> pd.DataFrame:
    feature_order = ["acceleration", "speed"] if reverse_order else ["speed", "acceleration"]
    importance_values = [0.2, 0.7] if reverse_order else [0.7, 0.2]
    return pd.DataFrame(
        {
            "feature": feature_order,
            "global_rank": [1, 2],
            "importance_metric": ["mean_abs_shap", "mean_abs_shap"],
            "importance_value": importance_values,
            "importance_ascending": [False, False],
            "mean_abs_shap": importance_values,
        }
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


def test_get_exported_model_info_prefers_explicit_export_fields():
    manifest = {
        "model_id": "gam",
        "target_col": "ml_ade_log",
        "variant_name": "Fallback Variant",
        "selection_metric_name": "mean_outer_rmse",
        "selection_metric_value": 0.4,
        "final_model": {
            "exported_model_name": "Published GAM",
            "exported_model_kind": "linear",
            "exported_model_target_mode": "raw",
            "exported_model_selection_metric_name": "published_metric",
            "exported_model_selection_metric_value": 0.2,
            "selected_variant_name": "Ignored Variant",
            "selected_variant_model_kind": "gamma",
        },
    }

    assert get_exported_model_info(manifest) == {
        "name": "Published GAM",
        "kind": "linear",
        "target_mode": "raw",
        "selection_metric_name": "published_metric",
        "selection_metric_value": 0.2,
    }


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


def test_summarize_nested_cv_requires_shared_metric_columns():
    nested_cv_df = pd.DataFrame(
        {
            "outer_rmse": [1.0],
            "outer_mae": [0.5],
        }
    )

    with pytest.raises(KeyError, match="outer_r2"):
        summarize_nested_cv(nested_cv_df)


def test_build_oof_frame_adds_shared_output_columns():
    model_df = pd.DataFrame({"data_idx": [42, 17], "feature_a": [10, 20], "target": [0.1, 0.2]})
    frame = build_oof_frame(
        model_df,
        row_ids=np.array([5, 6]),
        oof_pred=np.log1p(np.array([0.5, 0.7])),
        oof_fold=np.array([1, 2]),
        target_orig=np.array([0.4, 0.8]),
        pred_scale_kwargs={"target_mode": "log"},
    )

    assert frame["data_idx"].tolist() == [42, 17]
    assert frame["row_id"].tolist() == [5, 6]
    assert frame["outer_fold"].tolist() == [1, 2]
    np.testing.assert_allclose(frame["oof_pred_orig"], [0.5, 0.7])
    np.testing.assert_allclose(frame["target_orig"], [0.4, 0.8])


def test_build_oof_frame_rejects_misaligned_arrays():
    model_df = pd.DataFrame({"feature_a": [10, 20], "target": [0.1, 0.2]})

    with pytest.raises(ValueError, match="row_ids"):
        build_oof_frame(
            model_df,
            row_ids=np.array([5]),
            oof_pred=np.log1p(np.array([0.5, 0.7])),
            oof_fold=np.array([1, 2]),
            target_orig=np.array([0.4, 0.8]),
            pred_scale_kwargs={"target_mode": "log"},
        )


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


def test_prepare_single_target_model_data_falls_back_to_last_column():
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "custom_target": [0.1, 0.2],
        }
    )

    prepared = prepare_single_target_model_data(df, default_target="missing_target")

    assert prepared["target_col"] == "custom_target"
    assert prepared["feature_cols"] == ["feature_a"]


def test_prepare_single_target_model_data_filters_non_numeric_and_resolves_target():
    df = pd.DataFrame(
        {
            "data_idx": [42, 17],
            "feature_a": [1.0, 2.0],
            "feature_b": ["x", "y"],
            "ml_ade_log": [0.1, 0.2],
        }
    )

    prepared = prepare_single_target_model_data(df)

    assert prepared["target_col"] == "ml_ade_log"
    assert prepared["feature_cols"] == ["feature_a"]
    assert prepared["identity_cols"] == ["data_idx"]
    assert prepared["model_df"].columns.tolist() == ["data_idx", "feature_a", "ml_ade_log"]


def test_prepare_single_target_model_data_derives_requested_log_target():
    df = pd.DataFrame(
        {
            "data_idx": [42, 17],
            "feature_a": [1.0, 2.0],
            "ml_ade": [1.0, 3.0],
        }
    )

    prepared = prepare_single_target_model_data(df, target_col="ml_ade_log")

    assert prepared["target_col"] == "ml_ade_log"
    assert prepared["feature_cols"] == ["feature_a"]
    np.testing.assert_allclose(
        prepared["model_df"]["ml_ade_log"].to_numpy(),
        np.log1p([1.0, 3.0]),
    )
    assert "ml_ade_log" not in df.columns


def test_prepare_dual_target_model_data_derives_missing_raw_target():
    df = pd.DataFrame(
        {
            "run_name": ["run_a", "run_a"],
            "data_idx": [42, 17],
            "feature_a": [1.0, 2.0],
            "ml_ade_log": np.log1p([1.0, 3.0]),
        }
    )

    prepared = prepare_dual_target_model_data(df)

    assert prepared["raw_target_col"] == "ml_ade"
    assert prepared["identity_cols"] == ["run_name", "data_idx"]
    assert prepared["feature_cols"] == ["feature_a"]
    assert prepared["model_df"][["run_name", "data_idx"]].to_dict("list") == {
        "run_name": ["run_a", "run_a"],
        "data_idx": [42, 17],
    }
    np.testing.assert_allclose(prepared["y_raw"], [1.0, 3.0])


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


def test_load_run_context_requires_manifest_fields(tmp_path, monkeypatch):
    repo_root = tmp_path
    tables_dir = repo_root / "results" / "interpretable_model" / "gam" / "run_a" / "tables"
    tables_dir.mkdir(parents=True)
    model_data_path = tables_dir / "model_data_with_oof_ml_ade_log.csv"
    pd.DataFrame({"speed": [1.0], "target_orig": [2.0], "oof_pred_orig": [1.9]}).to_csv(model_data_path, index=False)

    manifest = {
        "model_id": "gam",
        "run_name": "run_a",
        "target_col": "ml_ade_log",
        "plots_dir": str(repo_root / "results" / "interpretable_model" / "gam" / "run_a" / "plots"),
        "nested_resampling": {
            "model_data_with_oof_path": str(model_data_path),
        },
        "final_model": {},
        "analysis": {},
    }
    (tables_dir / "run_manifest_ml_ade_log.json").write_text(json.dumps(manifest))

    monkeypatch.setattr("data_modelling.run_context._repo_root", lambda: repo_root)

    with pytest.raises(KeyError, match="feature_cols"):
        load_run_context("gam", "run_a", "ml_ade_log")


def test_prepare_feature_effect_export_names_effect_columns_and_base_value():
    model_df_oof = pd.DataFrame(
        {
            "run_name": ["run_a", "run_a"],
            "data_idx": [42, 17],
            "speed": [1.0, 2.0],
            "acceleration": [0.2, 0.4],
            "row_id": [0, 1],
            "outer_fold": [1, 2],
            "oof_pred_orig": [0.5, 0.7],
            "target_orig": [0.4, 0.8],
        }
    )

    export_df = feature_effect_performance_regimes_utils.prepare_feature_effect_export(
        model_df_oof=model_df_oof,
        feature_cols=["speed", "acceleration"],
        effect_values=np.array([[0.3, -0.1], [0.4, -0.2]], dtype=float),
        base_values=np.array([1.0, 1.0], dtype=float),
    )

    assert export_df.columns.tolist() == [
        "run_name",
        "data_idx",
        "speed",
        "acceleration",
        "row_id",
        "outer_fold",
        "oof_pred_orig",
        "target_orig",
        "effect__speed",
        "effect__acceleration",
        "effect_base_value",
    ]
    assert export_df[["run_name", "data_idx"]].to_dict("list") == {
        "run_name": ["run_a", "run_a"],
        "data_idx": [42, 17],
    }


def test_assemble_step1_analysis_table_aligns_metrics_by_data_idx_not_row_position():
    prepared_model_df = pd.DataFrame(
        {
            "data_idx": [20, 10],
            "speed": [2.0, 1.0],
            "ml_ade_log": np.log1p([20.0, 10.0]),
        }
    )
    joined_metrics_df = pd.DataFrame(
        {
            "data_idx": [10, 20],
            "speed": [1.0, 2.0],
            "ml_ade": [10.0, 20.0],
            "scene_id": ["scene_10", "scene_20"],
        }
    )
    feature_effects_df = pd.DataFrame(
        {
            "data_idx": [20, 10],
            "speed": [2.0, 1.0],
            "row_id": [0, 1],
            "outer_fold": [1, 1],
            "oof_pred_orig": [19.5, 9.5],
            "target_orig": [20.0, 10.0],
            "effect__speed": [0.2, 0.1],
        }
    )

    analysis_df, group_summary_df = feature_effect_performance_regimes_utils.assemble_step1_analysis_table(
        prepared_model_df=prepared_model_df,
        joined_metrics_df=joined_metrics_df,
        feature_effects_df=feature_effects_df,
        feature_cols=["speed"],
        target_col="ml_ade_log",
        performance_metric_col="ml_ade",
    )

    assert analysis_df["data_idx"].tolist() == [20, 10]
    assert analysis_df["ml_ade"].tolist() == [20.0, 10.0]
    assert analysis_df["scene_id"].tolist() == ["scene_20", "scene_10"]
    assert analysis_df["effect__speed"].tolist() == [0.2, 0.1]
    assert set(analysis_df["performance_group"]).issubset({"easy", "medium", "hard"})
    assert group_summary_df.loc[0, "n_total"] == 2


def test_assemble_step1_analysis_table_uses_run_name_with_duplicate_data_idx():
    prepared_model_df = pd.DataFrame(
        {
            "run_name": ["run_b", "run_a"],
            "data_idx": [0, 0],
            "speed": [2.0, 1.0],
            "ml_ade_log": np.log1p([20.0, 10.0]),
        }
    )
    joined_metrics_df = pd.DataFrame(
        {
            "run_name": ["run_a", "run_b"],
            "data_idx": [0, 0],
            "speed": [1.0, 2.0],
            "ml_ade": [10.0, 20.0],
        }
    )
    feature_effects_df = pd.DataFrame(
        {
            "run_name": ["run_b", "run_a"],
            "data_idx": [0, 0],
            "speed": [2.0, 1.0],
            "row_id": [0, 1],
            "outer_fold": [1, 1],
            "oof_pred_orig": [19.5, 9.5],
            "target_orig": [20.0, 10.0],
            "effect__speed": [0.2, 0.1],
        }
    )

    analysis_df, _ = feature_effect_performance_regimes_utils.assemble_step1_analysis_table(
        prepared_model_df=prepared_model_df,
        joined_metrics_df=joined_metrics_df,
        feature_effects_df=feature_effects_df,
        feature_cols=["speed"],
        target_col="ml_ade_log",
        performance_metric_col="ml_ade",
    )

    assert analysis_df[["run_name", "data_idx", "ml_ade"]].to_dict("records") == [
        {"run_name": "run_b", "data_idx": 0, "ml_ade": 20.0},
        {"run_name": "run_a", "data_idx": 0, "ml_ade": 10.0},
    ]


def test_assemble_step1_analysis_table_drops_non_key_identity_overlaps():
    prepared_model_df = pd.DataFrame(
        {
            "run_name": ["run_a", "run_a"],
            "eval_csv_name": ["eval_epoch_1.csv", "eval_epoch_1.csv"],
            "data_idx": [0, 1],
            "speed": [1.0, 2.0],
            "ml_ade_log": np.log1p([10.0, 20.0]),
        }
    )
    joined_metrics_df = pd.DataFrame(
        {
            "data_idx": [0, 1],
            "speed": [1.0, 2.0],
            "ml_ade": [10.0, 20.0],
        }
    )
    feature_effects_df = pd.DataFrame(
        {
            "run_name": ["run_a", "run_a"],
            "eval_csv_name": ["eval_epoch_1.csv", "eval_epoch_1.csv"],
            "data_idx": [0, 1],
            "speed": [1.0, 2.0],
            "row_id": [0, 1],
            "outer_fold": [1, 2],
            "oof_pred_orig": [9.5, 19.5],
            "target_orig": [10.0, 20.0],
            "effect__speed": [0.1, 0.2],
        }
    )

    analysis_df, _ = feature_effect_performance_regimes_utils.assemble_step1_analysis_table(
        prepared_model_df=prepared_model_df,
        joined_metrics_df=joined_metrics_df,
        feature_effects_df=feature_effects_df,
        feature_cols=["speed"],
        target_col="ml_ade_log",
        performance_metric_col="ml_ade",
    )

    assert analysis_df[["run_name", "eval_csv_name", "data_idx"]].to_dict("records") == [
        {"run_name": "run_a", "eval_csv_name": "eval_epoch_1.csv", "data_idx": 0},
        {"run_name": "run_a", "eval_csv_name": "eval_epoch_1.csv", "data_idx": 1},
    ]
    assert analysis_df["effect__speed"].tolist() == [0.1, 0.2]


def test_assemble_step1_analysis_table_rejects_legacy_prepared_data_without_data_idx():
    prepared_model_df = pd.DataFrame(
        {
            "speed": [2.0, 1.0],
            "ml_ade_log": np.log1p([20.0, 10.0]),
        }
    )
    joined_metrics_df = pd.DataFrame(
        {
            "data_idx": [10, 20],
            "speed": [1.0, 2.0],
            "ml_ade": [10.0, 20.0],
        }
    )
    feature_effects_df = pd.DataFrame(
        {
            "speed": [2.0, 1.0],
            "row_id": [0, 1],
            "outer_fold": [1, 1],
            "oof_pred_orig": [19.5, 9.5],
            "target_orig": [20.0, 10.0],
            "effect__speed": [0.2, 0.1],
        }
    )

    with pytest.raises(ValueError, match="missing 'data_idx'"):
        feature_effect_performance_regimes_utils.assemble_step1_analysis_table(
            prepared_model_df=prepared_model_df,
            joined_metrics_df=joined_metrics_df,
            feature_effects_df=feature_effects_df,
            feature_cols=["speed"],
            target_col="ml_ade_log",
            performance_metric_col="ml_ade",
        )


def test_build_feature_effect_importance_table_orders_xgboost_and_gam():
    xgboost_df = feature_effect_performance_regimes_utils.build_feature_effect_importance_table(
        model_id="xgboost",
        feature_cols=["speed", "acceleration"],
        effect_values=np.array([[0.8, 0.1], [0.6, 0.2], [0.4, 0.3]], dtype=float),
    )
    assert xgboost_df["feature"].tolist() == ["speed", "acceleration"]
    assert xgboost_df["global_rank"].tolist() == [1, 2]
    assert xgboost_df["importance_metric"].tolist() == ["mean_abs_shap", "mean_abs_shap"]
    assert not xgboost_df["importance_ascending"].any()

    gam_df = feature_effect_performance_regimes_utils.build_feature_effect_importance_table(
        model_id="gam",
        feature_cols=["speed", "acceleration"],
        p_values=np.array([0.20, 0.01], dtype=float),
    )
    assert gam_df["feature"].tolist() == ["acceleration", "speed"]
    assert gam_df["global_rank"].tolist() == [1, 2]
    assert gam_df["importance_metric"].tolist() == ["p_value", "p_value"]
    assert gam_df["importance_ascending"].all()


def test_compute_gam_feature_effects_reconstructs_link_predictor():
    class DummyTerms:
        def __init__(self):
            self._terms = [
                type("SplineTerm", (), {"isintercept": False, "feature": 0})(),
                type("SplineTerm", (), {"isintercept": False, "feature": 1})(),
                type("Intercept", (), {"isintercept": True, "feature": None})(),
            ]

        def __len__(self):
            return len(self._terms)

        def __getitem__(self, idx):
            return self._terms[idx]

        def get_coef_indices(self, idx):
            return [idx]

    class DummyModel:
        def __init__(self):
            self.terms = DummyTerms()
            self.coef_ = np.array([0.3, -0.2, 1.1], dtype=float)

        def _modelmat(self, X):
            return np.column_stack([X[:, 0], X[:, 1], np.ones(len(X))])

        def _linear_predictor(self, X):
            return self._modelmat(X) @ self.coef_

    X_scaled = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=float)
    effect_values, base_values = feature_effect_performance_regimes_utils.compute_gam_feature_effects(
        model=DummyModel(),
        X_scaled=X_scaled,
        feature_cols=["speed", "acceleration"],
    )

    np.testing.assert_allclose(effect_values, np.array([[0.6, -0.2], [1.2, -0.6]], dtype=float))
    np.testing.assert_allclose(base_values, np.array([1.1, 1.1], dtype=float))


def test_evaluate_umap_trustworthiness_by_group_emits_neighbor_views_and_mean(monkeypatch):
    umap_call_log: list[dict[str, float]] = []
    _patch_feature_effect_regime_dependencies(monkeypatch, umap_call_log)

    n_rows = 40
    analysis_df = pd.DataFrame(
        {
            "row_id": list(range(n_rows)),
            "performance_group": ["easy"] * n_rows,
            "effect__speed": np.linspace(0.1, 4.0, n_rows),
            "effect__heading": np.linspace(1.0, 5.0, n_rows),
            "effect__distance": np.linspace(2.0, 6.0, n_rows),
        }
    )
    cluster_spec = {
        "groups": ["easy"],
        "algorithms": ["hdbscan"],
        "evaluate_umap_latent_space": True,
        "umap_selected_n_components": {"easy": 2},
        "trustworthiness_neighbor_values": [5, 10, 15],
        "cluster_umap_n_neighbors": 30,
        "cluster_umap_min_dist": 0.0,
        "viz_umap_n_neighbors": 15,
        "viz_umap_min_dist": 0.1,
        "random_state": 42,
        "min_cluster_size": 5,
        "min_samples": 5,
        "optics_cluster_method": "xi",
        "optics_xi": 0.05,
        "distance_metric": "euclidean",
    }
    cluster_spec = feature_effect_performance_regimes_utils.resolve_cluster_spec(
        cluster_spec,
        effect_cols=["effect__speed", "effect__heading", "effect__distance"],
    )

    trustworthiness_df = feature_effect_performance_regimes_utils.evaluate_umap_trustworthiness_by_group(
        analysis_df,
        cluster_spec=cluster_spec,
        performance_group_col="performance_group",
    )

    assert set(trustworthiness_df["trustworthiness_view"]) == {"nn_5", "nn_10", "nn_15", "mean_5_10_15"}
    assert trustworthiness_df.groupby("n_components").size().to_dict() == {1: 4, 2: 4}
    assert trustworthiness_df.loc[trustworthiness_df["n_components"] == 2, "selected_for_clustering"].all()
    assert not trustworthiness_df.loc[trustworthiness_df["n_components"] == 1, "selected_for_clustering"].any()

    raw_df = trustworthiness_df.loc[trustworthiness_df["trustworthiness_n_neighbors"].notna()].copy()
    mean_df = trustworthiness_df.loc[trustworthiness_df["trustworthiness_view"] == "mean_5_10_15"].copy()
    expected_mean_by_dim = raw_df.groupby("n_components")["trustworthiness"].mean().to_dict()
    actual_mean_by_dim = mean_df.set_index("n_components")["trustworthiness"].to_dict()
    assert actual_mean_by_dim == pytest.approx(expected_mean_by_dim)
    assert set(raw_df["trustworthiness_n_neighbors"].astype(int)) == {5, 10, 15}
    assert len(umap_call_log) == 2
    assert {(entry["n_neighbors"], entry["min_dist"]) for entry in umap_call_log} == {(30, 0.0)}


def test_resolve_cluster_spec_requires_documented_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        feature_effect_performance_regimes_utils.resolve_cluster_spec(
            {
                "groups": ["easy"],
                "algorithms": ["hdbscan"],
            },
            effect_cols=["effect__speed", "effect__heading"],
        )


def test_resolve_cluster_spec_rejects_invalid_selected_umap_dim():
    with pytest.raises(ValueError, match="umap_selected_n_components"):
        feature_effect_performance_regimes_utils.resolve_cluster_spec(
            {
                "groups": ["easy"],
                "algorithms": ["hdbscan"],
                "evaluate_umap_latent_space": True,
                "umap_selected_n_components": {"easy": 5},
                "trustworthiness_neighbor_values": [5, 10, 15],
                "cluster_umap_n_neighbors": 30,
                "cluster_umap_min_dist": 0.0,
                "viz_umap_n_neighbors": 15,
                "viz_umap_min_dist": 0.1,
                "random_state": 42,
                "min_cluster_size": 5,
                "min_samples": 5,
                "optics_cluster_method": "xi",
                "optics_xi": 0.05,
                "distance_metric": "euclidean",
            },
            effect_cols=["effect__speed", "effect__heading", "effect__distance"],
        )


def test_resolve_inspection_config_rejects_umap_space_when_disabled():
    cluster_spec = feature_effect_performance_regimes_utils.resolve_cluster_spec(
        {
            "groups": ["easy"],
            "algorithms": ["hdbscan"],
            "evaluate_umap_latent_space": False,
            "umap_selected_n_components": {"easy": 1},
            "trustworthiness_neighbor_values": [5, 10, 15],
            "cluster_umap_n_neighbors": 30,
            "cluster_umap_min_dist": 0.0,
            "viz_umap_n_neighbors": 15,
            "viz_umap_min_dist": 0.1,
            "random_state": 42,
            "min_cluster_size": 5,
            "min_samples": 5,
            "optics_cluster_method": "xi",
            "optics_xi": 0.05,
            "distance_metric": "euclidean",
        },
        effect_cols=["effect__speed", "effect__heading"],
    )

    with pytest.raises(ValueError, match="inspection_cluster_space"):
        feature_effect_performance_regimes_utils.resolve_inspection_config(
            {
                "inspection_algorithm": "hdbscan",
                "inspection_cluster_space": "umap",
                "inspection_top_k_features": 8,
                "inspection_top_k_table": 3,
                "sort_cluster_profiles_by": "cluster_size",
            },
            cluster_spec=cluster_spec,
        )


def test_run_step2_clustering_separates_cluster_and_visual_umap_parameters(monkeypatch):
    umap_call_log: list[dict[str, float]] = []
    _patch_feature_effect_regime_dependencies(monkeypatch, umap_call_log)

    n_rows = 40
    analysis_df = pd.DataFrame(
        {
            "row_id": list(range(n_rows)),
            "performance_group": ["easy"] * n_rows,
            "effect__speed": np.linspace(0.1, 4.0, n_rows),
            "effect__heading": np.linspace(1.0, 5.0, n_rows),
            "effect__distance": np.linspace(2.0, 6.0, n_rows),
        }
    )
    cluster_spec = {
        "groups": ["easy"],
        "algorithms": ["hdbscan"],
        "evaluate_umap_latent_space": True,
        "umap_selected_n_components": {"easy": 2},
        "trustworthiness_neighbor_values": [5, 10, 15],
        "cluster_umap_n_neighbors": 30,
        "cluster_umap_min_dist": 0.0,
        "viz_umap_n_neighbors": 15,
        "viz_umap_min_dist": 0.1,
        "random_state": 42,
        "min_cluster_size": 10,
        "min_samples": 10,
        "optics_cluster_method": "xi",
        "optics_xi": 0.05,
        "distance_metric": "euclidean",
    }
    cluster_spec = feature_effect_performance_regimes_utils.resolve_cluster_spec(
        cluster_spec,
        effect_cols=["effect__speed", "effect__heading", "effect__distance"],
    )

    clustering_results = feature_effect_performance_regimes_utils.run_step2_clustering(
        analysis_df,
        cluster_spec=cluster_spec,
        performance_group_col="performance_group",
        row_id_col="row_id",
    )

    assert len(umap_call_log) == 3
    assert {"n_components": 1, "n_neighbors": 30, "min_dist": 0.0, "random_state": 42} in umap_call_log
    assert {"n_components": 2, "n_neighbors": 30, "min_dist": 0.0, "random_state": 42} in umap_call_log
    assert {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "random_state": 42} in umap_call_log
    assert clustering_results["clustered_df"].loc[0, "viz_umap_x"] == pytest.approx(15.1)
    assert clustering_results["clustered_df"].loc[0, "viz_umap_y"] == pytest.approx(16.1)
    assert set(clustering_results["trustworthiness_df"]["trustworthiness_view"]) == {
        "nn_5",
        "nn_10",
        "nn_15",
        "mean_5_10_15",
    }


def test_build_feature_effect_regime_export_layout_is_stable_for_equivalent_resolved_cluster_spec(tmp_path):
    cluster_spec_scalar = _resolved_feature_effect_regime_cluster_spec(
        {
            "groups": ["easy", "medium"],
            "algorithms": ["hdbscan", "optics"],
            "evaluate_umap_latent_space": True,
            "umap_selected_n_components": 2,
            "trustworthiness_neighbor_values": [5, 10, 15],
            "cluster_umap_n_neighbors": 30,
            "cluster_umap_min_dist": 0.0,
            "viz_umap_n_neighbors": 15,
            "viz_umap_min_dist": 0.1,
            "random_state": 42,
            "min_cluster_size": 5,
            "min_samples": 5,
            "optics_cluster_method": "xi",
            "optics_xi": 0.05,
            "distance_metric": "euclidean",
        }
    )
    cluster_spec_mapping = _resolved_feature_effect_regime_cluster_spec(
        {
            "groups": ["easy", "medium"],
            "algorithms": ["hdbscan", "optics"],
            "evaluate_umap_latent_space": True,
            "umap_selected_n_components": {"easy": 2, "medium": 2},
            "trustworthiness_neighbor_values": [5, 10, 15],
            "cluster_umap_n_neighbors": 30,
            "cluster_umap_min_dist": 0.0,
            "viz_umap_n_neighbors": 15,
            "viz_umap_min_dist": 0.1,
            "random_state": 42,
            "min_cluster_size": {"easy": 5, "medium": 5},
            "min_samples": {"easy": 5, "medium": 5},
            "optics_cluster_method": "xi",
            "optics_xi": 0.05,
            "distance_metric": "euclidean",
        }
    )
    export_context = feature_effect_performance_regimes_utils.resolve_feature_effect_regime_export_context(
        model_id="xgboost",
        run_name="run_a",
        target_col="ml_ade_log",
        eval_csv_name="eval_epoch_5.csv",
        lower_is_better=True,
        performance_group_col="performance_group",
        results_root=tmp_path / "results",
    )

    scalar_layout = feature_effect_performance_regimes_utils.build_feature_effect_regime_export_layout(
        export_context=export_context,
        cluster_spec=cluster_spec_scalar,
    )
    mapping_layout = feature_effect_performance_regimes_utils.build_feature_effect_regime_export_layout(
        export_context=export_context,
        cluster_spec=cluster_spec_mapping,
    )

    assert scalar_layout["cluster_spec_hash"] == mapping_layout["cluster_spec_hash"]
    assert scalar_layout["cluster_spec_dirname"] == mapping_layout["cluster_spec_dirname"]
    assert scalar_layout["cluster_spec_root"] == mapping_layout["cluster_spec_root"]
    assert scalar_layout["tables_dir"].is_dir()
    assert scalar_layout["plots_dir"].is_dir()


def test_resolve_feature_effect_regime_export_context_splits_on_non_cluster_inputs(tmp_path):
    base_kwargs = {
        "model_id": "xgboost",
        "run_name": "run_a",
        "target_col": "ml_ade_log",
        "eval_csv_name": "eval_epoch_5.csv",
        "lower_is_better": True,
        "performance_group_col": "performance_group",
        "results_root": tmp_path / "results",
    }
    base_context = feature_effect_performance_regimes_utils.resolve_feature_effect_regime_export_context(**base_kwargs)
    changed_eval_context = feature_effect_performance_regimes_utils.resolve_feature_effect_regime_export_context(
        **(base_kwargs | {"eval_csv_name": "eval_epoch_10.csv"})
    )
    changed_lower_is_better_context = feature_effect_performance_regimes_utils.resolve_feature_effect_regime_export_context(
        **(base_kwargs | {"lower_is_better": False})
    )
    changed_group_col_context = feature_effect_performance_regimes_utils.resolve_feature_effect_regime_export_context(
        **(base_kwargs | {"performance_group_col": "difficulty_band"})
    )
    changed_target_context = feature_effect_performance_regimes_utils.resolve_feature_effect_regime_export_context(
        **(base_kwargs | {"target_col": "ml_fde_log"})
    )
    changed_model_context = feature_effect_performance_regimes_utils.resolve_feature_effect_regime_export_context(
        **(base_kwargs | {"model_id": "gam"})
    )

    assert base_context["data_context_root"] != changed_eval_context["data_context_root"]
    assert base_context["data_context_root"] != changed_lower_is_better_context["data_context_root"]
    assert base_context["data_context_root"] != changed_group_col_context["data_context_root"]
    assert base_context["target_root"] != changed_target_context["target_root"]
    assert base_context["model_root"] != changed_model_context["model_root"]
    assert base_context["run_root"] != changed_model_context["run_root"]
    assert "target-ml_ade_log" in base_context["data_context_slug"]


def test_build_feature_effect_regime_artifact_names_return_candidate_wide_export_targets():
    cluster_spec = _resolved_feature_effect_regime_cluster_spec(
        {
            "groups": ["easy", "medium"],
            "algorithms": ["hdbscan", "optics"],
            "evaluate_umap_latent_space": True,
            "umap_selected_n_components": {"easy": 2, "medium": 2},
            "trustworthiness_neighbor_values": [5, 10, 15],
            "cluster_umap_n_neighbors": 30,
            "cluster_umap_min_dist": 0.0,
            "viz_umap_n_neighbors": 15,
            "viz_umap_min_dist": 0.1,
            "random_state": 42,
            "min_cluster_size": {"easy": 5, "medium": 5},
            "min_samples": {"easy": 5, "medium": 5},
            "optics_cluster_method": "xi",
            "optics_xi": 0.05,
            "distance_metric": "euclidean",
        }
    )
    artifact_names = feature_effect_performance_regimes_utils.build_feature_effect_regime_artifact_names(cluster_spec=cluster_spec)

    assert artifact_names["tables"]["regime_analysis"] == "regime_analysis.csv"
    assert artifact_names["tables"]["cluster_assignments"] == "cluster_assignments.csv"
    assert artifact_names["tables"]["cluster_feature_effect_profiles"] == "cluster_feature_effect_profiles.csv"
    assert artifact_names["tables"]["feature_effect_global_ranking"] == "feature_effect_global_ranking.csv"
    assert artifact_names["tables"]["cluster_catalog"] == "cluster_catalog.csv"
    assert "cluster_profile_barplots" not in artifact_names["plots"]
    assert "cluster_profile_heatmaps" not in artifact_names["plots"]
    assert artifact_names["plots"]["raw_algorithm_comparison_grid"] == "algorithm_comparison_grid__space-raw.png"
    assert (
        artifact_names["plots"]["umap_trustworthiness_curves"]["mean_5_10_15"]
        == "umap_trustworthiness_curve__view-mean_5_10_15.png"
    )


def test_merge_feature_effect_regime_artifact_records_upserts_tables_and_preserves_distinct_member_exports(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_context": {"run_name": "run_a"},
                "data_context": {"target_col": "ml_ade_log"},
                "cluster_spec": {"hash": "abc123"},
                "artifacts": [
                    {
                        "artifact_kind": "table",
                        "artifact_type": "cluster_scores",
                        "relative_path": "tables/cluster_scores.csv",
                        "absolute_path": "/old/tables/cluster_scores.csv",
                    },
                    {
                        "artifact_kind": "table",
                        "artifact_type": "cluster_members",
                        "relative_path": "tables/cluster_members__group-easy__alg-hdbscan__space-raw__label-cluster_0.csv",
                        "absolute_path": "/old/tables/cluster_0.csv",
                    },
                ],
            }
        )
    )

    manifest_data = feature_effect_performance_regimes_utils.load_or_initialize_feature_effect_regime_manifest(
        manifest_path,
        run_context={"run_name": "run_a"},
        data_context={"target_col": "ml_ade_log"},
        cluster_spec={"hash": "abc123"},
    )
    merged_manifest = feature_effect_performance_regimes_utils.merge_feature_effect_regime_artifact_records(
        manifest_data,
        artifact_records=[
            {
                "artifact_kind": "table",
                "artifact_type": "cluster_scores",
                "relative_path": "tables/cluster_scores.csv",
                "absolute_path": "/new/tables/cluster_scores.csv",
            },
            {
                "artifact_kind": "table",
                "artifact_type": "cluster_members",
                "relative_path": "tables/cluster_members__group-easy__alg-hdbscan__space-raw__label-noise.csv",
                "absolute_path": "/new/tables/noise.csv",
            },
        ],
    )

    merged_artifacts = {artifact["relative_path"]: artifact for artifact in merged_manifest["artifacts"]}
    assert len(merged_artifacts) == 3
    assert merged_artifacts["tables/cluster_scores.csv"]["absolute_path"] == "/new/tables/cluster_scores.csv"
    assert (
        "tables/cluster_members__group-easy__alg-hdbscan__space-raw__label-cluster_0.csv"
        in merged_artifacts
    )
    assert (
        "tables/cluster_members__group-easy__alg-hdbscan__space-raw__label-noise.csv"
        in merged_artifacts
    )


def test_build_cluster_feature_effect_profiles_include_noise_and_cluster_metadata():
    clustered_df, cluster_scores_df = _sample_cluster_export_inputs()

    profile_df = feature_effect_performance_regimes_utils.build_cluster_feature_effect_profiles(
        clustered_df,
        cluster_scores_df,
        performance_group_col="performance_group",
        effect_cols=["effect__speed", "effect__acceleration"],
        include_noise=True,
    )

    assert profile_df["cluster_id"].astype(int).tolist() == [0, 1, -1]
    assert profile_df["cluster_label"].tolist() == ["cluster_0", "cluster_1", "noise"]
    assert profile_df["cluster_size"].tolist() == [2, 2, 1]
    assert profile_df.loc[profile_df["cluster_id"] == 0, "cluster_rank_by_size"].iloc[0] == 1
    assert profile_df.loc[profile_df["cluster_id"] == 1, "cluster_rank_by_size"].iloc[0] == 2
    assert pd.isna(profile_df.loc[profile_df["cluster_id"] == -1, "cluster_rank_by_size"].iloc[0])
    assert profile_df.loc[profile_df["cluster_id"] == 0, "effect__speed"].iloc[0] == pytest.approx(0.7)
    assert profile_df.loc[profile_df["cluster_id"] == -1, "is_noise"].iloc[0]
    assert "dominant_feature_name" not in profile_df.columns


def test_write_cluster_exports_generates_catalog_member_files_and_artifact_records(tmp_path):
    clustered_df, cluster_scores_df = _sample_cluster_export_inputs()
    export_layout = {
        "cluster_spec_root": tmp_path,
        "tables_dir": tmp_path / "tables",
    }
    export_layout["tables_dir"].mkdir(parents=True)

    outputs = feature_effect_cluster_exports.write_cluster_exports(
        clustered_df,
        cluster_scores_df,
        export_layout=export_layout,
        performance_metric_col="ml_ade",
        performance_group_col="performance_group",
        effect_cols=["effect__speed", "effect__acceleration"],
    )

    catalog_df = outputs["cluster_catalog_df"]
    assert catalog_df["cluster_id"].astype(int).tolist() == [0, 1, -1]
    assert catalog_df["cluster_label"].tolist() == ["cluster_0", "cluster_1", "noise"]
    assert catalog_df["unique_scene_step_count"].tolist() == [1, 2, 1]
    assert catalog_df["unique_scene_count"].tolist() == [1, 2, 1]

    member_paths = catalog_df["members_relative_path"].tolist()
    assert member_paths == [
        "tables/cluster_members__group-easy__alg-hdbscan__space-raw__label-cluster_0.csv",
        "tables/cluster_members__group-easy__alg-hdbscan__space-raw__label-cluster_1.csv",
        "tables/cluster_members__group-easy__alg-hdbscan__space-raw__label-noise.csv",
    ]

    member_df = pd.read_csv(tmp_path / member_paths[0])
    assert member_df.columns.tolist() == [
        "row_id",
        "data_idx",
        "outer_fold",
        "performance_group",
        "ml_ade",
        "target_orig",
        "oof_pred_orig",
        "scene_id",
        "scene_path",
        "scene_ts",
        "agent_type",
    ]
    assert "effect__speed" not in member_df.columns
    assert "speed" not in member_df.columns

    artifact_types = [artifact["artifact_type"] for artifact in outputs["artifact_records"]]
    assert artifact_types.count("cluster_catalog") == 1
    assert artifact_types.count("cluster_feature_effect_profiles") == 1
    assert artifact_types.count("cluster_members") == 3


def test_build_scene_step_key_frame_dedupes_scene_steps_and_falls_back_to_scene_id():
    scene_df = pd.DataFrame(
        {
            "scene_id": ["scene_a", "scene_a", "scene_b", "scene_b"],
            "scene_ts": [0, 0, 1, 2],
        }
    )

    scene_steps_df = feature_effect_cluster_exports.build_scene_step_key_frame(scene_df)

    assert scene_steps_df.to_dict(orient="records") == [
        {"scene_key": "scene_a", "scene_id": "scene_a", "scene_ts": 0},
        {"scene_key": "scene_b", "scene_id": "scene_b", "scene_ts": 1},
        {"scene_key": "scene_b", "scene_id": "scene_b", "scene_ts": 2},
    ]
    assert feature_effect_cluster_exports.summarize_scene_steps(scene_df) == {
        "unique_scene_step_count": 3,
        "unique_scene_count": 2,
    }


def test_load_cluster_inspection_selection_supports_all_and_explicit_cluster_ids(tmp_path):
    clustered_df, cluster_scores_df = _sample_cluster_export_inputs()
    export_layout = {
        "cluster_spec_root": tmp_path,
        "tables_dir": tmp_path / "tables",
    }
    export_layout["tables_dir"].mkdir(parents=True)
    outputs = feature_effect_cluster_exports.write_cluster_exports(
        clustered_df,
        cluster_scores_df,
        export_layout=export_layout,
        performance_metric_col="ml_ade",
        performance_group_col="performance_group",
        effect_cols=["effect__speed", "effect__acceleration"],
    )
    cluster_assignments_path = export_layout["tables_dir"] / "cluster_assignments.csv"
    clustered_df.to_csv(cluster_assignments_path, index=False)
    ranking_path = export_layout["tables_dir"] / "feature_effect_global_ranking.csv"
    _sample_feature_effect_global_ranking_df(reverse_order=True).to_csv(ranking_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_context": {"run_name": "run_a", "model_id": "xgboost", "target_mode": "log"},
                "data_context": {"target_col": "ml_ade"},
                "cluster_spec": {"hash": "abc123"},
                "artifacts": [
                    {
                        "artifact_kind": "table",
                        "artifact_type": "cluster_assignments",
                        "relative_path": str(cluster_assignments_path.relative_to(tmp_path)),
                        "absolute_path": str(cluster_assignments_path.resolve()),
                    },
                    {
                        "artifact_kind": "table",
                        "artifact_type": "feature_effect_global_ranking",
                        "relative_path": str(ranking_path.relative_to(tmp_path)),
                        "absolute_path": str(ranking_path.resolve()),
                    },
                    *outputs["artifact_records"],
                ],
            }
        )
    )

    resolved_config = feature_effect_pr_cluster_inspection.resolve_cluster_inspection_config(
        {
            "cluster_spec_manifest_path": manifest_path,
            "performance_group": "easy",
            "inspection_algorithm": "hdbscan",
            "inspection_cluster_space": "raw",
            "cluster_ids": "all",
            "inspection_top_k_features": 8,
            "inspection_top_k_table": 3,
            "distribution_matrix_max_columns": 6,
            "sort_cluster_profiles_by": "cluster_size",
        }
    )
    bundle = feature_effect_pr_cluster_inspection.load_cluster_inspection_selection(resolved_config)
    assert bundle.ordered_cluster_ids == [0, 1, -1]
    assert bundle.trajectory_feature_cols == ["target_orig", "acceleration", "speed"]
    assert bundle.scene_metric_cols == ["scene_density", "scene_weather"]
    assert bundle.effect_title_label == "SHAP"
    assert bundle.effect_value_axis_label == "Mean SHAP value"

    explicit_config = feature_effect_pr_cluster_inspection.resolve_cluster_inspection_config(
        {
            "cluster_spec_manifest_path": manifest_path,
            "performance_group": "easy",
            "inspection_algorithm": "hdbscan",
            "inspection_cluster_space": "raw",
            "cluster_ids": [1, -1],
            "inspection_top_k_features": 8,
            "inspection_top_k_table": 3,
            "distribution_matrix_max_columns": 6,
            "sort_cluster_profiles_by": "cluster_size",
        }
    )
    explicit_bundle = feature_effect_pr_cluster_inspection.load_cluster_inspection_selection(explicit_config)
    assert explicit_bundle.ordered_cluster_ids == [1, -1]
    assert explicit_bundle.selected_catalog_df["cluster_id"].astype(int).tolist() == [1, -1]


def test_resolve_effect_display_context_switches_between_shap_and_additive_log_scale_labels():
    xgboost_context = feature_effect_pr_cluster_inspection.resolve_effect_display_context("xgboost", "log")
    assert xgboost_context["effect_title_label"] == "SHAP"
    assert xgboost_context["effect_value_axis_label"] == "Mean SHAP value"
    assert "SHAP contributions" in xgboost_context["effect_note"]

    gam_context = feature_effect_pr_cluster_inspection.resolve_effect_display_context("gam", "log")
    assert gam_context["effect_title_label"] == "Additive effect (log scale)"
    assert gam_context["effect_value_axis_label"] == "Mean additive effect (log scale)"
    assert "link/log scale" in gam_context["effect_note"]
    assert "multiplicative changes" in gam_context["effect_note"]


def test_resolve_metric_plot_type_distinguishes_continuous_and_discrete_series():
    continuous_series = pd.Series(np.arange(12, dtype=float))
    discrete_numeric_series = pd.Series([0, 1, 0, 1, 1, 0], dtype=int)
    categorical_series = pd.Series(["sunny", "rain", "rain"], dtype="string")

    assert feature_effect_pr_cluster_inspection.resolve_metric_plot_type(continuous_series) == "continuous"
    assert feature_effect_pr_cluster_inspection.resolve_metric_plot_type(discrete_numeric_series) == "categorical"
    assert feature_effect_pr_cluster_inspection.resolve_metric_plot_type(categorical_series) == "categorical"


def test_chunk_metric_columns_splits_overview_pages_deterministically():
    assert feature_effect_pr_cluster_inspection.chunk_metric_columns(
        ["f1", "f2", "f3", "f4", "f5"],
        max_columns=2,
    ) == [["f1", "f2"], ["f3", "f4"], ["f5"]]


def test_build_subset_style_map_assigns_noise_and_baseline_colors():
    style_map = feature_effect_pr_cluster_inspection.build_subset_style_map(
        ["Cluster 2", "Cluster 0", "Noise", feature_effect_pr_cluster_inspection.WHOLE_GROUP_LABEL]
    )

    assert set(style_map) == {"Cluster 2", "Cluster 0", "Noise", feature_effect_pr_cluster_inspection.WHOLE_GROUP_LABEL}
    assert style_map["Noise"]["color"] == feature_effect_pr_cluster_inspection.DEFAULT_NOISE_COLOR
    assert style_map[feature_effect_pr_cluster_inspection.WHOLE_GROUP_LABEL]["color"] == feature_effect_pr_cluster_inspection.DEFAULT_BASELINE_COLOR
    assert style_map["Cluster 2"]["color"] != style_map["Cluster 0"]["color"]


def test_load_cluster_inspection_selection_orders_trajectory_features_by_global_feature_effect_ranking(tmp_path):
    clustered_df, cluster_scores_df = _sample_cluster_export_inputs()
    export_layout = {
        "cluster_spec_root": tmp_path,
        "tables_dir": tmp_path / "tables",
    }
    export_layout["tables_dir"].mkdir(parents=True)
    outputs = feature_effect_cluster_exports.write_cluster_exports(
        clustered_df,
        cluster_scores_df,
        export_layout=export_layout,
        performance_metric_col="ml_ade",
        performance_group_col="performance_group",
        effect_cols=["effect__speed", "effect__acceleration"],
    )
    cluster_assignments_path = export_layout["tables_dir"] / "cluster_assignments.csv"
    clustered_df.to_csv(cluster_assignments_path, index=False)
    ranking_path = export_layout["tables_dir"] / "feature_effect_global_ranking.csv"
    _sample_feature_effect_global_ranking_df(reverse_order=True).to_csv(ranking_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_context": {"run_name": "run_a", "model_id": "xgboost", "target_mode": "log"},
                "data_context": {"target_col": "ml_ade"},
                "cluster_spec": {"hash": "abc123"},
                "artifacts": [
                    {
                        "artifact_kind": "table",
                        "artifact_type": "cluster_assignments",
                        "relative_path": str(cluster_assignments_path.relative_to(tmp_path)),
                        "absolute_path": str(cluster_assignments_path.resolve()),
                    },
                    {
                        "artifact_kind": "table",
                        "artifact_type": "feature_effect_global_ranking",
                        "relative_path": str(ranking_path.relative_to(tmp_path)),
                        "absolute_path": str(ranking_path.resolve()),
                    },
                    *outputs["artifact_records"],
                ],
            }
        )
    )

    bundle = feature_effect_pr_cluster_inspection.load_cluster_inspection_selection(
        {
            "cluster_spec_manifest_path": manifest_path,
            "performance_group": "easy",
            "inspection_algorithm": "hdbscan",
            "inspection_cluster_space": "raw",
            "cluster_ids": "all",
            "inspection_top_k_features": 8,
            "inspection_top_k_table": 3,
            "distribution_matrix_max_columns": 6,
            "sort_cluster_profiles_by": "cluster_size",
        }
    )

    assert bundle.trajectory_feature_cols == ["target_orig", "acceleration", "speed"]


def test_scene_metric_order_prefers_semantic_priority_then_alphabetical():
    df = pd.DataFrame(
        {
            "performance_group": ["easy"],
            "scene_weather": ["rain"],
            "scene_num_agents": [5],
            "scene_alpha": [1.0],
            "scene_density_PEDESTRIAN": [0.4],
            "scene_bbox_width": [12.0],
            "scene_misc": [2.0],
        }
    )

    ordered_cols = feature_effect_pr_cluster_inspection._resolve_scene_metric_cols(df)

    assert ordered_cols == [
        "scene_num_agents",
        "scene_bbox_width",
        "scene_density_PEDESTRIAN",
        "scene_alpha",
        "scene_misc",
        "scene_weather",
    ]
