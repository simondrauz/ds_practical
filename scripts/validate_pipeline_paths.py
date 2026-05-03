#!/usr/bin/env python3
"""Validate the full trainval and mini-sweep Trajectron++ analysis paths.

The script runs capped integration checks against the real nuScenes datasets.
It intentionally writes only ignored result artifacts and temporary validation
configs while leaving tracked pipeline configs untouched.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
TRAJDATA_PATH = ROOT / "unified-av-data-loader" / "src"
NOTEBOOK_DIR = ROOT / "src" / "data_modelling"
SHARED_CONFIG_PATH = ROOT / "config" / "shared_config.yaml"

MINI_RAW = ROOT / "data" / "raw"
MINI_CACHE = ROOT / "data" / "processed" / "trajdata_cache"
TRAINVAL_RAW = Path("/Volumes/LaCie 1TB/nuScenes/v1.0-trainval_raw")
TRAINVAL_CACHE = Path("/Volumes/LaCie 1TB/nuScenes/trajdata_cache")

MODEL_LOG_DIR = ROOT / "results" / "trajectory_prediction" / "nuScenes" / "models"
METRICS_ROOT = ROOT / "results" / "trajectory_prediction" / "trajectory_metrics"
JOINED_ROOT = ROOT / "results" / "trajectory_prediction" / "trajectory_metrics_joined"
COMBINED_OUTPUT = ROOT / "results" / "trajectory_prediction" / "combined_runs.csv"

TARGET_RAW = "ml_ade"
TARGET_MODEL = "ml_ade_log"
MODELS = ("gam", "xgboost")


PatchFn = Callable[[str], tuple[str, bool]]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "command"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _list_subdirs(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    return {item for item in path.iterdir() if item.is_dir()}


def _new_subdirs(path: Path, before: set[Path]) -> list[Path]:
    return sorted(_list_subdirs(path) - before, key=lambda item: item.stat().st_mtime)


def _line_assignment(name: str, value_repr: str) -> PatchFn:
    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=.*$")

    def patch(source: str) -> tuple[str, bool]:
        updated, count = pattern.subn(f"{name} = {value_repr}", source, count=1)
        return updated, count == 1

    return patch


def _dict_key_value(key: str, value_repr: str) -> PatchFn:
    pattern = re.compile(rf"(?m)^(\s*[\"']{re.escape(key)}[\"']\s*:\s*).*$")

    def patch(source: str) -> tuple[str, bool]:
        updated, count = pattern.subn(rf"\g<1>{value_repr},", source, count=1)
        return updated, count == 1

    return patch


def _block_assignment(name: str, value_repr: str) -> PatchFn:
    pattern = re.compile(
        rf"^{re.escape(name)}\s*=\s*\(.*?^\)\s*$|"
        rf"^{re.escape(name)}\s*=[^\n]*$",
        re.MULTILINE | re.DOTALL,
    )

    def patch(source: str) -> tuple[str, bool]:
        updated, count = pattern.subn(f"{name} = {value_repr}", source, count=1)
        return updated, count == 1

    return patch


def _patch_notebook(notebook_path: Path, patchers: list[PatchFn]):
    import nbformat

    notebook = nbformat.read(notebook_path, as_version=4)
    remaining = list(patchers)

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        applied: list[PatchFn] = []
        for patcher in remaining:
            updated, changed = patcher(source)
            if changed:
                source = updated
                applied.append(patcher)
        if applied:
            cell["source"] = source
            remaining = [patcher for patcher in remaining if patcher not in applied]
        if not remaining:
            break

    if remaining:
        raise ValueError(
            f"Failed to apply {len(remaining)} patch(es) for {notebook_path.name}."
        )
    return notebook


@dataclass
class ValidationContext:
    validation_id: str
    output_root: Path
    log_dir: Path
    summary_path: Path
    summary: dict[str, Any] = field(default_factory=dict)

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        pythonpath_entries = [str(SRC_PATH), str(TRAJDATA_PATH)]
        existing = env.get("PYTHONPATH")
        if existing:
            pythonpath_entries.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        env["WANDB_MODE"] = "disabled"
        env["MPLBACKEND"] = "Agg"
        return env

    def write_summary(self) -> None:
        _write_json(self.summary_path, self.summary)

    def record_command(
        self,
        *,
        label: str,
        cmd: list[str],
        returncode: int,
        log_path: Path,
        elapsed_s: float,
    ) -> None:
        self.summary.setdefault("commands", []).append(
            {
                "label": label,
                "cmd": cmd,
                "returncode": returncode,
                "elapsed_s": elapsed_s,
                "log_path": log_path,
            }
        )
        self.write_summary()


def _run_command(
    ctx: ValidationContext,
    label: str,
    cmd: list[str],
    *,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    ctx.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = ctx.log_dir / f"{len(ctx.summary.get('commands', [])) + 1:02d}_{_slug(label)}.log"
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=ctx.env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    elapsed_s = time.monotonic() - start
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    ctx.record_command(
        label=label,
        cmd=cmd,
        returncode=proc.returncode,
        log_path=log_path,
        elapsed_s=elapsed_s,
    )
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            output=f"Command failed; see {log_path}",
        )
    return proc


def _ensure_imports(ctx: ValidationContext) -> None:
    modules = [
        "trajdata",
        "trajectron",
        "data_preparation.join_characteristic_metrics",
        "data_modelling.run_context",
        "nbformat",
        "nbconvert",
        "pandas",
        "numpy",
        "yaml",
        "optuna",
        "xgboost",
        "pygam",
        "hdbscan",
        "umap",
        "statsmodels",
        "sklearn",
        "shap",
    ]
    package_by_module = {
        "hdbscan": "hdbscan>=0.8.33",
        "optuna": "optuna>=3.0.0",
        "pygam": "pygam>=0.9.0",
        "shap": "shap>=0.44.0",
        "sklearn": "scikit-learn>=1.3.0",
        "statsmodels": "statsmodels>=0.14.0",
        "umap": "umap-learn>=0.5.0",
        "xgboost": "xgboost>=1.7.0",
        "yaml": "pyyaml>=6.0",
    }
    editable_repo_modules = {
        "data_modelling.run_context",
        "data_preparation.join_characteristic_metrics",
        "trajectron",
    }
    editable_trajdata_modules = {"trajdata"}

    missing = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(module)

    if missing:
        ctx.summary["preflight"]["missing_imports_before_install"] = missing
        ctx.write_summary()
        packages_to_install = sorted(
            {package_by_module[module] for module in missing if module in package_by_module}
        )
        if packages_to_install:
            _run_command(
                ctx,
                "install missing validation dependencies",
                [sys.executable, "-m", "pip", "install", *packages_to_install],
            )
        unknown_missing = [
            module
            for module in missing
            if module not in package_by_module
            and module not in editable_repo_modules
            and module not in editable_trajdata_modules
        ]
        if unknown_missing:
            _run_command(
                ctx,
                "install requirements after unknown import failure",
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            )
            _run_command(
                ctx,
                "install l5kit without dependencies",
                [sys.executable, "-m", "pip", "install", "--no-dependencies", "l5kit==1.5.0"],
            )
        if editable_trajdata_modules.intersection(missing):
            _run_command(
                ctx,
                "editable install vendored trajdata",
                [sys.executable, "-m", "pip", "install", "-e", "unified-av-data-loader"],
            )
        if editable_repo_modules.intersection(missing):
            _run_command(
                ctx,
                "editable install repository",
                [sys.executable, "-m", "pip", "install", "-e", "."],
            )

    still_missing = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            still_missing.append({"module": module, "error": repr(exc)})
    if still_missing:
        raise ImportError(f"Required imports still failing: {still_missing}")


def _build_agent_dataset(
    desired_data: str,
    *,
    raw_root: Path,
    cache_root: Path,
    history_sec: float = 2.0,
    prediction_sec: float = 6.0,
):
    from trajdata import UnifiedDataset

    from shared_config.config_loader import (
        attention_radius_from_config,
        load_agent_type_defaults,
        load_attention_radius_config,
    )

    only_predict, no_types = load_agent_type_defaults()
    dataset_key = "nusc_trainval" if "trainval" in desired_data else "nusc_mini"
    return UnifiedDataset(
        desired_data=[desired_data],
        centric="agent",
        history_sec=(history_sec, history_sec),
        future_sec=(prediction_sec, prediction_sec),
        agent_interaction_distances=attention_radius_from_config(
            load_attention_radius_config()
        ),
        incl_robot_future=False,
        incl_raster_map=False,
        incl_vector_map=False,
        only_predict=only_predict,
        no_types=no_types,
        num_workers=0,
        cache_location=str(cache_root),
        data_dirs={dataset_key: str(raw_root)},
        verbose=False,
    )


def _dataset_samples(dataset) -> list[dict[str, Any]]:
    if len(dataset) == 0:
        return []
    candidate_indices = [0, len(dataset) // 2, len(dataset) - 1]
    rows = []
    for idx in dict.fromkeys(candidate_indices):
        scene_path, agent_id, scene_ts = dataset._data_index[int(idx)]
        try:
            elem = dataset[int(idx)]
            agent_type = getattr(elem, "agent_type", None)
            agent_type_name = getattr(agent_type, "name", str(agent_type))
        except Exception as exc:
            agent_type_name = f"unloaded: {exc!r}"
        rows.append(
            {
                "data_idx": int(idx),
                "scene_path": str(scene_path),
                "agent_id": str(agent_id),
                "scene_ts": int(scene_ts),
                "agent_type": agent_type_name,
            }
        )
    return rows


def _preflight(ctx: ValidationContext) -> None:
    ctx.summary["preflight"] = {
        "python": sys.executable,
        "python_version": sys.version,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "repo_root": ROOT,
    }
    _ensure_imports(ctx)

    required_paths = [
        MINI_RAW / "v1.0-mini",
        MINI_CACHE / "nusc_mini",
        TRAINVAL_RAW / "v1.0-trainval",
        TRAINVAL_CACHE / "nusc_trainval",
        ROOT / "config" / "experimental_setup" / "nuScenes" / "predchal_train_index.pkl",
        ROOT / "config" / "experimental_setup" / "nuScenes" / "predchal_train_val_index.pkl",
        ROOT / "config" / "experimental_setup" / "nuScenes" / "predchal_val_index.pkl",
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required validation paths: {missing_paths}")

    dataset_checks = [
        ("mini_train", "nusc_mini-mini_train", MINI_RAW, MINI_CACHE),
        ("mini_val", "nusc_mini-mini_val", MINI_RAW, MINI_CACHE),
        ("trainval_train", "nusc_trainval-train", TRAINVAL_RAW, TRAINVAL_CACHE),
        ("trainval_train_val", "nusc_trainval-train_val", TRAINVAL_RAW, TRAINVAL_CACHE),
    ]
    dataset_summary = {}
    for label, desired_data, raw_root, cache_root in dataset_checks:
        dataset = _build_agent_dataset(desired_data, raw_root=raw_root, cache_root=cache_root)
        dataset_summary[label] = {
            "desired_data": desired_data,
            "rows": len(dataset),
            "cache_path": str(dataset.cache_path),
            "samples": _dataset_samples(dataset),
        }
    ctx.summary["preflight"]["datasets"] = dataset_summary
    ctx.write_summary()


def _execute_notebook(
    ctx: ValidationContext,
    *,
    label: str,
    notebook_name: str,
    output_name: str,
    patchers: list[PatchFn],
) -> Path:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    notebook_path = NOTEBOOK_DIR / notebook_name
    output_path = ctx.output_root / "notebooks" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    old_env = os.environ.copy()
    os.environ.update(ctx.env())
    start = time.monotonic()
    try:
        notebook = _patch_notebook(notebook_path, patchers)
        nbformat.write(notebook, output_path)
        executor = ExecutePreprocessor(timeout=None, kernel_name="python3")
        executor.preprocess(notebook, {"metadata": {"path": str(NOTEBOOK_DIR)}})
        nbformat.write(notebook, output_path)
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    elapsed_s = time.monotonic() - start
    ctx.summary.setdefault("notebooks", []).append(
        {
            "label": label,
            "source": notebook_path,
            "executed": output_path,
            "elapsed_s": elapsed_s,
        }
    )
    ctx.write_summary()
    return output_path


def _run_preparation(
    ctx: ValidationContext,
    *,
    run_name: str,
    eval_csv_name: str,
    prefix: str,
) -> Path:
    _execute_notebook(
        ctx,
        label=f"{prefix} preparation",
        notebook_name="interpretable_model_data_preparation.ipynb",
        output_name=f"{prefix}__01_preparation.ipynb",
        patchers=[
            _line_assignment("RUN_NAME", repr(run_name)),
            _line_assignment("EVAL_CSV_NAME", repr(eval_csv_name)),
            _line_assignment("target_col", repr(TARGET_RAW)),
        ],
    )
    prepared_path = (
        ROOT
        / "results"
        / "interpretable_model"
        / "prepared_data"
        / run_name
        / f"prepared_data_{TARGET_RAW}.csv"
    )
    if not prepared_path.exists():
        raise FileNotFoundError(f"Preparation did not write expected data: {prepared_path}")
    return prepared_path


def _run_model_notebook(
    ctx: ValidationContext,
    *,
    model_id: str,
    run_name: str,
    prefix: str,
) -> Path:
    if model_id == "gam":
        notebook_name = "gam.ipynb"
        patchers = [
            _line_assignment("RUN_NAME", repr(run_name)),
            _line_assignment("PREPARED_TARGET_COL", repr(TARGET_RAW)),
            _line_assignment("TARGET_COL", repr(TARGET_MODEL)),
            _line_assignment("k_outer_fold", "2"),
            _line_assignment("k_inner_fold", "2"),
            _line_assignment("N_OPTUNA_TRIALS", "1"),
            _line_assignment("MIN_SPLINES", "5"),
            _line_assignment("MAX_SPLINES", "8"),
            _line_assignment("MIN_SPLINE_ORDER", "3"),
            _line_assignment("MAX_SPLINE_ORDER", "3"),
        ]
    elif model_id == "xgboost":
        notebook_name = "xgboost.ipynb"
        patchers = [
            _line_assignment("RUN_NAME", repr(run_name)),
            _line_assignment("PREPARED_TARGET_COL", repr(TARGET_RAW)),
            _line_assignment("TARGET_COL", repr(TARGET_MODEL)),
            _line_assignment("k_outer_fold", "2"),
            _line_assignment("k_inner_fold", "2"),
            _line_assignment("N_OPTUNA_TRIALS", "1"),
            _line_assignment("MAX_BOOST_ROUNDS", "20"),
            _line_assignment("EARLY_STOPPING_ROUNDS", "5"),
            _line_assignment("N_JOBS", "1"),
        ]
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")

    _execute_notebook(
        ctx,
        label=f"{prefix} {model_id}",
        notebook_name=notebook_name,
        output_name=f"{prefix}__02_{model_id}.ipynb",
        patchers=patchers,
    )
    manifest_path = (
        ROOT
        / "results"
        / "interpretable_model"
        / model_id
        / run_name
        / "tables"
        / f"run_manifest_{TARGET_MODEL}.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"{model_id} manifest not found: {manifest_path}")
    return manifest_path


def _run_model_inference(
    ctx: ValidationContext,
    *,
    model_id: str,
    run_name: str,
    prefix: str,
) -> dict[str, Path]:
    _execute_notebook(
        ctx,
        label=f"{prefix} {model_id} model inference",
        notebook_name="model_inference_analysis.ipynb",
        output_name=f"{prefix}__03_model_inference__{model_id}.ipynb",
        patchers=[
            _line_assignment("MODEL_ID", repr(model_id)),
            _line_assignment("RUN_NAME", repr(run_name)),
            _line_assignment("TARGET_COL", repr(TARGET_MODEL)),
        ],
    )
    tables_dir = ROOT / "results" / "interpretable_model" / model_id / run_name / "tables"
    outputs = {
        "feature_effects": tables_dir / f"feature_effects_{TARGET_MODEL}.csv",
        "feature_effect_importance": tables_dir / f"feature_effect_importance_{TARGET_MODEL}.csv",
    }
    for label, path in outputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Model inference did not write {label}: {path}")
    return outputs


def _run_regime_clustering(
    ctx: ValidationContext,
    *,
    model_id: str,
    run_name: str,
    eval_csv_name: str,
    prefix: str,
) -> Path:
    _execute_notebook(
        ctx,
        label=f"{prefix} {model_id} performance regimes",
        notebook_name="feature_effect_performance_regimes.ipynb",
        output_name=f"{prefix}__04_performance_regimes__{model_id}.ipynb",
        patchers=[
            _line_assignment("MODEL_ID", repr(model_id)),
            _line_assignment("RUN_NAME", repr(run_name)),
            _line_assignment("EVAL_CSV_NAME", repr(eval_csv_name)),
            _line_assignment("TARGET_COL", repr(TARGET_MODEL)),
            _dict_key_value("evaluate_umap_latent_space", "False"),
            _dict_key_value("umap_selected_n_components", "{'easy': 1, 'medium': 1, 'hard': 1}"),
            _dict_key_value("trustworthiness_neighbor_values", "[2]"),
            _dict_key_value("cluster_umap_n_neighbors", "5"),
            _dict_key_value("viz_umap_n_neighbors", "5"),
            _dict_key_value("min_cluster_size", "2"),
            _dict_key_value("min_samples", "2"),
            _dict_key_value("optics_xi", "0.05"),
        ],
    )
    search_root = (
        ROOT
        / "results"
        / "interpretable_model"
        / "feature_effect_performance_regimes"
        / model_id
        / run_name
        / TARGET_MODEL
    )
    manifests = list(search_root.glob("**/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No regime manifest found under {search_root}")
    return max(manifests, key=lambda path: path.stat().st_mtime)


def _artifact_from_manifest(manifest_path: Path, artifact_type: str, fallback: str) -> Path:
    manifest = _read_json(manifest_path)
    for artifact in manifest.get("artifacts", []):
        if artifact.get("artifact_type") == artifact_type:
            return (manifest_path.parent / artifact["relative_path"]).resolve()
    return manifest_path.parent / "tables" / fallback


def _select_cluster_candidate(manifest_path: Path) -> dict[str, Any]:
    import pandas as pd

    profiles_path = _artifact_from_manifest(
        manifest_path,
        "cluster_feature_effect_profiles",
        "cluster_feature_effect_profiles.csv",
    )
    profiles = pd.read_csv(profiles_path)
    if profiles.empty:
        raise ValueError(f"Cluster profile table is empty: {profiles_path}")

    profiles = profiles.loc[pd.to_numeric(profiles["cluster_size"], errors="coerce") > 0].copy()
    if profiles.empty:
        raise ValueError(f"No non-empty cluster profiles found: {profiles_path}")

    preferred = profiles.loc[
        (profiles["algorithm"].astype(str).str.lower() == "hdbscan")
        & (profiles["cluster_space"].astype(str).str.lower() == "raw")
        & (~profiles["is_noise"].astype(bool))
    ].copy()
    if preferred.empty:
        preferred = profiles.loc[
            (profiles["algorithm"].astype(str).str.lower() == "hdbscan")
            & (profiles["cluster_space"].astype(str).str.lower() == "raw")
        ].copy()
    if preferred.empty:
        preferred = profiles.loc[~profiles["is_noise"].astype(bool)].copy()
    if preferred.empty:
        preferred = profiles.copy()

    selected = preferred.sort_values(
        ["cluster_size", "performance_group", "algorithm", "cluster_space", "cluster_id"],
        ascending=[False, True, True, True, True],
    ).iloc[0]
    cluster_id = int(selected["cluster_id"])
    return {
        "manifest_path": manifest_path,
        "cluster_spec_dirname": manifest_path.parent.name,
        "performance_group": str(selected["performance_group"]),
        "algorithm": str(selected["algorithm"]).lower(),
        "cluster_space": str(selected["cluster_space"]).lower(),
        "cluster_ids": "all",
        "selected_cluster_id": cluster_id,
        "cluster_size": int(selected["cluster_size"]),
        "is_noise": bool(selected["is_noise"]),
        "profiles_path": profiles_path,
    }


def _run_cluster_inspection(
    ctx: ValidationContext,
    *,
    model_id: str,
    run_name: str,
    eval_csv_name: str,
    regime_manifest_path: Path,
    prefix: str,
) -> dict[str, Any]:
    candidate = _select_cluster_candidate(regime_manifest_path)
    _execute_notebook(
        ctx,
        label=f"{prefix} {model_id} cluster inspection",
        notebook_name="feature_effect_pr_cluster_inspection.ipynb",
        output_name=f"{prefix}__05_cluster_inspection__{model_id}.ipynb",
        patchers=[
            _line_assignment("MODEL_ID", repr(model_id)),
            _line_assignment("RUN_NAME", repr(run_name)),
            _line_assignment("EVAL_CSV_NAME", repr(eval_csv_name)),
            _line_assignment("TARGET_COL", repr(TARGET_MODEL)),
            _block_assignment("CLUSTER_SPEC_DIRNAME", repr(candidate["cluster_spec_dirname"])),
            _dict_key_value("performance_group", repr(candidate["performance_group"])),
            _dict_key_value("inspection_algorithm", repr(candidate["algorithm"])),
            _dict_key_value("inspection_cluster_space", repr(candidate["cluster_space"])),
            _dict_key_value("cluster_ids", repr(candidate["cluster_ids"])),
            _dict_key_value("inspection_top_k_features", "5"),
            _dict_key_value("inspection_top_k_table", "3"),
            _dict_key_value("distribution_matrix_max_columns", "5"),
        ],
    )
    return candidate


def _assert_eval_and_joined_outputs(
    *,
    eval_path: Path,
    joined_path: Path,
) -> dict[str, Any]:
    import pandas as pd

    eval_df = pd.read_csv(eval_path)
    joined_df = pd.read_csv(joined_path)
    required_eval_cols = {
        "data_idx",
        "scene_path",
        "agent_id",
        "scene_ts",
        "agent_type",
        "eval_data",
        "history_sec",
        "prediction_sec",
        "restrict_to_predchal",
        "ml_ade",
        "ml_fde",
    }
    required_joined_cols = {
        "data_idx",
        "ml_ade",
        "mean_speed",
        "path_efficiency",
        "scene_num_agents",
        "scene_spatial_density",
        "attention_radius_m",
        "history_sec",
        "prediction_sec",
    }
    missing_eval = sorted(required_eval_cols - set(eval_df.columns))
    missing_joined = sorted(required_joined_cols - set(joined_df.columns))
    if missing_eval:
        raise AssertionError(f"Eval CSV is missing columns: {missing_eval}")
    if missing_joined:
        raise AssertionError(f"Joined CSV is missing columns: {missing_joined}")
    if len(eval_df) != len(joined_df):
        raise AssertionError(
            f"Joined row count {len(joined_df)} does not match eval rows {len(eval_df)}"
        )
    if eval_df["data_idx"].nunique() != joined_df["data_idx"].nunique():
        raise AssertionError("Joined data_idx cardinality does not match eval data_idx")

    return {
        "eval_rows": len(eval_df),
        "joined_rows": len(joined_df),
        "eval_columns": list(eval_df.columns),
        "joined_columns": list(joined_df.columns),
    }


def _run_full_trainval(ctx: ValidationContext, args: argparse.Namespace) -> dict[str, Any]:
    before_models = _list_subdirs(MODEL_LOG_DIR)
    before_metrics = _list_subdirs(METRICS_ROOT)
    log_tag = f"validation_trainval_tpp_{ctx.validation_id}"
    data_loc_dict = json.dumps({"nusc_trainval": str(TRAINVAL_RAW)})

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        str(ROOT / "train_unified.py"),
        "--conf",
        "config/nuScenes.json",
        "--log_tag",
        log_tag,
        "--log_dir",
        str(MODEL_LOG_DIR),
        "--train_data",
        "nusc_trainval-train",
        "--eval_data",
        "nusc_trainval-train_val",
        "--trajdata_cache_dir",
        str(TRAINVAL_CACHE),
        "--data_loc_dict",
        data_loc_dict,
        "--history_sec",
        "2.0",
        "--prediction_sec",
        "6.0",
        "--train_epochs",
        "1",
        "--eval_every",
        "1",
        "--save_every",
        "1",
        "--batch_size",
        str(args.full_batch_size),
        "--eval_batch_size",
        str(args.full_batch_size),
        "--max_train_batches",
        str(args.full_max_train_batches),
        "--max_eval_batches",
        str(args.full_max_eval_batches),
        "--preprocess_workers",
        "0",
        "--K",
        "1",
        "--k_eval",
        "1",
    ]
    _run_command(ctx, "full trainval capped training", cmd)

    new_models = _new_subdirs(MODEL_LOG_DIR, before_models)
    new_metrics = _new_subdirs(METRICS_ROOT, before_metrics)
    if len(new_models) != 1 or len(new_metrics) != 1:
        raise RuntimeError(
            f"Expected one new model and metrics dir, got models={new_models}, metrics={new_metrics}"
        )
    model_dir = new_models[0]
    metrics_dir = new_metrics[0]
    run_name = metrics_dir.name
    config_path = model_dir / "config.json"
    eval_csv_name = "eval_epoch_1.csv"
    eval_path = metrics_dir / eval_csv_name
    if not config_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Missing full run config/eval output: config={config_path}, eval={eval_path}"
        )

    _run_command(
        ctx,
        "full trainval join",
        [
            sys.executable,
            "-m",
            "data_preparation.join_characteristic_metrics",
            "--conf",
            str(config_path),
            "--run_dir",
            run_name,
            "--output_root",
            str(JOINED_ROOT),
            "--format",
            "csv",
            "--incl_vector_map",
            "--trajdata_cache_dir",
            str(TRAINVAL_CACHE),
            "--data_loc_dict",
            data_loc_dict,
            "--preprocess_workers",
            "0",
        ],
    )
    joined_path = JOINED_ROOT / run_name / eval_csv_name
    output_summary = _assert_eval_and_joined_outputs(
        eval_path=eval_path,
        joined_path=joined_path,
    )

    prefix = "full_trainval"
    prepared_path = _run_preparation(
        ctx,
        run_name=run_name,
        eval_csv_name=eval_csv_name,
        prefix=prefix,
    )

    model_summaries = {}
    for model_id in MODELS:
        manifest_path = _run_model_notebook(
            ctx,
            model_id=model_id,
            run_name=run_name,
            prefix=prefix,
        )
        inference_outputs = _run_model_inference(
            ctx,
            model_id=model_id,
            run_name=run_name,
            prefix=prefix,
        )
        regime_manifest_path = _run_regime_clustering(
            ctx,
            model_id=model_id,
            run_name=run_name,
            eval_csv_name=eval_csv_name,
            prefix=prefix,
        )
        inspection_candidate = _run_cluster_inspection(
            ctx,
            model_id=model_id,
            run_name=run_name,
            eval_csv_name=eval_csv_name,
            regime_manifest_path=regime_manifest_path,
            prefix=prefix,
        )
        model_summaries[model_id] = {
            "manifest_path": manifest_path,
            "inference_outputs": inference_outputs,
            "regime_manifest_path": regime_manifest_path,
            "inspection_candidate": inspection_candidate,
        }

    return {
        "run_name": run_name,
        "model_dir": model_dir,
        "metrics_dir": metrics_dir,
        "config_path": config_path,
        "eval_path": eval_path,
        "joined_path": joined_path,
        "prepared_path": prepared_path,
        "output_summary": output_summary,
        "models": model_summaries,
    }


def _write_sweep_config(ctx: ValidationContext, args: argparse.Namespace) -> Path:
    sweep_config_path = ctx.output_root / "validation_sweep_config.yaml"
    config = {
        "base_args": {
            "conf": "config/nuScenes.json",
            "log_tag": f"validation_sweep_tpp_{ctx.validation_id}",
            "log_dir": str(MODEL_LOG_DIR),
            "train_epochs": 1,
            "eval_every": 1,
            "save_every": 1,
            "train_data": "nusc_mini-mini_train",
            "eval_data": "nusc_mini-mini_val",
            "trajdata_cache_dir": str(MINI_CACHE),
            "data_loc_dict": json.dumps({"nusc_mini": str(MINI_RAW)}),
            "batch_size": args.sweep_batch_size,
            "eval_batch_size": args.sweep_batch_size,
            "max_train_batches": args.sweep_max_train_batches,
            "max_eval_batches": args.sweep_max_eval_batches,
            "preprocess_workers": 0,
            "K": 1,
            "k_eval": 1,
        },
        "grid": {
            "history_sec": [2.0, 4.0],
            "prediction_sec": [2.0, 6.0],
            "attention_radius_scale": [0.5, 1.0],
        },
    }
    _write_yaml(sweep_config_path, config)
    return sweep_config_path


def _assert_sweep_combined(
    *,
    combined_path: Path,
    expected_run_names: list[str],
) -> dict[str, Any]:
    import pandas as pd

    combined = pd.read_csv(combined_path)
    if combined.empty:
        raise AssertionError(f"Combined sweep output is empty: {combined_path}")
    run_names = sorted(combined["run_name"].dropna().astype(str).unique().tolist())
    if run_names != sorted(expected_run_names):
        raise AssertionError(
            f"Combined run_names do not match current sweep outputs. "
            f"combined={run_names}, expected={sorted(expected_run_names)}"
        )
    required_cols = {
        "run_name",
        "eval_csv_name",
        "attention_radius_m",
        "history_sec",
        "prediction_sec",
        "agent_type",
        "mean_speed",
        "scene_num_agents",
        "ml_ade",
    }
    missing = sorted(required_cols - set(combined.columns))
    if missing:
        raise AssertionError(f"Combined sweep output missing columns: {missing}")

    grouped = (
        combined.groupby("run_name")[["history_sec", "prediction_sec"]]
        .nunique()
        .reset_index()
    )
    if not ((grouped["history_sec"] == 1) & (grouped["prediction_sec"] == 1)).all():
        raise AssertionError("Each sweep run must have one history/prediction setting.")

    run_settings = (
        combined.groupby("run_name", sort=False)
        .agg(
            history_sec=("history_sec", "first"),
            prediction_sec=("prediction_sec", "first"),
            attention_radius_m_values=(
                "attention_radius_m",
                lambda values: tuple(sorted({float(value) for value in values.dropna()})),
            ),
        )
        .reset_index()
        .sort_values(["history_sec", "prediction_sec", "attention_radius_m_values", "run_name"])
    )
    if len(run_settings) != 8:
        raise AssertionError(f"Expected 8 sweep run setting rows, got {len(run_settings)}")
    if run_settings["attention_radius_m_values"].map(len).eq(0).any():
        raise AssertionError("Each sweep run must have at least one realized attention_radius_m value.")

    expected_pairs = {
        (history_sec, prediction_sec)
        for history_sec in (2.0, 4.0)
        for prediction_sec in (2.0, 6.0)
    }
    actual_pairs = {
        (float(row.history_sec), float(row.prediction_sec))
        for row in run_settings.itertuples(index=False)
    }
    if actual_pairs != expected_pairs:
        raise AssertionError(
            f"Unexpected sweep history/prediction pairs. "
            f"actual={sorted(actual_pairs)}, expected={sorted(expected_pairs)}"
        )
    pair_counts = (
        run_settings.groupby(["history_sec", "prediction_sec"])["attention_radius_m_values"]
        .nunique()
        .reset_index(name="distinct_radius_settings")
    )
    if not (pair_counts["distinct_radius_settings"] == 2).all():
        raise AssertionError(
            "Each history/prediction pair should have two distinct realized "
            f"attention-radius settings. Found: {pair_counts.to_dict(orient='records')}"
        )

    realized_settings = (
        combined[["run_name", "agent_type", "history_sec", "prediction_sec", "attention_radius_m"]]
        .drop_duplicates()
        .sort_values(["history_sec", "prediction_sec", "attention_radius_m", "agent_type", "run_name"])
    )

    return {
        "rows": len(combined),
        "run_names": run_names,
        "setting_rows": run_settings.to_dict(orient="records"),
        "realized_setting_rows": realized_settings.to_dict(orient="records"),
        "columns": list(combined.columns),
    }


def _run_mini_sweep(ctx: ValidationContext, args: argparse.Namespace) -> dict[str, Any]:
    before_joined = _list_subdirs(JOINED_ROOT)
    before_shared_config = SHARED_CONFIG_PATH.read_bytes()
    sweep_config_path = _write_sweep_config(ctx, args)

    try:
        _run_command(
            ctx,
            "mini sweep capped run",
            [
                sys.executable,
                str(ROOT / "run_sweep.py"),
                "--sweep_config",
                str(sweep_config_path),
                "--metrics_root",
                str(METRICS_ROOT),
                "--joined_root",
                str(JOINED_ROOT),
                "--format",
                "csv",
            ],
        )
    finally:
        after_shared_config = SHARED_CONFIG_PATH.read_bytes()
        if after_shared_config != before_shared_config:
            SHARED_CONFIG_PATH.write_bytes(before_shared_config)
            ctx.summary.setdefault("mini_sweep", {})["shared_config_restored_by_harness"] = True
            ctx.write_summary()

    if SHARED_CONFIG_PATH.read_bytes() != before_shared_config:
        raise AssertionError("config/shared_config.yaml was not restored after sweep validation.")

    new_joined_dirs = _new_subdirs(JOINED_ROOT, before_joined)
    if len(new_joined_dirs) != 8:
        raise AssertionError(f"Expected exactly 8 new joined sweep dirs, got {len(new_joined_dirs)}")
    expected_run_names = [path.name for path in new_joined_dirs]

    if not COMBINED_OUTPUT.exists():
        raise FileNotFoundError(f"run_sweep did not write combined output: {COMBINED_OUTPUT}")
    combined_summary = _assert_sweep_combined(
        combined_path=COMBINED_OUTPUT,
        expected_run_names=expected_run_names,
    )

    combined_run_name = f"validation_sweep_combined_{ctx.validation_id}"
    combined_eval_csv_name = "eval_epoch_validation_sweep.csv"
    bridge_dir = JOINED_ROOT / combined_run_name
    bridge_dir.mkdir(parents=True, exist_ok=True)
    bridge_path = bridge_dir / combined_eval_csv_name
    shutil.copy2(COMBINED_OUTPUT, bridge_path)

    prefix = "mini_sweep"
    prepared_path = _run_preparation(
        ctx,
        run_name=combined_run_name,
        eval_csv_name=combined_eval_csv_name,
        prefix=prefix,
    )

    model_summaries = {}
    for model_id in MODELS:
        manifest_path = _run_model_notebook(
            ctx,
            model_id=model_id,
            run_name=combined_run_name,
            prefix=prefix,
        )
        inference_outputs = _run_model_inference(
            ctx,
            model_id=model_id,
            run_name=combined_run_name,
            prefix=prefix,
        )
        regime_root = (
            ROOT
            / "results"
            / "interpretable_model"
            / "feature_effect_performance_regimes"
            / model_id
            / combined_run_name
        )
        if regime_root.exists():
            raise AssertionError(
                f"Sweep validation should stop at model inference; found clustering root {regime_root}"
            )
        model_summaries[model_id] = {
            "manifest_path": manifest_path,
            "inference_outputs": inference_outputs,
        }

    return {
        "sweep_config_path": sweep_config_path,
        "new_joined_dirs": new_joined_dirs,
        "combined_output": COMBINED_OUTPUT,
        "combined_summary": combined_summary,
        "bridge_run_name": combined_run_name,
        "bridge_path": bridge_path,
        "prepared_path": prepared_path,
        "models": model_summaries,
        "shared_config_restored": SHARED_CONFIG_PATH.read_bytes() == before_shared_config,
    }


def _run_gates(ctx: ValidationContext) -> None:
    _run_command(
        ctx,
        "pytest tests",
        [sys.executable, "-m", "pytest", "tests"],
    )
    _run_command(
        ctx,
        "run_sweep dry run",
        [sys.executable, "run_sweep.py", "--dry_run"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run capped validation for the full trainval and mini sweep paths."
    )
    parser.add_argument("--skip-gates", action="store_true", help="Skip pytest and sweep dry-run gates.")
    parser.add_argument("--skip-full", action="store_true", help="Skip full trainval path validation.")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip mini sweep path validation.")
    parser.add_argument("--full-max-train-batches", type=int, default=5)
    parser.add_argument("--full-max-eval-batches", type=int, default=5)
    parser.add_argument("--full-batch-size", type=int, default=32)
    parser.add_argument("--sweep-max-train-batches", type=int, default=2)
    parser.add_argument("--sweep-max-eval-batches", type=int, default=2)
    parser.add_argument("--sweep-batch-size", type=int, default=32)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional validation artifact root. Defaults under ignored notebook_runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_id = _timestamp()
    output_root = (
        args.output_root
        if args.output_root is not None
        else ROOT / "results" / "interpretable_model" / "notebook_runs" / f"pipeline_validation_{validation_id}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    ctx = ValidationContext(
        validation_id=validation_id,
        output_root=output_root,
        log_dir=output_root / "logs",
        summary_path=output_root / "validation_summary.json",
        summary={
            "validation_id": validation_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "output_root": output_root,
        },
    )
    ctx.write_summary()

    try:
        _preflight(ctx)
        if not args.skip_gates:
            _run_gates(ctx)
        if not args.skip_full:
            ctx.summary["full_trainval"] = _run_full_trainval(ctx, args)
            ctx.write_summary()
        if not args.skip_sweep:
            ctx.summary["mini_sweep"] = {
                **ctx.summary.get("mini_sweep", {}),
                **_run_mini_sweep(ctx, args),
            }
            ctx.write_summary()
        ctx.summary["status"] = "passed"
        ctx.summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        ctx.write_summary()
        print(f"Validation passed. Summary: {ctx.summary_path}")
        return 0
    except Exception as exc:
        ctx.summary["status"] = "failed"
        ctx.summary["error"] = repr(exc)
        ctx.summary["traceback"] = traceback.format_exc()
        ctx.summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        ctx.write_summary()
        print(f"Validation failed. Summary: {ctx.summary_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
