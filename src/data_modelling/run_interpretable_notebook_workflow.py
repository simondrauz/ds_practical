from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "src" / "data_modelling"
NOTEBOOK_RUNS_ROOT = REPO_ROOT / "results" / "interpretable_model" / "notebook_runs"


def _python_literal(value: str | None) -> str:
    if value is None:
        return "None"
    return repr(value)


def _replace_assignment(source: str, name: str, value_literal: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=.*$")
    replacement = f"{name} = {value_literal}"
    updated, n_subs = pattern.subn(replacement, source, count=1)
    if n_subs != 1:
        raise ValueError(f"Expected exactly one assignment for {name!r}, found {n_subs}.")
    return updated


def _replace_first_literal(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Expected literal {old!r} to exist in notebook cell.")
    return source.replace(old, new, 1)


def _patch_notebook(notebook_path: Path, patchers: list[Callable[[str], str]]) -> nbformat.NotebookNode:
    notebook = nbformat.read(notebook_path, as_version=4)
    remaining_patchers = list(patchers)

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue

        source = cell["source"]
        applied_patchers: list[Callable[[str], str]] = []
        for patcher in remaining_patchers:
            try:
                updated = patcher(source)
            except ValueError:
                continue
            else:
                source = updated
                applied_patchers.append(patcher)

        if applied_patchers:
            cell["source"] = source
            remaining_patchers = [patcher for patcher in remaining_patchers if patcher not in applied_patchers]

        if not remaining_patchers:
            break

    if remaining_patchers:
        raise ValueError(
            f"Failed to apply {len(remaining_patchers)} notebook patch(es) for {notebook_path.name}."
        )

    return notebook


@dataclass(frozen=True)
class NotebookExecution:
    label: str
    notebook_path: Path
    patchers: list[Callable[[str], str]]
    output_name: str


def _build_workflow(
    *,
    run_name: str,
    eval_csv_name: str,
    prepared_target_col: str,
    target_col: str | None,
    include_model_settings_as_features: bool,
    include_gam: bool,
    include_xgboost: bool,
) -> list[NotebookExecution]:
    if not include_gam and not include_xgboost:
        raise ValueError("At least one model workflow must be selected.")

    workflow: list[NotebookExecution] = [
        NotebookExecution(
            label="interpretable_model_data_preparation",
            notebook_path=NOTEBOOK_DIR / "interpretable_model_data_preparation.ipynb",
            patchers=[
                lambda source: _replace_assignment(source, "RUN_NAME", _python_literal(run_name)),
                lambda source: _replace_assignment(source, "EVAL_CSV_NAME", _python_literal(eval_csv_name)),
                lambda source: _replace_assignment(
                    source,
                    "INCLUDE_MODEL_SETTINGS_AS_FEATURES",
                    repr(include_model_settings_as_features),
                ),
                lambda source: _replace_assignment(source, "target_col", _python_literal(prepared_target_col)),
            ],
            output_name="01_interpretable_model_data_preparation.ipynb",
        )
    ]

    if include_gam:
        workflow.extend(
            [
                NotebookExecution(
                    label="gam",
                    notebook_path=NOTEBOOK_DIR / "gam.ipynb",
                    patchers=[
                        lambda source: _replace_assignment(source, "RUN_NAME", _python_literal(run_name)),
                        lambda source: _replace_assignment(
                            source, "PREPARED_TARGET_COL", _python_literal(prepared_target_col)
                        ),
                        lambda source: _replace_assignment(source, "TARGET_COL", _python_literal(target_col)),
                    ],
                    output_name="02_gam.ipynb",
                ),
                NotebookExecution(
                    label="model_inference_analysis_gam",
                    notebook_path=NOTEBOOK_DIR / "model_inference_analysis.ipynb",
                    patchers=[
                        lambda source: _replace_assignment(source, "MODEL_ID", _python_literal("gam")),
                        lambda source: _replace_assignment(source, "RUN_NAME", _python_literal(run_name)),
                        lambda source: _replace_assignment(source, "TARGET_COL", _python_literal(target_col)),
                    ],
                    output_name="03_model_inference_analysis__gam.ipynb",
                ),
                NotebookExecution(
                    label="feature_effect_performance_regimes_gam",
                    notebook_path=NOTEBOOK_DIR / "feature_effect_performance_regimes.ipynb",
                    patchers=[
                        lambda source: _replace_assignment(source, "MODEL_ID", _python_literal("gam")),
                        lambda source: _replace_assignment(source, "RUN_NAME", _python_literal(run_name)),
                        lambda source: _replace_assignment(source, "EVAL_CSV_NAME", _python_literal(eval_csv_name)),
                        lambda source: _replace_assignment(source, "TARGET_COL", _python_literal(target_col)),
                    ],
                    output_name="04_feature_effect_performance_regimes__gam.ipynb",
                ),
            ]
        )

    if include_xgboost:
        workflow.extend(
            [
                NotebookExecution(
                    label="xgboost",
                    notebook_path=NOTEBOOK_DIR / "xgboost.ipynb",
                    patchers=[
                        lambda source: _replace_assignment(source, "RUN_NAME", _python_literal(run_name)),
                        lambda source: _replace_assignment(
                            source, "PREPARED_TARGET_COL", _python_literal(prepared_target_col)
                        ),
                        lambda source: _replace_assignment(source, "TARGET_COL", _python_literal(target_col)),
                    ],
                    output_name="05_xgboost.ipynb",
                ),
                NotebookExecution(
                    label="model_inference_analysis_xgboost",
                    notebook_path=NOTEBOOK_DIR / "model_inference_analysis.ipynb",
                    patchers=[
                        lambda source: _replace_assignment(source, "MODEL_ID", _python_literal("xgboost")),
                        lambda source: _replace_assignment(source, "RUN_NAME", _python_literal(run_name)),
                        lambda source: _replace_assignment(source, "TARGET_COL", _python_literal(target_col)),
                    ],
                    output_name="06_model_inference_analysis__xgboost.ipynb",
                ),
                NotebookExecution(
                    label="feature_effect_performance_regimes_xgboost",
                    notebook_path=NOTEBOOK_DIR / "feature_effect_performance_regimes.ipynb",
                    patchers=[
                        lambda source: _replace_assignment(source, "MODEL_ID", _python_literal("xgboost")),
                        lambda source: _replace_assignment(source, "RUN_NAME", _python_literal(run_name)),
                        lambda source: _replace_assignment(source, "EVAL_CSV_NAME", _python_literal(eval_csv_name)),
                        lambda source: _replace_assignment(source, "TARGET_COL", _python_literal(target_col)),
                    ],
                    output_name="07_feature_effect_performance_regimes__xgboost.ipynb",
                ),
            ]
        )

    return workflow


def _find_manifest_path(model_id: str, run_name: str) -> Path | None:
    tables_dir = REPO_ROOT / "results" / "interpretable_model" / model_id / run_name / "tables"
    manifest_paths = sorted(tables_dir.glob("run_manifest_*.json"))
    return manifest_paths[-1] if manifest_paths else None


def _find_latest_regime_manifest(model_id: str, run_name: str) -> Path | None:
    search_root = REPO_ROOT / "results" / "interpretable_model" / "feature_effect_performance_regimes" / model_id / run_name
    manifest_paths = list(search_root.glob("**/manifest.json"))
    if not manifest_paths:
        return None
    return max(manifest_paths, key=lambda path: path.stat().st_mtime)


def _write_summary(
    *,
    output_root: Path,
    run_name: str,
    eval_csv_name: str,
    prepared_target_col: str,
    target_col: str | None,
    include_model_settings_as_features: bool,
    include_gam: bool,
    include_xgboost: bool,
    executed_notebooks: list[dict[str, str]],
) -> Path:
    summary = {
        "run_name": run_name,
        "eval_csv_name": eval_csv_name,
        "prepared_target_col": prepared_target_col,
        "target_col_override": target_col,
        "include_model_settings_as_features": include_model_settings_as_features,
        "executed_notebooks": executed_notebooks,
        "artifacts": {
            "prepared_data_path": str(
                REPO_ROOT
                / "results"
                / "interpretable_model"
                / "prepared_data"
                / run_name
                / f"prepared_data_{prepared_target_col}.csv"
            ),
            "gam_manifest_path": str(_find_manifest_path("gam", run_name)) if include_gam else None,
            "xgboost_manifest_path": str(_find_manifest_path("xgboost", run_name)) if include_xgboost else None,
            "gam_regime_manifest_path": str(_find_latest_regime_manifest("gam", run_name)) if include_gam else None,
            "xgboost_regime_manifest_path": str(_find_latest_regime_manifest("xgboost", run_name))
            if include_xgboost
            else None,
        },
    }

    summary_path = output_root / "workflow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the interpretable-model notebook workflow against a specific run/epoch."
    )
    parser.add_argument("--run-name", required=True, help="Trajectory-metrics run name.")
    parser.add_argument("--eval-csv-name", required=True, help="Joined metrics CSV filename, e.g. eval_epoch_12.csv.")
    parser.add_argument(
        "--prepared-target-col",
        default="ml_ade",
        help="Raw target column used by the preparation notebook export.",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Optional target override for model/inference notebooks, e.g. ml_ade_log.",
    )
    model_setting_group = parser.add_mutually_exclusive_group(required=True)
    model_setting_group.add_argument(
        "--include-model-settings-as-features",
        dest="include_model_settings_as_features",
        action="store_true",
        help="Fit model-setting columns as predictors in GAM/XGBoost.",
    )
    model_setting_group.add_argument(
        "--exclude-model-settings-as-features",
        dest="include_model_settings_as_features",
        action="store_false",
        help="Exclude model-setting columns from the prepared GAM/XGBoost model frame.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gam", "xgboost"],
        default=["gam", "xgboost"],
        help="Which model workflows to execute.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional directory for executed notebook copies and the workflow summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    include_gam = "gam" in args.models
    include_xgboost = "xgboost" in args.models

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else NOTEBOOK_RUNS_ROOT / args.run_name / timestamp
    )
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")

    workflow = _build_workflow(
        run_name=args.run_name,
        eval_csv_name=args.eval_csv_name,
        prepared_target_col=args.prepared_target_col,
        target_col=args.target_col,
        include_model_settings_as_features=args.include_model_settings_as_features,
        include_gam=include_gam,
        include_xgboost=include_xgboost,
    )

    executed_notebooks: list[dict[str, str]] = []
    executor = ExecutePreprocessor(timeout=None, kernel_name="python3")

    for index, execution in enumerate(workflow, start=1):
        print(f"[{index}/{len(workflow)}] Executing {execution.label}...")
        patched_notebook = _patch_notebook(execution.notebook_path, execution.patchers)

        output_path = output_root / execution.output_name
        nbformat.write(patched_notebook, output_path)
        executor.preprocess(patched_notebook, {"metadata": {"path": str(NOTEBOOK_DIR)}})
        nbformat.write(patched_notebook, output_path)

        executed_notebooks.append(
            {
                "label": execution.label,
                "source_notebook": str(execution.notebook_path),
                "executed_notebook": str(output_path),
            }
        )
        print(f"    wrote executed notebook to {output_path}")

    summary_path = _write_summary(
        output_root=output_root,
        run_name=args.run_name,
        eval_csv_name=args.eval_csv_name,
        prepared_target_col=args.prepared_target_col,
        target_col=args.target_col,
        include_model_settings_as_features=args.include_model_settings_as_features,
        include_gam=include_gam,
        include_xgboost=include_xgboost,
        executed_notebooks=executed_notebooks,
    )
    print(f"Workflow summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
