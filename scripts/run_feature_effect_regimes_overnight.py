from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
UNIFIED_LOADER_SRC = REPO_ROOT / "unified-av-data-loader" / "src"
NOTEBOOK_DIR = SRC_ROOT / "data_modelling"
NOTEBOOK_RUNS_ROOT = REPO_ROOT / "results" / "interpretable_model" / "notebook_runs"
MODEL_INFERENCE_NOTEBOOK = NOTEBOOK_DIR / "model_inference_analysis.ipynb"
REGIME_NOTEBOOK = NOTEBOOK_DIR / "feature_effect_performance_regimes.ipynb"
DEFAULT_RUN_NAME = "full_trainval_12ep_1seed_vif_only_no_collision"
DEFAULT_EVAL_CSV_NAME = "eval_epoch_12.csv"
DEFAULT_MODELS = ("xgboost", "gam")
VALID_MODELS = ("xgboost", "gam")
VALID_INFERENCE_MODES = ("missing", "always", "never")
VALID_CLUSTER_SWEEP_MODES = (
    "promising_small",
    "promising_moderate",
    "promising_extended",
    "holistic_large",
)


def _ensure_import_paths() -> None:
    for path in [SRC_ROOT, UNIFIED_LOADER_SRC]:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_import_paths()

from data_modelling.run_context import load_run_context  # noqa: E402


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "na"


def _python_literal(value: str | None) -> str:
    if value is None:
        return "None"
    return repr(value)


def _replace_assignment(source: str, name: str, value_literal: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=.*$")
    updated, n_subs = pattern.subn(f"{name} = {value_literal}", source, count=1)
    if n_subs != 1:
        raise ValueError(f"Expected exactly one assignment for {name!r}, found {n_subs}.")
    return updated


def _patch_notebook(
    notebook_path: Path,
    patchers: list[Callable[[str], str]],
) -> nbformat.NotebookNode:
    notebook = nbformat.read(notebook_path, as_version=4)
    remaining_patchers = list(patchers)

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        applied_patchers: list[Callable[[str], str]] = []
        for patcher in remaining_patchers:
            try:
                source = patcher(source)
            except ValueError:
                continue
            applied_patchers.append(patcher)
        if applied_patchers:
            cell["source"] = source
            remaining_patchers = [
                patcher for patcher in remaining_patchers if patcher not in applied_patchers
            ]
        if not remaining_patchers:
            break

    if remaining_patchers:
        raise RuntimeError(
            f"Failed to apply {len(remaining_patchers)} notebook patch(es) to {notebook_path}."
        )
    return notebook


@dataclass(frozen=True)
class Job:
    model_id: str
    run_name: str


@dataclass
class StepRecord:
    label: str
    source_notebook: str
    executed_notebook: str
    status: str
    started_at: str
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


@dataclass
class JobRecord:
    model_id: str
    run_name: str
    target_col: str | None
    feature_effects_path: str | None = None
    feature_effect_importance_path: str | None = None
    steps: list[StepRecord] = field(default_factory=list)


def _configure_environment() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    path_parts = [
        str(SRC_ROOT),
        str(UNIFIED_LOADER_SRC),
        *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep)),
    ]
    deduped_parts = list(dict.fromkeys(path_parts))
    os.environ["PYTHONPATH"] = os.pathsep.join(deduped_parts)


def _resolve_feature_effect_paths(
    *,
    model_id: str,
    run_name: str,
    target_col: str | None,
) -> tuple[str, Path, Path]:
    run_ctx = load_run_context(model_id, run_name, target_col)
    resolved_target_col = str(run_ctx.target_col)
    feature_effects_path = run_ctx.tables_dir / f"feature_effects_{resolved_target_col}.csv"
    importance_path = run_ctx.tables_dir / f"feature_effect_importance_{resolved_target_col}.csv"
    return resolved_target_col, feature_effects_path, importance_path


def _feature_effects_exist(
    *,
    model_id: str,
    run_name: str,
    target_col: str | None,
) -> tuple[bool, str, Path, Path]:
    resolved_target_col, feature_effects_path, importance_path = _resolve_feature_effect_paths(
        model_id=model_id,
        run_name=run_name,
        target_col=target_col,
    )
    return (
        feature_effects_path.exists() and importance_path.exists(),
        resolved_target_col,
        feature_effects_path,
        importance_path,
    )


def _execute_notebook(
    *,
    notebook_path: Path,
    patchers: list[Callable[[str], str]],
    output_path: Path,
    kernel_name: str,
    label: str,
) -> StepRecord:
    started = time.time()
    record = StepRecord(
        label=label,
        source_notebook=str(notebook_path),
        executed_notebook=str(output_path),
        status="running",
        started_at=_timestamp(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = _patch_notebook(notebook_path, patchers)
    nbformat.write(notebook, output_path)

    print(f"[{_timestamp()}] Starting {label}; output={output_path}", flush=True)
    try:
        executor = ExecutePreprocessor(timeout=None, kernel_name=kernel_name)
        executor.preprocess(notebook, {"metadata": {"path": str(NOTEBOOK_DIR)}})
    except Exception as exc:
        record.status = "failed"
        record.error = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        raise
    else:
        record.status = "completed"
    finally:
        nbformat.write(notebook, output_path)
        record.completed_at = _timestamp()
        record.duration_seconds = round(time.time() - started, 3)
        print(
            f"[{_timestamp()}] {label} {record.status} "
            f"in {record.duration_seconds:.1f}s",
            flush=True,
        )

    return record


def _model_inference_patchers(
    *,
    model_id: str,
    run_name: str,
    target_col: str | None,
) -> list[Callable[[str], str]]:
    return [
        lambda source: _replace_assignment(source, "MODEL_ID", _python_literal(model_id)),
        lambda source: _replace_assignment(source, "RUN_NAME", _python_literal(run_name)),
        lambda source: _replace_assignment(source, "TARGET_COL", _python_literal(target_col)),
    ]


def _regime_patchers(
    *,
    model_id: str,
    run_name: str,
    eval_csv_name: str,
    target_col: str | None,
    cluster_sweep_mode: str,
) -> list[Callable[[str], str]]:
    return [
        lambda source: _replace_assignment(source, "MODEL_ID", _python_literal(model_id)),
        lambda source: _replace_assignment(source, "RUN_NAME", _python_literal(run_name)),
        lambda source: _replace_assignment(source, "EVAL_CSV_NAME", _python_literal(eval_csv_name)),
        lambda source: _replace_assignment(source, "TARGET_COL", _python_literal(target_col)),
        lambda source: _replace_assignment(
            source,
            "CLUSTER_SWEEP_MODE",
            _python_literal(cluster_sweep_mode),
        ),
    ]


def _write_summary(
    *,
    summary_path: Path,
    args: argparse.Namespace,
    status: str,
    jobs: list[JobRecord],
) -> None:
    summary = {
        "status": status,
        "updated_at": _timestamp(),
        "run_name": args.run_name,
        "eval_csv_name": args.eval_csv_name,
        "target_col_override": args.target_col,
        "models": args.models,
        "cluster_sweep_mode": args.cluster_sweep_mode,
        "inference_mode": args.inference_mode,
        "kernel_name": args.kernel_name,
        "jobs": [
            {
                "model_id": job.model_id,
                "run_name": job.run_name,
                "target_col": job.target_col,
                "feature_effects_path": job.feature_effects_path,
                "feature_effect_importance_path": job.feature_effect_importance_path,
                "steps": [step.__dict__ for step in job.steps],
            }
            for job in jobs
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2))


def _should_run_inference(
    *,
    inference_mode: str,
    feature_effects_present: bool,
) -> bool:
    if inference_mode == "always":
        return True
    if inference_mode == "never":
        return False
    if inference_mode == "missing":
        return not feature_effects_present
    raise ValueError(f"Unsupported inference mode: {inference_mode!r}")


def _run_job(
    *,
    job: Job,
    args: argparse.Namespace,
    output_root: Path,
) -> JobRecord:
    feature_effects_present, resolved_target_col, feature_effects_path, importance_path = (
        _feature_effects_exist(
            model_id=job.model_id,
            run_name=job.run_name,
            target_col=args.target_col,
        )
    )
    job_record = JobRecord(
        model_id=job.model_id,
        run_name=job.run_name,
        target_col=resolved_target_col,
        feature_effects_path=str(feature_effects_path),
        feature_effect_importance_path=str(importance_path),
    )

    job_label = f"{job.model_id}__{_safe_slug(job.run_name)}"
    job_output_root = output_root / job_label
    job_output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[{_timestamp()}] Job {job.model_id}/{job.run_name}: "
        f"feature effects present={feature_effects_present}",
        flush=True,
    )

    if _should_run_inference(
        inference_mode=args.inference_mode,
        feature_effects_present=feature_effects_present,
    ):
        inference_record = _execute_notebook(
            notebook_path=MODEL_INFERENCE_NOTEBOOK,
            patchers=_model_inference_patchers(
                model_id=job.model_id,
                run_name=job.run_name,
                target_col=args.target_col,
            ),
            output_path=job_output_root / f"01_model_inference_analysis__{job.model_id}.ipynb",
            kernel_name=args.kernel_name,
            label=f"model_inference_analysis[{job.model_id}]",
        )
        job_record.steps.append(inference_record)

        feature_effects_present, resolved_target_col, feature_effects_path, importance_path = (
            _feature_effects_exist(
                model_id=job.model_id,
                run_name=job.run_name,
                target_col=args.target_col,
            )
        )
        job_record.target_col = resolved_target_col
        job_record.feature_effects_path = str(feature_effects_path)
        job_record.feature_effect_importance_path = str(importance_path)

    if not feature_effects_present:
        raise FileNotFoundError(
            "Feature-effect exports are missing and inference was not run or did not produce them: "
            f"{feature_effects_path}, {importance_path}. "
            "Run with --inference-mode missing or --inference-mode always."
        )

    regime_record = _execute_notebook(
        notebook_path=REGIME_NOTEBOOK,
        patchers=_regime_patchers(
            model_id=job.model_id,
            run_name=job.run_name,
            eval_csv_name=args.eval_csv_name,
            target_col=args.target_col,
            cluster_sweep_mode=args.cluster_sweep_mode,
        ),
        output_path=(
            job_output_root
            / f"02_feature_effect_performance_regimes__{job.model_id}__{args.cluster_sweep_mode}.ipynb"
        ),
        kernel_name=args.kernel_name,
        label=f"feature_effect_performance_regimes[{job.model_id}:{args.cluster_sweep_mode}]",
    )
    job_record.steps.append(regime_record)
    return job_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run model-inference feature effects when needed, then execute "
            "feature_effect_performance_regimes.ipynb for the selected models."
        )
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--eval-csv-name", default=DEFAULT_EVAL_CSV_NAME)
    parser.add_argument("--target-col", default=None, help="Optional target override, e.g. ml_ade_log.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=VALID_MODELS,
        default=list(DEFAULT_MODELS),
        help="Models to run sequentially.",
    )
    parser.add_argument(
        "--cluster-sweep-mode",
        choices=VALID_CLUSTER_SWEEP_MODES,
        default="promising_moderate",
        help="Runtime sweep mode passed into feature_effect_performance_regimes.ipynb.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=VALID_INFERENCE_MODES,
        default="missing",
        help=(
            "'missing' runs model_inference_analysis.ipynb only when feature-effect "
            "exports are absent; 'always' regenerates them; 'never' requires them."
        ),
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help=(
            "Jupyter kernel name to use. When launched via "
            "`conda run -n adaptive-py310 python ...`, the default python3 kernel "
            "is the adaptive-py310 kernel used by these notebooks."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for executed notebook copies, logs, and summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_environment()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else NOTEBOOK_RUNS_ROOT
        / "overnight_feature_effect_regimes"
        / f"{timestamp}__{_safe_slug(args.run_name)}__{args.cluster_sweep_mode}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "overnight_regime_summary.json"

    jobs = [Job(model_id=model_id, run_name=args.run_name) for model_id in args.models]
    job_records: list[JobRecord] = []

    print(f"[{_timestamp()}] Output root: {output_root}", flush=True)
    print(
        f"[{_timestamp()}] Running models={args.models}, "
        f"cluster_sweep_mode={args.cluster_sweep_mode}, inference_mode={args.inference_mode}, "
        f"kernel_name={args.kernel_name}",
        flush=True,
    )

    status = "completed"
    try:
        for job in jobs:
            job_record = _run_job(job=job, args=args, output_root=output_root)
            job_records.append(job_record)
            _write_summary(
                summary_path=summary_path,
                args=args,
                status="running",
                jobs=job_records,
            )
    except Exception:
        status = "failed"
        _write_summary(
            summary_path=summary_path,
            args=args,
            status=status,
            jobs=job_records,
        )
        print(f"[{_timestamp()}] Failed; summary={summary_path}", flush=True)
        raise
    else:
        _write_summary(
            summary_path=summary_path,
            args=args,
            status=status,
            jobs=job_records,
        )
        print(f"[{_timestamp()}] Completed; summary={summary_path}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
