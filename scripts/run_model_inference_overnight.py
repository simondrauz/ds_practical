from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_feature_effect_regimes_overnight import (
    DEFAULT_MODELS,
    DEFAULT_RUN_NAME,
    MODEL_INFERENCE_NOTEBOOK,
    NOTEBOOK_RUNS_ROOT,
    VALID_MODELS,
    _configure_environment,
    _execute_notebook,
    _feature_effects_exist,
    _model_inference_patchers,
    _safe_slug,
    _timestamp,
)


VALID_INFERENCE_MODES = ("missing", "always")


@dataclass
class InferenceJobRecord:
    model_id: str
    run_name: str
    target_col: str | None
    status: str
    feature_effects_path: str
    feature_effect_importance_path: str
    executed_notebook: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


@dataclass
class InferenceSummary:
    status: str
    updated_at: str
    run_name: str
    target_col_override: str | None
    models: list[str]
    inference_mode: str
    kernel_name: str
    jobs: list[InferenceJobRecord] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "updated_at": self.updated_at,
            "run_name": self.run_name,
            "target_col_override": self.target_col_override,
            "models": self.models,
            "inference_mode": self.inference_mode,
            "kernel_name": self.kernel_name,
            "jobs": [job.__dict__ for job in self.jobs],
        }


def _write_summary(path: Path, summary: InferenceSummary) -> None:
    summary.updated_at = _timestamp()
    path.write_text(json.dumps(summary.to_json_dict(), indent=2))


def _run_inference_job(
    *,
    model_id: str,
    args: argparse.Namespace,
    output_root: Path,
) -> InferenceJobRecord:
    feature_effects_present, resolved_target_col, feature_effects_path, importance_path = (
        _feature_effects_exist(
            model_id=model_id,
            run_name=args.run_name,
            target_col=args.target_col,
        )
    )

    job_record = InferenceJobRecord(
        model_id=model_id,
        run_name=args.run_name,
        target_col=resolved_target_col,
        status="skipped" if feature_effects_present and args.inference_mode == "missing" else "pending",
        feature_effects_path=str(feature_effects_path),
        feature_effect_importance_path=str(importance_path),
    )

    if feature_effects_present and args.inference_mode == "missing":
        print(
            f"[{_timestamp()}] Skipping model_inference_analysis[{model_id}]; "
            "feature-effect exports already exist.",
            flush=True,
        )
        return job_record

    started = time.time()
    output_path = (
        output_root
        / f"{model_id}__{_safe_slug(args.run_name)}"
        / f"01_model_inference_analysis__{model_id}.ipynb"
    )
    job_record.started_at = _timestamp()
    job_record.executed_notebook = str(output_path)
    try:
        step_record = _execute_notebook(
            notebook_path=MODEL_INFERENCE_NOTEBOOK,
            patchers=_model_inference_patchers(
                model_id=model_id,
                run_name=args.run_name,
                target_col=args.target_col,
            ),
            output_path=output_path,
            kernel_name=args.kernel_name,
            label=f"model_inference_analysis[{model_id}]",
        )
    except Exception as exc:
        job_record.status = "failed"
        job_record.error = f"{type(exc).__name__}: {exc}"
        raise
    else:
        job_record.status = step_record.status
        job_record.completed_at = step_record.completed_at
        job_record.duration_seconds = step_record.duration_seconds

    feature_effects_present, resolved_target_col, feature_effects_path, importance_path = (
        _feature_effects_exist(
            model_id=model_id,
            run_name=args.run_name,
            target_col=args.target_col,
        )
    )
    job_record.target_col = resolved_target_col
    job_record.feature_effects_path = str(feature_effects_path)
    job_record.feature_effect_importance_path = str(importance_path)
    job_record.duration_seconds = job_record.duration_seconds or round(time.time() - started, 3)
    if not feature_effects_present:
        raise FileNotFoundError(
            "model_inference_analysis.ipynb completed, but feature-effect exports are missing: "
            f"{feature_effects_path}, {importance_path}"
        )
    return job_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model_inference_analysis.ipynb for one or more model/run combinations."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--target-col", default=None, help="Optional target override, e.g. ml_ade_log.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=VALID_MODELS,
        default=list(DEFAULT_MODELS),
        help="Models to run sequentially.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=VALID_INFERENCE_MODES,
        default="missing",
        help="'missing' skips existing feature-effect exports; 'always' regenerates them.",
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help=(
            "Jupyter kernel name to use. When launched via "
            "`conda run -n adaptive-py310 python ...`, python3 resolves to that env."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for executed notebook copies and inference summary JSON.",
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
        / "overnight_model_inference"
        / f"{timestamp}__{_safe_slug(args.run_name)}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "model_inference_summary.json"

    summary = InferenceSummary(
        status="running",
        updated_at=_timestamp(),
        run_name=args.run_name,
        target_col_override=args.target_col,
        models=args.models,
        inference_mode=args.inference_mode,
        kernel_name=args.kernel_name,
    )
    _write_summary(summary_path, summary)

    print(f"[{_timestamp()}] Output root: {output_root}", flush=True)
    print(
        f"[{_timestamp()}] Running model inference for models={args.models}, "
        f"inference_mode={args.inference_mode}, kernel_name={args.kernel_name}",
        flush=True,
    )

    try:
        for model_id in args.models:
            job_record = _run_inference_job(
                model_id=model_id,
                args=args,
                output_root=output_root,
            )
            summary.jobs.append(job_record)
            _write_summary(summary_path, summary)
    except Exception:
        summary.status = "failed"
        _write_summary(summary_path, summary)
        print(f"[{_timestamp()}] Failed; summary={summary_path}", flush=True)
        raise

    summary.status = "completed"
    _write_summary(summary_path, summary)
    print(f"[{_timestamp()}] Completed; summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
