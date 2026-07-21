from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_feature_effect_regimes_overnight import (
    DEFAULT_EVAL_CSV_NAME,
    DEFAULT_MODELS,
    DEFAULT_RUN_NAME,
    NOTEBOOK_RUNS_ROOT,
    VALID_CLUSTER_SWEEP_MODES,
    VALID_MODELS,
    _configure_environment,
    _safe_slug,
    _timestamp,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_INFERENCE_SCRIPT = REPO_ROOT / "scripts" / "run_model_inference_overnight.py"
REGIME_SCRIPT = REPO_ROOT / "scripts" / "run_feature_effect_regimes_overnight.py"


def _command_env() -> dict[str, str]:
    _configure_environment()
    return os.environ.copy()


def _run_stage(command: list[str], *, label: str, env: dict[str, str]) -> None:
    print(f"[{_timestamp()}] Starting stage {label}", flush=True)
    print(f"[{_timestamp()}] Command: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    print(f"[{_timestamp()}] Completed stage {label}", flush=True)


def _write_summary(
    *,
    path: Path,
    args: argparse.Namespace,
    status: str,
    inference_output_root: Path,
    regime_output_root: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "updated_at": _timestamp(),
                "run_name": args.run_name,
                "eval_csv_name": args.eval_csv_name,
                "target_col_override": args.target_col,
                "models": args.models,
                "cluster_sweep_mode": args.cluster_sweep_mode,
                "inference_mode": args.inference_mode,
                "kernel_name": args.kernel_name,
                "inference_output_root": str(inference_output_root),
                "regime_output_root": str(regime_output_root),
                "inference_summary": str(inference_output_root / "model_inference_summary.json"),
                "regime_summary": str(regime_output_root / "overnight_regime_summary.json"),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run model_inference_analysis.ipynb first, then "
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
        help="Models to run sequentially in both stages.",
    )
    parser.add_argument(
        "--cluster-sweep-mode",
        choices=VALID_CLUSTER_SWEEP_MODES,
        default="promising_moderate",
        help="Runtime sweep mode passed into feature_effect_performance_regimes.ipynb.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=("missing", "always"),
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
        help="Parent directory for stage outputs and pipeline summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = _command_env()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else NOTEBOOK_RUNS_ROOT
        / "overnight_feature_effect_pipeline"
        / f"{timestamp}__{_safe_slug(args.run_name)}__{args.cluster_sweep_mode}"
    )
    inference_output_root = output_root / "01_model_inference"
    regime_output_root = output_root / "02_feature_effect_regimes"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "pipeline_summary.json"

    _write_summary(
        path=summary_path,
        args=args,
        status="running",
        inference_output_root=inference_output_root,
        regime_output_root=regime_output_root,
    )

    inference_command = [
        sys.executable,
        str(MODEL_INFERENCE_SCRIPT),
        "--run-name",
        args.run_name,
        "--models",
        *args.models,
        "--inference-mode",
        args.inference_mode,
        "--kernel-name",
        args.kernel_name,
        "--output-root",
        str(inference_output_root),
    ]
    if args.target_col is not None:
        inference_command.extend(["--target-col", args.target_col])

    regime_command = [
        sys.executable,
        str(REGIME_SCRIPT),
        "--run-name",
        args.run_name,
        "--eval-csv-name",
        args.eval_csv_name,
        "--models",
        *args.models,
        "--cluster-sweep-mode",
        args.cluster_sweep_mode,
        "--inference-mode",
        "never",
        "--kernel-name",
        args.kernel_name,
        "--output-root",
        str(regime_output_root),
    ]
    if args.target_col is not None:
        regime_command.extend(["--target-col", args.target_col])

    try:
        _run_stage(inference_command, label="model_inference", env=env)
        _run_stage(regime_command, label="feature_effect_regimes", env=env)
    except Exception:
        _write_summary(
            path=summary_path,
            args=args,
            status="failed",
            inference_output_root=inference_output_root,
            regime_output_root=regime_output_root,
        )
        print(f"[{_timestamp()}] Pipeline failed; summary={summary_path}", flush=True)
        raise

    _write_summary(
        path=summary_path,
        args=args,
        status="completed",
        inference_output_root=inference_output_root,
        regime_output_root=regime_output_root,
    )
    print(f"[{_timestamp()}] Pipeline completed; summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
