from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


REQUIRED_ARTIFACTS = {
    "regime_analysis",
    "performance_group_summary",
    "cluster_scores",
    "cluster_scores_all_candidates",
    "cluster_assignments",
    "cluster_feature_effect_profiles",
    "cluster_catalog",
    "feature_effect_global_ranking",
    "candidate_score_heatmap_grid",
    "algorithm_candidate_umap",
    "algorithm_candidate_umap_no_noise",
    "optics_reachability_grid",
}

EMPIRICAL_OPTICS_XI_VALUES = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
EMPIRICAL_OPTICS_EPS_QUANTILES = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]
EMPIRICAL_GRID_BY_MODEL = {
    "gam": {
        "min_cluster_size_fractions": [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04],
        "min_cluster_size_floor": 20,
        "min_samples_values": [3, 5, 8, 10, 15, 20, 30, 40],
    },
    "xgboost": {
        "min_cluster_size_fractions": [0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03],
        "min_cluster_size_floor": 10,
        "min_samples_values": [2, 3, 5, 8, 10, 15, 20, 30],
    },
}


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
    *,
    notebook_path: Path,
    model_id: str,
    run_name: str,
    eval_csv_name: str,
    target_col: str | None,
    cluster_sweep_mode: str,
) -> Any:
    notebook = nbformat.read(notebook_path, as_version=4)
    replacements = [
        ("MODEL_ID", _python_literal(model_id)),
        ("RUN_NAME", _python_literal(run_name)),
        ("EVAL_CSV_NAME", _python_literal(eval_csv_name)),
        ("TARGET_COL", _python_literal(target_col)),
        ("CLUSTER_SWEEP_MODE", _python_literal(cluster_sweep_mode)),
    ]
    remaining = list(replacements)

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        applied = []
        for name, literal in remaining:
            try:
                source = _replace_assignment(source, name, literal)
            except ValueError:
                continue
            applied.append((name, literal))
        if applied:
            cell["source"] = source
            remaining = [item for item in remaining if item not in applied]
        if not remaining:
            break

    if remaining:
        raise RuntimeError(f"Could not patch notebook assignments: {[name for name, _ in remaining]}")
    return notebook


def _float_lists_equal(actual: Any, expected: list[float]) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    return all(abs(float(left) - float(right)) < 1e-12 for left, right in zip(actual, expected))


def _cluster_spec_matches_mode(data: dict[str, Any], *, model_id: str, cluster_sweep_mode: str) -> bool:
    if cluster_sweep_mode != "empirical":
        return True

    cluster_spec = data.get("cluster_spec", {})
    resolved = cluster_spec.get("resolved", {}) if isinstance(cluster_spec, dict) else {}
    empirical_grid = EMPIRICAL_GRID_BY_MODEL[model_id]

    return (
        resolved.get("effect_representations") == ["raw"]
        and resolved.get("distance_metrics") == {"hdbscan": ["euclidean"], "optics": ["euclidean"]}
        and resolved.get("optics_extraction_methods") == ["xi", "dbscan_eps"]
        and _float_lists_equal(resolved.get("optics_xi_values"), EMPIRICAL_OPTICS_XI_VALUES)
        and _float_lists_equal(resolved.get("optics_eps_quantiles"), EMPIRICAL_OPTICS_EPS_QUANTILES)
        and _float_lists_equal(
            resolved.get("min_cluster_size_fractions"),
            empirical_grid["min_cluster_size_fractions"],
        )
        and int(resolved.get("min_cluster_size_floor", -1)) == empirical_grid["min_cluster_size_floor"]
        and resolved.get("min_samples_values") == empirical_grid["min_samples_values"]
    )


def _manifest_complete(
    path: Path,
    *,
    started_after: float,
    model_id: str,
    cluster_sweep_mode: str,
) -> bool:
    try:
        if path.stat().st_mtime < started_after:
            return False
        data = json.loads(path.read_text())
    except Exception:
        return False

    if data.get("run_context", {}).get("model_id") != model_id:
        return False
    if not _cluster_spec_matches_mode(
        data,
        model_id=model_id,
        cluster_sweep_mode=cluster_sweep_mode,
    ):
        return False

    artifact_types = {
        artifact.get("artifact_type")
        for artifact in data.get("artifacts", [])
        if isinstance(artifact, dict)
    }
    return REQUIRED_ARTIFACTS <= artifact_types


def _latest_complete_manifest(
    search_root: Path,
    *,
    started_after: float,
    model_id: str,
    cluster_sweep_mode: str,
) -> Path | None:
    if not search_root.exists():
        return None
    manifests = [
        path
        for path in search_root.glob("cluster_spec__*/manifest.json")
        if _manifest_complete(
            path,
            started_after=started_after,
            model_id=model_id,
            cluster_sweep_mode=cluster_sweep_mode,
        )
    ]
    if not manifests:
        return None
    return max(manifests, key=lambda path: path.stat().st_mtime)


def _configure_pythonpath(repo_root: Path) -> None:
    path_entries = [
        str(repo_root / "src"),
        str(repo_root / "unified-av-data-loader" / "src"),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        path_entries.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(path_entries)


def _run_xgboost_notebook(
    *,
    repo_root: Path,
    run_name: str,
    eval_csv_name: str,
    target_col: str | None,
    trigger_manifest: Path,
    kernel_name: str,
    cluster_sweep_mode: str,
) -> Path:
    notebook_dir = repo_root / "src" / "data_modelling"
    notebook_path = notebook_dir / "feature_effect_performance_regimes.ipynb"
    output_root = (
        repo_root
        / "results"
        / "interpretable_model"
        / "notebook_runs"
        / run_name
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_xgboost_after_gam"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "feature_effect_performance_regimes__xgboost_after_gam.ipynb"
    summary_path = output_root / "xgboost_after_gam_summary.json"

    notebook = _patch_notebook(
        notebook_path=notebook_path,
        model_id="xgboost",
        run_name=run_name,
        eval_csv_name=eval_csv_name,
        target_col=target_col,
        cluster_sweep_mode=cluster_sweep_mode,
    )
    nbformat.write(notebook, output_path)

    _configure_pythonpath(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    executor = ExecutePreprocessor(timeout=None, kernel_name=kernel_name)
    executor.preprocess(notebook, {"metadata": {"path": str(notebook_dir)}})
    nbformat.write(notebook, output_path)

    summary_path.write_text(
        json.dumps(
            {
                "trigger_manifest": str(trigger_manifest),
                "executed_notebook": str(output_path),
                "cluster_sweep_mode": cluster_sweep_mode,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a completed GAM feature-effect regime run, then execute the XGBoost regime notebook."
    )
    parser.add_argument("--run-name", default="full_trainval_12ep_1seed")
    parser.add_argument("--eval-csv-name", default="eval_epoch_12.csv")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--target-root", default="ml_ade_log")
    parser.add_argument("--performance-group-col", default="performance_group")
    parser.add_argument("--cluster-sweep-mode", default="empirical")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=168.0)
    parser.add_argument(
        "--trigger-grace-seconds",
        type=int,
        default=900,
        help="Also accept matching GAM manifests completed shortly before this watcher starts.",
    )
    parser.add_argument("--kernel-name", default="python3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    started_after = time.time() - int(args.trigger_grace_seconds)
    data_context_slug = (
        f"target-{args.target_root}__eval-{args.eval_csv_name}"
        f"__lower-is-better-true__group-col-{args.performance_group_col}"
    )
    gam_context_root = (
        repo_root
        / "results"
        / "interpretable_model"
        / "feature_effect_performance_regimes"
        / "gam"
        / args.run_name
        / args.target_root
        / data_context_slug
    )
    timeout_seconds = args.timeout_hours * 60 * 60

    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] "
        f"Watching for completed GAM {args.cluster_sweep_mode!r} manifest under {gam_context_root}",
        flush=True,
    )
    while True:
        manifest_path = _latest_complete_manifest(
            gam_context_root,
            started_after=started_after,
            model_id="gam",
            cluster_sweep_mode=args.cluster_sweep_mode,
        )
        if manifest_path is not None:
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"GAM complete: {manifest_path}",
                flush=True,
            )
            break

        elapsed = time.time() - started_after - int(args.trigger_grace_seconds)
        if elapsed > timeout_seconds:
            raise TimeoutError("Timed out waiting for GAM feature-effect regime manifest.")
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"GAM not complete yet; sleeping {args.poll_seconds}s",
            flush=True,
        )
        time.sleep(args.poll_seconds)

    output_path = _run_xgboost_notebook(
        repo_root=repo_root,
        run_name=args.run_name,
        eval_csv_name=args.eval_csv_name,
        target_col=args.target_col,
        trigger_manifest=manifest_path,
        kernel_name=args.kernel_name,
        cluster_sweep_mode=args.cluster_sweep_mode,
    )
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] XGBoost notebook completed: {output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
