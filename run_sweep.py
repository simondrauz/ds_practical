"""Hyperparameter sweep runner for Trajectron++.

Runs training + metric-joining for every combination defined in
``config/sweep_config.yaml``, then combines all results into one file.

Attention radius is varied by writing a scaled copy of ``shared_config.yaml``
before each training run and restoring the original afterwards (runs are
sequential so this is safe). ``history_sec`` and ``prediction_sec`` are passed
directly as CLI arguments to ``train_unified.py``.

Usage
-----
    python run_sweep.py                                  # use default sweep_config.yaml
    python run_sweep.py --sweep_config config/my_sweep.yaml
    python run_sweep.py --dry_run                        # print commands without running
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent
SHARED_CONFIG_PATH = ROOT / "config" / "shared_config.yaml"
DEFAULT_SWEEP_CONFIG = ROOT / "config" / "sweep_config.yaml"
DEFAULT_LOG_DIR = ROOT / "results" / "trajectory_prediction" / "logs"
DEFAULT_METRICS_ROOT = ROOT / "results" / "trajectory_prediction" / "trajectory_metrics"
DEFAULT_JOINED_ROOT = ROOT / "results" / "trajectory_prediction" / "trajectory_metrics_joined"

# PYTHONPATH entry needed to import src packages (data_preparation, shared_config, etc.)
SRC_PATH = str(ROOT / "src")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(data: Dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _scale_attention_radii(shared_cfg: Dict, scale: float) -> Dict:
    """Returns a deep copy of shared_cfg with all attention radii multiplied by scale."""
    cfg = copy.deepcopy(shared_cfg)
    attn = cfg.get("attention_radius", {})
    if "default" in attn:
        attn["default"] = round(float(attn["default"]) * scale, 4)
    for _src, targets in attn.get("pairs", {}).items():
        for dst in targets:
            targets[dst] = round(float(targets[dst]) * scale, 4)
    return cfg


def _new_subdirs(directory: Path, known: set) -> List[Path]:
    """Returns subdirectories of ``directory`` not in ``known``."""
    if not directory.exists():
        return []
    return [p for p in directory.iterdir() if p.is_dir() and p not in known]


def _wait_for_new_subdir(
    directory: Path, known: set, timeout_s: int = 30, label: str = ""
) -> Optional[Path]:
    """Polls until a new subdirectory appears under ``directory``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        new = _new_subdirs(directory, known)
        if new:
            return max(new, key=lambda p: p.stat().st_mtime)
        time.sleep(1.0)
    print(f"  Warning: no new {label} directory appeared under {directory} within {timeout_s}s")
    return None


def _subprocess_env() -> Dict[str, str]:
    """Returns the environment with src/ prepended to PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_PATH}{os.pathsep}{existing}" if existing else SRC_PATH
    return env


def build_combinations(grid: Dict[str, List]) -> List[Dict]:
    """Returns the Cartesian product of all grid values."""
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[grid[k] for k in keys])]


def run_combination(
    combo: Dict,
    base_args: Dict,
    log_dir: Path,
    metrics_root: Path,
    joined_root: Path,
    output_format: str,
    dry_run: bool,
) -> bool:
    """Runs training + join for one hyperparameter combination. Returns True on success."""
    scale = combo.get("attention_radius_scale", 1.0)
    history_sec = combo["history_sec"]
    prediction_sec = combo["prediction_sec"]

    # ── Training ──────────────────────────────────────────────────────────────
    train_cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
                 str(ROOT / "train_unified.py")]
    for k, v in base_args.items():
        train_cmd += [f"--{k}", str(v)]
    train_cmd += [
        "--log_dir", str(log_dir),
        "--history_sec", str(history_sec),
        "--prediction_sec", str(prediction_sec),
    ]

    print(f"  Training: {' '.join(train_cmd)}")

    if not dry_run:
        before_log = set(log_dir.iterdir()) if log_dir.exists() else set()
        before_metrics = set(metrics_root.iterdir()) if metrics_root.exists() else set()
        subprocess.run(train_cmd, check=True, env=_subprocess_env())

        # Detect the model directory created by this run
        model_dir = _wait_for_new_subdir(log_dir, before_log, label="model")
        if model_dir is None:
            return False
        conf_path = model_dir / "config.json"
        if not conf_path.exists():
            print(f"  Warning: config.json not found at {conf_path}; skipping join step")
            return False

        # Detect the metrics directory created by this run's eval
        metrics_run_dir = _wait_for_new_subdir(metrics_root, before_metrics, label="metrics")
        if metrics_run_dir is None:
            return False
        run_name = metrics_run_dir.name
    else:
        conf_path = Path("<model_dir>/config.json")
        run_name = "<run_name>"

    # ── Metric joining ────────────────────────────────────────────────────────
    join_cmd = [
        sys.executable, "-m", "data_preparation.join_characteristic_metrics",
        "--conf", str(conf_path),
        "--run_dir", run_name,
        "--output_root", str(joined_root),
        "--format", output_format,
    ]

    print(f"  Joining:  {' '.join(join_cmd)}")

    if not dry_run:
        subprocess.run(join_cmd, check=True, env=_subprocess_env(), cwd=str(ROOT))

    return True


def run_sweep(args: argparse.Namespace) -> None:
    sweep_cfg = _load_yaml(args.sweep_config)
    original_shared = _load_yaml(SHARED_CONFIG_PATH)

    base_args = sweep_cfg.get("base_args", {})
    grid = sweep_cfg.get("grid", {})

    combos = build_combinations(grid)
    log_dir = Path(base_args.pop("log_dir", DEFAULT_LOG_DIR))
    metrics_root = Path(args.metrics_root)
    joined_root = Path(args.joined_root)

    print(f"Sweep: {len(combos)} combination(s)")
    for i, combo in enumerate(combos):
        scale = combo.get("attention_radius_scale", 1.0)
        print(f"\n{'─' * 60}")
        print(f"Run {i + 1}/{len(combos)}: history={combo['history_sec']}s  "
              f"prediction={combo['prediction_sec']}s  "
              f"attention_radius_scale={scale}×")
        print('─' * 60)

        if not args.dry_run:
            scaled_cfg = _scale_attention_radii(original_shared, scale)
            _write_yaml(scaled_cfg, SHARED_CONFIG_PATH)

        try:
            ok = run_combination(
                combo=combo,
                base_args=dict(base_args),
                log_dir=log_dir,
                metrics_root=metrics_root,
                joined_root=joined_root,
                output_format=args.format,
                dry_run=args.dry_run,
            )
            if not ok:
                print(f"  Skipped combination {i + 1} due to missing outputs.")
        finally:
            if not args.dry_run:
                _write_yaml(original_shared, SHARED_CONFIG_PATH)

    # ── Combine all runs ──────────────────────────────────────────────────────
    combine_cmd = [
        sys.executable, "-m", "data_preparation.combine_runs",
        "--joined_root", str(joined_root),
        "--format", args.format,
    ]
    print(f"\n{'─' * 60}")
    print(f"Combining: {' '.join(combine_cmd)}")

    if not args.dry_run:
        subprocess.run(combine_cmd, check=True, env=_subprocess_env(), cwd=str(ROOT))

    print("\nSweep complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Trajectron++ hyperparameter sweep over history_sec, "
                    "prediction_sec, and attention_radius_scale."
    )
    parser.add_argument(
        "--sweep_config",
        type=Path,
        default=DEFAULT_SWEEP_CONFIG,
        help="Path to sweep YAML config (default: config/sweep_config.yaml).",
    )
    parser.add_argument(
        "--metrics_root",
        type=Path,
        default=DEFAULT_METRICS_ROOT,
        help="Root directory where train_unified.py writes eval CSVs.",
    )
    parser.add_argument(
        "--joined_root",
        type=Path,
        default=DEFAULT_JOINED_ROOT,
        help="Root directory for joined metric outputs.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print all commands without executing anything.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
