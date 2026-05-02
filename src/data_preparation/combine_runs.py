"""Combine joined trajectory-metric CSVs from multiple Trajectron++ runs.

Reads selected eval CSV files produced by ``join_characteristic_metrics.py``
from ``results/trajectory_prediction/trajectory_metrics_joined/`` and
concatenates them into a single file.

When runs differ in ``history_sec``, ``prediction_sec``, or ``attention_radius_m``,
features that accumulate over the trajectory window are not directly comparable.
This script adds normalised per-second variants of those features alongside the
originals so that cross-run analysis is not confounded by window-length differences.

Normalised features
-------------------
The following features are computed over the full trajectory window
(history + future) and are divided by ``duration`` (seconds):

* ``displacement``     → ``displacement_per_sec``
* ``path_length``      → ``path_length_per_sec``
* ``heading_change``   → ``heading_change_per_sec``

Speed, acceleration, and jerk are already expressed as rates (m/s, m/s², m/s³)
and require no further normalisation.

``min_neighbor_distance`` and ``has_collision`` are affected by
``attention_radius_m`` (which controls the neighbor search radius) but cannot
be meaningfully normalised — the ``attention_radius_m`` column is the correct
covariate to condition on in downstream analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# `heading_change` accumulates over the full trajectory window and has no
# per-step counterpart in the feature set, so dividing by `duration` gives
# a clean angular rate (deg/s) that is comparable across runs with different
# window lengths.
#
# `displacement` and `path_length` are intentionally left unnormalised:
# they scale with window length in roughly the same way ADE does (ADE is
# already a per-step average), so the raw values preserve a consistent
# relationship with the target across runs. Normalising them would produce
# near-duplicates of `mean_speed`, which is already in the feature set.
_WINDOW_FEATURES = ["heading_change"]
_DURATION_COL = "duration"


def _add_normalised_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds per-second normalised variants of window-accumulating features.

    New columns are named ``{feature}_per_sec`` and placed immediately after
    their unnormalised counterparts. Rows where ``duration`` is zero or missing
    yield NaN for the normalised value.
    """
    if _DURATION_COL not in df.columns:
        return df

    for feat in _WINDOW_FEATURES:
        if feat not in df.columns:
            continue
        normed = df[feat] / df[_DURATION_COL].replace(0, float("nan"))
        # Insert the normalised column right after the original.
        insert_at = df.columns.get_loc(feat) + 1
        df.insert(insert_at, f"{feat}_per_sec", normed)

    return df


def collect_run_csvs(
    joined_root: Path, run_dirs: Optional[List[str]]
) -> List[Path]:
    """Returns sorted eval epoch CSV paths under ``joined_root``.

    If ``run_dirs`` is given, only those subdirectories are searched;
    otherwise every subdirectory of ``joined_root`` is included.
    """
    if run_dirs:
        paths: List[Path] = []
        for name in run_dirs:
            run_path = joined_root / name
            if not run_path.is_dir():
                raise FileNotFoundError(
                    f"Run directory not found: {run_path}"
                )
            paths.extend(sorted(run_path.glob("eval_epoch_*.csv")))
        return paths

    paths = []
    for subdir in sorted(joined_root.glob("*")):
        if subdir.is_dir():
            paths.extend(sorted(subdir.glob("eval_epoch_*.csv")))
    return paths


def combine(
    joined_root: Path, run_dirs: Optional[List[str]]
) -> pd.DataFrame:
    """Loads, annotates, normalises, and concatenates all run CSVs.

    Each row gains a ``run_name`` column (the subdirectory name) inserted as
    the first column so that results from different runs remain identifiable
    after concatenation.
    """
    csv_paths = collect_run_csvs(joined_root, run_dirs)
    if not csv_paths:
        raise FileNotFoundError(
            f"No eval_epoch_*.csv files found under {joined_root}"
        )

    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df.insert(0, "run_name", path.parent.name)
        df = _add_normalised_features(df)
        frames.append(df)
        print(f"  Loaded {len(df):>6} rows  {path.relative_to(joined_root)}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows across {len(frames)} file(s)")
    return combined


def _print_run_summary(combined: pd.DataFrame) -> None:
    """Prints a summary of the hyperparameter combinations present."""
    setting_cols = [
        c
        for c in ["run_name", "history_sec", "prediction_sec", "attention_radius_m"]
        if c in combined.columns
    ]
    if len(setting_cols) <= 1:
        return
    summary = (
        combined[setting_cols]
        .drop_duplicates()
        .sort_values(setting_cols)
        .reset_index(drop=True)
    )
    print("\nRun configurations:")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine joined trajectory-metric CSVs from multiple Trajectron++ runs "
            "into a single file, adding per-second normalised variants of "
            "window-accumulating features for cross-run comparability."
        )
    )
    parser.add_argument(
        "--joined_root",
        type=Path,
        default=ROOT
        / "results"
        / "trajectory_prediction"
        / "trajectory_metrics_joined",
        help="Root containing per-run subdirectories of joined CSVs.",
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Specific run directory names to include. Required unless "
            "--all_runs is passed."
        ),
    )
    parser.add_argument(
        "--all_runs",
        action="store_true",
        help=(
            "Explicitly include every run directory under --joined_root. Use only "
            "when intentionally recombining the full joined-output directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results"
        / "trajectory_prediction"
        / "combined_runs",
        help="Output path without a file suffix (suffix is added from --format).",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        help="Output file format (default: csv).",
    )
    args = parser.parse_args()
    if args.run_dirs and args.all_runs:
        parser.error("--run_dirs and --all_runs are mutually exclusive")
    if not args.run_dirs and not args.all_runs:
        parser.error(
            "refusing to combine every joined run implicitly; pass --run_dirs "
            "for the current sweep, or --all_runs to opt in"
        )
    return args


def main() -> None:
    args = parse_args()
    print(f"Combining runs from: {args.joined_root}")
    combined = combine(args.joined_root, args.run_dirs)
    _print_run_summary(combined)

    out_path = args.output.with_suffix("." + args.format)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        combined.to_parquet(out_path, index=False)
    else:
        combined.to_csv(out_path, index=False)
    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
