from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_sweep
from data_preparation import combine_runs


def test_combine_runs_preserves_eval_csv_name_in_row_identity(tmp_path):
    joined_root = tmp_path / "joined"
    run_dir = joined_root / "run_a"
    run_dir.mkdir(parents=True)
    pd.DataFrame({"data_idx": [0], "ml_ade": [1.0]}).to_csv(
        run_dir / "eval_epoch_1.csv",
        index=False,
    )
    pd.DataFrame({"data_idx": [0], "ml_ade": [2.0]}).to_csv(
        run_dir / "eval_epoch_2.csv",
        index=False,
    )

    combined = combine_runs.combine(joined_root, ["run_a"])

    assert combined[["run_name", "eval_csv_name", "data_idx", "ml_ade"]].to_dict("records") == [
        {"run_name": "run_a", "eval_csv_name": "eval_epoch_1.csv", "data_idx": 0, "ml_ade": 1.0},
        {"run_name": "run_a", "eval_csv_name": "eval_epoch_2.csv", "data_idx": 0, "ml_ade": 2.0},
    ]
    assert not combined.duplicated(["run_name", "eval_csv_name", "data_idx"]).any()


def test_sweep_combine_command_is_scoped_to_current_run_dirs():
    cmd = run_sweep.build_combine_command(
        joined_root=Path("/tmp/joined"),
        output_format="csv",
        run_names=["run_a", "run_b"],
    )

    assert cmd[:3] == [sys.executable, "-m", "data_preparation.combine_runs"]
    assert "--joined_root" in cmd
    assert cmd[cmd.index("--joined_root") + 1] == "/tmp/joined"
    assert "--format" in cmd
    assert cmd[cmd.index("--format") + 1] == "csv"
    assert "--run_dirs" in cmd
    assert cmd[cmd.index("--run_dirs") + 1 :] == ["run_a", "run_b"]
    assert "--all_runs" not in cmd


def test_sweep_combine_command_refuses_unscoped_combine():
    with pytest.raises(ValueError, match="refusing to combine all existing"):
        run_sweep.build_combine_command(
            joined_root=Path("/tmp/joined"),
            output_format="csv",
            run_names=[],
        )


def test_combine_runs_cli_requires_explicit_scope(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["combine_runs.py", "--joined_root", "/tmp/joined"],
    )

    with pytest.raises(SystemExit):
        combine_runs.parse_args()


def test_combine_runs_cli_accepts_explicit_all_runs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["combine_runs.py", "--joined_root", "/tmp/joined", "--all_runs"],
    )

    args = combine_runs.parse_args()

    assert args.joined_root == Path("/tmp/joined")
    assert args.run_dirs is None
    assert args.all_runs is True
