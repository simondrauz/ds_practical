from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_sweep
from data_preparation import combine_runs


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
