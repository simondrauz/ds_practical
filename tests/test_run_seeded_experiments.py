from pathlib import Path

import pandas as pd
import pytest

from scripts.run_seeded_experiments import aggregate_seeded_records
from scripts import run_prediction_result_set as result_sets


def _write_joined(path: Path, seed_offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_name": path.parent.name,
                "eval_csv_name": path.name,
                "data_idx": 10,
                "scene_path": "scene_a",
                "agent_id": "agent_1",
                "scene_ts": 5,
                "agent_type": "PEDESTRIAN",
                "eval_data": "nusc_mini-mini_val",
                "history_sec": 2.0,
                "prediction_sec": 4.0,
                "restrict_to_predchal": False,
                "attention_radius_m": 10.0,
                "mean_speed": 1.2,
                "ml_ade": 0.5 + seed_offset,
                "ml_fde": 0.9 + seed_offset,
            },
            {
                "run_name": path.parent.name,
                "eval_csv_name": path.name,
                "data_idx": 11,
                "scene_path": "scene_b",
                "agent_id": "agent_2",
                "scene_ts": 7,
                "agent_type": "PEDESTRIAN",
                "eval_data": "nusc_mini-mini_val",
                "history_sec": 2.0,
                "prediction_sec": 4.0,
                "restrict_to_predchal": False,
                "attention_radius_m": 10.0,
                "mean_speed": 2.4,
                "ml_ade": 1.5 + seed_offset,
                "ml_fde": 1.9 + seed_offset,
            },
        ]
    ).to_csv(path, index=False)


def _record(path: Path, seed: int) -> dict:
    return {
        "joined_path": str(path),
        "run_name": path.parent.name,
        "checkpoint_epoch": 30,
        "train_args": {"seed": seed},
    }


def test_aggregate_seeded_records_groups_by_setting_and_data_idx(tmp_path: Path) -> None:
    seed1 = tmp_path / "run_seed1" / "eval_epoch_30.csv"
    seed2 = tmp_path / "run_seed2" / "eval_epoch_30.csv"
    _write_joined(seed1, seed_offset=0.0)
    _write_joined(seed2, seed_offset=0.2)

    aggregated = aggregate_seeded_records(
        [_record(seed1, 123), _record(seed2, 456)],
        expected_seeds=2,
    )

    assert len(aggregated) == 2
    row = aggregated.loc[aggregated["agent_id"] == "agent_1"].iloc[0]
    assert row["n_seeds"] == 2
    assert row["ml_ade"] == pytest.approx(0.6)
    assert row["ml_ade_seed_min"] == pytest.approx(0.5)
    assert row["ml_ade_seed_max"] == pytest.approx(0.7)
    assert row["mean_speed"] == pytest.approx(1.2)
    assert row["seed_values"] == "123|456"


def test_aggregate_seeded_records_rejects_missing_stable_identifier(tmp_path: Path) -> None:
    path = tmp_path / "run_seed1" / "eval_epoch_30.csv"
    _write_joined(path)
    df = pd.read_csv(path).drop(columns=["scene_ts"])
    df.to_csv(path, index=False)

    with pytest.raises(KeyError, match="trajectory aggregation"):
        aggregate_seeded_records([_record(path, 123)], expected_seeds=1)


def test_aggregate_seeded_records_rejects_incomplete_seed_groups(tmp_path: Path) -> None:
    seed1 = tmp_path / "run_seed1" / "eval_epoch_30.csv"
    _write_joined(seed1)

    with pytest.raises(ValueError, match="do not have all 2 seeds"):
        aggregate_seeded_records([_record(seed1, 123)], expected_seeds=2)


def test_aggregate_seeded_records_validates_data_idx_identity(tmp_path: Path) -> None:
    seed1 = tmp_path / "run_seed1" / "eval_epoch_30.csv"
    seed2 = tmp_path / "run_seed2" / "eval_epoch_30.csv"
    _write_joined(seed1, seed_offset=0.0)
    _write_joined(seed2, seed_offset=0.2)
    df = pd.read_csv(seed2)
    df.loc[df["data_idx"] == 10, "agent_id"] = "different_agent"
    df.to_csv(seed2, index=False)

    with pytest.raises(ValueError, match="data_idx is not stable"):
        aggregate_seeded_records(
            [_record(seed1, 123), _record(seed2, 456)],
            expected_seeds=2,
        )


def test_prediction_result_set_experiment_definitions() -> None:
    small = result_sets.get_experiment_definition("sweep_small_3seeds")
    large = result_sets.get_experiment_definition("sweep_large_1seed")
    trainval = result_sets.get_experiment_definition("full_trainval_3seeds")

    assert len(result_sets.experiment_grid(small)) == 18
    assert len(result_sets.experiment_grid(large)) == 64
    assert result_sets.experiment_grid(trainval) == []
    assert len(result_sets.expected_training_keys(trainval)) == 3
    assert len(result_sets.expected_training_keys(small)) == 54
    assert len(result_sets.expected_training_keys(large)) == 64


def test_prediction_result_set_training_specs_apply_data_overrides() -> None:
    definition = result_sets.get_experiment_definition("full_trainval_1seed")
    cache_dir = Path("/external/trajdata_cache")
    data_loc_dict = '{"nusc_trainval":"/external/nuScenes"}'

    specs = result_sets.training_specs(
        definition,
        trajdata_cache_dir=cache_dir,
        data_loc_dict=data_loc_dict,
    )

    assert len(specs) == 1
    assert specs[0]["train_args"]["trajdata_cache_dir"] == cache_dir
    assert specs[0]["train_args"]["data_loc_dict"] == data_loc_dict


def test_single_seed_trainval_output_uses_direct_joined_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(result_sets, "DEFAULT_JOINED_ROOT", tmp_path / "joined")
    definition = result_sets.get_experiment_definition("full_trainval_1seed")
    source = pd.DataFrame(
        [
            {
                "run_name": "source_run",
                "eval_csv_name": "eval_epoch_12.csv",
                "data_idx": 1,
                "ml_ade": 0.4,
            }
        ]
    )

    output = result_sets._write_csv_for_notebook(  # noqa: SLF001
        source,
        definition,
        preserve_row_run_names=False,
    )
    written = pd.read_csv(output["path"])

    assert output["run_name"] == "full_trainval_12ep_1seed"
    assert output["eval_csv_name"] == "eval_epoch_12.csv"
    assert written["run_name"].tolist() == ["full_trainval_12ep_1seed"]
    assert not [col for col in written.columns if col.endswith("_seed_std")]


def test_single_seed_sweep_combines_without_seed_average_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joined_root = tmp_path / "joined"
    monkeypatch.setattr(result_sets, "DEFAULT_JOINED_ROOT", joined_root)
    definition = result_sets.get_experiment_definition("sweep_large_1seed")
    for run_name, history in [("run_a", 1.0), ("run_b", 2.0)]:
        path = joined_root / run_name / "eval_epoch_30.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "data_idx": 1,
                    "history_sec": history,
                    "prediction_sec": 4.0,
                    "attention_radius_m": 10.0,
                    "heading_change": 6.0,
                    "duration": 3.0,
                    "ml_ade": 0.7,
                }
            ]
        ).to_csv(path, index=False)

    combined = result_sets.combine_joined_runs(joined_root, ["run_a", "run_b"])
    output = result_sets._write_csv_for_notebook(  # noqa: SLF001
        combined,
        definition,
        preserve_row_run_names=True,
    )
    written = pd.read_csv(output["path"])

    assert sorted(written["run_name"].unique().tolist()) == ["run_a", "run_b"]
    assert "heading_change_per_sec" in written.columns
    assert not [col for col in written.columns if col.endswith("_seed_std")]


def test_archive_plan_keeps_referenced_candidates(tmp_path: Path) -> None:
    prediction_root = tmp_path / "trajectory_prediction"
    scan_root = prediction_root / "trajectory_metrics"
    keep_dir = scan_root / "keep_run"
    old_dir = scan_root / "old_run"
    keep_file = keep_dir / "eval_epoch_30.csv"
    old_file = old_dir / "eval_epoch_30.csv"
    keep_file.parent.mkdir(parents=True)
    old_file.parent.mkdir(parents=True)
    keep_file.write_text("keep\n", encoding="utf-8")
    old_file.write_text("old\n", encoding="utf-8")

    archive_root = tmp_path / "_archive" / "unused_prediction_results_test"
    plan = result_sets.build_archive_plan(
        keep_paths={keep_file.resolve()},
        archive_root=archive_root,
        scan_roots=[scan_root],
        prediction_root=prediction_root,
    )

    assert [Path(item["source"]).name for item in plan] == ["old_run"]
    assert Path(plan[0]["destination"]) == archive_root / "trajectory_prediction" / "trajectory_metrics" / "old_run"
