from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unified-av-data-loader" / "src"))

import pandas as pd
import pytest
from trajdata import AgentType

from data_preparation.join_characteristic_metrics import (
    drop_overlapping_columns_for_join,
    scene_keys_for_indices,
    trajectory_identity_for_index,
    validate_eval_context,
    validate_eval_identity,
)


class _FakeAgentElem:
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type


class _FakeAgentDataset:
    def __init__(self):
        self._data_index = [
            (Path("/cache/nusc_mini/scene-a"), "agent-1", 12),
            (Path("/cache/nusc_mini/scene-b"), "agent-2", 18),
        ]
        self._elems = [
            _FakeAgentElem(AgentType.VEHICLE),
            _FakeAgentElem(AgentType.PEDESTRIAN),
        ]

    def __getitem__(self, idx: int):
        return self._elems[idx]


def test_trajectory_identity_for_index_uses_trajdata_index_and_agent_type():
    dataset = _FakeAgentDataset()

    identity = trajectory_identity_for_index(dataset, 1, dataset[1])

    assert identity == {
        "data_idx": 1,
        "scene_path": "/cache/nusc_mini/scene-b",
        "agent_id": "agent-2",
        "scene_ts": 18,
        "agent_type": "PEDESTRIAN",
    }


def test_scene_keys_for_indices_preserves_data_idx_mapping():
    dataset = _FakeAgentDataset()

    scene_keys = scene_keys_for_indices(dataset, [1, 0])

    assert scene_keys.to_dict("records") == [
        {
            "data_idx": 1,
            "scene_path": "/cache/nusc_mini/scene-b",
            "scene_ts": 18,
        },
        {
            "data_idx": 0,
            "scene_path": "/cache/nusc_mini/scene-a",
            "scene_ts": 12,
        },
    ]


def test_validate_eval_identity_accepts_matching_reconstruction_with_duplicate_eval_rows():
    eval_df = pd.DataFrame(
        {
            "data_idx": [1, 0, 1],
            "scene_path": [
                "/cache/nusc_mini/scene-b",
                "/cache/nusc_mini/scene-a",
                "/cache/nusc_mini/scene-b",
            ],
            "agent_id": ["agent-2", "agent-1", "agent-2"],
            "scene_ts": [18, 12, 18],
            "agent_type": ["PEDESTRIAN", "VEHICLE", "PEDESTRIAN"],
        }
    )
    expected_identity_df = pd.DataFrame(
        {
            "data_idx": [0, 1],
            "scene_path": ["/cache/nusc_mini/scene-a", "/cache/nusc_mini/scene-b"],
            "agent_id": ["agent-1", "agent-2"],
            "scene_ts": [12, 18],
            "agent_type": ["VEHICLE", "PEDESTRIAN"],
        }
    )

    validate_eval_identity(
        eval_df,
        expected_identity_df,
        eval_file=Path("eval_epoch_1.csv"),
    )


def test_validate_eval_identity_rejects_mismatched_reconstruction():
    eval_df = pd.DataFrame(
        {
            "data_idx": [0],
            "scene_path": ["/cache/nusc_mini/scene-a"],
            "agent_id": ["agent-1"],
            "scene_ts": [12],
            "agent_type": ["VEHICLE"],
        }
    )
    expected_identity_df = pd.DataFrame(
        {
            "data_idx": [0],
            "scene_path": ["/cache/nusc_mini/scene-b"],
            "agent_id": ["agent-1"],
            "scene_ts": [12],
            "agent_type": ["VEHICLE"],
        }
    )

    with pytest.raises(ValueError, match="identity does not match"):
        validate_eval_identity(
            eval_df,
            expected_identity_df,
            eval_file=Path("eval_epoch_1.csv"),
        )


def test_validate_eval_identity_allows_legacy_data_idx_only_eval_csv():
    eval_df = pd.DataFrame({"data_idx": [0], "ml_ade": [1.0]})
    expected_identity_df = pd.DataFrame(
        {
            "data_idx": [0],
            "scene_path": ["/cache/nusc_mini/scene-a"],
            "agent_id": ["agent-1"],
            "scene_ts": [12],
            "agent_type": ["VEHICLE"],
        }
    )

    validate_eval_identity(
        eval_df,
        expected_identity_df,
        eval_file=Path("legacy_eval_epoch_1.csv"),
    )


def test_validate_eval_identity_rejects_partial_identity_columns():
    eval_df = pd.DataFrame(
        {
            "data_idx": [0],
            "scene_path": ["/cache/nusc_mini/scene-a"],
        }
    )
    expected_identity_df = pd.DataFrame(
        {
            "data_idx": [0],
            "scene_path": ["/cache/nusc_mini/scene-a"],
            "agent_id": ["agent-1"],
            "scene_ts": [12],
            "agent_type": ["VEHICLE"],
        }
    )

    with pytest.raises(ValueError, match="partial trajectory identity"):
        validate_eval_identity(
            eval_df,
            expected_identity_df,
            eval_file=Path("eval_epoch_1.csv"),
        )


def test_validate_eval_context_accepts_matching_context_columns():
    eval_df = pd.DataFrame(
        {
            "eval_data": ["nusc_mini-mini_val"],
            "history_sec": [2.0],
            "prediction_sec": [6.0],
            "restrict_to_predchal": [False],
        }
    )
    hyperparams = {
        "eval_data": "nusc_mini-mini_val",
        "history_sec": 2.0,
        "prediction_sec": 6.0,
        "restrict_to_predchal": False,
    }

    validate_eval_context(eval_df, hyperparams, eval_file=Path("eval_epoch_1.csv"))


def test_validate_eval_context_rejects_history_override_mismatch():
    eval_df = pd.DataFrame(
        {
            "eval_data": ["nusc_mini-mini_val"],
            "history_sec": [2.0],
            "prediction_sec": [6.0],
            "restrict_to_predchal": [False],
        }
    )
    hyperparams = {
        "eval_data": "nusc_mini-mini_val",
        "history_sec": 4.0,
        "prediction_sec": 6.0,
        "restrict_to_predchal": False,
    }

    with pytest.raises(ValueError, match="Eval context does not match"):
        validate_eval_context(eval_df, hyperparams, eval_file=Path("eval_epoch_1.csv"))


def test_validate_eval_context_allows_legacy_eval_without_context_columns():
    eval_df = pd.DataFrame({"data_idx": [0], "ml_ade": [1.0]})
    hyperparams = {
        "eval_data": "nusc_mini-mini_val",
        "history_sec": 2.0,
        "prediction_sec": 6.0,
    }

    validate_eval_context(eval_df, hyperparams, eval_file=Path("legacy_eval_epoch_1.csv"))


def test_drop_overlapping_columns_for_join_keeps_join_key_only():
    right_df = pd.DataFrame(
        {
            "data_idx": [0],
            "scene_path": ["/cache/nusc_mini/scene-a"],
            "agent_id": ["agent-1"],
            "mean_speed": [2.0],
        }
    )

    pruned = drop_overlapping_columns_for_join(
        right_df,
        existing_cols=["data_idx", "scene_path"],
        join_cols=["data_idx"],
    )

    assert pruned.columns.tolist() == ["data_idx", "agent_id", "mean_speed"]
