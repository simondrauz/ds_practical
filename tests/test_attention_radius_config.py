from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unified-av-data-loader" / "src"))

import pytest
from trajdata import AgentType

import data_preparation.join_characteristic_metrics as join_characteristic_metrics
from shared_config.config_loader import (
    attention_radius_from_config,
    load_attention_radius,
    load_attention_radius_config,
    normalise_attention_radius_config,
)


def _empty_overrides() -> argparse.Namespace:
    return argparse.Namespace(
        eval_data=None,
        history_sec=None,
        prediction_sec=None,
        trajdata_cache_dir=None,
        data_loc_dict=None,
        preprocess_workers=None,
        map_encoding=None,
        incl_robot_node=None,
    )


def test_normalise_attention_radius_config_preserves_serialisable_pair_map():
    cfg = {
        "default": 42,
        "pairs": {
            "vehicle": {"pedestrian": "7.5"},
            "AgentType.PEDESTRIAN": {"AgentType.VEHICLE": 11},
        },
    }

    normalised = normalise_attention_radius_config(cfg)

    assert normalised == {
        "default": 42.0,
        "pairs": {
            "VEHICLE": {"PEDESTRIAN": 7.5},
            "PEDESTRIAN": {"VEHICLE": 11.0},
        },
    }


def test_attention_radius_from_config_uses_persisted_values_and_default():
    radius = attention_radius_from_config(
        {
            "default": 42.0,
            "pairs": {
                "VEHICLE": {"PEDESTRIAN": 7.5},
            },
        }
    )

    assert radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] == pytest.approx(7.5)
    assert radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] == pytest.approx(42.0)


def test_load_attention_radius_config_round_trips_shared_yaml(tmp_path):
    shared_config = tmp_path / "shared_config.yaml"
    shared_config.write_text(
        """
attention_radius:
  default: 30
  pairs:
    VEHICLE:
      VEHICLE: 60
      PEDESTRIAN: 20
""",
        encoding="utf-8",
    )

    serialisable_cfg = load_attention_radius_config(shared_config)
    radius = load_attention_radius(shared_config)

    assert serialisable_cfg == {
        "default": 30.0,
        "pairs": {
            "VEHICLE": {
                "VEHICLE": 60.0,
                "PEDESTRIAN": 20.0,
            },
        },
    }
    assert radius[(AgentType.VEHICLE, AgentType.VEHICLE)] == pytest.approx(60.0)
    assert radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] == pytest.approx(30.0)


def test_join_load_hyperparams_keeps_persisted_attention_radius(tmp_path, monkeypatch):
    conf_path = tmp_path / "config.json"
    persisted_attention_radius = {
        "default": 42.0,
        "pairs": {"VEHICLE": {"PEDESTRIAN": 7.5}},
    }
    conf_path.write_text(
        json.dumps(
            {
                "eval_data": "nusc_mini-mini_val",
                "history_sec": 2.0,
                "prediction_sec": 6.0,
                "attention_radius": persisted_attention_radius,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        join_characteristic_metrics,
        "load_attention_radius_config",
        lambda: pytest.fail("fallback shared_config.yaml should not be read"),
    )

    hyperparams = join_characteristic_metrics.load_hyperparams(conf_path, _empty_overrides())
    radius = join_characteristic_metrics.resolve_attention_radius(hyperparams)

    assert hyperparams["attention_radius"] == persisted_attention_radius
    assert radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] == pytest.approx(7.5)


def test_join_load_hyperparams_backfills_legacy_config_from_shared_config(tmp_path, monkeypatch):
    conf_path = tmp_path / "legacy_config.json"
    conf_path.write_text(
        json.dumps(
            {
                "eval_data": "nusc_mini-mini_val",
                "history_sec": 2.0,
                "prediction_sec": 6.0,
            }
        ),
        encoding="utf-8",
    )
    fallback_attention_radius = {
        "default": 12.0,
        "pairs": {"PEDESTRIAN": {"PEDESTRIAN": 3.0}},
    }
    monkeypatch.setattr(
        join_characteristic_metrics,
        "load_attention_radius_config",
        lambda: fallback_attention_radius,
    )

    hyperparams = join_characteristic_metrics.load_hyperparams(conf_path, _empty_overrides())
    radius = join_characteristic_metrics.resolve_attention_radius(hyperparams)

    assert hyperparams["attention_radius"] == fallback_attention_radius
    assert radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] == pytest.approx(3.0)
    assert radius[(AgentType.VEHICLE, AgentType.VEHICLE)] == pytest.approx(12.0)
