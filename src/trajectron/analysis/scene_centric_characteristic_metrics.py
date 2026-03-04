"""Scene-centric characteristic metrics derived from trajdata SceneBatchElements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from trajectron.analysis.helper_functions_characteristic_metrics import (
    as_xy,
    bbox_from_positions,
    count_agent_types,
)


@dataclass
class SceneCharacteristicMetricConfig:
    """Configuration for scene-centric metric computation."""

    min_bbox_area_m2: float = 0.01


def _last_positions(agent_histories: Iterable[np.ndarray]) -> np.ndarray:
    """Extracts last observed x/y positions for each agent with history."""
    positions: List[np.ndarray] = []
    for agent_hist in agent_histories:
        if len(agent_hist) == 0:
            continue
        positions.append(as_xy(agent_hist)[-1])
    if len(positions) == 0:
        return np.empty((0, 2), dtype=float)
    return np.vstack(positions)


def compute_scene_characteristic_metrics(
    elem,
    scene_path: str,
    config: SceneCharacteristicMetricConfig,
) -> Dict[str, float]:
    """Computes scene-centric metrics for one scene timestep.

    Args:
        elem: A trajdata `SceneBatchElement` (scene-centric sample).
        scene_path: The scene cache path from the dataset's data index.
        config: Thresholds controlling density/bounding-box calculation.

    Returns:
        A flat dict with scene identifiers and instantaneous scene metrics.
    """
    scene_ts = int(getattr(elem, "scene_ts"))
    scene_id = getattr(elem, "scene_id", "")
    num_agents = int(getattr(elem, "num_agents"))

    agent_types_np = getattr(elem, "agent_types_np", None)
    type_counts = count_agent_types(agent_types_np) if agent_types_np is not None else {}

    metrics: Dict[str, float] = {
        "scene_path": scene_path,
        "scene_ts": scene_ts,
        "scene_id": scene_id,
        "scene_num_agents": float(num_agents),
    }
    for type_name, count in type_counts.items():
        metrics[f"scene_num_{type_name}"] = float(count)

    agent_histories = getattr(elem, "agent_histories", None)
    if agent_histories is None or len(agent_histories) == 0:
        metrics.update(
            {
                "scene_bbox_area": float(np.nan),
                "scene_bbox_width": float(np.nan),
                "scene_bbox_height": float(np.nan),
                "scene_spatial_density": float(np.nan),
            }
        )
        return metrics

    positions_xy = _last_positions(agent_histories)
    bbox_stats = bbox_from_positions(positions_xy, min_area=config.min_bbox_area_m2)
    if bbox_stats is None:
        metrics.update(
            {
                "scene_bbox_area": float(np.nan),
                "scene_bbox_width": float(np.nan),
                "scene_bbox_height": float(np.nan),
                "scene_spatial_density": float(np.nan),
            }
        )
        return metrics

    bbox_area = float(bbox_stats["bbox_area"])
    metrics.update(
        {
            "scene_bbox_area": bbox_area,
            "scene_bbox_width": float(bbox_stats["bbox_width"]),
            "scene_bbox_height": float(bbox_stats["bbox_height"]),
            "scene_spatial_density": float(num_agents / bbox_area),
        }
    )

    for type_name, count in type_counts.items():
        metrics[f"scene_density_{type_name}"] = float(count / bbox_area)

    return metrics

