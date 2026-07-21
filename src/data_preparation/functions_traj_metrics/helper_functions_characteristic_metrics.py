"""Shared helpers for characteristic metric computation.

These utilities centralize small, reusable pieces of logic used by both
agent-centric and scene-centric metric calculations.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from trajdata import AgentType


def as_xy(arr: np.ndarray) -> np.ndarray:
    """Returns only the x/y position slice.

    Assumes positions live in the first two state dimensions.
    """
    return arr[..., :2]


def full_traj_xy(agent_hist: np.ndarray, agent_fut: np.ndarray) -> np.ndarray:
    """Stacks history+future and keeps only position columns."""
    return as_xy(np.vstack([agent_hist, agent_fut]))


def safe_stat(values: np.ndarray, fn, default: float = np.nan) -> float:
    """Computes a scalar statistic while handling empty inputs."""
    if values.size == 0:
        return float(default)
    return float(fn(values))


def count_agent_types(agent_types: Iterable[int]) -> Dict[str, int]:
    """Counts agent types given integer-encoded AgentType values."""
    counts: Dict[str, int] = {}
    for agent_type_int in agent_types:
        try:
            type_name = AgentType(int(agent_type_int)).name
        except (ValueError, TypeError):
            continue
        counts[type_name] = counts.get(type_name, 0) + 1
    return counts


def bbox_from_positions(
    positions_xy: np.ndarray, min_area: float
) -> Optional[Dict[str, float]]:
    """Computes bounding-box statistics from positions.

    Returns None when there are fewer than two positions or the bounding-box
    area is below the provided minimum threshold.
    """
    if positions_xy.shape[0] < 2:
        return None

    x_min, x_max = positions_xy[:, 0].min(), positions_xy[:, 0].max()
    y_min, y_max = positions_xy[:, 1].min(), positions_xy[:, 1].max()

    bbox_width = float(x_max - x_min)
    bbox_height = float(y_max - y_min)
    bbox_area = bbox_width * bbox_height

    if bbox_area <= min_area:
        return None

    return {
        "bbox_area": float(bbox_area),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
    }

