"""Agent-centric characteristic metrics used to contextualize model errors.

This module lifts the metric logic from the analysis notebook into reusable,
testable functions. All metrics are computed from a single trajdata
`AgentBatchElement` (i.e., one agent at one reference timestep).

Key assumptions:
- Positions live in the first two state dimensions (x, y).
- Metrics operate on the full agent-centric window: history + future.
- `data_idx` alignment is handled outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from trajectron.analysis.helper_functions_characteristic_metrics import (
    as_xy,
    full_traj_xy,
    safe_stat,
)


def compute_speed_stats(full_traj_xy: np.ndarray, dt: float) -> Dict[str, float]:
    """Computes basic speed statistics from positions.

    Speed is derived from first differences of position divided by `dt`.
    """
    deltas = np.diff(full_traj_xy, axis=0) / dt
    speeds = np.linalg.norm(deltas, axis=1)
    return {
        "mean_speed": safe_stat(speeds, np.mean),
        "max_speed": safe_stat(speeds, np.max),
        "std_speed": safe_stat(speeds, np.std),
    }


def compute_accel_jerk_stats(full_traj_xy: np.ndarray, dt: float) -> Dict[str, float]:
    """Computes acceleration and jerk statistics from speeds.

    Acceleration is the first difference of speed over time, and jerk is the
    first difference of acceleration over time.
    """
    deltas = np.diff(full_traj_xy, axis=0) / dt
    speeds = np.linalg.norm(deltas, axis=1)
    accelerations = np.diff(speeds) / dt
    jerk = np.diff(accelerations) / dt
    return {
        "mean_acceleration": safe_stat(accelerations, np.mean),
        "max_acceleration": safe_stat(np.abs(accelerations), np.max),
        "mean_jerk": safe_stat(np.abs(jerk), np.mean),
        "max_jerk": safe_stat(np.abs(jerk), np.max),
    }


def compute_duration_stats(hist_len: int, fut_len: int, dt: float) -> Dict[str, float]:
    """Computes history/future/total durations in seconds."""
    total_len = hist_len + fut_len
    return {
        "duration": float(total_len * dt),
        "history_duration": float(hist_len * dt),
        "future_duration": float(fut_len * dt),
    }


def compute_ego_distance(agent_hist: np.ndarray, robot_fut: Optional[np.ndarray]) -> Dict[str, float]:
    """Distance between the agent and the ego/robot at the reference timestep.

    The reference point is the last history state for the agent and the first
    element of `robot_future_np` (which contains current + future).
    """
    if robot_fut is None or robot_fut.shape[0] == 0:
        return {"ego_distance": float(np.nan)}
    agent_pos = as_xy(agent_hist)[-1]
    ego_pos = as_xy(robot_fut)[0]
    return {"ego_distance": float(np.linalg.norm(agent_pos - ego_pos))}


def compute_path_efficiency(full_traj_xy: np.ndarray) -> Dict[str, float]:
    """Measures how straight the path is.

    Path efficiency is displacement / path length expressed as a percentage.
    """
    displacement = np.linalg.norm(full_traj_xy[-1] - full_traj_xy[0])
    path_length = np.sum(np.linalg.norm(np.diff(full_traj_xy, axis=0), axis=1))
    if path_length <= 1e-6:
        efficiency = 100.0
    else:
        efficiency = displacement / path_length * 100.0
    return {
        "path_efficiency": float(efficiency),
        "displacement": float(displacement),
        "path_length": float(path_length),
    }


def compute_heading_change(full_traj_xy: np.ndarray) -> Dict[str, float]:
    """Computes cumulative heading change in degrees along the trajectory."""
    deltas = np.diff(full_traj_xy, axis=0)
    if deltas.shape[0] <= 1:
        return {"heading_change": 0.0}
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    heading_diffs = np.diff(headings)
    heading_diffs = np.arctan2(np.sin(heading_diffs), np.cos(heading_diffs))
    cumulative_change_deg = np.degrees(np.abs(heading_diffs).sum())
    return {"heading_change": float(cumulative_change_deg)}


def compute_safety_metrics(
    agent_hist: np.ndarray,
    neighbor_histories,
    vec_map,
    collision_threshold_m: float,
    lane_threshold_m: float,
) -> Dict[str, float]:
    """Computes simple safety-style metrics from neighbors and lanes.

    - `has_collision` flags whether any neighbor is closer than
      `collision_threshold_m` at the reference timestep.
    - `min_neighbor_distance` records the closest neighbor distance (NaN if none).
    - `off_road` is a heuristic based on distance to any lane centerline
      (NaN if no vector map is available).
    """
    agent_pos = as_xy(agent_hist)[-1]

    min_neighbor_distance = np.inf
    has_collision = False
    if neighbor_histories is not None and len(neighbor_histories) > 0:
        for neighbor_hist in neighbor_histories:
            if len(neighbor_hist) == 0:
                continue
            neighbor_pos = as_xy(neighbor_hist)[-1]
            distance = float(np.linalg.norm(agent_pos - neighbor_pos))
            min_neighbor_distance = min(min_neighbor_distance, distance)
            if distance < collision_threshold_m:
                has_collision = True

    min_neighbor_distance_val = (
        float(min_neighbor_distance) if np.isfinite(min_neighbor_distance) else float(np.nan)
    )

    off_road = np.nan
    if vec_map is not None and hasattr(vec_map, "lanes") and vec_map.lanes is not None:
        min_dist_to_lane = np.inf
        for lane in vec_map.lanes:
            if not hasattr(lane, "center") or lane.center is None:
                continue
            lane_center = lane.center.xy
            if lane_center is None or len(lane_center) == 0:
                continue
            distances = np.linalg.norm(lane_center - agent_pos, axis=1)
            min_dist_to_lane = min(min_dist_to_lane, float(np.min(distances)))
        if np.isfinite(min_dist_to_lane):
            off_road = float(min_dist_to_lane >= lane_threshold_m)

    return {
        "has_collision": float(has_collision),
        "min_neighbor_distance": min_neighbor_distance_val,
        "off_road": off_road,
    }


@dataclass
class CharacteristicMetricConfig:
    """Configuration for threshold-based characteristic metrics."""
    collision_threshold_m: float = 0.75
    lane_threshold_m: float = 3.0


def compute_characteristic_metrics(elem, config: CharacteristicMetricConfig) -> Dict[str, float]:
    """Computes the notebook's characteristic metrics for one batch element.

    Args:
        elem: A trajdata `AgentBatchElement` (agent-centric sample).
        config: Thresholds for collision and off-road heuristics.

    Returns:
        A flat dict of scalar metrics keyed by column-friendly names.
    """
    dt = float(elem.dt)
    traj_xy = full_traj_xy(elem.agent_history_np, elem.agent_future_np)
    metrics: Dict[str, float] = dict()
    metrics.update(compute_speed_stats(traj_xy, dt))
    metrics.update(compute_accel_jerk_stats(traj_xy, dt))
    metrics.update(
        compute_duration_stats(
            hist_len=elem.agent_history_np.shape[0],
            fut_len=elem.agent_future_np.shape[0],
            dt=dt,
        )
    )
    metrics.update(compute_ego_distance(elem.agent_history_np, elem.robot_future_np))
    metrics.update(compute_path_efficiency(traj_xy))
    metrics.update(compute_heading_change(traj_xy))
    metrics.update(
        compute_safety_metrics(
            agent_hist=elem.agent_history_np,
            neighbor_histories=elem.neighbor_histories,
            vec_map=elem.vec_map,
            collision_threshold_m=config.collision_threshold_m,
            lane_threshold_m=config.lane_threshold_m,
        )
    )
    return metrics
