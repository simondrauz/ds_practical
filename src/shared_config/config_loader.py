from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from trajdata import AgentType

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_CONFIG_PATH = _REPO_ROOT / "config" / "shared_config.yaml"


def _load_shared_config(config_path: Path | str = DEFAULT_SHARED_CONFIG_PATH) -> Dict[str, Any]:
    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Shared config not found at {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_vector_map_settings(
    config_path: Path | str = DEFAULT_SHARED_CONFIG_PATH,
) -> Dict[str, Any]:
    """Loads shared raster-map settings from config/shared_config.yaml."""
    raw_cfg = _load_shared_config(config_path)

    vector_map_cfg = raw_cfg.get("vector_map", {})
    if not isinstance(vector_map_cfg, dict):
        raise ValueError("Invalid shared config: `vector_map` must be a mapping")

    raster_cfg = vector_map_cfg.get("raster_map_params", {})
    if not isinstance(raster_cfg, dict):
        raise ValueError(
            "Invalid shared config: `vector_map.raster_map_params` must be a mapping"
        )

    if "px_per_m" not in raster_cfg or "map_size_px" not in raster_cfg:
        raise ValueError(
            "Invalid shared config: raster_map_params must define `px_per_m` and `map_size_px`"
        )

    offset_xy = raster_cfg.get("offset_frac_xy", [-0.75, 0.0])
    if not isinstance(offset_xy, (list, tuple)) or len(offset_xy) != 2:
        raise ValueError(
            "Invalid shared config: `offset_frac_xy` must be a list/tuple of length 2"
        )

    return {
        "raster_map_params": {
            "px_per_m": raster_cfg["px_per_m"],
            "map_size_px": raster_cfg["map_size_px"],
            "offset_frac_xy": (float(offset_xy[0]), float(offset_xy[1])),
        }
    }


def load_attention_radius(
    config_path: Path | str = DEFAULT_SHARED_CONFIG_PATH,
) -> Dict[Tuple[AgentType, AgentType], float]:
    """Loads agent interaction attention radii from config/shared_config.yaml."""
    raw_cfg = _load_shared_config(config_path)

    attention_cfg = raw_cfg.get("attention_radius", {})
    if not isinstance(attention_cfg, dict):
        raise ValueError("Invalid shared config: `attention_radius` must be a mapping")

    default_radius = float(attention_cfg.get("default", 20.0))
    pairs_cfg = attention_cfg.get("pairs", {})
    if not isinstance(pairs_cfg, dict):
        raise ValueError(
            "Invalid shared config: `attention_radius.pairs` must be a mapping"
        )

    radius = defaultdict(lambda: default_radius)
    for src_name, targets in pairs_cfg.items():
        if not isinstance(targets, dict):
            raise ValueError(
                f"Invalid shared config: `attention_radius.pairs.{src_name}` must be a mapping"
            )
        src_agent = AgentType[src_name.upper()]
        for dst_name, value in targets.items():
            dst_agent = AgentType[dst_name.upper()]
            radius[(src_agent, dst_agent)] = float(value)

    return radius


def _parse_agent_type_names(names: List[str]) -> List[AgentType]:
    """Convert a list of agent type name strings to AgentType enums."""
    parsed = []
    for name in names:
        enum_name = name.split(".")[-1].upper()
        if enum_name not in AgentType.__members__:
            valid_types = ", ".join(AgentType.__members__.keys())
            raise ValueError(
                f"Unknown agent type `{name}`. Expected one of: {valid_types}"
            )
        parsed.append(AgentType[enum_name])
    return parsed


def load_agent_type_defaults(
    config_path: Path | str = DEFAULT_SHARED_CONFIG_PATH,
) -> Tuple[List[AgentType], List[AgentType]]:
    """Loads default only_predict and no_types lists from shared config.

    Returns:
        (only_predict, no_types) as lists of AgentType enums.
    """
    raw_cfg = _load_shared_config(config_path)

    defaults_cfg = raw_cfg.get("agent_type_defaults", {})
    if not isinstance(defaults_cfg, dict):
        raise ValueError("Invalid shared config: `agent_type_defaults` must be a mapping")

    only_predict_raw = defaults_cfg.get("only_predict")
    no_types_raw = defaults_cfg.get("no_types")

    if not isinstance(only_predict_raw, list) or not only_predict_raw:
        raise ValueError(
            "Invalid shared config: `agent_type_defaults.only_predict` must be a non-empty list"
        )
    if not isinstance(no_types_raw, list) or not no_types_raw:
        raise ValueError(
            "Invalid shared config: `agent_type_defaults.no_types` must be a non-empty list"
        )

    return _parse_agent_type_names(only_predict_raw), _parse_agent_type_names(no_types_raw)


def parse_agent_type_list(
    raw_values, default_values: List[AgentType], key_name: str
) -> List[AgentType]:
    """Parses config-provided agent type names into AgentType enums.

    If *raw_values* is ``None`` the *default_values* are returned unchanged.
    Otherwise each entry is resolved to an ``AgentType`` enum member.
    """
    if raw_values is None:
        return list(default_values)

    if not isinstance(raw_values, (list, tuple)):
        raise TypeError(
            f"`{key_name}` must be a list of agent type names, got {type(raw_values)}"
        )

    parsed: List[AgentType] = []
    for raw_val in raw_values:
        if isinstance(raw_val, AgentType):
            parsed.append(raw_val)
            continue
        if not isinstance(raw_val, str):
            raise TypeError(
                f"`{key_name}` entries must be strings or AgentType values, got {type(raw_val)}"
            )
        enum_name = raw_val.split(".")[-1].upper()
        if enum_name not in AgentType.__members__:
            valid_types = ", ".join(AgentType.__members__.keys())
            raise ValueError(
                f"Unknown agent type `{raw_val}` in `{key_name}`. "
                f"Expected one of: {valid_types}"
            )
        parsed.append(AgentType[enum_name])

    return parsed

