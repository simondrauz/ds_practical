from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from trajdata import AgentType

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_CONFIG_PATH = _REPO_ROOT / "config" / "shared_config.yaml"


def _agent_type_from_name(name: str) -> AgentType:
    enum_name = str(name).split(".")[-1].upper()
    if enum_name not in AgentType.__members__:
        valid_types = ", ".join(AgentType.__members__.keys())
        raise ValueError(
            f"Unknown agent type `{name}`. Expected one of: {valid_types}"
        )
    return AgentType[enum_name]


def _load_shared_config(config_path: Path | str = DEFAULT_SHARED_CONFIG_PATH) -> Dict[str, Any]:
    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Shared config not found at {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_json_config(config_path: Path | str) -> Dict[str, Any]:
    """Load a JSON config, resolving an optional ``extends`` parent first.

    ``extends`` is resolved relative to the child config file. Values in the
    child config override the parent; nested mappings are merged recursively.
    The returned config is fully resolved and does not include ``extends``.
    """
    return _load_json_config(Path(config_path).expanduser(), seen=set())


def _load_json_config(config_path: Path, seen: set[Path]) -> Dict[str, Any]:
    resolved_path = config_path.resolve()
    if resolved_path in seen:
        chain = " -> ".join(str(path) for path in [*seen, resolved_path])
        raise ValueError(f"Config inheritance cycle detected: {chain}")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config json at {resolved_path} not found!")

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config json at {resolved_path} must contain an object")

    parent = config.pop("extends", None)
    if parent is None:
        return config
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f"`extends` in {resolved_path} must be a non-empty string")

    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = resolved_path.parent / parent_path

    return _deep_merge(
        _load_json_config(parent_path, seen | {resolved_path}),
        config,
    )


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
    return attention_radius_from_config(raw_cfg.get("attention_radius", {}))


def load_attention_radius_config(
    config_path: Path | str = DEFAULT_SHARED_CONFIG_PATH,
) -> Dict[str, Any]:
    """Loads and normalises serialisable attention-radius config."""
    raw_cfg = _load_shared_config(config_path)
    return normalise_attention_radius_config(raw_cfg.get("attention_radius", {}))


def normalise_attention_radius_config(attention_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise serialisable attention-radius config."""
    if not isinstance(attention_cfg, dict):
        raise ValueError("Invalid shared config: `attention_radius` must be a mapping")

    default_radius = float(attention_cfg.get("default", 20.0))
    pairs_cfg = attention_cfg.get("pairs", {})
    if not isinstance(pairs_cfg, dict):
        raise ValueError(
            "Invalid shared config: `attention_radius.pairs` must be a mapping"
        )

    normalised_pairs: Dict[str, Dict[str, float]] = {}
    for src_name, targets in pairs_cfg.items():
        if not isinstance(targets, dict):
            raise ValueError(
                f"Invalid shared config: `attention_radius.pairs.{src_name}` must be a mapping"
            )
        src_agent = _agent_type_from_name(str(src_name))
        normalised_pairs[src_agent.name] = {}
        for dst_name, value in targets.items():
            dst_agent = _agent_type_from_name(str(dst_name))
            normalised_pairs[src_agent.name][dst_agent.name] = float(value)

    return {
        "default": default_radius,
        "pairs": normalised_pairs,
    }


def attention_radius_from_config(
    attention_cfg: Dict[str, Any],
) -> Dict[Tuple[AgentType, AgentType], float]:
    """Build trajdata attention-radius mapping from serialisable config."""
    attention_cfg = normalise_attention_radius_config(attention_cfg)

    default_radius = float(attention_cfg["default"])
    pairs_cfg = attention_cfg["pairs"]

    radius = defaultdict(lambda: default_radius)
    for src_name, targets in pairs_cfg.items():
        src_agent = _agent_type_from_name(src_name)
        for dst_name, value in targets.items():
            dst_agent = _agent_type_from_name(dst_name)
            radius[(src_agent, dst_agent)] = float(value)

    return radius


def _parse_agent_type_names(names: List[str]) -> List[AgentType]:
    """Convert a list of agent type name strings to AgentType enums."""
    return [_agent_type_from_name(name) for name in names]


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
