from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import yaml
from trajdata import AgentType

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTENTION_CONFIG_PATH = _REPO_ROOT / "config" / "attention_radii_config.yaml"


def load_attention_radius(
    config_path: Path | str = DEFAULT_ATTENTION_CONFIG_PATH,
) -> Dict[Tuple[AgentType, AgentType], float]:
    """Loads agent interaction attention radii from config/attention_radii_config.yaml."""
    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Attention radius config not found at {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    attention_cfg = raw_cfg.get("attention_radius", {})
    if not isinstance(attention_cfg, dict):
        raise ValueError("Invalid config: `attention_radius` must be a mapping")

    default_radius = float(attention_cfg.get("default", 20.0))
    pairs_cfg = attention_cfg.get("pairs", {})
    if not isinstance(pairs_cfg, dict):
        raise ValueError("Invalid config: `attention_radius.pairs` must be a mapping")

    radius = defaultdict(lambda: default_radius)
    for src_name, targets in pairs_cfg.items():
        if not isinstance(targets, dict):
            raise ValueError(
                f"Invalid config: `attention_radius.pairs.{src_name}` must be a mapping"
            )
        src_agent = AgentType[src_name.upper()]
        for dst_name, value in targets.items():
            dst_agent = AgentType[dst_name.upper()]
            radius[(src_agent, dst_agent)] = float(value)

    return radius

