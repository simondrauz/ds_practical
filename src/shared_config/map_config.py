from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VECTOR_MAP_CONFIG_PATH = _REPO_ROOT / "config" / "vector_map_config.yaml"


def load_vector_map_settings(
    config_path: Path | str = DEFAULT_VECTOR_MAP_CONFIG_PATH,
) -> Dict[str, Any]:
    """Loads shared raster-map settings from config/vector_map_config.yaml."""
    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Vector map config not found at {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    vector_map_cfg = raw_cfg.get("vector_map", {})
    if not isinstance(vector_map_cfg, dict):
        raise ValueError("Invalid vector map config: `vector_map` must be a mapping")

    raster_cfg = vector_map_cfg.get("raster_map_params", {})
    if not isinstance(raster_cfg, dict):
        raise ValueError(
            "Invalid vector map config: `vector_map.raster_map_params` must be a mapping"
        )

    if "px_per_m" not in raster_cfg or "map_size_px" not in raster_cfg:
        raise ValueError(
            "Invalid vector map config: raster_map_params must define `px_per_m` and `map_size_px`"
        )

    offset_xy = raster_cfg.get("offset_frac_xy", [-0.75, 0.0])
    if not isinstance(offset_xy, (list, tuple)) or len(offset_xy) != 2:
        raise ValueError(
            "Invalid vector map config: `offset_frac_xy` must be a list/tuple of length 2"
        )

    return {
        "raster_map_params": {
            "px_per_m": raster_cfg["px_per_m"],
            "map_size_px": raster_cfg["map_size_px"],
            "offset_frac_xy": (float(offset_xy[0]), float(offset_xy[1])),
        }
    }
