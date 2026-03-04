"""
User-specific path configuration for nuScenes experiments.

This module provides a centralized location for user-specific paths,
allowing multiple scripts to access the same configuration.
"""

from typing import Dict, Final

# User-specific paths configuration
USER_PATHS: Final[Dict[str, Dict[str, str]]] = {
    "simon": {
        "trajdata_cache": "/Users/simondrauz/Lokale Dokumente/Repositories/ds_practical/data/processed/trajdata_cache",
        "nusc_raw": "/Users/simondrauz/Lokale Dokumente/Repositories/ds_practical/data/raw",
    },
    "zoe": {
        "trajdata_cache": "/Users/zoe/.unified_data_cache",
        "nusc_raw": "/Users/zoe/Desktop/ds_practical/adaptive-prediction/v1.0-mini",
    },
}

DEFAULT_USER: Final[str] = "simon"


def get_user_paths(user: str) -> Dict[str, str]:
    """
    Get paths for a specific user.

    Args:
        user: Username to get paths for

    Returns:
        Dictionary containing 'trajdata_cache' and 'nusc_raw' paths

    Raises:
        ValueError: If user is not found in USER_PATHS
    """
    if user not in USER_PATHS:
        available_users = ", ".join(USER_PATHS.keys())
        raise ValueError(f"Unknown user '{user}'. Available users: {available_users}")
    return USER_PATHS[user]


def get_available_users() -> list:
    """Get list of available user configurations."""
    return list(USER_PATHS.keys())
