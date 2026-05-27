"""Utilities for loading teleoperation and IK configuration from YAML files."""

from pathlib import Path

import yaml  # type: ignore[import]


def load_ik_config(config_path: str) -> dict:
    """Loads the teleoperation and IK parameters from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(
            f"⚠️ Warning: Config file not found at {path}. Using safe fallback defaults."
        )
        return {}

    with open(path, "r") as f:
        return yaml.safe_load(f)
