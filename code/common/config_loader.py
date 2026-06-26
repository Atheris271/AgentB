"""Utility: load YAML / JSON configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Requires ``pyyaml`` to be installed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required for YAML configs. Run: pip install pyyaml")

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return its contents as a dict."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str | Path) -> str:
    """Read a plain-text file and return its contents."""
    return Path(path).read_text(encoding="utf-8")


def get_project_root() -> Path:
    """Return the absolute path to the agent/ project root."""
    return Path(__file__).resolve().parent.parent.parent  # code/common → code → agent/


def config_path(filename: str) -> Path:
    """Resolve a config filename relative to the project's configs/ directory."""
    return get_project_root() / "configs" / filename
