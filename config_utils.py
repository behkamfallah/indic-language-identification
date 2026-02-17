"""
Configuration helpers for YAML files.
This module has everything related to reading, overriding, and writing
configuration files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and guarantee a dict-like top-level structure.
    The training pipeline expects `key: value` pairs.
    """

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain \"key value\" pairs: {path}")
    return config


def parse_overrides(override_args: List[str]) -> Dict[str, Any]:
    """
    Parse CLI overrides.

    Example input:
      ["training.learning_rate=2e-5", "model.apply_dropout=true"]

    Example output:
      {
        "training": {"learning_rate": 2e-5},
        "model": {"apply_dropout": True}
      }
    """

    overrides: Dict[str, Any] = {}
    for item in override_args:
        if "=" not in item:
            raise ValueError(f"Override must be in \"key=value\" format: {item}")

        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)

        current = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return overrides


def merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries and return a new merged mapping.
    """

    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_nested(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """
    Fetch a nested config value via a dotted path.
    Example:
      get_nested(config, "training.learning_rate", 1e-5)
    """

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def save_json(path: Path, content: Dict[str, Any]) -> None:
    """
    Write a dictionary to disk as pretty-printed JSON.
    Persisting resolved config + metrics makes each run reproducible and easy
    to compare later.
    """

    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)
        handle.write("\n")
