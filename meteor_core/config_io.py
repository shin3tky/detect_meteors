#!/usr/bin/env python
#
# Detect Meteors CLI - Configuration I/O
# © 2025 Shinichi Morita (shin3tky)
#

"""Load pipeline configuration files for CLI usage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .exceptions import MeteorConfigError
from .schema import PipelineConfig

SUPPORTED_CONFIG_EXTENSIONS = {".json", ".yaml", ".yml"}


def _load_config_mapping(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MeteorConfigError(
                "Invalid JSON configuration file",
                filepath=str(path),
                original_error=exc,
            ) from exc
    if suffix in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise MeteorConfigError(
                "Invalid YAML configuration file",
                filepath=str(path),
                original_error=exc,
            ) from exc
        return data if data is not None else {}
    raise MeteorConfigError(
        "Unsupported configuration file format",
        filepath=str(path),
        context={"supported_extensions": sorted(SUPPORTED_CONFIG_EXTENSIONS)},
    )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML/JSON file."""
    config_path = Path(path)
    if not config_path.exists():
        raise MeteorConfigError(
            "Configuration file not found",
            filepath=str(config_path),
        )
    if config_path.is_dir():
        raise MeteorConfigError(
            "Configuration path must be a file, not a directory",
            filepath=str(config_path),
        )
    data = _load_config_mapping(config_path)
    if not isinstance(data, dict):
        raise MeteorConfigError(
            "Configuration file must define an object at the top level",
            filepath=str(config_path),
        )
    try:
        return PipelineConfig.from_dict(data)
    except (TypeError, ValueError, KeyError) as exc:
        raise MeteorConfigError(
            "Invalid pipeline configuration",
            filepath=str(config_path),
            original_error=exc,
        ) from exc
