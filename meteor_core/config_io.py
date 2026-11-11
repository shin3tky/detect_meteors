#!/usr/bin/env python
#
# Detect Meteors CLI - Configuration I/O
# © 2025 Shinichi Morita (shin3tky)
#

"""Read and serialize versioned pipeline configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal

import yaml

from .exceptions import MeteorConfigError
from .schema import PipelineConfig

SUPPORTED_CONFIG_EXTENSIONS = {".json", ".yaml", ".yml"}
_PIPELINE_CONFIG_SCHEMA_VERSION = 1


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
    """Load pipeline settings from YAML or JSON, filling omitted settings.

    A missing top-level schema_version identifies the original v1 format.

    Args:
        path: Path to the pipeline configuration file.

    Returns:
        Pipeline settings with defaults applied.

    Raises:
        MeteorConfigError: The file, its schema version, or its settings are invalid.
    """
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
    version = data.pop("schema_version", 1)
    if type(version) is not int or version != _PIPELINE_CONFIG_SCHEMA_VERSION:
        raise MeteorConfigError(
            f"Unsupported pipeline configuration schema_version {version!r}. "
            f"Expected integer {_PIPELINE_CONFIG_SCHEMA_VERSION}.",
            filepath=str(config_path),
            config_key="schema_version",
            context={"supported_versions": [_PIPELINE_CONFIG_SCHEMA_VERSION]},
        )
    try:
        return PipelineConfig.from_partial_dict(data)
    except (TypeError, ValueError, KeyError) as exc:
        raise MeteorConfigError(
            "Invalid pipeline configuration",
            filepath=str(config_path),
            original_error=exc,
        ) from exc


def dump_pipeline_config(
    config: PipelineConfig, *, format: Literal["yaml", "json"] = "yaml"
) -> str:
    """Return pipeline configuration text with the current file schema version.

    Args:
        config: Pipeline settings to serialize.
        format: Output format, either "yaml" or "json".

    Returns:
        Text suitable for writing to a UTF-8 configuration file.
    """
    data = {"schema_version": _PIPELINE_CONFIG_SCHEMA_VERSION, **config.to_dict()}
    if format == "yaml":
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    if format == "json":
        return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    raise ValueError(f"Unsupported pipeline configuration output format {format!r}")
