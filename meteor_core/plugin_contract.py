"""Shared plugin contract helpers for meteor_core plugins."""

from __future__ import annotations

import importlib.util
from typing import Any

_PYDANTIC_SPEC = importlib.util.find_spec("pydantic")
if _PYDANTIC_SPEC:
    import pydantic
    from pydantic import BaseModel

    Extra = getattr(pydantic, "Extra", None)
else:
    BaseModel = None
    Extra = None


def require_plugin_name(cls: type[Any], *, kind: str) -> str:
    """Validate and return the plugin name defined on a class.

    Args:
        cls: Class declaring the plugin.
        kind: Human-readable plugin kind (inputs/outputs/detectors).

    Returns:
        The validated plugin name.

    Raises:
        ValueError: If the plugin name is missing or empty.
    """

    plugin_name = getattr(cls, "plugin_name", "")
    if not isinstance(plugin_name, str) or not plugin_name:
        raise ValueError("Subclasses must define a non-empty 'plugin_name' string.")
    return plugin_name


def require_config_type(cls: type[Any]) -> Any:
    """Fetch the ConfigType declared on a plugin class (if any)."""

    return getattr(cls, "ConfigType", None)


def forbid_unknown_keys(model: type[Any]) -> type[Any]:
    """Force a Pydantic model to reject unknown fields at parse time.

    Args:
        model: The Pydantic model class to modify.

    Returns:
        The modified model class.

    Raises:
        ImportError: If pydantic is not installed.
        TypeError: If model is not a Pydantic BaseModel.
    """

    if BaseModel is None:
        raise ImportError(
            "pydantic is not installed; cannot enforce unknown key behavior."
        )
    if not issubclass(model, BaseModel):
        raise TypeError("Model must inherit from pydantic.BaseModel.")

    extra_value = Extra.forbid if Extra is not None else "forbid"
    if hasattr(model, "model_config"):
        existing_config = getattr(model, "model_config") or {}
        merged_config = dict(existing_config)
        merged_config["extra"] = extra_value
        model.model_config = merged_config
        if hasattr(model, "model_rebuild"):
            model.model_rebuild(force=True)
        return model

    config_class = getattr(model, "Config", None)
    if config_class is None:

        class Config:
            extra = extra_value

        model.Config = Config
    else:
        setattr(config_class, "extra", extra_value)

    if hasattr(model, "model_rebuild"):
        model.model_rebuild(force=True)

    return model


__all__ = [
    "require_plugin_name",
    "require_config_type",
    "forbid_unknown_keys",
]
