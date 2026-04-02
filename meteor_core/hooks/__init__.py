"""Hooks for extending meteor_core pipeline behavior."""

from .base import BaseHook, DataclassHook, PydanticHook, _is_valid_hook
from .aircraft_trail import AircraftTrailHook
from .file_found import AllowAllFilesFoundHook
from .registry import HookRegistry
from .discovery import PLUGIN_DIR, PLUGIN_GROUP

# Deprecated: use HookRegistry.discover() instead
from .discovery import discover_hooks

__all__ = [
    "AircraftTrailHook",
    "AllowAllFilesFoundHook",
    "BaseHook",
    "DataclassHook",
    "PydanticHook",
    "HookRegistry",
    "_is_valid_hook",
    "PLUGIN_DIR",
    "PLUGIN_GROUP",
    "discover_hooks",
]
