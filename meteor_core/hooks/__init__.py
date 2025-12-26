"""Hooks for extending meteor_core pipeline behavior."""

from .base import BaseHook, DataclassHook, PydanticHook, _is_valid_hook
from .file_found import AllowAllFilesFoundHook
from .registry import HookRegistry

HookRegistry.register(AllowAllFilesFoundHook)

__all__ = [
    "AllowAllFilesFoundHook",
    "BaseHook",
    "DataclassHook",
    "PydanticHook",
    "HookRegistry",
    "_is_valid_hook",
]
