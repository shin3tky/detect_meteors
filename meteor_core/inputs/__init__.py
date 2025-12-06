"""Input loader protocols and helper base classes."""

from .base import (
    ConfigType,
    DataclassInputLoader,
    InputLoader,
    PydanticInputLoader,
    forbid_unknown_keys,
)
from .raw import RawImageLoader, RawLoaderConfig, create_raw_loader
from .discovery import PLUGIN_DIR, PLUGIN_GROUP, discover_input_loaders

__all__ = [
    "ConfigType",
    "DataclassInputLoader",
    "InputLoader",
    "PydanticInputLoader",
    "forbid_unknown_keys",
    "RawImageLoader",
    "RawLoaderConfig",
    "create_raw_loader",
    "PLUGIN_DIR",
    "PLUGIN_GROUP",
    "discover_input_loaders",
]
