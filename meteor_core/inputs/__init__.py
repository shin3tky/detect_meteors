"""Input loader protocols and helper base classes."""

from .base import (
    ConfigType,
    DataclassInputLoader,
    InputLoader,
    MetadataExtractor,
    PydanticInputLoader,
    forbid_unknown_keys,
    supports_metadata_extraction,
    _is_valid_input_loader,
)
from .raw import RawImageLoader, RawLoaderConfig, create_raw_loader
from .discovery import PLUGIN_DIR, PLUGIN_GROUP, discover_input_loaders

__all__ = [
    # Type variables
    "ConfigType",
    # Protocols
    "InputLoader",
    "MetadataExtractor",
    # Base classes
    "DataclassInputLoader",
    "PydanticInputLoader",
    # Utility functions
    "forbid_unknown_keys",
    "supports_metadata_extraction",
    "_is_valid_input_loader",
    # RAW loader
    "RawImageLoader",
    "RawLoaderConfig",
    "create_raw_loader",
    # Discovery
    "PLUGIN_DIR",
    "PLUGIN_GROUP",
    "discover_input_loaders",
]
