"""Input loader abstract base classes and helpers."""

from .base import (
    ConfigType,
    BaseInputLoader,
    BaseMetadataExtractor,
    DataclassInputLoader,
    PydanticInputLoader,
    forbid_unknown_keys,
    supports_metadata_extraction,
    _is_valid_input_loader,
)
from .raw import RawImageLoader, RawLoaderConfig, create_raw_loader
from .registry import LoaderRegistry
from .discovery import PLUGIN_DIR, PLUGIN_GROUP

# Deprecated: use LoaderRegistry.discover() instead
from .discovery import discover_input_loaders

__all__ = [
    # Type variables
    "ConfigType",
    # Abstract base classes
    "BaseInputLoader",
    "BaseMetadataExtractor",
    # Concrete base classes (for specific config types)
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
    # Registry (recommended)
    "LoaderRegistry",
    # Discovery constants
    "PLUGIN_DIR",
    "PLUGIN_GROUP",
    # Discovery function (deprecated)
    "discover_input_loaders",
]
