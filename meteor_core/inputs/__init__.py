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
from .discovery import PLUGIN_DIR, PLUGIN_GROUP, discover_input_loaders

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
    # Discovery
    "PLUGIN_DIR",
    "PLUGIN_GROUP",
    "discover_input_loaders",
]
