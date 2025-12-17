"""Abstract base classes and helpers for input loader plugins."""

from abc import ABC, abstractmethod
from dataclasses import is_dataclass
import importlib.util
import logging
from typing import Any, Dict, Generic, Type, TypeVar

from meteor_core.plugin_contract import (
    forbid_unknown_keys as _forbid_unknown_keys,
    require_config_type,
    require_plugin_name,
)

ConfigType = TypeVar("ConfigType")

_PYDANTIC_SPEC = importlib.util.find_spec("pydantic")
if _PYDANTIC_SPEC:
    from pydantic import BaseModel
else:
    BaseModel = None

# Module-level logger for diagnostics shared by input loaders
logger = logging.getLogger(__name__)


# Exported helper for compatibility with existing imports
forbid_unknown_keys = _forbid_unknown_keys


class BaseInputLoader(ABC, Generic[ConfigType]):
    """Abstract base class for input loader plugins.

    All input loaders must inherit from this class to be discoverable
    and usable by the detection pipeline.

    See :doc:`PLUGIN_AUTHOR_GUIDE` for lifecycle details shared across
    plugin kinds (discovery order, config coercion, and hooks).

    Subclasses must define:
        - plugin_name: str - Unique identifier for the loader
        - load(filepath: str) -> Any - The loading method

    Attributes:
        plugin_name: Unique identifier for the loader plugin (used in registry).
        name: Human-readable name of the loader.
        version: Version string of the loader.
        config: Configuration instance for this loader.

    Example:
        >>> class MyLoader(BaseInputLoader):
        ...     plugin_name = "my_loader"
        ...     name = "My Custom Loader"
        ...     version = "1.0.0"
        ...
        ...     def __init__(self, config):
        ...         self.config = config
        ...
        ...     def load(self, filepath: str) -> np.ndarray:
        ...         return load_image(filepath)
    """

    #: Unique name identifying this loader plugin
    plugin_name: str = ""

    #: Human-readable name of the loader
    name: str = "BaseInputLoader"

    #: Version string of the loader
    version: str = "1.0.0"

    #: Configuration instance for this loader
    config: ConfigType

    @abstractmethod
    def load(self, filepath: str) -> Any:
        """Load an image from the given filepath.

        Args:
            filepath: Path to the image file to load.

        Returns:
            Loaded image data (typically a numpy array).
        """
        pass

    def get_info(self) -> Dict[str, str]:
        """Get information about the loader.

        Returns:
            Dictionary with loader metadata.
        """
        return {
            "plugin_name": self.plugin_name,
            "name": self.name,
            "version": self.version,
            "class": self.__class__.__name__,
        }


class BaseMetadataExtractor(ABC):
    """Abstract base class for metadata extraction.

    Loaders that need to provide metadata extraction should also inherit
    from this class. This is useful for extracting EXIF data or other
    file-specific information.

    Example:
        >>> class MyLoader(BaseInputLoader, BaseMetadataExtractor):
        ...     plugin_name = "my_loader"
        ...
        ...     def load(self, filepath: str) -> np.ndarray:
        ...         return load_image(filepath)
        ...
        ...     def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        ...         return extract_exif(filepath)
    """

    @abstractmethod
    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from the given filepath.

        Args:
            filepath: Path to the file to extract metadata from.

        Returns:
            Dictionary containing extracted metadata.
        """
        pass


def supports_metadata_extraction(loader: Any) -> bool:
    """Check if a loader supports metadata extraction.

    Args:
        loader: The loader instance to check.

    Returns:
        True if the loader is an instance of BaseMetadataExtractor.
    """
    return isinstance(loader, BaseMetadataExtractor)


def _is_valid_input_loader(cls: Type[Any]) -> bool:
    """Check if class is a valid InputLoader subclass.

    This function checks that a class inherits from BaseInputLoader
    and has a valid plugin_name defined.

    Args:
        cls: The class to check.

    Returns:
        True if the class is a valid BaseInputLoader subclass.
    """
    if not isinstance(cls, type):
        logger.debug(
            "_is_valid_input_loader: %r is not a type",
            cls,
        )
        return False
    if not issubclass(cls, BaseInputLoader):
        logger.debug(
            "_is_valid_input_loader: %s does not inherit from BaseInputLoader",
            cls.__name__,
        )
        return False
    # Check that plugin_name is defined and non-empty
    plugin_name = getattr(cls, "plugin_name", None)
    if not (
        plugin_name is not None and isinstance(plugin_name, str) and plugin_name != ""
    ):
        logger.debug(
            "_is_valid_input_loader: %s has invalid plugin_name: %r",
            cls.__name__,
            plugin_name,
        )
        return False
    return True


class DataclassInputLoader(BaseInputLoader[ConfigType], Generic[ConfigType]):
    """Abstract base class for loaders configured by dataclasses.

    This base class provides common functionality for loaders that use
    dataclasses for their configuration.

    Subclasses must define:
        - plugin_name: str - Unique identifier for the loader
        - ConfigType: Type - The dataclass type for configuration
        - load(filepath: str) -> Any - The loading method

    Example:
        >>> @dataclass
        ... class MyConfig:
        ...     option: str = "default"
        ...
        >>> class MyLoader(DataclassInputLoader[MyConfig]):
        ...     plugin_name = "my_loader"
        ...     ConfigType = MyConfig
        ...
        ...     def load(self, filepath: str) -> np.ndarray:
        ...         return load_with_option(filepath, self.config.option)
    """

    ConfigType: Type[ConfigType]

    def __init__(self, config: ConfigType) -> None:
        """Initialize the loader with configuration.

        Args:
            config: Configuration instance (must match ConfigType).

        Raises:
            ValueError: If plugin_name is not defined.
            TypeError: If ConfigType is not a dataclass or config type mismatches.
        """
        require_plugin_name(self.__class__, kind="input loader")
        config_type = require_config_type(self.__class__)
        if config_type is not None:
            if not is_dataclass(config_type):
                logger.error(
                    "ConfigType %s is not a dataclass for loader %s",
                    config_type,
                    self.__class__.__name__,
                )
                raise TypeError(
                    "ConfigType must be a dataclass type for DataclassInputLoader."
                )
            if not isinstance(config, config_type):
                logger.error(
                    "Config instance %s does not match dataclass %s for loader %s",
                    type(config).__name__,
                    config_type.__name__,
                    self.__class__.__name__,
                )
                raise TypeError(
                    f"config must be an instance of {config_type.__name__}."
                )
        self.config = config
        logger.debug(
            "%s initialized with dataclass config %s",
            self.__class__.__name__,
            config,
        )


class PydanticInputLoader(BaseInputLoader[ConfigType], Generic[ConfigType]):
    """Abstract base class for loaders configured by Pydantic models.

    This base class provides common functionality for loaders that use
    Pydantic models for their configuration with validation support.

    Subclasses must define:
        - plugin_name: str - Unique identifier for the loader
        - ConfigType: Type - The Pydantic model type for configuration
        - load(filepath: str) -> Any - The loading method

    Requires pydantic to be installed.
    """

    ConfigType: Type[ConfigType]

    def __init__(self, config: ConfigType) -> None:
        """Initialize the loader with configuration.

        Args:
            config: Configuration instance (must match ConfigType).

        Raises:
            ImportError: If pydantic is not installed.
            ValueError: If plugin_name is not defined.
            TypeError: If ConfigType is not a Pydantic model or config type mismatches.
        """
        require_plugin_name(self.__class__, kind="input loader")
        if BaseModel is None:
            logger.error(
                "PydanticInputLoader requires pydantic but it is not installed (%s)",
                self.__class__.__name__,
            )
            raise ImportError("pydantic must be installed to use PydanticInputLoader.")
        config_type = require_config_type(self.__class__)
        if config_type is not None:
            if not issubclass(config_type, BaseModel):
                logger.error(
                    "ConfigType %s is not a pydantic BaseModel for loader %s",
                    config_type,
                    self.__class__.__name__,
                )
                raise TypeError(
                    "ConfigType must inherit from pydantic.BaseModel for PydanticInputLoader."
                )
            if not isinstance(config, config_type):
                logger.error(
                    "Config instance %s does not match Pydantic model %s for loader %s",
                    type(config).__name__,
                    config_type.__name__,
                    self.__class__.__name__,
                )
                raise TypeError(
                    f"config must be an instance of {config_type.__name__}."
                )
        self.config = config
        logger.debug(
            "%s initialized with pydantic config %s",
            self.__class__.__name__,
            config,
        )
