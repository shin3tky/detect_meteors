"""Abstract base classes and helpers for input loader plugins."""

from abc import ABC, abstractmethod
from dataclasses import is_dataclass
import importlib.util
from typing import Any, Dict, Generic, Type, TypeVar

ConfigType = TypeVar("ConfigType")

_PYDANTIC_SPEC = importlib.util.find_spec("pydantic")
if _PYDANTIC_SPEC:
    import pydantic
    from pydantic import BaseModel

    Extra = getattr(pydantic, "Extra", None)
else:
    BaseModel = None
    Extra = None


class BaseInputLoader(ABC, Generic[ConfigType]):
    """Abstract base class for input loader plugins.

    All input loaders must inherit from this class to be discoverable
    and usable by the detection pipeline.

    Subclasses must define:
        - plugin_name: str - Unique identifier for the loader
        - load(filepath: str) -> Any - The loading method

    Attributes:
        plugin_name: Unique string identifier for this loader plugin.
        config: Configuration instance for this loader.

    Example:
        >>> class MyLoader(BaseInputLoader):
        ...     plugin_name = "my_loader"
        ...
        ...     def __init__(self, config):
        ...         self.config = config
        ...
        ...     def load(self, filepath: str) -> np.ndarray:
        ...         return load_image(filepath)
    """

    #: Unique name identifying this loader plugin
    plugin_name: str = ""

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
        return False
    if not issubclass(cls, BaseInputLoader):
        return False
    # Check that plugin_name is defined and non-empty
    plugin_name = getattr(cls, "plugin_name", None)
    return (
        plugin_name is not None and isinstance(plugin_name, str) and plugin_name != ""
    )


def _require_plugin_name(cls: Type[Any]) -> str:
    """Extract and validate plugin_name from a loader class.

    Args:
        cls: The loader class to check.

    Returns:
        The plugin_name string.

    Raises:
        ValueError: If plugin_name is missing or invalid.
    """
    plugin_name = getattr(cls, "plugin_name", "")
    if not isinstance(plugin_name, str) or not plugin_name:
        raise ValueError("Subclasses must define a non-empty 'plugin_name' string.")
    return plugin_name


def _require_config_type(cls: Type[Any]) -> Type[ConfigType]:
    """Extract ConfigType from a loader class (optional).

    Args:
        cls: The loader class to check.

    Returns:
        The ConfigType class, or None if not defined.
    """
    config_type = getattr(cls, "ConfigType", None)
    # ConfigType is optional - loaders may not require configuration
    return config_type


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
        _require_plugin_name(self.__class__)
        config_type = _require_config_type(self.__class__)
        if config_type is not None:
            if not is_dataclass(config_type):
                raise TypeError(
                    "ConfigType must be a dataclass type for DataclassInputLoader."
                )
            if not isinstance(config, config_type):
                raise TypeError(
                    f"config must be an instance of {config_type.__name__}."
                )
        self.config = config


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
        _require_plugin_name(self.__class__)
        if BaseModel is None:
            raise ImportError("pydantic must be installed to use PydanticInputLoader.")
        config_type = _require_config_type(self.__class__)
        if config_type is not None:
            if not issubclass(config_type, BaseModel):
                raise TypeError(
                    "ConfigType must inherit from pydantic.BaseModel for PydanticInputLoader."
                )
            if not isinstance(config, config_type):
                raise TypeError(
                    f"config must be an instance of {config_type.__name__}."
                )
        self.config = config


def forbid_unknown_keys(model: Type[Any]) -> Type[Any]:
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
        return model

    config_class = getattr(model, "Config", None)
    if config_class is None:

        class Config:
            extra = extra_value

        model.Config = Config
    else:
        setattr(config_class, "extra", extra_value)

    return model
