"""Base classes for meteor_core hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import is_dataclass
import importlib.util
import logging
from typing import Dict, Generic, Type, TypeVar

from ..plugin_contract import require_plugin_name
from ..schema import InputContext

logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType")

_PYDANTIC_SPEC = importlib.util.find_spec("pydantic")
if _PYDANTIC_SPEC:
    from pydantic import BaseModel
else:
    BaseModel = None


class BaseHook(ABC, Generic[ConfigType]):
    """Base class for pipeline hooks.

    Subclasses should define a non-empty ``plugin_name`` string.
    """

    plugin_name: str = ""
    name: str = "BaseHook"
    version: str = "1.0.0"

    @abstractmethod
    def on_file_found(self, filepath: str) -> bool:
        """Hook called for each discovered file path.

        Args:
            filepath: Absolute, normalized path to the file.

        Returns:
            True to keep the file, False to drop it.
        """

    def on_image_loaded(self, context: InputContext) -> InputContext:
        """Hook called after an image is loaded and normalized.

        Args:
            context: InputContext bundle for the loaded image.

        Returns:
            Updated InputContext (default: unchanged).
        """
        return context

    def get_info(self) -> Dict[str, str]:
        return {
            "plugin_name": self.plugin_name,
            "name": self.name,
            "version": self.version,
            "class": self.__class__.__name__,
        }


class DataclassHook(BaseHook[ConfigType], Generic[ConfigType]):
    """Base class for hooks configured by dataclasses."""

    ConfigType: Type[ConfigType]
    config: ConfigType

    def __init__(self, config: ConfigType) -> None:
        require_plugin_name(self.__class__, kind="hook")

        config_type = getattr(self.__class__, "ConfigType", None)
        if config_type is not None:
            if not is_dataclass(config_type):
                logger.error(
                    "ConfigType %s is not a dataclass for hook %s",
                    config_type,
                    self.__class__.__name__,
                )
                raise TypeError(
                    "ConfigType must be a dataclass type for DataclassHook."
                )
            if not isinstance(config, config_type):
                logger.error(
                    "Config instance %s does not match dataclass %s for hook %s",
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


class PydanticHook(BaseHook[ConfigType], Generic[ConfigType]):
    """Base class for hooks configured by Pydantic models."""

    ConfigType: Type[ConfigType]
    config: ConfigType

    def __init__(self, config: ConfigType) -> None:
        if BaseModel is None:
            logger.error(
                "PydanticHook requires pydantic but it is not installed (%s)",
                self.__class__.__name__,
            )
            raise ImportError(
                "pydantic is required to use PydanticHook. Install pydantic first."
            )

        require_plugin_name(self.__class__, kind="hook")

        config_type = getattr(self.__class__, "ConfigType", None)
        if config_type is not None:
            if not issubclass(config_type, BaseModel):
                logger.error(
                    "ConfigType %s is not a pydantic BaseModel for hook %s",
                    config_type,
                    self.__class__.__name__,
                )
                raise TypeError(
                    "ConfigType must be a pydantic BaseModel for PydanticHook."
                )
            if not isinstance(config, config_type):
                logger.error(
                    "Config instance %s does not match Pydantic model %s for hook %s",
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


def _is_valid_hook(cls: type[BaseHook]) -> bool:
    try:
        require_plugin_name(cls, kind="hook")
    except ValueError:
        return False
    return issubclass(cls, BaseHook)


__all__ = [
    "BaseHook",
    "DataclassHook",
    "PydanticHook",
    "_is_valid_hook",
]
