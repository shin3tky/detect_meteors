"""Protocols and helpers for input loader plugins."""

from dataclasses import is_dataclass
import importlib.util
from typing import Any, Generic, Protocol, Type, TypeVar, runtime_checkable

ConfigType = TypeVar("ConfigType")

_PYDANTIC_SPEC = importlib.util.find_spec("pydantic")
if _PYDANTIC_SPEC:
    import pydantic
    from pydantic import BaseModel

    Extra = getattr(pydantic, "Extra", None)
else:
    BaseModel = None
    Extra = None


@runtime_checkable
class InputLoader(Protocol[ConfigType]):
    """Protocol describing an input loader plugin."""

    plugin_name: str
    ConfigType: Type[ConfigType]

    def __init__(self, config: ConfigType) -> None: ...

    def load(self, filepath: str) -> Any: ...


def _require_plugin_name(cls: Type[Any]) -> str:
    plugin_name = getattr(cls, "plugin_name", "")
    if not isinstance(plugin_name, str) or not plugin_name:
        raise ValueError("Subclasses must define a non-empty 'plugin_name' string.")
    return plugin_name


def _require_config_type(cls: Type[Any]) -> Type[ConfigType]:
    config_type = getattr(cls, "ConfigType", None)
    if config_type is None:
        raise TypeError(
            "Subclasses must define a ConfigType for their loader configuration."
        )
    return config_type


class DataclassInputLoader(Generic[ConfigType]):
    """Base class for loaders configured by dataclasses."""

    ConfigType: Type[ConfigType]
    plugin_name: str

    def __init__(self, config: ConfigType) -> None:
        _require_plugin_name(self.__class__)
        config_type = _require_config_type(self.__class__)
        if not is_dataclass(config_type):
            raise TypeError(
                "ConfigType must be a dataclass type for DataclassInputLoader."
            )
        if not isinstance(config, config_type):
            raise TypeError(f"config must be an instance of {config_type.__name__}.")
        self.config = config


class PydanticInputLoader(Generic[ConfigType]):
    """Base class for loaders configured by Pydantic models."""

    ConfigType: Type[ConfigType]
    plugin_name: str

    def __init__(self, config: ConfigType) -> None:
        _require_plugin_name(self.__class__)
        if BaseModel is None:
            raise ImportError("pydantic must be installed to use PydanticInputLoader.")
        config_type = _require_config_type(self.__class__)
        if not issubclass(config_type, BaseModel):
            raise TypeError(
                "ConfigType must inherit from pydantic.BaseModel for PydanticInputLoader."
            )
        if not isinstance(config, config_type):
            raise TypeError(f"config must be an instance of {config_type.__name__}.")
        self.config = config


def forbid_unknown_keys(model: Type[Any]) -> Type[Any]:
    """Force a Pydantic model to reject unknown fields at parse time."""

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
