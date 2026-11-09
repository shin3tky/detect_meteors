"""Carry the parent's plugin selections and schema converters to worker batches."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
import sys
from types import ModuleType
from typing import Any, Callable, Generator

from ._plugin_import import PluginModule
from .exceptions import MeteorConfigError
from .hooks import BaseHook, HookRegistry
from .schema import HookConfig, _converter_registries


class WorkerPluginError(MeteorConfigError):
    """A worker could not prepare the plugin configuration chosen by the parent."""


@dataclass(frozen=True)
class PluginReference:
    module: PluginModule
    qualname: str

    @classmethod
    def capture(cls, obj: Callable) -> "PluginReference":
        module = PluginModule.from_name(obj.__module__)
        reference = cls(module, obj.__qualname__)
        if reference.resolve(sys.modules[module.name]) is not obj:
            raise ValueError(f"{module.name}.{reference.qualname} is not importable")
        return reference

    def resolve(self, module: ModuleType) -> Any:
        obj: Any = module
        for part in self.qualname.split("."):
            obj = getattr(obj, part)
        return obj


@dataclass(frozen=True)
class WorkerPlugins:
    """The resolved plugin sources and converters for a single pipeline run."""

    modules: tuple[PluginModule, ...]
    hook_classes: tuple[PluginReference, ...]
    converters: tuple[tuple[str, int, PluginReference], ...]
    plugin_names: tuple[str, ...]

    @classmethod
    def capture(
        cls, loader: Any, detector: Any, hooks: list[BaseHook]
    ) -> "WorkerPlugins":
        modules = dict.fromkeys(
            PluginModule.from_name(type(plugin).__module__)
            for plugin in (loader, detector)
            if plugin is not None
        )
        hook_classes = []
        for hook in hooks:
            try:
                reference = PluginReference.capture(type(hook))
            except (AttributeError, KeyError, ValueError) as exc:
                raise WorkerPluginError(
                    "Hook class is not importable in a worker",
                    config_key="hooks",
                    plugin_name=hook.plugin_name,
                    original_error=exc,
                ) from exc
            hook_classes.append(reference)
            modules[reference.module] = None

        converters = []
        for contract, registry in _converter_registries().items():
            for version, converter in registry.items():
                try:
                    reference = PluginReference.capture(converter)
                except (AttributeError, KeyError, ValueError) as exc:
                    raise WorkerPluginError(
                        f"Converter for {contract} schema version {version} "
                        "is not importable in a worker",
                        config_key="schema_converters",
                        original_error=exc,
                    ) from exc
                converters.append((contract, version, reference))
                modules[reference.module] = None
        names = tuple(
            plugin.plugin_name
            for plugin in (loader, detector, *hooks)
            if plugin is not None
        )
        return cls(tuple(modules), tuple(hook_classes), tuple(converters), names)


@dataclass(frozen=True)
class WorkerPluginState:
    """Defer snapshots until the pool serializes each queued batch."""

    loader: Any
    detector: Any
    hooks: tuple[HookConfig, ...]

    def __reduce__(self):
        # The worker receives bytes, so unpickling plugin objects can wait
        # until the selected modules and converters are ready.
        return bytes, (self.pack(),)

    def pack(self) -> bytes:
        """Snapshot plugin state with the reducers used by the process pool."""
        try:
            return bytes(ForkingPickler.dumps((self.loader, self.detector, self.hooks)))
        except Exception as exc:
            raise WorkerPluginError(
                "Could not serialize plugin state for a worker",
                config_key="plugins",
                context={
                    "input_loader": getattr(self.loader, "plugin_name", None),
                    "detector": getattr(self.detector, "plugin_name", None),
                    "hooks": [hook.name for hook in self.hooks],
                },
                original_error=exc,
            ) from exc


@contextmanager
def _use_converters(
    converters: dict[str, dict[int, Callable]],
) -> Generator[None, None, None]:
    registries = _converter_registries()
    previous = {name: registry.copy() for name, registry in registries.items()}
    try:
        for name, registry in registries.items():
            registry.clear()
            registry.update(converters[name])
        yield
    finally:
        for name, registry in registries.items():
            registry.clear()
            registry.update(previous[name])


@contextmanager
def restore_worker_plugins(
    plan: WorkerPlugins, payload: bytes | WorkerPluginState
) -> Generator[tuple[Any, Any, list[BaseHook]], None, None]:
    """Load selected modules before unpickling instances and constructing hooks."""
    with ExitStack() as stack:
        try:
            modules = {reference: reference.load() for reference in plan.modules}
            converters: dict[str, dict[int, Callable]] = {
                name: {} for name in _converter_registries()
            }
            for contract, version, reference in plan.converters:
                converter = reference.resolve(modules[reference.module])
                if not callable(converter):
                    raise TypeError(
                        f"Invalid converter for {contract} schema version {version}"
                    )
                converters[contract][version] = converter
            stack.enter_context(_use_converters(converters))
            if isinstance(payload, WorkerPluginState):
                payload = payload.pack()
            loader, detector, configs = ForkingPickler.loads(payload)
            hooks = [
                HookRegistry._create_from_class(
                    reference.resolve(modules[reference.module]), config.config
                )
                for reference, config in zip(plan.hook_classes, configs, strict=True)
            ]
        except Exception as exc:
            # The exception sent back to the parent must itself be picklable.
            raise WorkerPluginError(
                f"Could not prepare worker plugins: {exc}",
                config_key="plugins",
                context={
                    "plugins": plan.plugin_names,
                    "modules": [module.path or module.name for module in plan.modules],
                },
            ) from exc
        yield loader, detector, hooks
