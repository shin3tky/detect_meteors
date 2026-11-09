"""Identify plugin modules so workers can load the sources selected by the parent."""

from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType


@dataclass(frozen=True)
class PluginModule:
    """Import name and, for a plugin file, the source used during discovery."""

    name: str
    path: str | None = None
    digest: str | None = None

    @classmethod
    def from_name(cls, name: str) -> "PluginModule":
        module = sys.modules.get(name)
        local_source = getattr(module, "_detect_meteors_plugin_source", None)
        if local_source is not None:
            return local_source
        origin = getattr(module, "__file__", None)
        return cls(name, str(Path(origin).resolve()) if origin else None)

    def load(self) -> ModuleType:
        if self.path is not None and self.digest is not None:
            return load_local_plugin(Path(self.path), expected_digest=self.digest)
        module = import_module(self.name)
        if self.path is not None:
            origin = getattr(module, "__file__", None)
            if origin is None or Path(origin).resolve() != Path(self.path):
                raise ImportError(
                    f"Expected {self.name} from {self.path}, got {origin}"
                )
        return module


def load_local_plugin(path: Path, *, expected_digest: str | None = None) -> ModuleType:
    """Load the exact file as an importable module, including during spawn."""
    path = path.resolve()
    source = path.read_bytes()
    digest = sha256(source).hexdigest()
    if expected_digest is not None and digest != expected_digest:
        raise ImportError(f"Selected plugin source changed: {path}")

    path_key = sha256(str(path).encode()).hexdigest()
    name = f"_detect_meteors_plugin_{path_key}_{digest}"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached

    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    module = module_from_spec(spec)
    module._detect_meteors_plugin_source = PluginModule(name, str(path), digest)
    # Dataclasses and pickle both look up the defining module in sys.modules.
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module
