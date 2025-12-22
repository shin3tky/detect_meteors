# Version 1.6 Release Notes

## Version 1.6.2 (2025-12-23)

### 🔧 Input/Output Context Contracts

Version 1.6.2 extends schema versioning to input loaders and output handlers, completing the contract standardization across all plugin types.

### Highlights

- **Input context contracts**: `InputContext` formalizes loader return values with `schema_version`, `loader_info`, and `to_dict()`
- **Output result contracts**: `OutputResult` formalizes handler return values with `schema_version`, `handler_info`, `metrics`, and `to_dict()`
- **Complete plugin contract coverage**: All three plugin types (loaders, detectors, handlers) now have versioned contracts
- **Documentation updates**: Plugin Author Guide updated with comprehensive `InputContext`/`OutputResult` documentation

### Why This Change?

This release completes the schema versioning initiative started in v1.6.1:

| Plugin Type | v1.6.1 | v1.6.2 |
|-------------|--------|--------|
| **Detectors** | ✅ `DetectionContext`, `DetectionResult` | — |
| **Input Loaders** | — | ✅ `InputContext` |
| **Output Handlers** | — | ✅ `OutputResult` |

Benefits:
1. **Consistent contracts**: All plugin types follow the same pattern (schema_version, info dict, to_dict())
2. **Future migration support**: Version field enables gradual migration without breaking existing plugins
3. **Improved diagnostics**: `metrics` field in `OutputResult` for standardized performance tracking
4. **Serialization support**: `to_dict()` methods for JSON-compatible logging and debugging

### Schema Changes (v1.6.2)

**InputContext** (new in v1.6.2):
```python
@dataclass
class InputContext:
    image_data: ImageLike              # Loaded image (numpy, torch, or PIL)
    filepath: str                      # Original file path
    metadata: Dict[str, Any] = {}      # Loader-extracted metadata (EXIF, etc.)
    loader_info: Dict[str, Any] = {}   # Loader identity (name, version)
    schema_version: int = 1            # INPUT_CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for JSON/logging (excludes image_data)."""
        ...
```

**OutputResult** (new in v1.6.2):
```python
@dataclass
class OutputResult:
    saved: bool                        # Whether file was persisted
    output_path: Optional[str]         # Path to saved candidate
    debug_path: Optional[str]          # Path to saved debug image
    handler_info: Dict[str, Any] = {}  # Handler identity (name, version)
    metrics: Dict[str, Any] = {}       # Performance diagnostics
    schema_version: int = 1            # OUTPUT_RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result for JSON/logging."""
        ...
```

**Constants added**:
```python
INPUT_CONTEXT_SCHEMA_VERSION = 1
OUTPUT_RESULT_SCHEMA_VERSION = 1
```

### Contract Pattern Summary

All plugin contracts now follow a consistent pattern:

| Contract | Plugin Type | Info Field | Metrics | Schema Version |
|----------|-------------|------------|---------|----------------|
| `DetectionContext` | Detector (input) | — | — | ✅ |
| `DetectionResult` | Detector (output) | — | ✅ | ✅ |
| `InputContext` | Loader (output) | `loader_info` | — | ✅ |
| `OutputResult` | Handler (output) | `handler_info` | ✅ | ✅ |

### Migration Guide for Plugin Authors

Plugin architecture remains **experimental**, and v1.6.2 introduces a breaking change for output handlers used by the pipeline.

**Required for Output Handler authors**:
1. Return `OutputResult` from `save_candidate`. The pipeline expects `.saved`, `.output_path`, and `.debug_path`.
2. Update any legacy `OutputWriter`-based handlers (or wrappers returning `bool`) to return `OutputResult` instead.

**For Input Loader authors** (optional enhancements):
1. Return `InputContext` instead of raw image array
2. Populate `loader_info` using `self.get_info()` from `BaseInputLoader`
3. Use `context.to_dict()` for logging

**Recommended for Output Handler authors**:
1. Populate `handler_info` using `self.get_info()` from `BaseOutputHandler`
2. Use `metrics` for performance tracking (duration_ms, bytes_written, etc.)
3. Use `result.to_dict()` for logging

### Example: Updated Loader Implementation

```python
from meteor_core.inputs import DataclassInputLoader
from meteor_core.schema import InputContext

class MyLoader(DataclassInputLoader[MyConfig]):
    def load(self, filepath: str) -> InputContext:
        image = self._load_image(filepath)
        return InputContext(
            image_data=image,
            filepath=filepath,
            metadata=self.extract_metadata(filepath),
            loader_info=self.get_info(),  # {"name": ..., "version": ...}
        )
```

### Example: Updated Handler Implementation

```python
from meteor_core.outputs import DataclassOutputHandler
from meteor_core.schema import OutputResult
import time

class MyHandler(DataclassOutputHandler[MyConfig]):
    def save_candidate(self, source_path, filename, ...) -> OutputResult:
        start = time.perf_counter()
        dest_path = self._save_file(source_path, filename)
        duration_ms = (time.perf_counter() - start) * 1000

        return OutputResult(
            saved=True,
            output_path=dest_path,
            debug_path=None,
            handler_info=self.get_info(),
            metrics={"duration_ms": duration_ms},
        )
```

### Files Changed

| File | Changes |
|------|---------|
| `meteor_core/schema.py` | Added `InputContext`, `OutputResult`, schema version constants |
| `PLUGIN_AUTHOR_GUIDE.md` | Updated loader/handler documentation with new contracts |
| `CHANGELOG.md` | Added v1.6.2 entry |
| `README.md` | Updated "What's New" section |
| `ROADMAP.md` | Added v1.6.2 milestone |

### Backward Compatibility

✅ **Fully backward compatible** with v1.6.1 and v1.6.0:

- **CLI**: No changes
- **Runtime**: No changes to detection behavior
- **API**: Existing loader/handler plugins work unchanged
- **Configuration**: No changes


---

## Version 1.6.1 (2025-12-22)

### 🔧 Plugin Schema Versioning and ML-Ready Architecture

Version 1.6.1 introduces schema versioning for detector plugin contracts and multi-framework image support, laying the groundwork for ML-based detectors in v2.x.

### Highlights

- **Schema versioning**: `DetectionContext` and `DetectionResult` now include `schema_version` field

- **Multi-framework images**: New `ImageLike` type supports numpy, PyTorch, and PIL
- **Enhanced diagnostics**: `DetectionResult.metrics` for standardized performance metrics
- **Serialization support**: `DetectionResult.to_dict()` for JSON-compatible output

### Why This Change?

This release prepares the plugin architecture for:
1. **Future schema migrations**: Version field enables gradual migration without breaking existing plugins
2. **ML detector support**: `ImageLike` type allows detectors to work with PyTorch tensors natively
3. **Standardized metrics**: Consistent diagnostics across different detector implementations

### Schema Changes (v1.6.1)

**DetectionContext** (v1.6.1):
```python
@dataclass
class DetectionContext:
    current_image: ImageLike      # NEW: Union[np.ndarray, torch.Tensor, PIL.Image]
    previous_image: ImageLike     # NEW: Union[np.ndarray, torch.Tensor, PIL.Image]
    roi_mask: Any
    runtime_params: Dict[str, Any]
    metadata: Dict[str, Any]
    schema_version: int = 1       # NEW: For migration support
```

**DetectionResult** (v1.6.1):
```python
@dataclass
class DetectionResult:
    is_candidate: bool
    score: float
    lines: List[Tuple[int, int, int, int]]
    aspect_ratio: float
    debug_image: Optional[Any]
    extras: Dict[str, Any]
    metrics: Dict[str, Any]       # NEW: Standard diagnostics
    schema_version: int = 1       # NEW: For migration support
    def to_dict(self) -> Dict[str, Any]:  # NEW: Serialization
        ...
```

### New Utility Functions

**`ensure_numpy()`** in `meteor_core.utils`:
```python
from meteor_core.utils import ensure_numpy

# Convert any ImageLike to numpy array
image = ensure_numpy(context.current_image)  # Works with numpy, torch, PIL
```

### Migration Guide for Plugin Authors

No migration required. Existing plugins work without modification.

**Optional enhancements**:
1. Use `ensure_numpy()` for type-safe image handling
2. Populate `metrics` dict with standard diagnostics
3. Use `result.to_dict()` for logging

### Files Changed

| File | Changes |
|------|---------|
| `meteor_core/schema.py` | Added `ImageLike`, schema versions, `to_dict()` |
| `meteor_core/utils.py` | Added `ensure_numpy()` function |
| `PLUGIN_AUTHOR_GUIDE.md` | Updated DetectionContext/DetectionResult documentation |
| `CHANGELOG.md` | Added v1.6.1 entry |
| `README.md` | Updated "What's New" section |

### Backward Compatibility

✅ **Fully backward compatible** with v1.6.0:

- **CLI**: No changes
- **Runtime**: No changes to detection behavior
- **API**: All existing detector plugins work unchanged
- **Configuration**: No changes

---

## Version 1.6.0 (2025-12-21)

### ⚡ Development Toolchain Modernization

Version 1.6.0 modernizes the development toolchain by migrating from pip/black/flake8 to uv/Ruff. This change dramatically improves developer experience with faster dependency management and unified linting/formatting.

### Highlights

- **uv**: Rust-based Python package manager (10-100× faster than pip)
- **Ruff**: Rust-based linter and formatter replacing both black and flake8
- **Unified configuration**: All tool settings in `pyproject.toml`
- **Simplified setup**: Single command installs everything

### Why This Change?

| Aspect | Before (v1.5.x) | After (v1.6.0) |
|--------|-----------------|----------------|
| **Package manager** | pip | uv |
| **Formatter** | black | Ruff |
| **Linter** | flake8 | Ruff |
| **Config files** | `pyproject.toml` + `.flake8` | `pyproject.toml` only |
| **Install speed** | ~30-60 seconds | ~2-5 seconds |
| **Lint + format** | Two separate tools | Single unified tool |

### Performance Comparison

**Dependency installation:**
```bash
# Before (pip)
pip install -e ".[dev]"  # ~30-60 seconds

# After (uv)
uv sync --all-extras     # ~2-5 seconds
```

**Linting and formatting:**
```bash
# Before (black + flake8)
black .                  # ~2 seconds
flake8 .                 # ~1 second

# After (Ruff)
ruff check --fix .       # ~0.1 seconds
ruff format .            # ~0.1 seconds
```

### Migration Guide for Contributors

#### New Developer Setup

```bash
# 1. Install uv (if not already installed)
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone and setup
git clone https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
uv sync --all-extras

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Verify setup
uv run pre-commit run --all-files
```

#### For Existing Contributors

If you have an existing development environment:

```bash
# Update your local repository
git pull origin main

# Remove old virtual environment (optional but recommended)
rm -rf .venv

# Create new environment with uv
uv sync --all-extras

# Reinstall pre-commit hooks
uv run pre-commit install
```

### New Development Workflow

#### Running Commands

All Python commands should be prefixed with `uv run`:

```bash
# Run tests
uv run python run_tests.py

# Run the CLI
uv run python detect_meteors_cli.py --help

# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .
```

#### Pre-commit Hooks

The pre-commit configuration now uses local Ruff hooks:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff check
        entry: .venv/bin/ruff check --fix
        language: system
        types: [python]
      - id: ruff-format
        name: ruff format
        entry: .venv/bin/ruff format
        language: system
        types: [python]
```

### Configuration Changes

#### pyproject.toml

The `[tool.ruff]` section replaces both black and flake8 configurations:

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "C90"]
ignore = ["E203", "E501", "E226"]

[tool.ruff.lint.mccabe]
max-complexity = 40

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### Ruff vs black/flake8 Compatibility

Ruff is designed to be a drop-in replacement for black and flake8. Key compatibility notes:

| Feature | black/flake8 | Ruff |
|---------|--------------|------|
| Line length | `max-line-length = 88` | `line-length = 88` |
| Quote style | black default (double) | `quote-style = "double"` |
| Complexity | `max-complexity = 40` | `[tool.ruff.lint.mccabe] max-complexity = 40` |
| Ignored rules | `.flake8` ignore list | `[tool.ruff.lint] ignore` |

### Dependencies Updated

**Development dependencies (optional):**

```toml
[project.optional-dependencies]
dev = [
    "ruff==0.14.10",      # Replaces black + flake8
    "pre-commit>=4.5.0",
    "coverage>=7.6.0",
]
```

**Removed from dev dependencies:**
- `black>=25.12.0`
- `flake8>=7.3.0`
- `flake8-pyproject>=1.2.4`

### FAQ

#### Q: Do I need to install uv?

Yes, uv is now the recommended way to manage dependencies for this project. It's a single binary with no dependencies.

#### Q: Can I still use pip?

While pip still works for basic installation (`pip install -e .`), the development workflow is optimized for uv. Pre-commit hooks expect Ruff to be available via the uv-managed virtual environment.

#### Q: Will Ruff format code differently than black?

Ruff's formatter is designed to be compatible with black. In most cases, the output is identical. Minor differences may occur in edge cases, but these are cosmetic and don't affect functionality.

#### Q: Is the linting stricter or more lenient?

The same rule set is applied. Ruff supports the same error codes as flake8 (E, W, F, C90), and the ignore list has been preserved.

### Files Changed

| File | Changes |
|------|---------|
| `pyproject.toml` | Added `[tool.ruff]` section, updated dev dependencies |
| `.pre-commit-config.yaml` | Replaced black/flake8 with local Ruff hooks |
| `INSTALL_DEV.md` | Rewritten for uv/Ruff workflow |
| `CHANGELOG.md` | Added v1.6.0 entry |
| `README.md` | Updated "What's New" section |

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.x:

- **CLI**: No changes to command-line interface
- **Runtime**: No changes to detection behavior or output
- **API**: No changes to `meteor_core` module interfaces
- **Configuration**: No changes to detection parameters or sensor presets

This release only affects the development toolchain. Users who install the package without development dependencies will not notice any difference.

### Links

- [uv documentation](https://docs.astral.sh/uv/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [INSTALL_DEV.md](INSTALL_DEV.md) - Updated developer setup guide

---

## Version Information

### v1.6.2 (Latest)
- **Version**: 1.6.2
- **Release Date**: 2025-12-23
- **Major Changes**:
  - InputContext for standardized loader return values
  - OutputResult for standardized handler return values
  - Schema versioning for all plugin contracts
  - Complete plugin contract documentation

### v1.6.1
- **Version**: 1.6.1
- **Release Date**: 2025-12-22
- **Major Changes**:
  - Schema versioning for DetectionContext/DetectionResult
  - ImageLike type for multi-framework support
  - DetectionResult.metrics for standardized diagnostics
  - ensure_numpy() utility for type-safe conversion

### v1.6.0
- **Version**: 1.6.0
- **Release Date**: 2025-12-21
- **Major Changes**:
  - pip → uv (package management)
  - black + flake8 → Ruff (formatting + linting)
  - Unified configuration in pyproject.toml
  - Simplified developer setup workflow


---

**Status**: Production Ready  
**Compatibility**: Fully backward compatible with v1.5.x  
**Recommendation**: Contributors should migrate to uv/Ruff workflow; plugin authors should adopt new contract types

Happy meteor hunting! 🌠
