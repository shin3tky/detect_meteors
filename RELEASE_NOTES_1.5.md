# Version 1.5 Release Notes

## Version 1.5.13 (2025-12-19) ⛄️

### 🌐 Internationalization (i18n)

Version 1.5.13 adds multi-language support for CLI user-facing messages. The localization follows a clear policy: UI/UX messages (error descriptions, progress indicators, completion summaries) are translated, while system-level output and debug information remain in English for consistency and easier troubleshooting.

### Highlights

- **New `--locale` option**: Specify the display language for CLI messages
- **Environment variable support**: Set default locale via `DETECT_METEORS_LOCALE`
- **ICU-style message templates**: Full support for plural rules and parameter substitution
- **YAML-based catalogs**: Easy-to-edit locale files under `meteor_core/locales/`
- **Supported languages**: English (`en`), Japanese (`ja`)

### Localization Policy

| Category | Language | Examples |
|----------|----------|----------|
| **UI/UX messages** | Localized | Error headers, progress messages, completion summaries |
| **System output** | English only | Stack traces, debug logs, diagnostic reports |
| **Technical terms** | English only | Exception class names, parameter names |

### Usage

#### Setting the Locale via CLI

```bash
# Use Japanese for user messages
python detect_meteors_cli.py --auto-params --sensor-type MFT --locale ja

# Use English (default)
python detect_meteors_cli.py --auto-params --sensor-type MFT --locale en
```

#### Setting the Default Locale via Environment Variable

```bash
# Set default locale for all runs
export DETECT_METEORS_LOCALE=ja

# Now runs use Japanese by default
python detect_meteors_cli.py --auto-params --sensor-type MFT
```

### Example Output

**English (`--locale en`):**
```
Complete! 3 candidates extracted
```

**Japanese (`--locale ja`):**
```
完了！候補を 3 件抽出しました
```

### Message Catalog Structure

Locale catalogs use YAML format with nested keys:

```yaml
# meteor_core/locales/ja/messages.yaml
ui:
  error:
    header: "エラー: {message}"
    filepath: "ファイル: {filepath}"
  run:
    summary: "{count, plural, =0 {完了！抽出された候補はありませんでした} other {完了！候補を # 件抽出しました}}"
```

### ICU Plural Rules

The i18n system supports ICU-style plural rules:

```yaml
# English (has singular/plural distinction)
ui:
  run:
    summary: "{count, plural, =0 {No candidates} one {# candidate} other {# candidates}}"

# Japanese (no grammatical number)
ui:
  run:
    summary: "{count, plural, =0 {候補なし} other {# 件}}"
```

### Adding a New Language

1. Create a new directory: `meteor_core/locales/<lang>/`
2. Add `__init__.py` (empty file)
3. Add `messages.yaml` with translated messages
4. Test with `--locale <lang>`

### Technical Details

#### New Module: `meteor_core/i18n.py`

Key functions:
- `get_message(key, locale, **params)`: Retrieve and format a localized message
- `log_warning(logger, key, locale, **params)`: Log a localized warning message

#### Locale Resolution

The system tries locales in order:
1. Exact match (e.g., `ja-JP`)
2. Language code (e.g., `ja`)
3. Default locale (`en`)

### Progress File Normalization

Version 1.5.13 also adds `normalize_progress_data()` helper that:
- Sanitizes `progress.json` contents
- Recomputes totals from actual file lists
- Handles malformed entries gracefully
- Logs warnings for invalid data (in English, following the localization policy)

### Validation Improvement

`validate_and_apply_sensor_preset()` now raises `MeteorValidationError` directly instead of returning error tuples, providing cleaner error handling and better integration with the exception hierarchy.

### New Dependency

- **PyYAML**: Required for parsing YAML-based locale catalogs

### Test Coverage

New test files:
- `test_i18n.py`: i18n module functionality tests
- `test_infrastructure_v1x.py`: Progress normalization tests

### Files Changed

| File | Changes |
|------|---------|
| `meteor_core/i18n.py` | New i18n module with ICU-style formatting |
| `meteor_core/locales/en/messages.yaml` | English message catalog |
| `meteor_core/locales/ja/messages.yaml` | Japanese message catalog |
| `meteor_core/outputs/progress.py` | Progress normalization helpers |
| `meteor_core/exceptions.py` | Updated for localized error formatting |
| `detect_meteors_cli.py` | Added `--locale` option, integrated i18n |
| `detect_meteors_cli_completion.bash` | Added `--locale` completion |
| `_detect_meteors_cli` (zsh) | Added `--locale` completion |
| `pyproject.toml` | Added PyYAML dependency |
| `requirements.txt` | Added PyYAML dependency |

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.12 and earlier:
- Default locale is English (`en`)
- All existing commands work without modification
- No changes to output file formats

---

## Version 1.5.12 (2025-12-18) 🔮

### 🛡️ Stability & Error Handling

Version 1.5.12 focuses on code stability, introducing a comprehensive exception hierarchy and structured diagnostic reporting to improve troubleshooting and bug reporting workflows.

### Highlights

- **Custom exception hierarchy**: Structured error classes for inputs (`MeteorLoadError`, `MeteorUnsupportedFormatError`), outputs (`MeteorOutputError`, `MeteorWriteError`, `MeteorProgressError`), and configuration (`MeteorValidationError`, `MeteorConfigError`)
- **Diagnostic reporting**: GitHub issue-ready diagnostic reports with system information
- **New CLI options**: `--verbose` for detailed logging, `--save-diagnostic` for error reports
- **Structured logging**: Standard Python logging throughout all modules

### Exception Hierarchy

```
MeteorError (base)
├── MeteorLoadError (image loading failures)
│   └── MeteorUnsupportedFormatError (unsupported file formats)
├── MeteorOutputError (output operation failures)
│   ├── MeteorWriteError (file write failures)
│   └── MeteorProgressError (progress tracking errors)
├── MeteorValidationError (parameter/input validation)
└── MeteorConfigError (configuration errors)
```

Each exception includes:
- Human-readable error message
- File path (if applicable)
- Original exception (for chained errors)
- Additional context information
- Full diagnostic info for bug reporting

### New CLI Options

#### `--verbose`

Enable detailed diagnostic information on errors and DEBUG-level logging:

```bash
python detect_meteors_cli.py --auto-params --sensor-type MFT --verbose
```

When `--verbose` is enabled:
- DEBUG-level logs from input handling modules are displayed
- Full diagnostic information is shown on errors
- Stack traces are included for unexpected errors

#### `--save-diagnostic [FILE]`

Save a diagnostic report file on error:

```bash
python detect_meteors_cli.py --auto-params --sensor-type MFT --save-diagnostic
# Creates: meteor_diagnostic_20251218_123456.md

python detect_meteors_cli.py --auto-params --sensor-type MFT --save-diagnostic my_report.md
# Creates: my_report.md
```

The diagnostic report includes:
- meteor_core version
- Python version and platform
- Dependency versions (numpy, opencv-python, rawpy, etc.)
- File information (if applicable)
- Error details and context
- Formatted for easy GitHub issue submission

### Diagnostic Report Example

```markdown
## Diagnostic Information

```
meteor_core version: 1.5.12
Python version: 3.12.0 (main, Oct  2 2024, 00:00:00) [Clang 16.0.0]
Platform: Darwin 23.0.0 (arm64)
Timestamp: 2025-12-18T12:34:56+00:00
```

### File Information

```
Path: /path/to/corrupted.CR2
Exists: True
Size: 25,432,100 bytes
```

### Error Details

```
Type: MeteorLoadError
Message: Failed to load RAW file
Original Error Type: LibRawIOError
Original Error Message: Cannot open file
```

### Dependencies

```
numpy: 2.2.6
opencv-python: 4.12.0
rawpy: 0.25.1
pillow: 12.0.0
psutil: 7.1.3
pydantic: 2.10.0
```
```

### Structured Logging

All `meteor_core` modules now use Python's standard `logging` module:

```python
import logging

# Enable debug logging for troubleshooting
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('meteor_core')
logger.setLevel(logging.DEBUG)
```

Modules with logging support:
- `meteor_core.inputs` (loader discovery, configuration)
- `meteor_core.detectors` (detector discovery, detection process)
- `meteor_core.outputs` (output handling, file writing)
- `meteor_core.pipeline` (processing orchestration)
- `meteor_core.image_io` (RAW file loading, EXIF extraction)

### Exception Usage Examples

**Catching specific exceptions**:

```python
from meteor_core.exceptions import (
    MeteorError,
    MeteorLoadError,
    MeteorOutputError,
    MeteorWriteError,
    MeteorProgressError,
    MeteorValidationError,
)

try:
    result = process_images(target_folder)
except MeteorLoadError as e:
    # Handle image loading failures
    print(f"Failed to load: {e.filepath}")
    print(e.format_for_issue())  # Get diagnostic report
except MeteorWriteError as e:
    # Handle file write failures
    print(f"Failed to write: {e.destination_path}")
    print(f"Operation: {e.operation}")
except MeteorProgressError as e:
    # Handle progress tracking errors
    print(f"Progress file error: {e.filepath}")
except MeteorOutputError as e:
    # Catch-all for output errors
    print(f"Output error: {e.message}")
except MeteorValidationError as e:
    # Handle validation errors
    print(f"Invalid parameter: {e.parameter_name}")
except MeteorError as e:
    # Catch-all for meteor_core errors
    print(f"Error: {e.message}")
```

**Creating custom exceptions with context**:

```python
from meteor_core.exceptions import MeteorLoadError, MeteorWriteError, MeteorProgressError

# Input loading error
raise MeteorLoadError(
    "Failed to load RAW file",
    filepath="/path/to/image.CR2",
    original_error=original_exception,
    context={
        "loader": "raw",
        "binning": 2,
        "normalize": True,
    },
)

# File write error
raise MeteorWriteError(
    "Failed to copy candidate file",
    filepath="/source/image.CR2",
    destination_path="/output/image.CR2",
    operation="copy",
    original_error=os_error,
    context={"error_category": "copy_failed"},
)

# Progress tracking error
raise MeteorProgressError(
    "Failed to parse progress file",
    filepath="progress.json",
    operation="parse",
    original_error=json_error,
    context={"error_category": "parse_failed"},
)
```

### Test Coverage

New test files added:
- `test_exceptions_v1x.py`: Exception hierarchy and diagnostic info tests
- `test_inputs_logging_v1x.py`: Logging configuration tests

### Files Changed

| File | Changes |
|------|---------|
| `meteor_core/exceptions.py` | New exception module with hierarchy and diagnostics |
| `meteor_core/__init__.py` | Logging configuration and exception exports |
| `meteor_core/inputs/*.py` | Added logging throughout |
| `meteor_core/detectors/*.py` | Added logging throughout |
| `meteor_core/outputs/*.py` | Added logging throughout |
| `meteor_core/pipeline.py` | Added logging, exception handling |
| `meteor_core/image_io.py` | Added logging, exception wrapping |
| `detect_meteors_cli.py` | Added `--verbose`, `--save-diagnostic`, error handling |

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.11 and earlier:
- CLI interface unchanged except for new options
- All existing commands work without modification
- Exception handling is opt-in for library users

---

## Version 1.5.11 (2025-12-15) 📜

### 🔧 Cross-Package Consistency & Plugin Metadata

Version 1.5.11 focuses on unifying plugin registry behaviors and improving the plugin development experience.

### Highlights

- **Case-insensitive registry lookup**: Both `LoaderRegistry` and `DetectorRegistry` now support case-insensitive name lookup (e.g., `get("raw")`, `get("RAW")`, `get("Raw")` all return the same class).
- **Unified plugin directories**: Standardized naming convention for plugin directories (`~/.detect_meteors/input_plugins/`, `~/.detect_meteors/detector_plugins/`, `~/.detect_meteors/output_plugins/`).
- **BaseInputLoader metadata**: Added `name`, `version` attributes and `get_info()` method to match `BaseDetector` interface.
- **Early config validation**: Registry config coercion now raises explicit `TypeError` or `ValueError` instead of silently returning `None`.
- **Deprecated function rename**: `discover_input_loaders()` renamed to `discover_loaders()` for naming consistency.

### For Plugin Developers

For detailed information on creating custom plugins, including complete examples and best practices, see [PLUGIN_AUTHOR_GUIDE.md](PLUGIN_AUTHOR_GUIDE.md).

### Migration Notes

⚠️ **For plugin developers**:
- Move plugins from `~/.detect_meteors/plugins/` to `~/.detect_meteors/input_plugins/`
- Update imports if using `discover_input_loaders()` → use `LoaderRegistry.discover()` instead
- Consider adding `name` and `version` attributes to your loaders for metadata consistency

### Backward Compatibility

✅ **Internal refactoring only** - no breaking changes to CLI or runtime behavior.


---

## Version 1.5.10 (2025-12-11) ⛰️

### 🏗️ Plugin Architecture: ABC Migration

Version 1.5.10 migrates all plugin interfaces from Protocol-based to Abstract Base Classes (ABC), providing a more robust foundation for plugin development.

> ⚠️ **Important**: The plugin architecture is experimental and may undergo changes before the v2.0 stable release. Plugin interfaces, discovery mechanisms, and base class signatures could be modified based on feedback and evolving requirements.

### Why ABC over Protocol?

| Aspect | ABC (New) | Protocol (Old) |
|--------|-----------|----------------|
| **Error detection** | Immediate at instantiation | Runtime only |
| **IDE support** | Full (auto-complete, warnings) | Limited |
| **Learning curve** | Familiar pattern | Requires typing knowledge |
| **Shared implementation** | Supported (default methods) | Not supported |
| **Discoverability** | Clear inheritance hierarchy | Implicit structural matching |

### Migration Summary

| Old (Protocol) | New (ABC) | Location |
|----------------|-----------|----------|
| `InputLoader` | `BaseInputLoader` | `meteor_core/inputs/base.py` |
| `MetadataExtractor` | `BaseMetadataExtractor` | `meteor_core/inputs/base.py` |
| `OutputHandler` | `BaseOutputHandler` | `meteor_core/outputs/base.py` |
| `BaseDetector` | `BaseDetector` (unchanged) | `meteor_core/detectors/base.py` |

### Class Hierarchy

```
BaseInputLoader (ABC)
├── DataclassInputLoader (ABC + Generic)
│   └── RawImageLoader (+ BaseMetadataExtractor)
└── PydanticInputLoader (ABC + Generic)

BaseMetadataExtractor (ABC)
└── RawImageLoader (multiple inheritance)

BaseOutputHandler (ABC)
├── DataclassOutputHandler (ABC + Generic)
│   └── FileOutputHandler
└── PydanticOutputHandler (ABC + Generic)

BaseDetector (ABC)
├── DataclassDetector (ABC + Generic)
│   ├── HoughDetector
│   └── SimpleThresholdDetector
└── PydanticDetector (ABC + Generic)
```

### Creating Custom Plugins

#### Custom Input Loader

```python
from typing import Dict, Any
import numpy as np
from meteor_core.inputs.base import BaseInputLoader, BaseMetadataExtractor


class MyCustomLoader(BaseInputLoader, BaseMetadataExtractor):
    """Custom loader for a specific image format."""

    plugin_name = "my_format"  # Required: unique identifier

    def __init__(self, config: Any = None):
        self.config = config

    def load(self, filepath: str) -> np.ndarray:
        """Load an image from the given filepath."""
        # Your implementation here
        pass

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from the file (optional)."""
        return {"format": "my_format"}
```

#### Custom Detector

```python
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from meteor_core.detectors.base import BaseDetector


class MyCustomDetector(BaseDetector):
    """Custom meteor detection algorithm."""

    name = "MyCustomDetector"
    version = "1.0.0"

    def detect(
        self,
        current_image: np.ndarray,
        previous_image: np.ndarray,
        roi_mask: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[bool, float, List[Tuple[int, int, int, int]], float, Optional[np.ndarray]]:
        """Detect meteor candidates."""
        # Your detection logic here
        pass

    def compute_line_score(
        self,
        mask: np.ndarray,
        hough_params: Dict[str, int],
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """Compute line detection score."""
        # Your line scoring logic here
        pass
```

#### Custom Output Handler

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from meteor_core.outputs import DataclassOutputHandler, OutputHandlerRegistry


@dataclass
class CloudOutputConfig:
    bucket_name: str
    prefix: str = "meteors/"


class MyCustomOutputHandler(DataclassOutputHandler[CloudOutputConfig]):
    """Custom output handler (e.g., for cloud storage)."""

    plugin_name = "cloud"
    ConfigType = CloudOutputConfig

    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: Optional[np.ndarray] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> bool:
        """Save a meteor candidate."""
        return True

    def save_debug_image(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> str:
        """Save a debug visualization."""
        return f"s3://{self.config.bucket_name}/{self.config.prefix}{filename}"


OutputHandlerRegistry.register(MyCustomOutputHandler)
handler = OutputHandlerRegistry.create("cloud", {"bucket_name": "my-bucket"})
handler.save_candidate("/tmp/source.CR2", "source.CR2")
```

### Benefits for Plugin Developers

1. **Immediate feedback**: Missing abstract methods cause `TypeError` at instantiation, not at runtime
2. **IDE integration**: Full auto-complete and "implement abstract methods" quick fixes
3. **Clear contracts**: Explicit inheritance makes required methods obvious
4. **Documentation**: ABC docstrings appear in IDE tooltips and help()

### Files Changed

| File | Changes |
|------|---------|
| `meteor_core/inputs/base.py` | `InputLoader` → `BaseInputLoader`, `MetadataExtractor` → `BaseMetadataExtractor` |
| `meteor_core/inputs/raw.py` | Updated inheritance |
| `meteor_core/inputs/__init__.py` | Updated exports |
| `meteor_core/inputs/discovery.py` | Updated type hints and skip classes |
| `meteor_core/outputs/base.py` | `OutputHandler` → `BaseOutputHandler` |
| `meteor_core/outputs/writer.py` | Updated inheritance |
| `meteor_core/outputs/__init__.py` | Updated exports |
| `meteor_core/__init__.py` | Updated exports |
| `meteor_core/pipeline.py` | Updated type hints |
| `INSTALL_DEV.md` | Added comprehensive plugin architecture documentation |

### Backward Compatibility

✅ **Internal refactoring only** - no breaking changes to CLI or runtime behavior:
- All existing commands work unchanged
- Detection results are identical
- Progress files remain compatible

⚠️ **For plugin developers**: If you have custom plugins using the old Protocol-based interfaces, update imports:

```python
# Old (v1.5.6-v1.5.9)
from meteor_core.inputs import InputLoader, MetadataExtractor
from meteor_core.outputs import OutputHandler

# New (v1.5.10+)
from meteor_core.inputs import BaseInputLoader, BaseMetadataExtractor
from meteor_core.outputs import BaseOutputHandler
```

---

## Version 1.5.9 (2025-12-10) 👤

### 📦 PEP 621 Project Configuration

Version 1.5.9 modernizes the project configuration by migrating to PEP 621 compliant `pyproject.toml`, consolidating all tool configurations and project metadata in a single file.

### Highlights

- **PEP 621 compliance**: Full project metadata now defined in `pyproject.toml` including name, description, authors, maintainers, keywords, and classifiers.
- **Unified tool configuration**: Consolidated flake8 settings from `.flake8` into `pyproject.toml` via flake8-pyproject, providing single-file configuration.
- **Dependency management**: Runtime and optional dev dependencies clearly defined for clearer package requirements.
- **Test infrastructure**: Testing and coverage configurations added to `pyproject.toml`.

### Configuration Overview

#### Project Metadata

```toml
[project]
name = "detect-meteors"
version = "1.5.9"
description = "Automated meteor detection in RAW astrophotography images using frame-to-frame difference analysis"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.12"
```

#### Dependencies

**Runtime dependencies:**
- `numpy>=2.2.6` - Numerical computing
- `opencv-python>=4.12.0` - Image processing
- `rawpy>=0.25.1` - RAW image reading
- `psutil>=7.1.3` - System utilities
- `pillow>=12.0.0` - Image handling and EXIF extraction

**Development dependencies (optional):**
```toml
[project.optional-dependencies]
dev = [
    "black>=25.12.0",
    "flake8>=7.3.0",
    "flake8-pyproject>=1.2.4",
    "pre-commit>=4.5.0",
    "coverage>=7.6.0",
]
```

#### Flake8 Configuration (Migrated)

The flake8 configuration has been moved from `.flake8` to `pyproject.toml`:

```toml
[tool.flake8]
max-line-length = 88
max-complexity = 70
exclude = [
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    # ... other exclusions
]
ignore = ["E203", "W503", "E501", "E226"]
```

**Note**: `flake8-pyproject>=1.2.4` is required to read flake8 configuration from `pyproject.toml`.

#### Testing Configuration

```toml
[tool.coverage.run]
source = ["meteor_core"]
branch = true
omit = ["tests/*", "*/__pycache__/*", ".venv/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
show_missing = true
```

#### Project URLs

```toml
[project.urls]
Homepage = "https://github.com/shin3tky/detect_meteors"
Repository = "https://github.com/shin3tky/detect_meteors.git"
Documentation = "https://github.com/shin3tky/detect_meteors#readme"
Issues = "https://github.com/shin3tky/detect_meteors/issues"
Changelog = "https://github.com/shin3tky/detect_meteors/blob/main/CHANGELOG.md"
```

### Pre-commit Hook Update

The `.pre-commit-config.yaml` now includes `Flake8-pyproject` as an additional dependency:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 25.12.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 7.3.0
    hooks:
      - id: flake8
        additional_dependencies: [Flake8-pyproject]
```

### Files Changed

| File | Changes |
|------|---------|
| `pyproject.toml` | Full PEP 621 metadata, flake8 config, testing config |
| `.flake8` | Removed (migrated to pyproject.toml) |
| `.pre-commit-config.yaml` | Added Flake8-pyproject dependency |
| `CHANGELOG.md` | Added v1.5.9 entry |
| `README.md` | Added v1.5.9 section |

### Benefits

1. **Single source of truth**: All project configuration in one file
2. **Modern Python packaging**: PEP 621 compliance for future pip/PyPI compatibility
3. **Clearer dependencies**: Explicit version requirements for all packages
4. **Reduced file clutter**: Removed separate `.flake8` configuration file
5. **Standardized testing**: Coverage configuration for consistent test measurement

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.8 and earlier:
- No changes to runtime behavior
- No breaking changes to API or CLI options
- Development workflow unchanged (flake8 reads from pyproject.toml transparently)

### Developer Notes

When installing development dependencies:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or install individual tools
pip install black flake8 flake8-pyproject coverage pre-commit
```

---

## Version 1.5.8 (2025-12-09) ♿️

### 🔍 Code Quality Improvements

Version 1.5.8 strengthens the development workflow by integrating flake8 linter alongside the existing Black formatter, establishing comprehensive code quality standards for the project.

### Highlights

- **flake8 linter integration**: Added flake8 to complement Black formatter, providing static code analysis and style checking beyond automatic formatting.
- **Project-specific configuration**: Implemented `.flake8` configuration file with rules optimized for Black compatibility and project requirements.
- **Enhanced developer workflow**: Manual linting checks before commits help maintain consistent code quality and catch potential issues early.

### Configuration Details

#### `.flake8` Configuration
```ini
[flake8]
exclude =
    .git,
    __pycache__,
    .venv,
    venv,
    .serena,
    .ruff_cache,
    .github,
    build,
    dist,
    *.egg-info,
    rawfiles,
    candidates,
    debug_masks,
    tests

max-line-length = 88
ignore = E203,W503,E501,E226
max-complexity = 70
```

**Key Settings:**

| Setting | Value | Reason |
|---------|-------|--------|
| `max-line-length` | 88 | Matches Black's default line length |
| `ignore: E203` | Whitespace before ':' | Black compatibility (slice formatting) |
| `ignore: W503` | Line break before binary operator | Black compatibility |
| `ignore: E501` | Line too long | Black handles line length |
| `ignore: E226` | Missing whitespace around operator | Black compatibility |
| `max-complexity` | 70 | Appropriate for image processing algorithms |

### Usage

#### Manual Linting
```bash
# Check all Python files
flake8 .

# Check specific files or directories
flake8 detect_meteors_cli.py
flake8 meteor_core/
flake8 tests/

# Show statistics
flake8 --statistics --count .
```

#### Development Workflow
```bash
# 1. Make your changes
vim detect_meteors_cli.py

# 2. Format with Black (via pre-commit or manually)
black detect_meteors_cli.py

# 3. Check code quality with flake8
flake8 detect_meteors_cli.py

# 4. Fix any issues reported

# 5. Commit your changes
git add detect_meteors_cli.py
git commit -m "Add new feature"
```

### Benefits

1. **Early issue detection**: Catch potential bugs, unused imports, and code smells before they reach production
2. **Consistent style**: Enforce coding standards across the entire codebase
3. **Black compatibility**: Carefully configured to avoid conflicts with Black formatter
4. **Developer guidance**: Provides helpful feedback for maintaining code quality

### Toolchain Overview

| Tool | Purpose | When It Runs |
|------|---------|--------------|
| **Black** | Automatic code formatting | Pre-commit hook (automatic) |
| **flake8** | Static code analysis & style checking | Manual before commit (recommended) |

### Common flake8 Checks

**What flake8 catches:**

- Unused imports: `F401`
- Undefined names: `F821`
- Syntax errors: `E999`
- Code complexity: `C901`
- PEP 8 violations: `E***`, `W***`
- Logical errors: Missing return statements, undefined variables

**Example:**
```bash
$ flake8 detect_meteors_cli.py
detect_meteors_cli.py:45:1: F401 'os' imported but unused
detect_meteors_cli.py:234:5: F841 local variable 'result' is assigned to but never used
```

### Ignored Warnings (Black Compatibility)

Some flake8 warnings are intentionally ignored to avoid conflicts with Black:

- `E203`: Black's slice formatting uses no space before `:`
- `W503`: Black prefers line breaks before binary operators
- `E501`: Black automatically handles line length
- `E226`: Black handles operator spacing its own way

### Test Files Exclusion

Test files are excluded from flake8 checks by default to allow more flexible testing patterns. This prevents false positives from test-specific code structures.

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.7 and earlier:
- No changes to runtime behavior
- No breaking changes to API or CLI options
- Pure development tooling enhancement

### Files Updated

| File | Changes |
|------|---------|
| `.flake8` | New configuration file for flake8 linter |
| `INSTALL_DEV.md` | Updated with flake8 integration details |
| `CHANGELOG.md` | Added v1.5.8 entry |
| `README.md` | Added v1.5.8 section |

---

## Version 1.5.7 (2025-12-08) 🪡

### 📝 Progress metadata enrichment

Version 1.5.7 enriches `progress.json` with additional metadata to help users review past runs and adjust parameters for future sessions.

### Highlights

- **Recorded configuration**: `progress.json` now stores CLI parameters (`params`), the selected ROI (`roi`), and the finalized processing parameters (`processing_params`). This information serves as a reference when reviewing detection results or planning subsequent runs with adjusted settings.
- **Unified pipeline behavior**: Both the CLI path and `PipelineConfig`/`DetectionPipeline` flows write the same metadata fields, keeping progress files consistent across entry points.

### New `progress.json` Fields

| Field | Description |
|-------|-------------|
| `params` | Original CLI parameter string for reference |
| `roi` | Selected ROI polygon or `"full_image"` |
| `processing_params` | Finalized detection parameters used |

### Example `progress.json`

```json
{
  "version": "1.5.7",
  "params_hash": "abc123...",
  "params": "--auto-params --sensor-type MFT",
  "roi": [[100, 100], [1000, 100], [1000, 800], [100, 800]],
  "processing_params": {
    "diff_threshold": 7,
    "min_area": 5,
    "min_aspect_ratio": 3.0,
    "hough_threshold": 10,
    "hough_min_line_length": 15,
    "hough_max_line_gap": 5,
    "min_line_score": 35.0
  },
  "processed_files": ["IMG_0001.ORF", "IMG_0002.ORF"],
  "detected_files": ["IMG_0002.ORF"],
  "total_processed": 2,
  "total_detected": 1
}
```

---

## Version 1.5.6 (2025-12-06) ☃️

### 🧩 Input/Output Plugin Preparation

Version 1.5.6 refines the `meteor_core` interfaces so that input and output handling now follow the same plugin-ready structure as detectors.

### Motivation

- **v2.0 plugin architecture**: Align input/output with the detector plugin plan
- **Configurable loaders**: Allow different RAW readers and metadata providers to coexist
- **Extensible outputs**: Formalize how candidate and debug artifacts are persisted

### Input Loader Architecture

- **Protocols**: New `InputLoader` and `MetadataExtractor` protocols define the loader contract (`plugin_name`, `config`, `load()`, optional `extract_metadata()`).
- **Helper bases**: Dataclass/Pydantic-friendly base classes validate configs automatically for loader authors.
- **Built-in plugin**: `RawImageLoader` wraps the existing RAW helpers with configurable binning/normalization and EXIF metadata extraction.
- **Discovery**: Deterministic discovery order—built-in loaders, then Python entry points (`detect_meteors.input`), then local plugin files under `~/.detect_meteors/plugins`.

### Pipeline Configuration & Metadata Flow

- **PipelineConfig dataclass** centralizes all runtime settings for the detection pipeline (folders, params, workers, batch sizing, progress tracking, overwrite policy).
- **DetectionPipeline protocol** exposes a structured interface for future pipeline implementations.
- **Loader resolution**: Pipelines now resolve input loaders by instance, name, or config, and use loader-provided metadata when available, falling back to EXIF helpers otherwise.

### Output Handling Contract

- **OutputHandler protocol** formalizes candidate and debug image persistence, paving the way for pluggable output destinations.

### Backward Compatibility

✅ Fully compatible with v1.5.5: defaults preserve existing behavior (RAW loader, EXIF extraction, output writer), while new hooks prepare the jump to v2.0 plugins.

---

## Version 1.5.5 (2025-12-05) 👖

### 🏗️ Code Architecture Refactoring

Version 1.5.5 introduces a major internal restructuring of the codebase, separating the CLI interface from core logic modules. This refactoring prepares the foundation for the v2.x plugin architecture while maintaining full backward compatibility.

### Motivation

- **Plugin Architecture Preparation**: The monolithic `detect_meteors_cli.py` needed to be split to enable future plugin support (v2.x roadmap)
- **Separation of Concerns**: CLI parsing and user interaction are now cleanly separated from detection logic
- **Type Safety**: Enhanced type definitions improve code reliability and IDE support
- **Maintainability**: Smaller, focused modules are easier to test, understand, and modify

### New Project Structure

```
detect_meteors/
├── detect_meteors_cli.py          # CLI interface only
└── meteor_core/                   # Core logic modules
    ├── __init__.py
    ├── schema.py                  # Type definitions (TypedDict, constants)
    ├── pipeline.py                # Processing pipeline orchestration
    ├── image_io.py                # RAW image loading, EXIF extraction
    ├── roi_selector.py            # ROI selection interface
    ├── utils.py                   # Utility functions
    ├── detectors/                 # Detection algorithms
    │   ├── __init__.py
    │   ├── base.py                # Abstract base detector class
    │   └── hough_default.py       # Default Hough transform detector
    └── outputs/                   # Output handling
        ├── __init__.py
        └── writer.py              # Result file writer
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `detect_meteors_cli.py` | CLI argument parsing, user interaction, main entry point |
| `meteor_core/schema.py` | Type definitions using TypedDict, constants, data structures |
| `meteor_core/pipeline.py` | Processing pipeline orchestration, batch processing |
| `meteor_core/image_io.py` | RAW image loading, EXIF metadata extraction |
| `meteor_core/roi_selector.py` | Interactive ROI selection interface |
| `meteor_core/utils.py` | Utility functions (NPF calculations, parameter estimation) |
| `meteor_core/detectors/base.py` | Abstract base class for detection algorithms |
| `meteor_core/detectors/hough_default.py` | Default Hough transform-based meteor detector |
| `meteor_core/outputs/writer.py` | Result file writing, output management |

### Type Safety Improvements

Enhanced type hints throughout the codebase using `TypedDict` for structured data:

```python
from typing import TypedDict, Optional

class NPFMetrics(TypedDict):
    pixel_pitch: float
    npf_recommended: float
    actual_exposure: float
    npf_ratio: float
    compliance: str
    impact: str

class EXIFData(TypedDict, total=False):
    camera: Optional[str]
    focal_length: Optional[float]
    iso: Optional[int]
    exposure_time: Optional[float]
    aperture: Optional[float]
    width: int
    height: int
```

### Detector Plugin Infrastructure

The `detectors/` subpackage introduces an abstract base class pattern for future extensibility:

```python
# meteor_core/detectors/base.py
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, diff_image, params):
        """Detect meteors in difference image."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return detector name."""
        pass
```

This enables v2.x plugin architecture where users can implement custom detection algorithms.

### Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `detect_meteors_cli.py` | Modified | Reduced to CLI interface only |
| `meteor_core/__init__.py` | New | Package initialization |
| `meteor_core/schema.py` | New | Type definitions |
| `meteor_core/pipeline.py` | New | Pipeline orchestration |
| `meteor_core/image_io.py` | New | Image I/O operations |
| `meteor_core/roi_selector.py` | New | ROI selection |
| `meteor_core/utils.py` | New | Utility functions |
| `meteor_core/detectors/__init__.py` | New | Detectors subpackage |
| `meteor_core/detectors/base.py` | New | Abstract base detector |
| `meteor_core/detectors/hough_default.py` | New | Default Hough detector |
| `meteor_core/outputs/__init__.py` | New | Outputs subpackage |
| `meteor_core/outputs/writer.py` | New | Output writer |

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.4 and all earlier versions:
- CLI interface unchanged - all existing commands work without modification
- No breaking changes to command-line options
- No changes to input/output formats
- Shell completion scripts unchanged

### For Developers

If you have custom scripts that import from `detect_meteors_cli.py`:

```python
# Old import (still works via re-exports)
from detect_meteors_cli import calculate_npf_metrics

# New recommended import
from meteor_core.utils import calculate_npf_metrics
```

### Future Plans (v2.x)

This restructuring enables:
- Custom detector plugins via `detectors/` subpackage
- Custom output writers via `outputs/` subpackage
- Configuration file support for detector selection
- Third-party integration capabilities

---

## Version 1.5.4 (2025-12-03) 👩

### 🔆 ROI Display Improvement

Version 1.5.4 improves the ROI (Region of Interest) selection experience by brightening the displayed image, making it easier to select regions in dark astrophotography images.

### Changes

- **Brightened ROI Selection Image**: The image displayed during ROI selection is now enhanced for better visibility, helping users accurately select regions even in very dark night sky images

### 📄 NOTICE Document

- **Added NOTICE file**: New NOTICE document containing third-party license attributions and acknowledgments for dependencies used in this project

### Files Updated

| File | Changes |
|------|---------|
| `detect_meteors_cli.py` | Brightened ROI selection display |
| `NOTICE` | New file for third-party attributions |
| `CHANGELOG.md` | Added v1.5.4 entry |
| `README.md` | Added v1.5.4 section |

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.3 and earlier:
- No breaking changes to API or CLI options
- All existing commands work without modification

---

## Version 1.5.3 (2025-12-02) 🪒

### 🐟 Fisheye Lens Correction

Version 1.5.3 adds fisheye lens support with equisolid angle projection compensation. This feature accounts for the varying effective focal length across fisheye images, providing more accurate NPF calculations and star trail estimations.

### The Problem with Fisheye Lenses

Fisheye lenses use special projection geometries that cause the effective focal length to vary across the image:

- **Center**: Uses the nominal focal length (e.g., 8mm → 16mm equiv. on MFT)
- **Edge/Corner**: Shorter effective focal length due to projection compression

For **equisolid angle projection** (most common fisheye type):
- Formula: `r = 2f × sin(θ/2)`
- Edge effective focal length: ~0.707× nominal (at 90° from center)
- Star trails at edges: ~1.414× longer than at center

Without correction, NPF calculations based on nominal focal length are too conservative for the image center but potentially insufficient for the edges.

### New `--fisheye` Flag

Add `--fisheye` to enable equisolid angle projection compensation:

```bash
# MFT camera with 8mm fisheye (16mm equiv.)
python detect_meteors_cli.py --auto-params --sensor-type MFT --focal-length 16 --fisheye

# Full Frame with 8mm fisheye
python detect_meteors_cli.py --auto-params --sensor-type FF --focal-length 8 --fisheye

# Check NPF analysis with fisheye correction
python detect_meteors_cli.py --show-npf --sensor-type MFT --focal-length 16 --fisheye
```

### Effect on NPF Calculations

When `--fisheye` is enabled:
- NPF recommended exposure uses **edge focal length** (worst case)
- Star trail estimation uses **edge trail length** (longest trails)
- More lenient/conservative NPF recommendations

### Real-World Example

**Equipment**: OM-D E-M1 Mark II + M.ZUIKO 8mm F1.8 Fisheye PRO

**Without `--fisheye`**:
```
NPF Rule Analysis
============================================================
  Pixel pitch:      3.70μm (sensor: 17.3mm)
  NPF recommended:  10.9s
  Actual exposure:  6.0s ✓ OK
  Star trail est.:  ~1.4 pixels
============================================================
```

**With `--fisheye`**:
```
Fisheye Correction
============================================================
  Projection model:   Equisolid Angle Projection
  Nominal focal:      16.0mm (center)
  Effective focal:    11.3mm (edge)
  Trail length ratio: 1.41× (edge vs center)
  NPF calculation:    Based on edge (worst case)
============================================================

NPF Rule Analysis
============================================================
  Pixel pitch:      3.70μm (sensor: 17.3mm)
  NPF recommended:  15.4s
  Actual exposure:  6.0s ✓ OK
  Star trail est.:  ~1.9 pixels
============================================================
```

**Detection Results**: 308 images → 2 candidates (successfully detected expected meteors)

### Technical Details

#### New Functions

| Function | Description |
|----------|-------------|
| `calculate_fisheye_effective_focal_length()` | Position-dependent focal length calculation |
| `calculate_fisheye_edge_focal_length()` | Edge focal length for NPF (worst case) |
| `calculate_fisheye_trail_length_ratio()` | Trail length variation across image |
| `get_fisheye_max_trail_ratio()` | Maximum trail ratio at image edge |
| `display_fisheye_info()` | Display fisheye correction parameters |

#### New NPF Metrics Fields

When `--fisheye` is enabled, `calculate_npf_metrics()` returns additional fields:

```python
{
    'fisheye': True,
    'fisheye_model': 'EQUISOLID',
    'effective_focal_length': 11.3,  # Edge focal length
    'trail_length_ratio': 1.414,     # Edge vs center ratio
    # ... existing fields ...
}
```

#### Projection Model Infrastructure

The implementation uses an extensible design for future projection models:

```python
FISHEYE_PROJECTION_MODELS = {
    "EQUISOLID": {
        "name": "Equisolid Angle Projection",
        "description": "Equal-area projection (r = 2f × sin(θ/2))",
    },
    # Future: EQUIDISTANT, STEREOGRAPHIC, etc.
}
```

### Supported Lenses

The equisolid angle projection covers most common fisheye lenses:

- Olympus M.ZUIKO 8mm F1.8 Fisheye PRO
- Samyang/Rokinon 8mm F2.8 Fisheye
- Canon EF 8-15mm F4L Fisheye USM
- Nikon AF-S Fisheye NIKKOR 8-15mm f/3.5-4.5E ED
- Sigma 15mm F2.8 EX DG Diagonal Fisheye
- And most other circular/diagonal fisheye lenses

### Test Coverage

New test file `test_fisheye_v1x.py` with 27 test cases covering:
- Effective focal length calculations
- Edge focal length calculations
- Trail length ratio calculations
- NPF metrics integration with fisheye
- Projection model configuration

**Total test count**: 228 tests (previously 201)

### Files Updated

| File | Changes |
|------|---------|
| `detect_meteors_cli.py` | Added fisheye functions, `--fisheye` flag, NPF integration |
| `detect_meteors_cli_completion.bash` | Added `--fisheye` completion |
| `_detect_meteors_cli` (zsh) | Added `--fisheye` completion |
| `test_fisheye_v1x.py` | New test file (27 tests) |
| `CHANGELOG.md` | Added v1.5.3 entry |
| `COMMAND_OPTIONS.md` | Added Fisheye Correction Options section |
| `NPF_RULE.md` | Added Fisheye Lens Correction section |
| `README.md` | Added v1.5.3 section, fisheye usage examples |
| `ROADMAP.md` | Added v1.5.3 milestone |

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.2 and earlier:
- `--fisheye` is optional; default behavior unchanged
- All existing commands work without modification
- No breaking changes to API or CLI options

---

## Version 1.5.2 (2025-12-01) ♥️️

### 🛡️ Sensor Override Validation

Version 1.5.2 adds automatic validation when users override `--sensor-type` preset values with individual parameters. This helps catch potential misconfiguration while preserving the flexibility to use custom values when needed.

### New Validation Feature

When using `--sensor-type` with `--sensor-width` or `--pixel-pitch` overrides, the system now checks if the overridden values deviate significantly from the preset values:

**Warning Thresholds:**
- `--sensor-width`: ±30% deviation from preset
- `--pixel-pitch`: ±50% deviation from preset

**Important**: Warnings are informational only - processing continues normally. This design preserves user flexibility while providing helpful feedback about potential configuration issues.

### Usage Examples

#### Example 1: No Warning (Small Deviation)
```bash
# MFT preset: sensor_width=17.3mm
python detect_meteors_cli.py --auto-params \
  --sensor-type MFT \
  --sensor-width 17.5  # 1.2% deviation → no warning
```

#### Example 2: Warning Displayed (Large Deviation)
```bash
# MFT preset: sensor_width=17.3mm
python detect_meteors_cli.py --auto-params \
  --sensor-type MFT \
  --sensor-width 23.5  # 35.8% deviation → warning
```

**Output:**
```
======================================================================
⚠ Warning: --sensor-width 23.5mm deviates 35.8% from --sensor-type MFT preset (17.3mm)
======================================================================
```

#### Example 3: Multiple Warnings
```bash
# FF preset: sensor_width=36.0mm, pixel_pitch=4.3μm
python detect_meteors_cli.py --auto-params \
  --sensor-type FF \
  --sensor-width 23.5 \
  --pixel-pitch 7.0
```

**Output:**
```
======================================================================
⚠ Warning: --sensor-width 23.5mm deviates 34.7% from --sensor-type FF preset (36.0mm)
⚠ Warning: --pixel-pitch 7.0μm deviates 62.8% from --sensor-type FF preset (4.3μm)
======================================================================
```

### Why These Thresholds?

#### sensor_width (±30%)
- Sensor sizes are standardized (1-inch, MFT, APS-C, FF, MF)
- 30% deviation typically indicates selecting wrong sensor type
- Example: MFT (17.3mm) vs APS-C (23.5mm) = 36% deviation

#### pixel_pitch (±50%)
- Pixel pitch varies with resolution even within same sensor type
- 20MP vs 45MP can differ by >2×
- More lenient threshold accommodates this natural variation

### When Warnings Appear

**Common scenarios that trigger warnings:**

1. **Wrong sensor type selected**
   ```bash
   # User has APS-C camera but selected MFT
   --sensor-type MFT --sensor-width 23.5  # ⚠ Warning
   ```

2. **Typo in manual override**
   ```bash
   # Meant 3.7 but typed 37
   --sensor-type MFT --pixel-pitch 37.0  # ⚠ Warning
   ```

3. **Using values from different camera**
   ```bash
   # Using FF values with MFT sensor type
   --sensor-type MFT --sensor-width 36.0  # ⚠ Warning
   ```

### When NOT to Worry

**Legitimate use cases that may show warnings:**

1. **Precise measured values**
   ```bash
   # User measured their specific camera's sensor
   --sensor-type MFT --sensor-width 17.4  # Small deviation, no warning
   ```

2. **High-resolution variants**
   ```bash
   # GFX100 II (102MP) vs GFX50S (51MP)
   --sensor-type MF44X33 --pixel-pitch 3.3  # Different resolution
   ```

3. **Custom sensor configurations**
   ```bash
   # Specialized or modified camera
   --sensor-type FF --pixel-pitch 8.5  # May be intentional
   ```

### Technical Details

#### New Function: `validate_sensor_overrides()`

```python
def validate_sensor_overrides(
    args,
    preset: Optional[Dict[str, any]],
    sensor_width_value: Optional[float],
    pixel_pitch_value: Optional[float],
) -> None:
    """
    Validate that overridden --sensor-width and --pixel-pitch values
    are not significantly different from --sensor-type preset values.
    
    Prints warnings if discrepancies are detected, but does not stop processing.
    """
```

#### Updated Function: `apply_sensor_preset()`

Now returns 5-tuple instead of 4-tuple:

```python
# v1.5.1 and earlier
focal_factor, sensor_width, focal_length, pixel_pitch = apply_sensor_preset(args)

# v1.5.2
focal_factor, sensor_width, focal_length, pixel_pitch, preset = apply_sensor_preset(args)
```

The additional `preset` return value enables validation by providing access to the original preset values.

### Test Coverage

New test file `test_sensor_validation_v1x.py` with 23 test cases:

- Basic validation behavior (no warnings without overrides)
- Threshold boundary testing (exactly at and just over limits)
- Multiple sensor types (1INCH, MFT, APS-C, FF, MF44X33, MF54X40)
- Real-world scenarios (correct setups, wrong sensor type, custom values)
- Edge cases (None values, missing preset keys)

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.1 and earlier versions:
- All existing commands work unchanged
- New validation is purely additive
- No breaking changes to API or CLI options

### Migration Notes

No action required. Existing code and scripts continue to work.

**Optional**: If you have custom scripts that call `apply_sensor_preset()`, update to handle the new 5-tuple return value:

```python
# Update this
focal_factor, sensor_width, focal_length, pixel_pitch = apply_sensor_preset(args)

# To this
focal_factor, sensor_width, focal_length, pixel_pitch, preset = apply_sensor_preset(args)
```

Or use tuple unpacking with underscore:

```python
# If you don't need the preset
focal_factor, sensor_width, focal_length, pixel_pitch, _ = apply_sensor_preset(args)
```

### Files Updated

| File | Changes |
|------|---------|
| `detect_meteors_cli.py` | Added `validate_sensor_overrides()`, updated `apply_sensor_preset()` |
| `test_sensor_validation_v1x.py` | New test file (23 tests) |
| `test_sensor_presets_v1x.py` | Updated for 5-tuple return value |

---

## Version 1.5.1 (2025-11-30) 📷

### 📷 Medium Format Sensor Support

Version 1.5.1 adds support for medium format sensors, extending the tool's capabilities to professional medium format camera systems from Fujifilm, Pentax/Ricoh, and Hasselblad.

### New Sensor Types

| Sensor Type | Crop Factor | Sensor Width | Pixel Pitch | Supported Cameras |
|-------------|-------------|--------------|-------------|-------------------|
| `MF44X33` | 0.79 | 43.8mm | 3.76μm | Fujifilm GFX series, Pentax 645Z, Hasselblad X2D/X1D |
| `MF54X40` | 0.64 | 53.4mm | 4.6μm | Hasselblad H6D-100c, H6D-400c MS |

### Usage Examples

```bash
# Fujifilm GFX100 II
python detect_meteors_cli.py --auto-params --sensor-type MF44X33

# Pentax 645Z
python detect_meteors_cli.py --auto-params --sensor-type MF44X33

# Hasselblad X2D 100C
python detect_meteors_cli.py --auto-params --sensor-type MF44X33

# Hasselblad H6D-100c
python detect_meteors_cli.py --auto-params --sensor-type MF54X40
```

### Sensor Size Ordering

All sensor types are now ordered by sensor size (smallest to largest):

```
1INCH → MFT → APSC → APSC_CANON → APSH → FF → MF44X33 → MF54X40
```

This ordering is reflected in `--list-sensor-types` output and shell completions.

### Aliases

- `MF44-33`, `MF44_33` → `MF44X33`
- `MF54-40`, `MF54_40` → `MF54X40`

---

## Version 1.5.0 (2025-11-29 🦃)

### 🎯 Sensor Type Presets - Simplified NPF Configuration

Version 1.5.0 introduces **sensor type presets** that dramatically simplify NPF Rule configuration. Instead of manually specifying multiple parameters, users can now use a single `--sensor-type` option to configure all sensor-related settings at once.

## Evolution from v1.4

### v1.4.x (Manual Parameter Specification)
- ✅ NPF Rule-based optimization
- ✅ EXIF metadata integration
- ❌ Required separate specification of `--sensor-width`, `--pixel-pitch`, `--focal-factor`
- ❌ Users needed to look up sensor specifications
- ❌ Easy to misconfigure parameters

### v1.5.0 (Sensor Type Presets)
- ✅ **NEW**: Single `--sensor-type` option for all sensor parameters
- ✅ **NEW**: Unified `SENSOR_PRESETS` configuration
- ✅ **NEW**: `--list-sensor-types` to display available presets
- ✅ **NEW**: Individual parameters override presets when needed
- ✅ Backward compatible with v1.4.x options

## Major Changes

### 1. Unified Sensor Presets (NEW)

**Purpose**: Consolidate sensor-related parameters into easy-to-use presets

**Available Presets** (ordered by sensor size):

| Sensor Type | Crop Factor | Sensor Width | Pixel Pitch | Description |
|-------------|-------------|--------------|-------------|-------------|
| `1INCH` | 2.7 | 13.2mm | 2.4μm | 1-inch sensor |
| `MFT` | 2.0 | 17.3mm | 3.7μm | Micro Four Thirds |
| `APS-C` | 1.5 | 23.5mm | 3.9μm | APS-C (Sony/Nikon/Fuji) |
| `APS-C_CANON` | 1.6 | 22.3mm | 3.2μm | APS-C (Canon) |
| `APS-H` | 1.3 | 27.9mm | 5.7μm | APS-H (Canon) |
| `FF` | 1.0 | 36.0mm | 4.3μm | Full Frame 35mm |
| `MF44X33` | 0.79 | 43.8mm | 3.76μm | Medium Format 44×33 |
| `MF54X40` | 0.64 | 53.4mm | 4.6μm | Medium Format 54×40 |

**Aliases**: `FULLFRAME` → `FF`, `APS_C` → `APS-C`, `MF44-33` → `MF44X33`, etc.

### 2. New `--sensor-type` Option (NEW)

**Purpose**: One-stop configuration for sensor parameters

**Usage**:
```bash
# Before (v1.4.x) - Required multiple parameters
python detect_meteors_cli.py --auto-params \
  --sensor-width 17.3 \
  --focal-factor 2.0 \
  --pixel-pitch 3.7

# After (v1.5.0) - Single parameter
python detect_meteors_cli.py --auto-params --sensor-type MFT
```

### 3. New `--list-sensor-types` Option (NEW)

**Purpose**: Display available sensor presets and their configurations

**Usage**:
```bash
python detect_meteors_cli.py --list-sensor-types
```

## Technical Details

### New Functions

#### `get_sensor_preset(sensor_type: str) -> Optional[Dict]`
Retrieves sensor preset configuration by type name.

#### `apply_sensor_preset(args, verbose=False) -> Tuple`
Applies sensor preset with CLI argument priority.

#### `list_sensor_types() -> None`
Displays available sensor presets in formatted output, ordered by sensor size.

### Backward Compatibility

- `CROP_FACTORS` dictionary preserved (auto-generated from `SENSOR_PRESETS`)
- `DEFAULT_SENSOR_WIDTHS` dictionary preserved (auto-generated from `SENSOR_PRESETS`)
- All v1.4.x command-line options work unchanged
- `--focal-factor` still accepts sensor type strings (e.g., `MFT`, `APS-C`)

## Breaking Changes

**None** - v1.5.x is fully backward compatible with v1.4.x.

---

## Version Information

- **Latest Version**: 1.5.13
- **Release Date**: 2025-12-19
- **Major Changes**:
  - Internationalization (i18n) with `--locale` option
  - English and Japanese message catalogs
  - ICU-style plural rule support
  - Progress file normalization helpers
  - UI/UX messages localized; system/debug in English

- **Version**: 1.5.12
- **Release Date**: 2025-12-18
- **Major Changes**:
  - Custom exception hierarchy for inputs (`MeteorLoadError`, `MeteorUnsupportedFormatError`), outputs (`MeteorOutputError`, `MeteorWriteError`, `MeteorProgressError`), and config (`MeteorValidationError`, `MeteorConfigError`)
  - Diagnostic reporting with `DiagnosticInfo` dataclass
  - `--verbose` flag for detailed error info and DEBUG logging
  - `--save-diagnostic` option for bug reporting
  - Standard Python logging throughout all modules

- **Version**: 1.5.11
- **Release Date**: 2025-12-15
- **Major Changes**:
  - Cross-package consistency for plugin registries
  - Case-insensitive registry lookup
  - BaseInputLoader metadata enhancement (`name`, `version`, `get_info()`)
  - Early config validation with explicit errors
  - Plugin directory naming standardization

- **Version**: 1.5.10
- **Release Date**: 2025-12-11
- **Major Changes**:
  - Plugin architecture migrated from Protocol to ABC
  - Enhanced IDE support and immediate error detection
  - Comprehensive plugin development documentation
  - ⚠️ Plugin architecture is experimental until v2.0

- **Version**: 1.5.9
- **Release Date**: 2025-12-10
- **Major Changes**:
  - PEP 621 compliant project configuration
  - Flake8 config migrated to pyproject.toml
  - Dependency and testing configuration centralized

- **Version**: 1.5.8
- **Release Date**: 2025-12-09
- **Major Changes**:
  - flake8 linter integration with Black formatter
  - Project-specific linting configuration
  - Enhanced developer workflow

- **Version**: 1.5.7
- **Release Date**: 2025-12-08
- **Major Changes**:
  - Progress metadata enrichment (params, roi, processing_params)
  - Unified pipeline behavior for consistent progress files

- **Version**: 1.5.6
- **Release Date**: 2025-12-07
- **Major Changes**:
  - InputLoader/MetadataExtractor protocols
  - PipelineConfig and DetectionPipeline protocol
  - OutputHandler protocol for extensibility

- **Version**: 1.5.5
- **Release Date**: 2025-12-05
- **Major Changes**:
  - Code architecture refactoring (CLI/core separation)
  - New `meteor_core/` package with modular components
  - Enhanced type safety with TypedDict
  - Detector plugin infrastructure preparation

- **Version**: 1.5.4
- **Release Date**: 2025-12-03
- **Major Changes**:
  - Brightened ROI selection image for better visibility
  - Added NOTICE document for third-party attributions

- **Version**: 1.5.3
- **Release Date**: 2025-12-02
- **Major Changes**:
  - Fisheye lens correction (`--fisheye` flag)
  - Equisolid angle projection compensation
  - Position-dependent effective focal length calculation
  - Enhanced NPF calculation for fisheye lenses
  - New test coverage (27 fisheye tests)

- **Version**: 1.5.2
- **Release Date**: 2025-12-01
- **Major Changes**:
  - Sensor override validation (automatic warning for misconfigurations)
  - Enhanced test coverage (23 new validation tests)
  - Improved `apply_sensor_preset()` function

- **Version**: 1.5.1
- **Release Date**: 2025-11-30
- **Major Changes**:
  - Medium format sensor support (MF44X33, MF54X40)
  - Sensor size ordering (smallest to largest)
  - Updated shell completion scripts

## Quick Reference

| Feature | Command | Example |
|---------|---------|---------|
| Use sensor preset | `--sensor-type TYPE` | `--sensor-type MFT` |
| List presets | `--list-sensor-types` | `python detect_meteors_cli.py --list-sensor-types` |
| Preset + override | `--sensor-type TYPE --PARAM VALUE` | `--sensor-type FF --pixel-pitch 5.9` |
| Full auto (MFT) | `--auto-params --sensor-type MFT` | `python detect_meteors_cli.py --auto-params --sensor-type MFT` |
| Medium Format | `--auto-params --sensor-type MF44X33` | `python detect_meteors_cli.py --auto-params --sensor-type MF44X33` |
| Fisheye lens | `--auto-params --fisheye` | `python detect_meteors_cli.py --auto-params --sensor-type MFT --focal-length 16 --fisheye` |
| Verbose mode | `--verbose` | `python detect_meteors_cli.py --auto-params --sensor-type MFT --verbose` |
| Save diagnostic | `--save-diagnostic [FILE]` | `python detect_meteors_cli.py --auto-params --save-diagnostic my_report.md` |
| NPF check | `--show-npf --sensor-type TYPE` | `python detect_meteors_cli.py --show-npf --sensor-type APS-C` |
| NPF + Fisheye | `--show-npf --fisheye` | `python detect_meteors_cli.py --show-npf --sensor-type MFT --focal-length 16 --fisheye` |
| Set locale | `--locale LANG` | `python detect_meteors_cli.py --auto-params --sensor-type MFT --locale ja` |
| Locale (env) | `DETECT_METEORS_LOCALE` | `export DETECT_METEORS_LOCALE=ja` |

## Files Updated (v1.5.x Summary)

| File | Changes |
|------|---------|
| `meteor_core/i18n.py` | New i18n module with ICU-style formatting (v1.5.13) |
| `meteor_core/locales/en/messages.yaml` | English message catalog (v1.5.13) |
| `meteor_core/locales/ja/messages.yaml` | Japanese message catalog (v1.5.13) |
| `meteor_core/outputs/progress.py` | Progress normalization helpers (v1.5.13) |
| `meteor_core/exceptions.py` | New exception hierarchy and diagnostic reporting (v1.5.12) |
| `meteor_core/__init__.py` | Logging configuration and exception exports (v1.5.12) |
| `meteor_core/inputs/*.py` | Added logging throughout (v1.5.12) |
| `meteor_core/detectors/*.py` | Added logging throughout (v1.5.12) |
| `meteor_core/outputs/*.py` | Added logging throughout (v1.5.12) |
| `meteor_core/pipeline.py` | Added logging and exception handling (v1.5.12) |
| `meteor_core/image_io.py` | Added logging and exception wrapping (v1.5.12) |
| `detect_meteors_cli.py` | CLI interface, `--verbose`, `--save-diagnostic` (v1.5.12) |
| `pyproject.toml` | PEP 621 metadata, tool configs (v1.5.9) |
| `.flake8` | Removed, migrated to pyproject.toml (v1.5.9) |
| `.pre-commit-config.yaml` | Added Flake8-pyproject (v1.5.9) |
| `meteor_core/` | New package with modular components (v1.5.5) |
| `detect_meteors_cli_completion.bash` | Shell completions |
| `_detect_meteors_cli` (zsh) | Shell completions |
| `COMMAND_OPTIONS.md` | CLI options reference |
| `NPF_RULE.md` | NPF Rule documentation |
| `test_exceptions_v1x.py` | Exception hierarchy tests (v1.5.12) |
| `test_inputs_logging_v1x.py` | Logging configuration tests (v1.5.12) |
| `test_fisheye_v1x.py` | Fisheye tests (27 tests) |
| `test_sensor_validation_v1x.py` | Validation tests (23 tests) |

---

**Status**: Production Ready  
**Compatibility**: Fully backward compatible with v1.4.x  
**Recommendation**: Use `--sensor-type` for simplified configuration; add `--fisheye` for fisheye lenses; use `--locale ja` for Japanese messages

Happy meteor hunting! 🌠
