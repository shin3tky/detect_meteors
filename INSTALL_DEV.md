# Developer Installation Guide

This guide provides setup instructions for contributors and developers working on Detect Meteors CLI.

For basic user installation, see [INSTALL.md](INSTALL.md).
For plugin development, see [PLUGIN_AUTHOR_GUIDE.md](PLUGIN_AUTHOR_GUIDE.md).

## Prerequisites

Complete the basic installation from [INSTALL.md](INSTALL.md) first, then follow this guide for development-specific setup.

## Development Environment Setup

### Step 1: Install Development Dependencies

After cloning the repository, install all dependencies including dev tools:

```bash
uv sync --all-extras
```

This installs the project in editable mode plus the dev toolchain (Ruff, pre-commit, coverage).

### Step 2: Set Up Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) with [Ruff](https://docs.astral.sh/ruff/) for automatic code formatting and linting.

```bash
# Install the git hooks
uv run pre-commit install

# Verify installation
uv run pre-commit --version
```

### Step 3: Verify Setup

Run the linter and formatter on all files to verify the setup:

```bash
uv run pre-commit run --all-files
```

## Pre-commit Hooks

### How It Works

Once installed, pre-commit will automatically run Ruff (linter and formatter) before each commit:

1. You make changes and run `git commit`
2. Pre-commit runs Ruff linter with auto-fix and formatter on staged files
3. If changes are made or errors are found, the commit is aborted
4. You stage the changes and commit again

### Example Workflow

```bash
# Make your changes
vim detect_meteors_cli.py

# Stage and commit
git add detect_meteors_cli.py
git commit -m "Add new feature"

# If Ruff makes changes or finds errors, you'll see:
# - Files were modified by this hook
# - or linting errors
# - Commit aborted

# Stage the changes and commit again
git add detect_meteors_cli.py
git commit -m "Add new feature"
```

### Manual Formatting and Linting

```bash
# Run both linter and formatter via pre-commit
uv run pre-commit run --all-files

# Or run Ruff directly
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check format without changing
```

## Running Tests

### Using the Test Runner

```bash
uv run python run_tests.py
```

### Using unittest Directly

```bash
# Run all tests
uv run python -m unittest discover -s tests -p "test_*.py" -v

# Run specific test file
uv run python -m unittest tests.test_calculations_v1x -v

# Run specific test class
uv run python -m unittest tests.test_calculations_v1x.TestCalculateNPFRule -v
```

### Test Coverage

```bash
# Run tests with coverage measurement
uv run coverage run -m unittest discover tests

# View coverage report
uv run coverage report

# Generate HTML report (outputs to htmlcov/)
uv run coverage html
```

### Test Files Overview

| File | Tests | Description |
|------|-------|-------------|
| `test_exceptions_v1x.py` | 70 | Exception hierarchy and diagnostic info |
| `test_calculations_v1x.py` | 54 | NPF Rule, pixel pitch, star/meteor trail estimation |
| `test_output_handler_registry_v1x.py` | 53 | Output handler registry |
| `test_integration_v1x.py` | 44 | End-to-end meteor detection |
| `test_sensor_presets_v1x.py` | 38 | Sensor type presets |
| `test_loader_registry_v1x.py` | 30 | Input loader registry |
| `test_detector_registry_v1x.py` | 30 | Detector registry |
| `test_sensor_validation_v1x.py` | 27 | Sensor override validation |
| `test_fisheye_v1x.py` | 27 | Fisheye lens correction |
| `test_detector_plugin_v1x.py` | 26 | Detector plugin architecture |
| `test_raw_loader_v1x.py` | 23 | RAW image loader |
| `test_sensor_npf_integration_v1x.py` | 16 | Sensor/NPF integration |
| `test_infrastructure_v1x.py` | 25 | ROI, progress, file collection |
| `test_inputs_logging_v1x.py` | 8 | Logging configuration |
| `test_memory_batch_size_v1x.py` | 6 | Memory-based batch sizing |
| `test_i18n.py` | 5 | Localization lookup and pluralization |
| `test_plugin_registry_base.py` | 4 | Plugin registry base class |
| `test_plugin_contract_validation_v1x.py` | 3 | Plugin contract validation |
| `test_cli_options_v1x.py` | 2 | CLI option parsing |
| `test_discovery_parity.py` | 2 | Plugin discovery parity |
| `test_registry_default_contracts_v1x.py` | 2 | Registry default contracts |
| `test_image_io_helpers_v1x.py` | 3 | Image I/O helpers |
| `test_plugin_contract_helpers_v1x.py` | 5 | Plugin contract helpers |
| `test_roi_selector_helpers_v1x.py` | 2 | ROI selector helpers |
| `test_utils_display_width_v1x.py` | 3 | Unicode display width helpers |
| `test_inputs_base_v1x.py` | 7 | Input loader base class |
| `test_roi_selector_ui_v1x.py` | 2 | ROI selector UI flow |
| `test_pipeline_helpers_v1x.py` | 12 | Pipeline helpers |

**Total: 529 tests**

## Code Style

### Standards

- **Ruff** for linting and formatting (88 character line length)
- **Python 3.12+** required
- **Google-style docstrings**
- **Type hints** throughout

### Example

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class NPFMetrics:
    """NPF Rule calculation results."""
    pixel_pitch_um: Optional[float] = None
    npf_recommended_sec: Optional[float] = None
    compliance_level: str = "UNKNOWN"


def estimate_star_trail_length(
    focal_length_mm: float,
    exposure_time_sec: float,
    image_width_px: int,
) -> float:
    """Estimate star trail length in pixels during exposure.

    Args:
        focal_length_mm: Focal length in 35mm equivalent (mm)
        exposure_time_sec: Exposure time in seconds
        image_width_px: Image width in pixels

    Returns:
        Star trail length in pixels
    """
    ...
```

## Localization and logging

- The default locale is English (`en`). Locale codes are normalized (e.g., `en_US` → `en-us`), and lookups fall back to the base language and then to English. If no translation exists, the message key itself is emitted so untranslated strings remain discoverable.
- Message formatting is performed before records reach the logger via helpers such as `get_message` and `log_warning`, ensuring log records contain final strings without `%` placeholders.

## Internationalization (i18n)

- User-facing strings are keyed under `ui.*` (e.g., `ui.error.header`, `ui.run.summary`). Technical logs stay under `log.*` and are kept in English even when a locale is set.
- Translations live in JSON-compatible YAML at `meteor_core/locales/<locale>/messages.yaml`. Keep placeholder names consistent across locales and prefer dot-delimited nesting.
- Use ICU-style placeholders such as `{path}` or plural templates like `{count, plural, =0 {Complete! No candidates extracted} one {Complete! # candidate extracted} other {Complete! # candidates extracted}}`. The `#` token is replaced with the numeric value.
- Fetch messages with `meteor_core.i18n.get_message(key, locale=..., params={...})` (also re-exported as `meteor_core.get_message`). Missing locales or keys fall back to English, and unknown placeholders remain visible for easier debugging.
- When adding new strings: update `locales/en/messages.yaml`, mirror the key in other locales (even with an English placeholder), and add a unit test if the message uses parameters or plural rules.

## Project Structure

```
detect_meteors/
├── _detect_meteors_cli            # Generated completion output (for install)
├── detect_meteors_cli.py          # CLI interface
├── detect_meteors_cli_completion.bash  # Bash completion script
├── meteor_core/                   # Core logic modules
│   ├── i18n.py                    # Locale resolution and message formatting
│   ├── schema.py                  # Type definitions, constants
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── pipeline.py                # Pipeline orchestration
│   ├── image_io.py                # Image IO, EXIF utilities
│   ├── roi_selector.py            # ROI selection
│   ├── utils.py                   # Utility functions
│   ├── messages.py                # User-facing message helpers
│   ├── plugin_registry_base.py    # Base class for plugin registries
│   ├── plugin_registry.py         # Unified plugin registry
│   ├── plugin_contract.py         # Plugin contract definitions
│   ├── inputs/                    # Input loader plugins
│   ├── outputs/                   # Output handler plugins
│   ├── detectors/                 # Detection algorithm plugins
│   ├── locales/                   # Translations
│   └── templates/                 # Report templates and assets
├── candidates/                    # Default output folder for detections
├── debug_masks/                   # Debug output masks
├── rawfiles/                      # Sample/raw image inputs
├── tests/                         # Test suite
├── pyproject.toml                 # Project configuration
├── run_tests.py                   # Test runner helper
└── PLUGIN_AUTHOR_GUIDE.md         # Plugin development guide
```

## Plugin Development

For creating custom plugins (input loaders, detectors, output handlers), see the comprehensive [PLUGIN_AUTHOR_GUIDE.md](PLUGIN_AUTHOR_GUIDE.md).

> ⚠️ **Note**: The plugin architecture is experimental and may change before v2.0.

## Contribution Workflow

1. **Fork** the repository
2. **Clone** your fork locally
3. **Set up** development environment (this guide)
4. **Create** a feature branch: `git checkout -b feature/your-feature`
5. **Make** changes and add tests
6. **Run** tests: `uv run python run_tests.py`
7. **Measure** coverage: `uv run coverage run -m unittest discover tests && uv run coverage report`
8. **Commit** (pre-commit will run Ruff automatically)
9. **Push** to your fork
10. **Create** a Pull Request

## Troubleshooting

### Updating Ruff Version

Ruff version is pinned in two locations for consistency across all environments:
- `pyproject.toml` (`ruff==X.Y.Z` in dev dependencies)
- `.pre-commit-config.yaml` (`rev: vX.Y.Z`)

When updating Ruff, change both files to the same version.

### Pre-commit Issues

```bash
# Reinstall hooks
uv run pre-commit uninstall
uv run pre-commit install

# Update to latest versions
uv run pre-commit autoupdate
```

### Test Issues

```bash
# Ensure dependencies are installed
uv sync --all-extras

# Reinstall if needed
uv sync --reinstall
```

### Import Errors

```bash
# Test imports
uv run python -c "from meteor_core import BaseInputLoader, BaseOutputHandler, BaseDetector"
```

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

When redistributing, include the [NOTICE](NOTICE) file.

## Resources

- [PLUGIN_AUTHOR_GUIDE.md](PLUGIN_AUTHOR_GUIDE.md) - Plugin development
- [README.md](README.md) - User documentation
- [CHANGELOG.md](CHANGELOG.md) - Release history
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Pre-commit Documentation](https://pre-commit.com/)
