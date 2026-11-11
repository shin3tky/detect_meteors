# Developer Installation Guide

[日本語版](INSTALL_DEV_ja.md)

This guide provides setup instructions for contributors and developers working on Detect Meteors CLI.

Run the commands in this guide from the repository root or the root of an
extracted source distribution, not from `docs/`.

For basic user installation, see [INSTALL.md](INSTALL.md).
For plugin development, see [PLUGIN_AUTHOR_GUIDE.md](PLUGIN_AUTHOR_GUIDE.md).

## Prerequisites

Install Git and uv, then clone the repository or your fork and check out the
branch you want to develop. For platform setup and where to keep the checkout
on macOS, see [INSTALL.md](INSTALL.md).

```bash
git clone https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

The CLI version comes from the checked-out source. Use that checkout's
`uv.lock` to install dependencies.

## Development Environment Setup

### Step 1: Install Development Dependencies

From the checkout root, install the locked runtime and development dependencies.
The repository selects Python 3.12 in `.python-version`. CI runs the test suite
with both Python 3.12 and 3.13.

```bash
uv python install 3.12
uv sync --locked --all-extras
uv run detect-meteors --version
uv run detect-meteors --help
```

This installs the project in editable mode in `.venv`, including Ruff, ty,
pre-commit, and coverage. To test on Python 3.13, install it with
`uv python install 3.13` and add `--python 3.13` to the sync and run commands.

<a id="development-tool-versions-v1611"></a>

### Development Tool Versions

| Package | Pinned version |
| --- | --- |
| `ruff` | `0.17.4` |
| `ty` | `0.0.100` |
| `pre-commit` | `4.7.0` |
| `coverage` | `7.17.2` |

These match the `dev` extra in `pyproject.toml` and `uv.lock`. Runtime pins are
listed in [installation](INSTALL.md#dependency-versions-v1611).

After switching branches or pulling a lockfile update, run
`uv sync --locked --all-extras` again from the checkout root.

### Step 2: Set Up Pre-commit Hooks

From a Git checkout on macOS or Linux, install the pre-commit hook. The hook
invokes Ruff and ty from `.venv/bin`. On native Windows or in an extracted
source distribution, use the direct checks in [Step 3](#step-3-verify-setup).

```bash
# Install the git hooks
uv run pre-commit install

# Verify installation
uv run pre-commit --version
```

### Step 3: Verify Setup

Run the formatter check, lint, type check, and tests from the project root.

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python run_tests.py
```

In a Git checkout with the POSIX environment layout, run the same hook suite
as the lint workflow with `uv run pre-commit run --all-files`.

### IDE Interpreter

Open the repository root as the project in VS Code or PyCharm. Select the
existing project interpreter at `.venv/bin/python` on macOS/Linux or
`.venv\Scripts\python.exe` on Windows. Use the interpreter from your checkout
so imports, docstrings, and test runs resolve against the synced environment.
To check the path uv is using, run
`uv run python -c "import sys; print(sys.executable)"`.

## Pre-commit Hooks

### How It Works

Once installed, pre-commit runs Ruff (linter and formatter) and ty before each commit:

1. You make changes and run `git commit`
2. Pre-commit runs Ruff linter with auto-fix and formatter on staged Python files, then ty over the configured source paths
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
# Run lint, formatting, and type checking via pre-commit
uv run pre-commit run --all-files

# Or run Ruff directly
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check format without changing
```

### Type Checking

This project uses [ty](https://docs.astral.sh/ty/) for type checking:

```bash
# Run type checker
uv run ty check

# Check specific file or directory
uv run ty check meteor_core/
```

Type checking rules and source paths are configured under `[tool.ty.*]` in
`pyproject.toml`. The `ty-check` hook checks those source paths as a whole.

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

#### Core calculations and exceptions

| File | Tests | Description |
|------|-------|-------------|
| `test_calculations_v1x.py` | 54 | NPF Rule, pixel pitch, star/meteor trail estimation |
| `test_exceptions_v1x.py` | 70 | Exception hierarchy and diagnostic info |
| `test_schema_context_results_v1x.py` | 4 | Schema context/results helpers |

#### Sensors, optics, and image corrections

| File | Tests | Description |
|------|-------|-------------|
| `test_fisheye_v1x.py` | 27 | Fisheye lens correction |
| `test_sensor_npf_integration_v1x.py` | 16 | Sensor/NPF integration |
| `test_sensor_presets_v1x.py` | 38 | Sensor type presets |
| `test_sensor_validation_v1x.py` | 27 | Sensor override validation |

#### Inputs, outputs, and image I/O

| File | Tests | Description |
|------|-------|-------------|
| `test_image_io_helpers_v1x.py` | 4 | RAW binning and EXIF helpers |
| `test_inputs_base_v1x.py` | 12 | Input loader base class |
| `test_inputs_logging_v1x.py` | 8 | Logging configuration |
| `test_loader_registry_v1x.py` | 30 | Input loader registry |
| `test_output_handler_registry_v1x.py` | 53 | Output handler registry |
| `test_outputs_base_v1x.py` | 7 | Output handler base class |
| `test_raw_loader_v1x.py` | 23 | RAW image loader |

#### Pipeline execution and integration

| File | Tests | Description |
|------|-------|-------------|
| `test_infrastructure_v1x.py` | 27 | ROI, progress, file collection, parameter estimation |
| `test_integration_v1x.py` | 44 | End-to-end meteor detection |
| `test_memory_batch_size_v1x.py` | 6 | Memory-based batch sizing |
| `test_pipeline_execution_v1x.py` | 7 | Pipeline execution and results |
| `test_pipeline_helpers_v1x.py` | 12 | Pipeline helpers |
| `test_pipeline_hooks_v1x.py` | 12 | Hook order, lifetime, process boundaries, interruption/resume |
| `test_pipeline_hook_errors_v1x.py` | 16 | Invalid hook returns, worker and save failures, and resume after detector compatibility errors |
| `test_aircraft_trail_hook_v1x.py` | 3 | Aircraft tracking metadata and progress persistence |

#### Plugins, registries, and contracts

| File | Tests | Description |
|------|-------|-------------|
| `test_detector_plugin_v1x.py` | 26 | Detector plugin architecture |
| `test_detector_registry_v1x.py` | 30 | Detector registry |
| `test_detectors_base_v1x.py` | 12 | Detector base class |
| `test_discovery_parity.py` | 2 | Plugin discovery parity |
| `test_plugin_contract_helpers_v1x.py` | 5 | Plugin contract helpers |
| `test_plugin_contract_validation_v1x.py` | 3 | Plugin contract validation |
| `test_plugin_registry_base.py` | 4 | Plugin registry base class |
| `test_registry_default_contracts_v1x.py` | 2 | Registry default contracts |
| `test_worker_plugin_compatibility_v1x.py` | 4 | Plugin sources, converters, state transport, and recovery with spawn workers |

#### CLI, config, localization, and utilities

| File | Tests | Description |
|------|-------|-------------|
| `test_cli_options_v1x.py` | 4 | CLI option parsing |
| `test_config_io_v1x.py` | 4 | Legacy and versioned YAML/JSON, CLI overrides, and unsupported versions |
| `test_i18n.py` | 5 | Localization lookup and pluralization |
| `test_roi_selector_helpers_v1x.py` | 2 | ROI selector helpers |
| `test_roi_selector_ui_v1x.py` | 2 | ROI selector UI flow |
| `test_utils_display_width_v1x.py` | 3 | Unicode display width helpers |
| `test_utils_roi_hash_v1x.py` | 5 | ROI hash utilities |

#### Missed-candidate analysis, tuning, and export

| File | Tests | Description |
| --- | --- | --- |
| `test_hough_analysis_v1x.py` | 7 | Shared measurement/decision/rendering, normalized inputs |
| `test_analysis_reproduction_v1x.py` | 14 | Read-only pair replay and provenance |
| `test_analysis_cli_v1x.py` | 27 | Analysis/tuning CLI and ROI overrides |
| `test_tuning_v1x.py` | 16 | Bounded proposals and comparison |
| `test_analysis_reports_v1x.py` | 21 | Bundles, fingerprints, and export and replay of versioned sessions |

The test runner reports the total for the current checkout.

## Code Style

### Building a Release

Set the release version in `pyproject.toml` and `meteor_core/schema.py`, then
regenerate `uv.lock`. The setuptools backend packages the CLI module and installs
the `detect-meteors` command. `MANIFEST.in` includes documentation, configuration
examples, and tests in the source distribution.

```bash
uv lock
uv sync --locked --all-extras
uv run python run_tests.py
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv build
```

The output filenames in `dist/` include the release version. To verify the wheel,
create a virtual environment in a separate directory, install the wheel there
with `uv pip install /path/to/the-built.whl`, activate that environment, and run
`detect-meteors --version` and `detect-meteors --help` from that directory.
Check the source archive for the sample configuration, documentation, tests,
and the aircraft sample's README and checksums. RAW images and generated
candidates belong outside the distribution. For commands to run against the
local sample, see the [aircraft guide](aircraft_light_trails_hook_design.md).

### Standards

- **Ruff** for linting and formatting (88 character line length)
- Markdown code examples are excluded from Ruff formatting. Preserve manually aligned inline `#` comments for readability.
- **Python 3.12 or 3.13** required (`>=3.12,<3.14`)
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
│   ├── __init__.py                # Package entry point
│   ├── i18n.py                    # Locale resolution and message formatting
│   ├── schema.py                  # Type definitions, constants
│   ├── config_io.py               # YAML/JSON pipeline configuration
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── pipeline.py                # Pipeline orchestration
│   ├── analysis.py                # Reproduce a Hough decision for an explicit pair
│   ├── analysis_cli.py            # Analysis input selection and CLI reporting
│   ├── analysis_reports.py        # Analysis bundles and replay configuration
│   ├── tuning.py                   # Bounded parameter search and comparisons
│   ├── image_io.py                # Image IO, EXIF utilities
│   ├── roi_selector.py            # ROI selection
│   ├── utils.py                   # Utility functions
│   ├── messages.py                # User-facing message helpers
│   ├── plugin_registry_base.py    # Base class for plugin registries
│   ├── plugin_registry.py         # Unified plugin registry
│   ├── plugin_contract.py         # Plugin contract definitions
│   ├── inputs/                    # Input loader plugins
│   ├── detectors/                 # Detection algorithm plugins
│   ├── outputs/                   # Output handler plugins
│   ├── hooks/                     # Pipeline hooks, including aircraft trail analysis
│   ├── locales/                   # Translations
│   └── templates/                 # Report templates and assets
├── candidates/                    # Default output folder for detections
├── config_examples/               # Sample configuration files
├── docs/                          # Guides, changelogs, release notes, and translations
│   ├── INSTALL_DEV.md             # Developer setup guide
│   └── PLUGIN_AUTHOR_GUIDE.md     # Plugin development guide
├── debug_masks/                   # Debug output masks
├── rawfiles/                      # Sample/raw image inputs
├── tests/                         # Test suite
├── pyproject.toml                 # Project configuration
├── run_tests.py                   # Test runner helper
├── README.md                      # English overview
└── README_ja.md                   # Japanese overview
```

## Plugin Development

For creating custom plugins (input loaders, detectors, output handlers, pipeline hooks), see the comprehensive [PLUGIN_AUTHOR_GUIDE.md](PLUGIN_AUTHOR_GUIDE.md).

> ⚠️ **Note**: The plugin architecture is experimental and may change before v2.0.

## Contribution Workflow

1. **Fork** the repository
2. **Clone** your fork locally
3. **Set up** development environment (this guide)
4. **Create** a feature branch: `git checkout -b feature/your-feature`
5. **Make** changes and add tests
6. **Run** tests: `uv run python run_tests.py`
7. **Measure** coverage: `uv run coverage run -m unittest discover tests && uv run coverage report`
8. **Commit** (the installed hook runs Ruff and ty)
9. **Push** to your fork
10. **Create** a Pull Request

## Troubleshooting

### Updating Ruff Version

Ruff is pinned in `pyproject.toml` (`ruff==X.Y.Z` in dev dependencies).
The local hooks in `.pre-commit-config.yaml` invoke `.venv/bin/ruff` and
`.venv/bin/ty`; they do not have separate version pins. Update the dependency
pin, regenerate `uv.lock` with `uv lock`, and run
`uv sync --locked --all-extras` to refresh the tools used by pre-commit.

### Pre-commit Issues

```bash
# Reinstall hooks
uv run pre-commit uninstall
uv run pre-commit install

# Refresh the locally pinned tools
uv sync --locked --all-extras
```

### Test Issues

```bash
# Ensure dependencies are installed
uv sync --locked --all-extras

# Reinstall if needed
uv sync --locked --all-extras --reinstall
```

### Import Errors

```bash
# Test imports
uv run python -c "from meteor_core import BaseInputLoader, BaseOutputHandler, BaseDetector"
```

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](../LICENSE) for details.

When redistributing, include the [NOTICE](../NOTICE) file.

## Resources

- [PLUGIN_AUTHOR_GUIDE.md](PLUGIN_AUTHOR_GUIDE.md) — Plugin development
- [README.md](../README.md) — User documentation
- [CHANGELOG.md](CHANGELOG.md) — Release history
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Pre-commit Documentation](https://pre-commit.com/)
