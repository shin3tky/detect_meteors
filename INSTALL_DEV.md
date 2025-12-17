# Developer Installation Guide

This guide provides setup instructions for contributors and developers working on Detect Meteors CLI.

For basic user installation, see [INSTALL.md](INSTALL.md).
For plugin development, see [PLUGIN_AUTHOR_GUIDE.md](PLUGIN_AUTHOR_GUIDE.md).

## Prerequisites

Complete the basic installation from [INSTALL.md](INSTALL.md) first, then follow this guide for development-specific setup.

## Development Environment Setup

### Step 1: Install Development Dependencies

After activating your virtual environment, install development dependencies (defined in `pyproject.toml`):

```bash
pip install -e ".[dev]"
```

This installs the project in editable mode plus the dev toolchain (Black, flake8, flake8-pyproject, coverage, pre-commit). If you prefer a minimal install, the equivalent explicit command is:

```bash
pip install pre-commit black flake8 flake8-pyproject coverage
```

### Step 2: Set Up Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) with [Black](https://github.com/psf/black) for automatic code formatting.

```bash
# Install the git hooks
pre-commit install

# Verify installation
pre-commit --version
```

### Step 3: Verify Setup

Run the formatter on all files to verify the setup:

```bash
pre-commit run --all-files
```

## Pre-commit Hooks

### How It Works

Once installed, pre-commit will automatically run Black (formatter) and flake8 (linter) before each commit:

1. You make changes and run `git commit`
2. Pre-commit runs Black (formatting) and flake8 (linting) on staged files
3. If formatting changes are made or linting errors are found, the commit is aborted
4. You fix any issues, stage the changes, and commit again

### Example Workflow

```bash
# Make your changes
vim detect_meteors_cli.py

# Stage and commit
git add detect_meteors_cli.py
git commit -m "Add new feature"

# If Black reformats files or flake8 finds errors, you'll see:
# - Files were modified by this hook (Black)
# - or linting errors (flake8)
# - Commit aborted

# Fix any issues, stage the changes, and commit again
git add detect_meteors_cli.py
git commit -m "Add new feature"
```

### Manual Formatting and Linting

```bash
# Run both Black and flake8 via pre-commit
pre-commit run --all-files

# Or run tools directly
black detect_meteors_cli.py meteor_core/ tests/
flake8 .
```

## Running Tests

### Using the Test Runner

```bash
python run_tests.py
```

### Using unittest Directly

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py" -v

# Run specific test file
python -m unittest tests.test_calculations_v1x -v

# Run specific test class
python -m unittest tests.test_calculations_v1x.TestCalculateNPFRule -v
```

### Test Coverage

```bash
# Run tests with coverage measurement
coverage run -m unittest discover tests

# View coverage report
coverage report

# Generate HTML report (outputs to htmlcov/)
coverage html
```

### Test Files Overview

| File | Tests | Description |
|------|-------|-------------|
| `test_exceptions_v1x.py` | 66 | Exception hierarchy and diagnostic info |
| `test_calculations_v1x.py` | 54 | NPF Rule, pixel pitch, star/meteor trail estimation |
| `test_output_handler_registry_v1x.py` | 51 | Output handler registry |
| `test_integration_v1x.py` | 44 | End-to-end meteor detection |
| `test_sensor_presets_v1x.py` | 38 | Sensor type presets |
| `test_loader_registry_v1x.py` | 30 | Input loader registry |
| `test_detector_registry_v1x.py` | 30 | Detector registry |
| `test_sensor_validation_v1x.py` | 27 | Sensor override validation |
| `test_fisheye_v1x.py` | 27 | Fisheye lens correction |
| `test_detector_plugin_v1x.py` | 24 | Detector plugin architecture |
| `test_raw_loader_v1x.py` | 19 | RAW image loader |
| `test_sensor_npf_integration_v1x.py` | 16 | Sensor/NPF integration |
| `test_infrastructure_v1x.py` | 16 | ROI, progress, file collection |
| `test_inputs_logging_v1x.py` | 8 | Logging configuration |
| `test_memory_batch_size_v1x.py` | 6 | Memory-based batch sizing |
| `test_plugin_registry_base.py` | 4 | Plugin registry base class |
| `test_plugin_contract_validation_v1x.py` | 3 | Plugin contract validation |
| `test_cli_options_v1x.py` | 2 | CLI option parsing |
| `test_discovery_parity.py` | 2 | Plugin discovery parity |
| `test_registry_default_contracts_v1x.py` | 2 | Registry default contracts |

**Total: 469 tests**

## Code Style

### Standards

- **Black** with default settings (88 character line length)
- **flake8** for static code analysis
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

## Project Structure

```
detect_meteors/
├── detect_meteors_cli.py          # CLI interface
├── meteor_core/                   # Core logic modules
│   ├── schema.py                  # Type definitions, constants
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── pipeline.py                # Pipeline orchestration
│   ├── image_io.py                # Image IO, EXIF utilities
│   ├── roi_selector.py            # ROI selection
│   ├── utils.py                   # Utility functions
│   ├── plugin_registry_base.py    # Base class for plugin registries
│   ├── plugin_registry.py         # Unified plugin registry
│   ├── plugin_contract.py         # Plugin contract definitions
│   ├── inputs/                    # Input loader plugins
│   ├── outputs/                   # Output handler plugins
│   └── detectors/                 # Detection algorithm plugins
├── tests/                         # Test suite
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Dependencies
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
6. **Run** tests: `python run_tests.py`
7. **Measure** coverage: `coverage run -m unittest discover tests && coverage report`
8. **Commit** (pre-commit will run Black and flake8 automatically)
9. **Push** to your fork
10. **Create** a Pull Request

## Troubleshooting

### Pre-commit Issues

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Update to latest versions
pre-commit autoupdate
```

### Test Issues

```bash
# Ensure virtual environment is activated
source ./venv/bin/activate  # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows

# Install missing dependencies
pip install -r requirements.txt
pip install pre-commit black flake8 flake8-pyproject coverage
```

### Import Errors

```bash
# Test imports
python -c "from meteor_core import BaseInputLoader, BaseOutputHandler, BaseDetector"
```

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

When redistributing, include the [NOTICE](NOTICE) file.

## Resources

- [PLUGIN_AUTHOR_GUIDE.md](PLUGIN_AUTHOR_GUIDE.md) - Plugin development
- [README.md](README.md) - User documentation
- [CHANGELOG.md](CHANGELOG.md) - Release history
- [Black Documentation](https://black.readthedocs.io/)
- [flake8 Documentation](https://flake8.pycqa.org/)
- [Pre-commit Documentation](https://pre-commit.com/)
