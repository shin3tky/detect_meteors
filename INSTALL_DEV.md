# Developer Installation Guide

This guide provides setup instructions for contributors and developers working on Detect Meteors CLI.

For basic user installation, see [INSTALL.md](INSTALL.md).

## Prerequisites

Complete the basic installation from [INSTALL.md](INSTALL.md) first, then follow this guide for development-specific setup.

## Development Environment Setup

### Step 1: Install Development Dependencies

After activating your virtual environment, install development dependencies:

```bash
pip install pre-commit black flake8
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

Once installed, Black will automatically format your code before each commit:

1. You make changes and run `git commit`
2. Pre-commit runs Black on staged files
3. If formatting changes are made, the commit is aborted
4. You review changes, stage them, and commit again

### Example Workflow

```bash
# Make your changes
vim detect_meteors_cli.py

# Stage and commit
git add detect_meteors_cli.py
git commit -m "Add new feature"

# If Black reformats files, you'll see:
# - Files were modified by this hook
# - Commit aborted

# Stage the formatted files and commit again
git add detect_meteors_cli.py
git commit -m "Add new feature"
```

### Manual Formatting

To manually format files:

```bash
# Format all files via pre-commit
pre-commit run --all-files

# Or run Black directly on specific files
black detect_meteors_cli.py
black meteor_core/
black tests/
```

### Code Quality Tools

This project uses two complementary tools for code quality:

- **Black** (v1.5.0+): Automatic code formatting via pre-commit hooks
- **flake8** (v1.5.8+): Static code analysis and style checking

As of v1.5.8, flake8 is formally integrated with project-specific configuration in `.flake8`.

### Manual Linting

To manually check code quality with flake8:

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

### Configuration

The project uses multiple configuration files for code quality and formatting:

#### `.pre-commit-config.yaml`

Pre-commit hooks configuration:

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
```

#### `pyproject.toml`

Black formatter configuration:

```toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311', 'py312']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.venv
  | venv
  | __pycache__
  | \.serena
  | \.ruff_cache
  | \.github
  | build
  | dist
  | \.egg-info
  | rawfiles
  | candidates
  | debug_masks
)/
'''
```

#### `.flake8`

Flake8 linter configuration:

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
    debug_masks

max-line-length = 88
ignore = E203,W503,E501,E226
max-complexity = 70
```

## Running Tests

### Using the Test Runner

The project includes a version-aware test runner:

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

# Run specific test method
python -m unittest tests.test_calculations_v1x.TestCalculateNPFRule.test_npf_rule_basic -v
```

### Test Files

| File | Tests | Description |
|------|-------|-------------|
| `test_calculations_v1x.py` | 54 | NPF Rule, pixel pitch, star/meteor trail estimation |
| `test_integration_v1x.py` | 44 | End-to-end meteor detection with various parameters |
| `test_sensor_presets_v1x.py` | 42 | Sensor type presets (`--sensor-type`) |
| `test_sensor_validation_v1x.py` | 23 | Sensor override validation (v1.5.2+) |
| `test_sensor_npf_integration_v1x.py` | 16 | Sensor preset integration with NPF calculations |
| `test_infrastructure_v1x.py` | 16 | ROI parsing, progress tracking, file collection, auto-estimation logic |
| `test_memory_batch_size_v1x.py` | 6 | Memory-based batch size adjustment |
| `test_fisheye_v1x.py` | 27 | Fisheye lens correction (v1.5.3+) |

**Total: 228 tests**

### Writing Tests

When adding new features, follow these guidelines:

1. **Location**: Place tests in `tests/` directory
2. **Naming**: Use `test_<feature>_v1x.py` format
3. **Structure**: One test class per logical unit
4. **Coverage**: Test both success and failure cases

Example:

```python
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meteor_core.utils import your_function

class TestYourFunction(unittest.TestCase):
    def test_basic_case(self):
        """Test basic functionality."""
        result = your_function(input_value)
        self.assertEqual(result, expected_value)

    def test_edge_case(self):
        """Test edge case handling."""
        result = your_function(edge_input)
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
```

## Code Style

### Formatter

- **Black** with default settings (88 character line length)
- Automatically enforced via pre-commit hooks

### Linter

- **flake8** for code quality and style checking
- Helps catch common errors and enforce PEP 8 compliance
- Run manually before committing

### Python Version

- **Python 3.12+** recommended
- Tested on Python 3.12.12

### Type Hints

Type hints are used throughout the codebase. As of v1.5.5, structured data uses TypedDict for improved type safety:

```python
from typing import TypedDict

class NPFMetrics(TypedDict):
    pixel_pitch: float
    npf_recommended: float
    actual_exposure: float
    npf_ratio: float
    compliance: str

def calculate_npf_metrics(...) -> NPFMetrics:
    """Calculate NPF Rule metrics."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def estimate_star_trail_length(
    focal_length_mm: float,
    exposure_time_sec: float,
    image_width_px: int,
    declination_deg: float = 0.0,
) -> float:
    """
    Estimate star trail length in pixels during exposure.

    Args:
        focal_length_mm: Focal length in 35mm equivalent (mm)
        exposure_time_sec: Exposure time in seconds
        image_width_px: Image width in pixels
        declination_deg: Declination in degrees (default: 0°)

    Returns:
        Star trail length in pixels
    """
```

## Project Structure

```
detect_meteors/
├── detect_meteors_cli.py          # CLI interface (argument parsing, user interaction)
├── meteor_core/                   # Core logic modules (v1.5.6 plugin-ready)
│   ├── __init__.py
│   ├── schema.py                  # Type definitions (TypedDict, constants)
│   ├── pipeline.py                # Processing pipeline orchestration + PipelineConfig
│   ├── image_io.py                # Shared image IO helpers, EXIF utilities
│   ├── inputs/                    # Input loader plugins (RAW readers, metadata)
│   │   ├── __init__.py
│   │   ├── base.py                # InputLoader/MetadataExtractor protocols, helpers
│   │   ├── discovery.py           # Plugin discovery (entry points, ~/.detect_meteors/plugins)
│   │   └── raw.py                 # Built-in RAW loaders and configs
│   ├── outputs/                   # Output handlers + writer orchestration
│   │   ├── __init__.py
│   │   ├── handler.py             # OutputHandler protocol + plugin helpers
│   │   └── writer.py              # Result writer, output formatting
│   ├── detectors/                 # Detection algorithm implementations
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract base detector class
│   │   └── hough_default.py       # Default Hough transform detector
│   ├── roi_selector.py            # ROI selection interface
│   └── utils.py                   # Utility functions (NPF calculations, estimation)
├── candidates/                    # Sample candidate outputs
├── rawfiles/                      # Sample input files
├── debug_masks/                   # Debug masks for tests/examples
├── tests/                         # Test suite
│   ├── test_calculations_v1x.py
│   ├── test_integration_v1x.py
│   ├── test_sensor_presets_v1x.py
│   ├── test_sensor_validation_v1x.py
│   ├── test_sensor_npf_integration_v1x.py
│   ├── test_infrastructure_v1x.py
│   ├── test_memory_batch_size_v1x.py
│   └── test_fisheye_v1x.py
├── run_tests.py                   # Version-aware test runner
├── .pre-commit-config.yaml        # Pre-commit hooks configuration
├── pyproject.toml                 # Black formatter configuration
├── .flake8                        # Flake8 linter configuration
├── requirements.txt               # Python dependencies
├── CHANGELOG.md                   # Release history
├── README.md                      # User documentation
├── INSTALL.md                     # User installation guide
├── INSTALL_DEV.md                 # Developer installation guide (this file)
├── ROADMAP.md                     # Development roadmap
├── RELEASE_NOTES_*.md             # Version-specific release notes
├── COMMAND_OPTIONS.md             # CLI options reference
├── NPF_RULE.md                    # NPF Rule documentation
├── detect_meteors_cli_completion.bash  # Bash completion
├── _detect_meteors_cli            # Zsh completion
├── workflow.png                   # Pipeline overview diagram
├── LICENSE                        # Apache License 2.0
└── NOTICE                         # Attribution notices
```

### Module Responsibilities (v1.5.6+)

| Module | Responsibility |
|--------|----------------|
| `detect_meteors_cli.py` | CLI argument parsing, user interaction, main entry point |
| `meteor_core/schema.py` | Type definitions (TypedDict), constants, data structures |
| `meteor_core/pipeline.py` | DetectionPipeline protocol, PipelineConfig, orchestration hooks |
| `meteor_core/image_io.py` | Shared image loading helpers, EXIF metadata utilities |
| `meteor_core/roi_selector.py` | Interactive ROI selection interface |
| `meteor_core/utils.py` | Utility functions (NPF calculations, parameter estimation) |
| `meteor_core/inputs/base.py` | InputLoader/MetadataExtractor protocols and validation helpers |
| `meteor_core/inputs/discovery.py` | Plugin discovery for input loaders (entry points, plugin dir) |
| `meteor_core/inputs/raw.py` | Built-in RAW loader configs and factory helpers |
| `meteor_core/detectors/base.py` | Abstract base class for detection algorithms |
| `meteor_core/detectors/hough_default.py` | Default Hough transform-based meteor detector |
| `meteor_core/outputs/handler.py` | OutputHandler protocol, plugin helpers |
| `meteor_core/outputs/writer.py` | Result writer, output management |

This modular structure prepares for the v2.x plugin architecture by separating concerns and enabling future extensibility.

## Contribution Workflow

1. **Fork** the repository
2. **Clone** your fork locally
3. **Set up** development environment (this guide)
4. **Create** a feature branch: `git checkout -b feature/your-feature`
5. **Make** changes and add tests
6. **Run** tests: `python run_tests.py`
7. **Check** code quality: `flake8 .`
8. **Commit** (pre-commit will format code automatically)
9. **Push** to your fork
10. **Create** a Pull Request

## Troubleshooting

### Pre-commit Issues

**Hook not running:**
```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install
```

**Outdated hooks:**
```bash
# Update to latest versions
pre-commit autoupdate
```

### Test Issues

**Import errors:**
```bash
# Ensure you're in the project root
cd /path/to/detect_meteors

# Ensure virtual environment is activated
source ./venv/bin/activate  # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
pip install pre-commit black flake8
```

### Linting Issues

**Too many errors:**
```bash
# Focus on specific error types first
flake8 --select=E9,F63,F7,F82 .

# Ignore specific errors temporarily
flake8 --extend-ignore=E501,W503 .
```

**Configuration conflicts:**
```bash
# Check flake8 configuration
cat .flake8

# Check Black configuration
cat pyproject.toml | grep -A 20 "\[tool.black\]"
```

### Module Import Issues (v1.5.5+)

If you encounter import errors after the v1.5.5 restructuring:

```bash
# Ensure meteor_core is recognized as a package
ls meteor_core/__init__.py

# Test imports
python -c "from meteor_core import pipeline, utils"
```

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

### NOTICE File

When redistributing this software (in source or binary form), you must include the [NOTICE](NOTICE) file. The NOTICE file contains required attribution notices for this project and any bundled third-party components.

For more information about Apache License 2.0 compliance, see the [Apache License FAQ](https://www.apache.org/foundation/license-faq.html).

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [flake8 Documentation](https://flake8.pycqa.org/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Python unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Project README](README.md)
- [CHANGELOG](CHANGELOG.md)
