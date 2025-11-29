# Developer Installation Guide

This guide provides setup instructions for contributors and developers working on Detect Meteors CLI.

For basic user installation, see [INSTALL.md](INSTALL.md).

## Prerequisites

Complete the basic installation from [INSTALL.md](INSTALL.md) first, then follow this guide for development-specific setup.

## Development Environment Setup

### Step 1: Install Development Dependencies

After activating your virtual environment, install development dependencies:

```bash
pip install pre-commit black
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
black tests/
```

### Configuration

The pre-commit configuration is in `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 25.11.0
    hooks:
      - id: black
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
| `test_calculations_v1x.py` | 56 | NPF Rule, pixel pitch, star/meteor trail estimation |
| `test_integration_v1x.py` | 44 | End-to-end meteor detection with various parameters |
| `test_sensor_presets_v1x.py` | 28 | Sensor type presets (`--sensor-type`) |
| `test_sensor_npf_integration_v1x.py` | 8 | Sensor preset integration with NPF calculations |
| `test_memory_batch_size_v1x.py` | 6 | Memory-based batch size adjustment |

**Total: 142 tests**

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

from detect_meteors_cli import your_function

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

### Python Version

- **Python 3.12+** recommended
- Tested on Python 3.12.12

### Type Hints

Type hints are used throughout the codebase:

```python
def calculate_pixel_pitch(sensor_width_mm: float, image_width_px: int) -> float:
    """Calculate pixel pitch in micrometers."""
    return (sensor_width_mm * 1000.0) / image_width_px
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
├── detect_meteors_cli.py          # Main CLI application
├── tests/                         # Test suite
│   ├── test_calculations_v1x.py
│   ├── test_integration_v1x.py
│   ├── test_sensor_presets_v1x.py
│   ├── test_sensor_npf_integration_v1x.py
│   └── test_memory_batch_size_v1x.py
├── run_tests.py                   # Version-aware test runner
├── .pre-commit-config.yaml        # Pre-commit hooks configuration
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
└── _detect_meteors_cli            # Zsh completion
```

## Contribution Workflow

1. **Fork** the repository
2. **Clone** your fork locally
3. **Set up** development environment (this guide)
4. **Create** a feature branch: `git checkout -b feature/your-feature`
5. **Make** changes and add tests
6. **Run** tests: `python run_tests.py`
7. **Commit** (pre-commit will format code automatically)
8. **Push** to your fork
9. **Create** a Pull Request

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
pip install pre-commit black
```

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Python unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Project README](README.md)
- [CHANGELOG](CHANGELOG.md)
