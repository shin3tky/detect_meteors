# Developer Installation Guide

This guide provides setup instructions for contributors and developers working on Detect Meteors CLI.

For basic user installation, see [INSTALL.md](INSTALL.md).

## Prerequisites

Complete the basic installation from [INSTALL.md](INSTALL.md) first, then follow this guide for development-specific setup.

## Development Environment Setup

### Step 1: Install Development Dependencies

After activating your virtual environment, install development dependencies:

```bash
pip install pre-commit black flake8 flake8-pyproject coverage
```

**Note**: `flake8-pyproject` enables flake8 to read configuration from `pyproject.toml`.

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

### Code Quality Tools

This project uses two complementary tools for code quality, both automatically executed via pre-commit hooks:

- **Black**: Automatic code formatting
- **flake8**: Static code analysis and style checking

All tool configurations are consolidated in `pyproject.toml`.

### Manual Formatting

To manually run formatting:

```bash
# Run both Black and flake8 via pre-commit
pre-commit run --all-files

# Or run Black directly on specific files
black detect_meteors_cli.py
black meteor_core/
black tests/
```

### Manual Linting

To manually run linting:

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

The project uses configuration files for code quality and formatting:

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

All tool configurations are centralized in `pyproject.toml`:

**Black formatter:**

```toml
[tool.black]
line-length = 88
target-version = ["py312"]
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

**Flake8 linter:**

```toml
[tool.flake8]
max-line-length = 88
max-complexity = 40
exclude = [
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".serena",
    ".ruff_cache",
    ".github",
    "build",
    "dist",
    "*.egg-info",
    "rawfiles",
    "candidates",
    "debug_masks",
]
# E203: whitespace before ':'
# W503: line break before binary operator
# E501: line too long (handled by Black)
# E226: missing whitespace around arithmetic operator
ignore = ["E203", "W503", "E501", "E226"]
```

**Coverage** (for test coverage measurement):

```toml
[tool.coverage.run]
source = ["meteor_core"]
branch = true
omit = [
    "tests/*",
    "*/__pycache__/*",
    ".venv/*",
]

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

### Test Coverage

To measure test coverage:

```bash
# Run tests with coverage measurement
coverage run -m unittest discover tests

# View coverage report in terminal
coverage report

# Generate HTML report (outputs to htmlcov/)
coverage html
```

### Test Files

| File | Tests | Description |
|------|-------|-------------|
| `test_calculations_v1x.py` | 54 | NPF Rule, pixel pitch, star/meteor trail estimation |
| `test_integration_v1x.py` | 44 | End-to-end meteor detection with various parameters |
| `test_sensor_presets_v1x.py` | 42 | Sensor type presets (`--sensor-type`) |
| `test_loader_registry_v1x.py` | 30 | Input loader registry (v1.5.6+) |
| `test_fisheye_v1x.py` | 27 | Fisheye lens correction (v1.5.3+) |
| `test_detector_registry_v1x.py` | 26 | Detector registry (v1.5.10+) |
| `test_detector_plugin_v1x.py` | 24 | Detector plugin architecture (v1.5.10+) |
| `test_sensor_validation_v1x.py` | 23 | Sensor override validation (v1.5.2+) |
| `test_sensor_npf_integration_v1x.py` | 16 | Sensor preset integration with NPF calculations |
| `test_infrastructure_v1x.py` | 16 | ROI parsing, progress tracking, file collection, auto-estimation logic |
| `test_memory_batch_size_v1x.py` | 6 | Memory-based batch size adjustment |

**Total: 308 tests**

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
- Automatically executed via pre-commit hooks

### Python Version

- **Python 3.12+** required
- Tested on Python 3.12.12

### Type Hints

Type hints are used throughout the codebase. As of v1.5.5, structured data uses dataclasses for improved type safety:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class NPFMetrics:
    pixel_pitch_um: Optional[float] = None
    npf_recommended_sec: Optional[float] = None
    star_trail_px: Optional[float] = None
    compliance_level: str = "UNKNOWN"
    overshoot_factor: float = 0.0
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
├── meteor_core/                   # Core logic modules (v1.5.11+ ABC-based plugin architecture)
│   ├── __init__.py
│   ├── schema.py                  # Type definitions (dataclasses), constants
│   ├── pipeline.py                # Processing pipeline orchestration + PipelineConfig
│   ├── image_io.py                # Shared image IO helpers, EXIF utilities
│   ├── inputs/                    # Input loader plugins (RAW readers, metadata)
│   │   ├── __init__.py
│   │   ├── base.py                # BaseInputLoader/BaseMetadataExtractor ABCs, helpers
│   │   ├── registry.py            # LoaderRegistry (plugin registration and lookup)
│   │   ├── discovery.py           # Plugin discovery (entry points, ~/.detect_meteors/input_plugins)
│   │   └── raw.py                 # Built-in RAW loader (RawImageLoader)
│   ├── outputs/                   # Output handlers + writer orchestration
│   │   ├── __init__.py
│   │   ├── handler.py             # BaseOutputHandler ABC
│   │   └── writer.py              # OutputWriter, ProgressManager
│   ├── detectors/                 # Detection algorithm implementations
│   │   ├── __init__.py
│   │   ├── base.py                # BaseDetector ABC
│   │   ├── registry.py            # DetectorRegistry (plugin registration and lookup)
│   │   ├── discovery.py           # Plugin discovery (entry points, ~/.detect_meteors/detector_plugins)
│   │   └── hough_default.py       # HoughDetector (default implementation)
│   ├── roi_selector.py            # ROI selection interface
│   └── utils.py                   # Utility functions (NPF calculations, estimation)
├── candidates/                    # Sample candidate outputs
├── rawfiles/                      # Sample input files
├── debug_masks/                   # Debug masks for tests/examples
├── tests/                         # Test suite
│   ├── test_calculations_v1x.py
│   ├── test_detector_plugin_v1x.py
│   ├── test_detector_registry_v1x.py
│   ├── test_fisheye_v1x.py
│   ├── test_infrastructure_v1x.py
│   ├── test_integration_v1x.py
│   ├── test_loader_registry_v1x.py
│   ├── test_memory_batch_size_v1x.py
│   ├── test_sensor_npf_integration_v1x.py
│   ├── test_sensor_presets_v1x.py
│   └── test_sensor_validation_v1x.py
├── run_tests.py                   # Version-aware test runner
├── .pre-commit-config.yaml        # Pre-commit hooks configuration
├── pyproject.toml                 # Project metadata and tool configurations
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

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `detect_meteors_cli.py` | CLI argument parsing, user interaction, main entry point |
| `meteor_core/schema.py` | Type definitions (dataclasses), constants, data structures |
| `meteor_core/pipeline.py` | DetectionPipeline protocol, PipelineConfig, orchestration hooks |
| `meteor_core/image_io.py` | Shared image loading helpers, EXIF metadata utilities |
| `meteor_core/roi_selector.py` | Interactive ROI selection interface |
| `meteor_core/utils.py` | Utility functions (NPF calculations, parameter estimation) |
| `meteor_core/inputs/base.py` | BaseInputLoader/BaseMetadataExtractor ABCs and validation helpers |
| `meteor_core/inputs/registry.py` | LoaderRegistry for plugin registration and case-insensitive lookup |
| `meteor_core/inputs/discovery.py` | Plugin discovery for input loaders (entry points, input_plugins dir) |
| `meteor_core/inputs/raw.py` | Built-in RAW loader (RawImageLoader) |
| `meteor_core/detectors/base.py` | BaseDetector ABC for detection algorithms |
| `meteor_core/detectors/registry.py` | DetectorRegistry for plugin registration and case-insensitive lookup |
| `meteor_core/detectors/discovery.py` | Plugin discovery for detectors (entry points, detector_plugins dir) |
| `meteor_core/detectors/hough_default.py` | HoughDetector (default Hough transform-based detector) |
| `meteor_core/outputs/handler.py` | BaseOutputHandler ABC |
| `meteor_core/outputs/writer.py` | OutputWriter, ProgressManager |

This modular structure prepares for the v2.x plugin architecture by separating concerns and enabling future extensibility.

## Plugin Architecture (v1.5.10+)

> ⚠️ **Experimental**: The plugin architecture is under active development and **may undergo breaking changes before the v2.0 stable release**. Plugin interfaces, discovery mechanisms, base class signatures, and configuration formats could be modified based on feedback and evolving requirements. If you are developing custom plugins, please be prepared to update your code when upgrading to future versions.

As of v1.5.10, the project uses **Abstract Base Classes (ABC)** for all plugin interfaces. This provides clear contracts for plugin developers with immediate error detection for missing implementations.

### Abstract Base Classes Overview

| ABC | Location | Purpose |
|-----|----------|---------|
| `BaseInputLoader` | `meteor_core/inputs/base.py` | Image loading plugins |
| `BaseMetadataExtractor` | `meteor_core/inputs/base.py` | Metadata extraction (optional mixin) |
| `BaseOutputHandler` | `meteor_core/outputs/handler.py` | Output handling plugins |
| `BaseDetector` | `meteor_core/detectors/base.py` | Detection algorithm plugins |

### Class Hierarchy

```
BaseInputLoader (ABC)
├── DataclassInputLoader (ABC + Generic) - for dataclass-based config
│   └── RawImageLoader (+ BaseMetadataExtractor)
└── PydanticInputLoader (ABC + Generic) - for Pydantic-based config

BaseMetadataExtractor (ABC)
└── RawImageLoader (multiple inheritance)

BaseOutputHandler (ABC)
└── OutputWriter

BaseDetector (ABC)
└── HoughDetector
```

### Creating a Custom Input Loader

To create a custom input loader, inherit from `BaseInputLoader` and implement the required `load()` method:

```python
from typing import Dict, Any
import numpy as np
from meteor_core.inputs.base import BaseInputLoader, BaseMetadataExtractor


class MyCustomLoader(BaseInputLoader, BaseMetadataExtractor):
    """Custom loader for a specific image format."""

    plugin_name = "my_format"  # Required: unique identifier
    name = "My Format Loader"  # Human-readable name
    version = "1.0.0"          # Version string

    def __init__(self, config: Any = None):
        self.config = config

    def load(self, filepath: str) -> np.ndarray:
        """Load an image from the given filepath.

        Args:
            filepath: Path to the image file.

        Returns:
            Image data as numpy array.
        """
        # Your implementation here
        import my_format_library
        return my_format_library.load(filepath)

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from the file.

        Args:
            filepath: Path to the image file.

        Returns:
            Dictionary containing metadata.
        """
        # Optional: implement if your format has metadata
        return {"format": "my_format", "version": "1.0"}


# Get plugin metadata
loader = MyCustomLoader()
info = loader.get_info()
# Returns: {'plugin_name': 'my_format', 'name': 'My Format Loader',
#           'version': '1.0.0', 'class': 'MyCustomLoader'}
```

**Key points:**

- `plugin_name` must be defined as a non-empty string
- `name` and `version` are optional but recommended for plugin metadata
- `load()` is an abstract method - instantiation fails without it
- `get_info()` returns plugin metadata dictionary (inherited from base class)
- `BaseMetadataExtractor` is optional - use when metadata extraction is needed
- IDE will show errors for missing abstract methods
- Registry lookup is case-insensitive: `get("my_format")` and `get("MY_FORMAT")` both work

### Creating a Custom Detector

To create a custom detection algorithm, inherit from `BaseDetector`:

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
        """Detect meteor candidates.

        Args:
            current_image: Current frame (uint16 grayscale)
            previous_image: Previous frame (uint16 grayscale)
            roi_mask: Binary mask for region of interest (uint8)
            params: Detection parameters dictionary

        Returns:
            Tuple of (is_candidate, line_score, line_segments, max_aspect_ratio, debug_image)
        """
        # Your detection logic here
        pass

    def compute_line_score(
        self,
        mask: np.ndarray,
        hough_params: Dict[str, int],
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """Compute line detection score.

        Args:
            mask: Binary mask of detected changes
            hough_params: Hough transform parameters

        Returns:
            Tuple of (score, line_segments)
        """
        # Your line scoring logic here
        pass
```

### Creating a Custom Output Handler

To create a custom output handler, inherit from `BaseOutputHandler`:

```python
from typing import List, Optional
import numpy as np
from meteor_core.outputs.handler import BaseOutputHandler


class MyCustomOutputHandler(BaseOutputHandler):
    """Custom output handler (e.g., for cloud storage)."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: Optional[np.ndarray] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> bool:
        """Save a meteor candidate.

        Args:
            source_path: Path to the source file.
            filename: Output filename.
            debug_image: Optional debug visualization.
            roi_polygon: Optional ROI polygon.

        Returns:
            True if saved successfully, False if skipped.
        """
        # Upload to cloud storage
        pass

    def save_debug_image(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> str:
        """Save a debug visualization.

        Args:
            debug_image: Debug visualization image.
            filename: Base filename.
            roi_polygon: Optional ROI polygon.

        Returns:
            Path or URL to the saved debug image.
        """
        # Upload debug image
        pass
```

### Using Dataclass-based Configuration

For loaders with structured configuration, use `DataclassInputLoader`:

```python
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from meteor_core.inputs.base import DataclassInputLoader, BaseMetadataExtractor


@dataclass
class TiffLoaderConfig:
    """Configuration for TIFF loader."""
    normalize: bool = False
    bit_depth: int = 16


class TiffImageLoader(DataclassInputLoader[TiffLoaderConfig], BaseMetadataExtractor):
    """TIFF image loader with configuration."""

    plugin_name = "tiff"
    ConfigType = TiffLoaderConfig

    def load(self, filepath: str) -> np.ndarray:
        """Load a TIFF image."""
        import tifffile
        image = tifffile.imread(filepath)
        if self.config.normalize:
            max_val = 2 ** self.config.bit_depth - 1
            image = image.astype(np.float32) / max_val
        return image

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract TIFF metadata."""
        import tifffile
        with tifffile.TiffFile(filepath) as tif:
            return {"pages": len(tif.pages), "shape": tif.pages[0].shape}
```

### Plugin Discovery

Plugins are discovered automatically from:

1. **Built-in plugins** (e.g., `RawImageLoader`, `HoughDetector`)
2. **Entry points** in `pyproject.toml`:
   ```toml
   [project.entry-points."detect_meteors.input"]
   my_loader = "my_package.loaders:MyCustomLoader"
   
   [project.entry-points."detect_meteors.detector"]
   my_detector = "my_package.detectors:MyCustomDetector"
   ```
3. **Plugin directories**:
   - Input loaders: `~/.detect_meteors/input_plugins/*.py`
   - Detectors: `~/.detect_meteors/detector_plugins/*.py`

**Note**: Registry lookup is case-insensitive. Both `LoaderRegistry.get("raw")` and `LoaderRegistry.get("RAW")` return the same class.

### Why ABC over Protocol?

The project uses ABC instead of Protocol for plugin interfaces because:

| Aspect | ABC | Protocol |
|--------|-----|----------|
| **Error detection** | Immediate at instantiation | Runtime only |
| **IDE support** | Full (auto-complete, warnings) | Limited |
| **Learning curve** | Familiar pattern | Requires typing knowledge |
| **Shared implementation** | Supported (default methods) | Not supported |
| **Discoverability** | Clear inheritance hierarchy | Implicit structural matching |

For plugin developers, ABC provides clearer guidance on what to implement and catches errors earlier in the development cycle.

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
pip install pre-commit black flake8 flake8-pyproject coverage
```

### Linting Issues

**Too many errors:**
```bash
# Focus on specific error types first
flake8 --select=E9,F63,F7,F82 .

# Ignore specific errors temporarily
flake8 --extend-ignore=E501,W503 .
```

**Configuration not loading:**
```bash
# Verify flake8-pyproject is installed
pip show flake8-pyproject

# Check pyproject.toml has [tool.flake8] section
cat pyproject.toml | grep -A 20 "\[tool.flake8\]"
```

### Module Import Issues

If you encounter import errors after the v1.5.5 restructuring:

```bash
# Ensure meteor_core is recognized as a package
ls meteor_core/__init__.py

# Test imports
python -c "from meteor_core import BaseInputLoader, BaseOutputHandler, BaseDetector"
```

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

### NOTICE File

When redistributing this software (in source or binary form), you must include the [NOTICE](NOTICE) file. The NOTICE file contains required attribution notices for this project and any bundled third-party components.

For more information about Apache License 2.0 compliance, see the [Apache License FAQ](https://www.apache.org/foundation/license-faq.html).

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [flake8 Documentation](https://flake8.pycqa.org/)
- [flake8-pyproject Documentation](https://github.com/john-googletmp/flake8-pyproject)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Python unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Python ABC Documentation](https://docs.python.org/3/library/abc.html)
- [Project README](README.md)
- [CHANGELOG](CHANGELOG.md)
