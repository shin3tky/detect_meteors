# Version 1.5 Release Notes

## Version 1.5.6 (2025-12-06)

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

## Version 1.5.5 (2025-12-05)

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

## Version 1.5.4 (2025-12-03)

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

## Version 1.5.3 (2025-12-02)

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

## Version 1.5.2 (2025-12-01)

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

## Version 1.5.1 (2025-11-30)

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

- **Latest Version**: 1.5.5
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
| NPF check | `--show-npf --sensor-type TYPE` | `python detect_meteors_cli.py --show-npf --sensor-type APS-C` |
| NPF + Fisheye | `--show-npf --fisheye` | `python detect_meteors_cli.py --show-npf --sensor-type MFT --focal-length 16 --fisheye` |

## Files Updated (v1.5.x Summary)

| File | Changes |
|------|---------|
| `detect_meteors_cli.py` | CLI interface (v1.5.5: reduced to CLI only) |
| `meteor_core/` | New package with modular components (v1.5.5) |
| `detect_meteors_cli_completion.bash` | Shell completions |
| `_detect_meteors_cli` (zsh) | Shell completions |
| `COMMAND_OPTIONS.md` | CLI options reference |
| `NPF_RULE.md` | NPF Rule documentation |
| `test_fisheye_v1x.py` | Fisheye tests (27 tests) |
| `test_sensor_validation_v1x.py` | Validation tests (23 tests) |

---

**Status**: Production Ready  
**Compatibility**: Fully backward compatible with v1.4.x  
**Recommendation**: Use `--sensor-type` for simplified configuration; add `--fisheye` for fisheye lenses

Happy meteor hunting! 🌠
