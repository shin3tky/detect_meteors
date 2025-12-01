# Version 1.5 Release Notes

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

**Implementation**:
```python
SENSOR_PRESETS = {
    "1INCH": {
        "focal_factor": 2.7,
        "sensor_width": 13.2,
        "pixel_pitch": 2.4,
        "description": "1-inch sensor (13.2×8.8mm)",
    },
    "MFT": {
        "focal_factor": 2.0,
        "sensor_width": 17.3,
        "pixel_pitch": 3.7,
        "description": "Micro Four Thirds (17.3×13mm)",
    },
    # ... additional presets ordered by sensor size
    "MF44X33": {
        "focal_factor": 0.79,
        "sensor_width": 43.8,
        "pixel_pitch": 3.76,
        "description": "Medium Format 44×33 (43.8×32.9mm) - GFX/645Z/X2D",
    },
    "MF54X40": {
        "focal_factor": 0.64,
        "sensor_width": 53.4,
        "pixel_pitch": 4.6,
        "description": "Medium Format 54×40 (53.4×40mm) - Hasselblad H6D-100c",
    },
}
```

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

**What it sets automatically**:
- `--focal-factor` (crop factor for 35mm equivalent)
- `--sensor-width` (physical sensor width in mm)
- `--pixel-pitch` (typical pixel pitch in μm)

### 3. Parameter Override Priority (NEW)

Individual parameters always take priority over `--sensor-type` presets:

```bash
# Use MFT preset but override pixel pitch
python detect_meteors_cli.py --auto-params \
  --sensor-type MFT \
  --pixel-pitch 3.3  # Override for specific camera
```

**Priority Order** (highest to lowest):
1. Individual CLI arguments (`--focal-factor`, `--sensor-width`, `--pixel-pitch`, `--focal-length`)
2. `--sensor-type` preset values
3. Default/calculated values

### 4. New `--list-sensor-types` Option (NEW)

**Purpose**: Display available sensor presets and their configurations

**Usage**:
```bash
python detect_meteors_cli.py --list-sensor-types
```

**Output**:
```
======================================================================
Available Sensor Types (--sensor-type)
======================================================================

  1INCH         1-inch sensor (13.2×8.8mm)
                  focal_factor=2.7, sensor_width=13.2mm, pixel_pitch=2.4μm

  MFT           Micro Four Thirds (17.3×13mm)
                  focal_factor=2.0, sensor_width=17.3mm, pixel_pitch=3.7μm

  APSC          APS-C Sony/Nikon/Fuji (23.5×15.6mm)
                  focal_factor=1.5, sensor_width=23.5mm, pixel_pitch=3.9μm

  APSC_CANON    APS-C Canon (22.3×14.9mm)
                  focal_factor=1.6, sensor_width=22.3mm, pixel_pitch=3.2μm

  APSH          APS-H Canon (27.9×18.6mm)
                  focal_factor=1.3, sensor_width=27.9mm, pixel_pitch=5.7μm

  FF            Full Frame 35mm (36×24mm)
                  focal_factor=1.0, sensor_width=36.0mm, pixel_pitch=4.3μm

  MF44X33       Medium Format 44×33 (43.8×32.9mm) - GFX/645Z/X2D
                  focal_factor=0.79, sensor_width=43.8mm, pixel_pitch=3.76μm

  MF54X40       Medium Format 54×40 (53.4×40mm) - Hasselblad H6D-100c
                  focal_factor=0.64, sensor_width=53.4mm, pixel_pitch=4.6μm

======================================================================
Aliases:
  1-INCH, 1_INCH      → 1INCH
  APS-C, APS_C        → APSC
  APS-C_CANON         → APSC_CANON
  APS-H, APS_H        → APSH
  FULLFRAME           → FF
  MF44-33, MF44_33    → MF44X33
  MF54-40, MF54_40    → MF54X40
======================================================================

Usage Examples:
  --sensor-type MFT
  --sensor-type APS-C
  --sensor-type FF --pixel-pitch 5.9   # Override pixel pitch
  --sensor-type MF44X33                # Fujifilm GFX / Pentax 645Z
======================================================================
```

## Usage Examples

### Basic Usage (Recommended)

```bash
# 1-inch sensor camera
python detect_meteors_cli.py --auto-params --sensor-type 1INCH

# Micro Four Thirds camera
python detect_meteors_cli.py --auto-params --sensor-type MFT

# Sony/Nikon/Fuji APS-C camera
python detect_meteors_cli.py --auto-params --sensor-type APS-C

# Canon APS-C camera
python detect_meteors_cli.py --auto-params --sensor-type APS-C_CANON

# Full Frame camera
python detect_meteors_cli.py --auto-params --sensor-type FF

# Medium Format (Fujifilm GFX, Pentax 645Z, Hasselblad X2D)
python detect_meteors_cli.py --auto-params --sensor-type MF44X33

# Medium Format (Hasselblad H6D-100c)
python detect_meteors_cli.py --auto-params --sensor-type MF54X40
```

### With Override

```bash
# MFT with custom pixel pitch for high-resolution sensor
python detect_meteors_cli.py --auto-params \
  --sensor-type MFT \
  --pixel-pitch 3.3

# Full Frame with specific sensor width
python detect_meteors_cli.py --auto-params \
  --sensor-type FF \
  --sensor-width 35.9
```

### NPF Analysis Only

```bash
# Quick NPF analysis with sensor preset
python detect_meteors_cli.py --show-npf --sensor-type MFT

# Medium format NPF analysis
python detect_meteors_cli.py --show-npf --sensor-type MF44X33
```

### Legacy Usage (Still Supported)

```bash
# v1.4.x style - still works
python detect_meteors_cli.py --auto-params \
  --sensor-width 17.3 \
  --focal-factor MFT
```

## Comparison: Before and After

### Configuration Effort

| Task | v1.4.x | v1.5.x |
|------|--------|--------|
| Basic MFT setup | 3 parameters | 1 parameter |
| Full Frame setup | 3 parameters | 1 parameter |
| Medium Format setup | Not supported | 1 parameter |
| NPF analysis | Manual lookup | Preset available |
| Custom override | Same | Same (+ preset base) |

### Command Length

```bash
# v1.4.x (verbose)
python detect_meteors_cli.py --auto-params \
  --sensor-width 17.3 --focal-factor 2.0 --pixel-pitch 3.7

# v1.5.x (concise)
python detect_meteors_cli.py --auto-params --sensor-type MFT
```

## Technical Details

### New Functions

#### `get_sensor_preset(sensor_type: str) -> Optional[Dict]`
Retrieves sensor preset configuration by type name.

```python
>>> get_sensor_preset("MFT")
{
    'focal_factor': 2.0,
    'sensor_width': 17.3,
    'pixel_pitch': 3.7,
    'description': 'Micro Four Thirds (17.3×13mm)'
}

>>> get_sensor_preset("MF44X33")
{
    'focal_factor': 0.79,
    'sensor_width': 43.8,
    'pixel_pitch': 3.76,
    'description': 'Medium Format 44×33 (43.8×32.9mm) - GFX/645Z/X2D'
}
```

#### `apply_sensor_preset(args, verbose=False) -> Tuple`
Applies sensor preset with CLI argument priority.

```python
# v1.5.2+: Returns (focal_factor, sensor_width, focal_length, pixel_pitch, preset)
>>> apply_sensor_preset(args)
(2.0, 17.3, None, 3.7, {...})

# v1.5.1 and earlier: Returns (focal_factor, sensor_width, focal_length, pixel_pitch)
>>> apply_sensor_preset(args)
(2.0, 17.3, None, 3.7)
```

#### `list_sensor_types() -> None`
Displays available sensor presets in formatted output, ordered by sensor size.

### Backward Compatibility

- `CROP_FACTORS` dictionary preserved (auto-generated from `SENSOR_PRESETS`)
- `DEFAULT_SENSOR_WIDTHS` dictionary preserved (auto-generated from `SENSOR_PRESETS`)
- All v1.4.x command-line options work unchanged
- `--focal-factor` still accepts sensor type strings (e.g., `MFT`, `APS-C`)

## Breaking Changes

**None** - v1.5.x is fully backward compatible with v1.4.x.

All existing command-line options and behaviors are preserved. The new `--sensor-type` option is purely additive.

## Migration Guide

### From v1.4.x to v1.5.x

No changes required. Existing commands continue to work.

**Optional improvement**:
```bash
# Replace this (v1.4.x)
python detect_meteors_cli.py --auto-params --sensor-width 17.3 --focal-factor MFT

# With this (v1.5.x)
python detect_meteors_cli.py --auto-params --sensor-type MFT
```

## Shell Completion Updates

Both bash and zsh completion scripts have been updated:

### New Completions
- `--sensor-type` with preset suggestions (ordered by sensor size)
- `--list-sensor-types` flag
- Medium format sensor types (`MF44X33`, `MF54X40`)

### Updated Completions
- `--sensor-width` values updated to include medium format (43.8, 53.4)
- `--pixel-pitch` values updated to include medium format (3.76, 4.6)
- `--focal-factor` values updated to include medium format crop factors (0.64, 0.79)

## Future Enhancements

### Planned for v1.6

1. **Camera Model Auto-Detection**
   - Automatic sensor type detection from EXIF camera model
   - No manual `--sensor-type` needed for known cameras

2. **Custom Preset Definition**
   - User-defined sensor presets via configuration file
   - Support for less common sensor sizes

3. **Preset Validation**
   - Cross-check preset values against EXIF data
   - Warn if detected parameters differ significantly

## Version Information

- **Latest Version**: 1.5.2
- **Release Date**: 2024-12-01
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
| NPF check | `--show-npf --sensor-type TYPE` | `python detect_meteors_cli.py --show-npf --sensor-type APS-C` |

## Files Updated

| File | Changes |
|------|---------|
| `detect_meteors_cli.py` | Added `SENSOR_PRESETS`, new functions, `--sensor-type`, `--list-sensor-types`, medium format support |
| `detect_meteors_cli_completion.bash` | Added `--sensor-type`, `--list-sensor-types`, medium format completions |
| `_detect_meteors_cli` (zsh) | Added `--sensor-type`, `--list-sensor-types`, medium format completions |
| `COMMAND_OPTIONS.md` | Updated NPF Rule Options section with medium format |

---

**Status**: Production Ready  
**Compatibility**: Fully backward compatible with v1.4.x  
**Recommendation**: Use `--sensor-type` for simplified configuration

Happy meteor hunting! 🌠
