# Version 1.5 Release Notes

## Version 1.5.0 (2025-11-29 🍖)

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

**Available Presets**:

| Sensor Type | Crop Factor | Sensor Width | Pixel Pitch | Description |
|-------------|-------------|--------------|-------------|-------------|
| `MFT` | 2.0 | 17.3mm | 3.7μm | Micro Four Thirds |
| `APS-C` | 1.5 | 23.5mm | 3.9μm | APS-C (Sony/Nikon/Fuji) |
| `APS-C_CANON` | 1.6 | 22.3mm | 3.2μm | APS-C (Canon) |
| `APS-H` | 1.3 | 27.9mm | 5.7μm | APS-H (Canon) |
| `FF` | 1.0 | 36.0mm | 4.3μm | Full Frame 35mm |
| `1INCH` | 2.7 | 13.2mm | 2.4μm | 1-inch sensor |

**Aliases**: `FULLFRAME` → `FF`, `APS_C` → `APS-C`, etc.

**Implementation**:
```python
SENSOR_PRESETS = {
    "MFT": {
        "focal_factor": 2.0,
        "sensor_width": 17.3,
        "pixel_pitch": 3.7,
        "description": "Micro Four Thirds (17.3×13mm)",
    },
    "APSC": {
        "focal_factor": 1.5,
        "sensor_width": 23.5,
        "pixel_pitch": 3.9,
        "description": "APS-C Sony/Nikon/Fuji (23.5×15.6mm)",
    },
    # ... additional presets
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

  1INCH         1-inch sensor (13.2×8.8mm)
                  focal_factor=2.7, sensor_width=13.2mm, pixel_pitch=2.4μm

======================================================================
Aliases:
  APS-C, APS_C        → APSC
  APS-C_CANON         → APSC_CANON
  APS-H, APS_H        → APSH
  FULLFRAME           → FF
  1-INCH, 1_INCH      → 1INCH
======================================================================

Usage Examples:
  --sensor-type MFT
  --sensor-type APS-C
  --sensor-type FF --pixel-pitch 5.9   # Override pixel pitch
======================================================================
```

## Usage Examples

### Basic Usage (Recommended)

```bash
# Micro Four Thirds camera
python detect_meteors_cli.py --auto-params --sensor-type MFT

# Sony/Nikon/Fuji APS-C camera
python detect_meteors_cli.py --auto-params --sensor-type APS-C

# Canon APS-C camera
python detect_meteors_cli.py --auto-params --sensor-type APS-C_CANON

# Full Frame camera
python detect_meteors_cli.py --auto-params --sensor-type FF
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

| Task | v1.4.x | v1.5.0 |
|------|--------|--------|
| Basic MFT setup | 3 parameters | 1 parameter |
| Full Frame setup | 3 parameters | 1 parameter |
| NPF analysis | Manual lookup | Preset available |
| Custom override | Same | Same (+ preset base) |

### Command Length

```bash
# v1.4.x (verbose)
python detect_meteors_cli.py --auto-params \
  --sensor-width 17.3 --focal-factor 2.0 --pixel-pitch 3.7

# v1.5.0 (concise)
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
```

#### `apply_sensor_preset(args, verbose=False) -> Tuple`
Applies sensor preset with CLI argument priority.

```python
# Returns (focal_factor, sensor_width, focal_length, pixel_pitch)
>>> apply_sensor_preset(args)
(2.0, 17.3, None, 3.7)
```

#### `list_sensor_types() -> None`
Displays available sensor presets in formatted output.

### Backward Compatibility

- `CROP_FACTORS` dictionary preserved (auto-generated from `SENSOR_PRESETS`)
- `DEFAULT_SENSOR_WIDTHS` dictionary preserved (auto-generated from `SENSOR_PRESETS`)
- All v1.4.x command-line options work unchanged
- `--focal-factor` still accepts sensor type strings (e.g., `MFT`, `APS-C`)

## Breaking Changes

**None** - v1.5.0 is fully backward compatible with v1.4.x.

All existing command-line options and behaviors are preserved. The new `--sensor-type` option is purely additive.

## Migration Guide

### From v1.4.x to v1.5.0

No changes required. Existing commands continue to work.

**Optional improvement**:
```bash
# Replace this (v1.4.x)
python detect_meteors_cli.py --auto-params --sensor-width 17.3 --focal-factor MFT

# With this (v1.5.0)
python detect_meteors_cli.py --auto-params --sensor-type MFT
```

## Shell Completion Updates

Both bash and zsh completion scripts have been updated:

### New Completions
- `--sensor-type` with preset suggestions (`MFT`, `APS-C`, `APS-C_CANON`, `APS-H`, `FF`, `FULLFRAME`, `1INCH`)
- `--list-sensor-types` flag

### Updated Completions
- `--sensor-width` values updated to match presets
- `--pixel-pitch` values updated to match presets

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

- **Version**: 1.5.0
- **Release Date**: 2025-11-27
- **Major Changes**:
  - Unified `SENSOR_PRESETS` configuration
  - New `--sensor-type` option for simplified setup
  - New `--list-sensor-types` option
  - Parameter override priority system
  - Updated shell completion scripts

## Quick Reference

| Feature | Command | Example |
|---------|---------|---------|
| Use sensor preset | `--sensor-type TYPE` | `--sensor-type MFT` |
| List presets | `--list-sensor-types` | `python detect_meteors_cli.py --list-sensor-types` |
| Preset + override | `--sensor-type TYPE --PARAM VALUE` | `--sensor-type FF --pixel-pitch 5.9` |
| Full auto (MFT) | `--auto-params --sensor-type MFT` | `python detect_meteors_cli.py --auto-params --sensor-type MFT` |
| NPF check | `--show-npf --sensor-type TYPE` | `python detect_meteors_cli.py --show-npf --sensor-type APS-C` |

## Files Updated

| File | Changes |
|------|---------|
| `detect_meteors_cli.py` | Added `SENSOR_PRESETS`, new functions, `--sensor-type`, `--list-sensor-types` |
| `detect_meteors_cli_completion.bash` | Added `--sensor-type`, `--list-sensor-types` completions |
| `_detect_meteors_cli` (zsh) | Added `--sensor-type`, `--list-sensor-types` completions |
| `COMMAND_OPTIONS.md` | Updated NPF Rule Options section |

---

**Status**: Production Ready  
**Compatibility**: Fully backward compatible with v1.4.x  
**Recommendation**: Use `--sensor-type` for simplified configuration

Happy meteor hunting! 🌠
