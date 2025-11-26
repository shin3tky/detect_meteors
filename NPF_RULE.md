# Understanding NPF Rule

The **NPF Rule** (developed by Frédéric Michaud) is a scientific method to calculate the maximum exposure time before stars show trailing due to Earth's rotation. Unlike traditional "500 Rule" or "600 Rule", it accounts for modern camera sensor pixel pitch:

```
NPF Exposure (seconds) = (35 × F-number + 30 × Pixel Pitch) / Focal Length
```

Where:
- **F-number**: Aperture (e.g., f/2.8 → 2.8)
- **Pixel Pitch**: Physical pixel size in micrometers (μm)
- **Focal Length**: 35mm equivalent in millimeters

The software uses this to:
- Assess whether your exposure settings are optimal
- Estimate star trail length during actual exposure
- Adjust detection parameters based on NPF compliance

## Sensor Width Specification

For best accuracy, specify your camera's sensor width:

```bash
# Micro Four Thirds (17.3mm)
python detect_meteors_cli.py --auto-params --sensor-width 17.3

# APS-C (23.5mm)
python detect_meteors_cli.py --auto-params --sensor-width 23.5

# Full Frame (36.0mm)
python detect_meteors_cli.py --auto-params --sensor-width 36.0
```

Or use sensor type shortcuts:

```bash
# Predefined sensor types
python detect_meteors_cli.py --auto-params --focal-factor MFT    # Micro Four Thirds
python detect_meteors_cli.py --auto-params --focal-factor APS-C  # APS-C
python detect_meteors_cli.py --auto-params --focal-factor FF     # Full Frame

# Or specify numeric crop factor
python detect_meteors_cli.py --auto-params --focal-factor 2.0    # 2.0× crop
```

If neither `--sensor-width` nor `--focal-factor` is provided, the system uses a default pixel pitch of 4.0μm.

## Focal Length Handling

The software extracts focal length from EXIF metadata automatically. If not available or incorrect:

```bash
# Specify 35mm equivalent focal length directly
python detect_meteors_cli.py --auto-params --sensor-width 17.3 --focal-length 24

# Or use crop factor (converts actual focal length to 35mm equivalent)
python detect_meteors_cli.py --auto-params --sensor-width 17.3 --focal-factor 2.0
```

## NPF Compliance Analysis

Use `--show-npf` to see detailed NPF analysis without processing:

```bash
python detect_meteors_cli.py --show-npf --sensor-width 17.3
```

Example output:
```
NPF Rule Analysis
============================================================
  Pixel pitch:      3.30μm (sensor: 17.3mm)
  NPF recommended:  8.2s
  Actual exposure:  5.0s ✓ OK
  Star trail est.:  ~1.5 pixels
  Impact:           LOW
============================================================
```

This helps you understand:
- Whether your exposure time is optimal
- Expected star trail length in your images
- Impact level on meteor detection quality

## Common Sensor Widths Reference

| Camera System | Sensor Width (mm) | Crop Factor | Typical Pixel Pitch (μm) |
|--------------|-------------------|-------------|-------------------------|
| Micro Four Thirds | 17.3 | 2.0 | 3.3-3.7 |
| APS-C (Sony, Nikon, Fuji) | 23.5 | 1.5 | 3.9-4.3 |
| APS-C (Canon) | 22.3 | 1.6 | 4.1-4.5 |
| APS-H (Canon) | 27.9 | 1.3 | 5.0-6.4 |
| Full Frame | 36.0 | 1.0 | 4.3-8.4 |
| 1-inch | 13.2 | 2.7 | 2.4-2.9 |
