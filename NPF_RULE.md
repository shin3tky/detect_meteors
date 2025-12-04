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

If `--sensor-width` is not provided, the system uses a default pixel pitch of 4.0μm.

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
| 1-inch | 13.2 | 2.7 | 2.4-2.9 |
| Micro Four Thirds | 17.3 | 2.0 | 3.3-3.7 |
| APS-C (Canon) | 22.3 | 1.6 | 4.1-4.5 |
| APS-C (Sony, Nikon, Fuji) | 23.5 | 1.5 | 3.9-4.3 |
| APS-H (Canon) | 27.9 | 1.3 | 5.0-6.4 |
| Full Frame | 36.0 | 1.0 | 4.3-8.4 |
| Medium Format 44×33 | 43.8 | 0.79 | 3.3-5.3 |
| Medium Format 54×40 | 53.4 | 0.64 | 4.0-4.6 |

## Fisheye Lens Correction

Fisheye lenses have a unique characteristic: the effective focal length varies across the image due to the projection geometry. The center of the image has the nominal focal length, while the edges have a shorter effective focal length.

### Why Fisheye Needs Special Treatment

For **equisolid angle projection** (the most common fisheye type):
- **Formula**: r = 2f × sin(θ/2)
- **Center (θ=0°)**: Effective focal length = nominal focal length
- **Edge (θ=90°)**: Effective focal length ≈ 0.707× nominal (cos(45°))

This means:
- Stars at the image edge move across more pixels per unit time
- Star trails are approximately **1.414× longer** at corners compared to center
- NPF Rule must use the edge focal length for conservative recommendations

### Using Fisheye Correction

Add the `--fisheye` flag to enable equisolid angle projection compensation:

```bash
# MFT camera with 8mm fisheye (16mm equiv.)
python detect_meteors_cli.py --auto-params --sensor-type MFT --focal-length 16 --fisheye

# Full Frame with 8mm fisheye
python detect_meteors_cli.py --auto-params --sensor-type FF --focal-length 8 --fisheye

# Check NPF analysis with fisheye correction
python detect_meteors_cli.py --show-npf --sensor-type MFT --focal-length 16 --fisheye
```

### Example: Fisheye vs Standard NPF Analysis

**Without `--fisheye`** (8mm F1.8 on MFT, 16mm equiv.):
```
NPF Rule Analysis
============================================================
  Pixel pitch:      3.70μm (sensor: 17.3mm)
  NPF recommended:  10.9s
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
  Star trail est.:  ~1.9 pixels
============================================================
```

### When to Use `--fisheye`

Use the `--fisheye` flag when:
- Using a dedicated fisheye lens (circular or diagonal)
- Using an ultra-wide rectilinear lens with significant barrel distortion
- The lens has 180° or greater diagonal field of view

### Supported Projection Model

Currently, only **equisolid angle projection** is implemented. This covers most common fisheye lenses including:
- Olympus M.ZUIKO 8mm F1.8 Fisheye PRO
- Samyang/Rokinon 8mm F2.8 Fisheye
- Canon EF 8-15mm F4L Fisheye USM
- Nikon AF-S Fisheye NIKKOR 8-15mm f/3.5-4.5E ED

Future versions may add support for other projection models (equidistant, stereographic).
