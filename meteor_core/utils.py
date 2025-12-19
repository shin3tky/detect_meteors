#!/usr/bin/env python
#
# Detect Meteors CLI - Utilities
# © 2025 Shinichi Morita (shin3tky)
#

"""
Utility functions for meteor detection.
Includes sensor presets, NPF calculations, fisheye corrections, memory estimation, etc.
"""

import math
import json
import hashlib
import os
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

from .schema import (
    SENSOR_PRESETS,
    CROP_FACTORS,
    FISHEYE_PROJECTION_MODELS,
    DEFAULT_PIXEL_PITCH_UM,
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_LINE_SCORE,
    AUTO_BATCH_MEMORY_FRACTION,
)
from .i18n import DEFAULT_LOCALE, get_message


def _resolve_locale(locale: Optional[str]) -> str:
    if locale:
        return locale
    return os.environ.get("DETECT_METEORS_LOCALE", DEFAULT_LOCALE)


# ==========================================
# Sensor Preset Functions
# ==========================================


def parse_focal_factor(focal_factor_str: str) -> Optional[float]:
    """
    Parse --focal-factor argument and return the coefficient.

    Args:
        focal_factor_str: Numeric string (e.g., "2.0", "1.5")
                         or sensor type (e.g., "MFT", "APS-C")

    Returns:
        Focal length conversion factor, or None if parsing fails

    Examples:
        >>> parse_focal_factor("2.0")
        2.0
        >>> parse_focal_factor("MFT")
        2.0
        >>> parse_focal_factor("APS-C")
        1.5
    """
    if not focal_factor_str:
        return None

    # Try to interpret as numeric value
    try:
        factor = float(focal_factor_str)
        if 0.5 <= factor <= 10.0:  # Check valid range
            return factor
        else:
            return None
    except ValueError:
        pass

    # Interpret as sensor type string
    # Normalize hyphens/spaces to underscores, convert to uppercase
    key = focal_factor_str.upper().replace("-", "_").replace(" ", "_")
    return CROP_FACTORS.get(key)


def get_sensor_preset(sensor_type: str) -> Optional[Dict[str, Any]]:
    """
    Get sensor preset configuration by sensor type name.

    Args:
        sensor_type: Sensor type string (e.g., "MFT", "APS-C", "FF")

    Returns:
        Sensor preset dictionary or None if not found

    Examples:
        >>> get_sensor_preset("MFT")
        {'focal_factor': 2.0, 'sensor_width': 17.3, 'pixel_pitch': 3.7, ...}
        >>> get_sensor_preset("APS-C")
        {'focal_factor': 1.5, 'sensor_width': 23.5, 'pixel_pitch': 3.9, ...}
    """
    if not sensor_type:
        return None

    # Normalize hyphens/spaces to underscores, convert to uppercase
    key = sensor_type.upper().replace("-", "_").replace(" ", "_")
    return SENSOR_PRESETS.get(key)


def apply_sensor_preset(
    args,
    verbose: bool = False,
    locale: Optional[str] = None,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[Dict[str, Any]],
]:
    """
    Apply sensor preset values, with individual arguments taking priority.

    Priority order (highest to lowest):
    1. Individual CLI arguments (--focal-factor, --sensor-width, --focal-length, --pixel-pitch)
    2. --sensor-type preset values
    3. None (not specified)

    Args:
        args: Parsed argparse namespace with:
            - sensor_type: str or None
            - focal_factor: str or None
            - sensor_width: float or None
            - focal_length: float or None
            - pixel_pitch: float or None
        verbose: If True, print which values are being used

    Returns:
        Tuple of (focal_factor, sensor_width, focal_length, pixel_pitch, preset)
        Each value is either from CLI argument, preset, or None.
        preset is the sensor preset dictionary or None.

    Examples:
        # --sensor-type MFT (no overrides)
        >>> apply_sensor_preset(args)  # args.sensor_type="MFT"
        (2.0, 17.3, None, 3.7, {...})

        # --sensor-type MFT --sensor-width 18.0 (override sensor_width)
        >>> apply_sensor_preset(args)  # args.sensor_type="MFT", args.sensor_width=18.0
        (2.0, 18.0, None, 3.7, {...})
    """
    # Initialize with CLI argument values (may be None)
    focal_factor_value = None
    sensor_width_value = args.sensor_width
    focal_length_value = args.focal_length
    pixel_pitch_value = args.pixel_pitch

    # Parse focal_factor from CLI argument
    if args.focal_factor:
        focal_factor_value = parse_focal_factor(args.focal_factor)

    # Get preset if --sensor-type is specified
    preset = None
    if hasattr(args, "sensor_type") and args.sensor_type:
        preset = get_sensor_preset(args.sensor_type)
        if preset is None:
            # Invalid sensor type - will be handled by caller
            return (
                focal_factor_value,
                sensor_width_value,
                focal_length_value,
                pixel_pitch_value,
                None,
            )

    locale = _resolve_locale(locale)

    # Apply preset values where CLI arguments are not specified
    if preset:
        preset_applied = []

        # focal_factor: CLI --focal-factor takes priority
        if focal_factor_value is None:
            focal_factor_value = preset.get("focal_factor")
            if focal_factor_value is not None:
                preset_applied.append(f"focal_factor={focal_factor_value}")

        # sensor_width: CLI --sensor-width takes priority
        if sensor_width_value is None:
            sensor_width_value = preset.get("sensor_width")
            if sensor_width_value is not None:
                preset_applied.append(f"sensor_width={sensor_width_value}mm")

        # pixel_pitch: CLI --pixel-pitch takes priority
        if pixel_pitch_value is None:
            pixel_pitch_value = preset.get("pixel_pitch")
            if pixel_pitch_value is not None:
                preset_applied.append(f"pixel_pitch={pixel_pitch_value}μm")

        # focal_length is not in preset (it depends on actual lens used)

        if verbose and preset_applied:
            print(
                get_message(
                    "ui.utils.sensor_preset.applied",
                    locale=locale,
                    sensor_type=args.sensor_type,
                    values=", ".join(preset_applied),
                )
            )

    return (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    )


def validate_sensor_overrides(
    args,
    preset: Optional[Dict[str, Any]],
    sensor_width_value: Optional[float],
    pixel_pitch_value: Optional[float],
    locale: Optional[str] = None,
) -> None:
    """
    Validate that overridden --sensor-width and --pixel-pitch values
    are not significantly different from --sensor-type preset values.

    Prints warnings if discrepancies are detected, but does not stop processing.

    Thresholds for warnings:
    - sensor_width: ±30% deviation from preset
    - pixel_pitch: ±50% deviation from preset

    Args:
        args: Parsed argparse namespace
        preset: Sensor preset dictionary (from get_sensor_preset)
        sensor_width_value: Final sensor width value (mm)
        pixel_pitch_value: Final pixel pitch value (μm)
    """
    if not preset or not args.sensor_type:
        return

    locale = _resolve_locale(locale)
    warnings = []

    # Check sensor_width deviation
    if args.sensor_width is not None and sensor_width_value is not None:
        preset_width = preset.get("sensor_width")
        if preset_width:
            deviation_pct = abs(sensor_width_value - preset_width) / preset_width * 100
            if deviation_pct > 30.0:  # 30% threshold
                warnings.append(
                    get_message(
                        "ui.utils.sensor_override.warning.sensor_width",
                        locale=locale,
                        sensor_width=sensor_width_value,
                        deviation=deviation_pct,
                        sensor_type=args.sensor_type,
                        preset_width=preset_width,
                    )
                )

    # Check pixel_pitch deviation
    if args.pixel_pitch is not None and pixel_pitch_value is not None:
        preset_pitch = preset.get("pixel_pitch")
        if preset_pitch:
            deviation_pct = abs(pixel_pitch_value - preset_pitch) / preset_pitch * 100
            if deviation_pct > 50.0:  # 50% threshold
                warnings.append(
                    get_message(
                        "ui.utils.sensor_override.warning.pixel_pitch",
                        locale=locale,
                        pixel_pitch=pixel_pitch_value,
                        deviation=deviation_pct,
                        sensor_type=args.sensor_type,
                        preset_pitch=preset_pitch,
                    )
                )

    # Print warnings if any
    if warnings:
        print(f"\n{'='*70}")
        for warning in warnings:
            print(warning)
        print(f"{'='*70}\n")


def list_sensor_types(locale: Optional[str] = None) -> None:
    """
    Display available sensor type presets and their configurations.
    Ordered by sensor size (smallest to largest).
    """
    locale = _resolve_locale(locale)
    print(f"\n{'='*70}")
    print(get_message("ui.utils.sensor_types.header", locale=locale))
    print(f"{'='*70}\n")

    # Group by primary types (exclude aliases), ordered by sensor size
    primary_types = [
        "1INCH",
        "MFT",
        "APSC",
        "APSC_CANON",
        "APSH",
        "FF",
        "MF44X33",
        "MF54X40",
    ]

    for sensor_type in primary_types:
        preset = SENSOR_PRESETS.get(sensor_type)
        if preset:
            print(
                get_message(
                    "ui.utils.sensor_types.entry",
                    locale=locale,
                    sensor_type=f"{sensor_type:12}",
                    description=preset["description"],
                )
            )
            print(
                get_message(
                    "ui.utils.sensor_types.details",
                    locale=locale,
                    focal_factor=preset["focal_factor"],
                    sensor_width=preset["sensor_width"],
                    pixel_pitch=preset["pixel_pitch"],
                )
            )
            print()

    print(f"{'='*70}")
    print(get_message("ui.utils.sensor_types.aliases.header", locale=locale))
    print(get_message("ui.utils.sensor_types.aliases.1inch", locale=locale))
    print(get_message("ui.utils.sensor_types.aliases.apsc", locale=locale))
    print(get_message("ui.utils.sensor_types.aliases.apsc_canon", locale=locale))
    print(get_message("ui.utils.sensor_types.aliases.apsh", locale=locale))
    print(get_message("ui.utils.sensor_types.aliases.ff", locale=locale))
    print(get_message("ui.utils.sensor_types.aliases.mf44x33", locale=locale))
    print(get_message("ui.utils.sensor_types.aliases.mf54x40", locale=locale))
    print(f"{'='*70}")
    print(get_message("ui.utils.sensor_types.examples.header", locale=locale))
    print(get_message("ui.utils.sensor_types.examples.mft", locale=locale))
    print(get_message("ui.utils.sensor_types.examples.apsc", locale=locale))
    print(get_message("ui.utils.sensor_types.examples.ff", locale=locale))
    print(get_message("ui.utils.sensor_types.examples.mf44x33", locale=locale))
    print(f"{'='*70}\n")


# ==========================================
# Fisheye Correction Functions
# ==========================================


def calculate_fisheye_effective_focal_length(
    nominal_focal_length_mm: float,
    normalized_radius: float,
    projection_model: str = "EQUISOLID",
) -> float:
    """
    Calculate effective focal length at a given position in fisheye image.

    For equisolid angle projection: r = 2f × sin(θ/2)
    The effective focal length decreases toward the edges because the same
    angular distance covers fewer pixels (compression effect).

    Args:
        nominal_focal_length_mm: Nominal (center) focal length in mm
        normalized_radius: Distance from image center as fraction of image half-diagonal (0.0-1.0)
                          0.0 = center, 1.0 = corner
        projection_model: Fisheye projection model (currently only "EQUISOLID" supported)

    Returns:
        Effective focal length at the given position (mm)

    Note:
        For equisolid projection with typical full-frame circular fisheye (180° diagonal):
        - Center (r=0): effective_f = nominal_f
        - Edge (r=1, θ=90°): effective_f ≈ nominal_f × 0.707 (cos(45°))

        The derivative dr/dθ = f × cos(θ/2) shows the compression effect.
        At edges, angular changes map to smaller pixel distances.
    """
    if normalized_radius < 0 or normalized_radius > 1:
        normalized_radius = max(0.0, min(1.0, normalized_radius))

    if projection_model.upper() != "EQUISOLID":
        # Future: support other projection models
        # For now, only EQUISOLID is implemented
        return nominal_focal_length_mm

    # For equisolid: r = 2f × sin(θ/2)
    # Typical fisheye covers 180° diagonal, so max θ = 90° at corners
    # At normalized_radius = 1.0, θ = 90°

    # Calculate the angle θ at the given radius
    # Assuming 180° diagonal coverage (typical for circular fisheye)
    max_theta_rad = math.pi / 2  # 90° at the edge
    theta_rad = normalized_radius * max_theta_rad

    # The effective focal length is reduced by the compression factor
    # dr/dθ = f × cos(θ/2), so effective scale = cos(θ/2)
    # This means the same angular motion covers fewer pixels at edges
    compression_factor = math.cos(theta_rad / 2)

    # Effective focal length (shorter at edges = wider effective FOV)
    effective_focal_length = nominal_focal_length_mm * compression_factor

    return effective_focal_length


def calculate_fisheye_edge_focal_length(
    nominal_focal_length_mm: float,
    projection_model: str = "EQUISOLID",
) -> float:
    """
    Calculate effective focal length at the edge/corner of fisheye image.

    This is used for NPF Rule calculation to ensure the recommended
    exposure time is based on the worst-case (edge) condition.

    Args:
        nominal_focal_length_mm: Nominal (center) focal length in mm
        projection_model: Fisheye projection model

    Returns:
        Effective focal length at image edge (mm)
    """
    return calculate_fisheye_effective_focal_length(
        nominal_focal_length_mm,
        normalized_radius=1.0,
        projection_model=projection_model,
    )


def calculate_fisheye_trail_length_ratio(
    normalized_radius: float,
    projection_model: str = "EQUISOLID",
) -> float:
    """
    Calculate the ratio of star trail length at given position vs center.

    For fisheye images, star trails are longer at the edges due to the
    compression effect of the projection.

    Args:
        normalized_radius: Distance from image center (0.0-1.0)
        projection_model: Fisheye projection model

    Returns:
        Ratio of trail length at position vs center (>= 1.0)
        1.0 at center, increasing toward edges
    """
    if normalized_radius <= 0:
        return 1.0

    if projection_model.upper() != "EQUISOLID":
        return 1.0

    # Trail length is inversely proportional to effective focal length
    # Longer trails where effective focal length is shorter
    max_theta_rad = math.pi / 2
    theta_rad = normalized_radius * max_theta_rad
    compression_factor = math.cos(theta_rad / 2)

    # Ratio = center_f / edge_f = 1 / compression_factor
    if compression_factor > 0:
        return 1.0 / compression_factor
    else:
        return 1.0


def get_fisheye_max_trail_ratio(projection_model: str = "EQUISOLID") -> float:
    """
    Get the maximum trail length ratio (at image edge vs center).

    Args:
        projection_model: Fisheye projection model

    Returns:
        Maximum trail length ratio at edge
    """
    return calculate_fisheye_trail_length_ratio(1.0, projection_model)


def display_fisheye_info(
    nominal_focal_length_mm: float,
    projection_model: str = "EQUISOLID",
    locale: Optional[str] = None,
) -> None:
    """
    Display fisheye correction information.

    Args:
        nominal_focal_length_mm: Nominal focal length in mm
        projection_model: Fisheye projection model
    """
    locale = _resolve_locale(locale)
    edge_focal = calculate_fisheye_edge_focal_length(
        nominal_focal_length_mm, projection_model
    )
    max_ratio = get_fisheye_max_trail_ratio(projection_model)

    model_info = FISHEYE_PROJECTION_MODELS.get(
        projection_model.upper(), {"name": projection_model, "description": ""}
    )

    print(f"\n{'='*60}")
    print(get_message("ui.utils.fisheye.header", locale=locale))
    print(f"{'='*60}")
    print(
        get_message(
            "ui.utils.fisheye.projection_model",
            locale=locale,
            model=model_info["name"],
        )
    )
    print(
        get_message(
            "ui.utils.fisheye.nominal_focal",
            locale=locale,
            focal_length=nominal_focal_length_mm,
        )
    )
    print(
        get_message(
            "ui.utils.fisheye.effective_focal",
            locale=locale,
            focal_length=edge_focal,
        )
    )
    print(
        get_message(
            "ui.utils.fisheye.trail_ratio",
            locale=locale,
            ratio=max_ratio,
        )
    )
    print(get_message("ui.utils.fisheye.npf_basis", locale=locale))
    print(f"{'='*60}")


# ==========================================
# NPF Rule Functions
# ==========================================


def calculate_pixel_pitch(sensor_width_mm: float, image_width_px: int) -> float:
    """
    Calculate pixel pitch in micrometers (μm).

    Args:
        sensor_width_mm: Sensor width in millimeters
        image_width_px: Image width in pixels

    Returns:
        Pixel pitch in micrometers (μm)
    """
    return (sensor_width_mm * 1000.0) / image_width_px


def calculate_npf_rule(
    focal_length_mm: float, aperture: float, pixel_pitch_um: float
) -> float:
    """
    Calculate recommended maximum exposure time based on NPF Rule.

    NPF Rule formula:
        Exposure Time (seconds) = (35 × Aperture + 30 × Pixel Pitch) / Focal Length

    Args:
        focal_length_mm: Focal length in 35mm equivalent (mm)
        aperture: F-number (aperture value)
        pixel_pitch_um: Pixel pitch in micrometers (μm)

    Returns:
        Recommended maximum exposure time in seconds

    Examples:
        >>> calculate_npf_rule(25, 1.8, 3.7)  # MFT, 25mm f/1.8
        8.24
        >>> calculate_npf_rule(50, 2.8, 5.9)  # FF, 50mm f/2.8
        5.84
    """
    if focal_length_mm <= 0 or aperture <= 0 or pixel_pitch_um <= 0:
        return 0.0

    numerator = (35 * aperture) + (30 * pixel_pitch_um)
    npf_time = numerator / focal_length_mm

    return npf_time


def estimate_star_trail_length(
    focal_length_mm: float,
    exposure_time_sec: float,
    image_width_px: int,
    declination_deg: float = 0.0,
) -> float:
    """
    Estimate star trail length in pixels during exposure.

    Due to Earth's rotation, stars move approximately 15 degrees per hour.
    This function converts this movement to pixels on the image.

    Args:
        focal_length_mm: Focal length in 35mm equivalent (mm)
        exposure_time_sec: Exposure time in seconds
        image_width_px: Image width in pixels
        declination_deg: Declination in degrees (default: 0° = celestial equator)

    Returns:
        Star trail length in pixels

    Note:
        - Earth's rotation rate: 15°/hour = 0.00417°/second
        - Field of view: FOV = 2 × arctan(36mm / (2 × focal_length))
        - Declination correction: multiply by cos(declination)
    """
    if focal_length_mm <= 0 or exposure_time_sec <= 0 or image_width_px <= 0:
        return 0.0

    # Earth's rotation rate (degrees per second)
    EARTH_ROTATION_DEG_PER_SEC = 15.0 / 3600.0  # 15°/hour

    # Angular movement of stars during exposure time (degrees)
    star_movement_deg = EARTH_ROTATION_DEG_PER_SEC * exposure_time_sec

    # Declination correction (apparent motion slows near celestial poles)
    declination_factor = math.cos(math.radians(declination_deg))
    star_movement_deg *= declination_factor

    # Field of view in 35mm equivalent (degrees)
    # FOV = 2 × arctan(sensor width / (2 × focal length))
    sensor_width_35mm = 36.0
    fov_rad = 2 * math.atan(sensor_width_35mm / (2 * focal_length_mm))
    fov_deg = math.degrees(fov_rad)

    # Pixels per degree
    pixels_per_degree = image_width_px / fov_deg

    # Star trail length in pixels
    trail_length_px = star_movement_deg * pixels_per_degree

    return trail_length_px


def evaluate_npf_compliance(
    exposure_time_sec: float, npf_recommended_sec: float
) -> Tuple[str, float]:
    """
    Evaluate NPF Rule compliance.

    Args:
        exposure_time_sec: Actual exposure time in seconds
        npf_recommended_sec: NPF recommended exposure time in seconds

    Returns:
        (compliance_level, overshoot_factor)
        compliance_level: "OK", "WARNING", or "CRITICAL"
        overshoot_factor: actual_exposure / npf_recommended
    """
    if npf_recommended_sec <= 0:
        return "UNKNOWN", 0.0

    overshoot_factor = exposure_time_sec / npf_recommended_sec

    if overshoot_factor <= 1.0:
        return "OK", overshoot_factor
    elif overshoot_factor <= 1.5:
        return "WARNING", overshoot_factor
    else:
        return "CRITICAL", overshoot_factor


def calculate_npf_metrics(
    exif_data: Dict[str, Any],
    sensor_width_mm: Optional[float] = None,
    pixel_pitch_um: Optional[float] = None,
    fisheye: bool = False,
    fisheye_model: str = "EQUISOLID",
) -> Dict[str, Any]:
    """
    Calculate NPF Rule-related metrics from EXIF information.

    Args:
        exif_data: Return value from extract_exif_metadata()
        sensor_width_mm: Sensor width in millimeters (None if cannot estimate)
        pixel_pitch_um: Pixel pitch in micrometers (μm) (calculated or default if None)
        fisheye: Whether to apply fisheye correction
        fisheye_model: Fisheye projection model (default: "EQUISOLID")

    Returns:
        Dictionary of NPF metrics:
        {
            'pixel_pitch_um': float,           # Pixel pitch (μm)
            'npf_recommended_sec': float,      # NPF recommended exposure time (seconds)
            'star_trail_px': float,            # Star trail length in pixels (at edge for fisheye)
            'compliance_level': str,           # "OK", "WARNING", "CRITICAL"
            'overshoot_factor': float,         # Overshoot factor
            'sensor_width_mm': float,          # Sensor width used
            'has_complete_data': bool,         # Whether complete data is available
            'fisheye': bool,                   # Whether fisheye correction is applied
            'fisheye_model': str,              # Fisheye projection model used
            'effective_focal_length': float,   # Effective focal length (edge for fisheye)
            'trail_length_ratio': float,       # Trail length ratio (edge vs center for fisheye)
        }
    """
    result = {
        "pixel_pitch_um": None,
        "npf_recommended_sec": None,
        "star_trail_px": None,
        "compliance_level": "UNKNOWN",
        "overshoot_factor": 0.0,
        "sensor_width_mm": sensor_width_mm,
        "has_complete_data": False,
        "fisheye": fisheye,
        "fisheye_model": fisheye_model if fisheye else None,
        "effective_focal_length": None,
        "trail_length_ratio": 1.0,
    }

    focal_length = exif_data.get("focal_length_35mm")
    aperture = exif_data.get("f_number")
    exposure_time = exif_data.get("exposure_time")
    image_width = exif_data.get("image_width")

    # Check for required data
    if not focal_length or not aperture or not exposure_time:
        return result

    # Apply fisheye correction: use effective focal length at edge
    effective_focal_length = focal_length
    trail_length_ratio = 1.0

    if fisheye:
        effective_focal_length = calculate_fisheye_edge_focal_length(
            focal_length, fisheye_model
        )
        trail_length_ratio = get_fisheye_max_trail_ratio(fisheye_model)

    result["effective_focal_length"] = effective_focal_length
    result["trail_length_ratio"] = trail_length_ratio

    # Determine pixel pitch
    if pixel_pitch_um is not None:
        # Use explicitly specified value
        result["pixel_pitch_um"] = pixel_pitch_um
    elif sensor_width_mm and image_width:
        # Calculate from sensor width and image width
        result["pixel_pitch_um"] = calculate_pixel_pitch(sensor_width_mm, image_width)
        result["sensor_width_mm"] = sensor_width_mm
    else:
        # Use default value
        result["pixel_pitch_um"] = DEFAULT_PIXEL_PITCH_UM

    # Calculate NPF recommended exposure time
    # For fisheye, use effective focal length at edge (worst case)
    result["npf_recommended_sec"] = calculate_npf_rule(
        effective_focal_length, aperture, result["pixel_pitch_um"]
    )

    # Estimate star trail length
    # For fisheye, estimate at edge (longest trail)
    if image_width:
        center_trail = estimate_star_trail_length(
            focal_length, exposure_time, image_width
        )
        # Apply fisheye trail length ratio
        result["star_trail_px"] = center_trail * trail_length_ratio

    # Evaluate NPF compliance
    result["compliance_level"], result["overshoot_factor"] = evaluate_npf_compliance(
        exposure_time, result["npf_recommended_sec"]
    )

    # Check if complete data is available
    result["has_complete_data"] = all(
        [focal_length, aperture, exposure_time, result["pixel_pitch_um"], image_width]
    )

    return result


# ==========================================
# NPF-based Auto-params Functions
# ==========================================


def optimize_diff_threshold_npf(
    exif_data: Dict[str, Any], npf_metrics: Dict[str, Any], base_threshold: float = 5.0
) -> int:
    """
    Optimize diff_threshold based on ISO sensitivity and NPF overshoot.

    Args:
        exif_data: EXIF information
        npf_metrics: NPF metrics
        base_threshold: Base threshold value

    Returns:
        Optimized diff_threshold

    Algorithm:
        1. ISO sensitivity adjustment: +2 per ISO doubling
        2. Exposure time adjustment: noise increases with long exposures
        3. NPF overshoot adjustment: relax when star trails are long
    """
    iso_value = exif_data.get("iso")
    exposure_time = exif_data.get("exposure_time")
    overshoot_factor = npf_metrics.get("overshoot_factor", 1.0)

    # Base value
    threshold = base_threshold

    # Adjust for ISO sensitivity
    if iso_value:
        # Based on ISO 800, +2 per doubling
        iso_factor = math.log2(max(iso_value, 100) / 800.0) * 2.0
        threshold += iso_factor

    # Adjust for exposure time (gradually increase for ≥15s)
    if exposure_time and exposure_time > 15.0:
        exp_factor = math.log2(exposure_time / 15.0) * 1.0
        threshold += exp_factor

    # Adjust for NPF overshoot (longer star trails harder to distinguish from noise)
    if overshoot_factor > 1.5:
        # Increase threshold for large overshoot to prevent false positives
        npf_factor = (overshoot_factor - 1.5) * 1.5
        threshold += npf_factor

    # Final range limit
    threshold = max(3, min(25, threshold))

    return int(round(threshold))


def optimize_min_area_npf(
    exif_data: Dict[str, Any], npf_metrics: Dict[str, Any]
) -> int:
    """
    Optimize min_area based on star trail length and focal length.

    Args:
        exif_data: EXIF information
        npf_metrics: NPF metrics

    Returns:
        Optimized min_area

    Algorithm:
        1. Set base on star trail length
        2. Adjust for focal length (wide→small, telephoto→large)
        3. Adjust for NPF compliance
    """
    focal_length = exif_data.get("focal_length_35mm")
    star_trail_px = npf_metrics.get("star_trail_px")
    overshoot_factor = npf_metrics.get("overshoot_factor", 1.0)

    # Default value
    min_area = 10

    if star_trail_px:
        # Base on 40-60% of star trail
        # Meteors move faster than stars, leaving longer trails even in short time
        base_area = star_trail_px * 0.5
        min_area = max(3, int(base_area))

    # Adjust for focal length
    if focal_length:
        if focal_length < 20:
            # Ultra-wide: detect smaller objects
            min_area = max(3, int(min_area * 0.7))
        elif focal_length < 35:
            # Wide: slightly smaller than standard
            min_area = max(4, int(min_area * 0.85))
        elif focal_length > 70:
            # Telephoto: only larger trails
            min_area = int(min_area * 1.3)

    # Adjust for NPF overshoot
    if overshoot_factor > 2.0:
        # When heavily exceeded, star trails are long so raise threshold
        min_area = int(min_area * 1.2)
    elif overshoot_factor < 0.8:
        # When under NPF, star trails are short so lower threshold
        min_area = max(3, int(min_area * 0.8))

    # Final range limit
    min_area = max(3, min(50, min_area))

    return min_area


def estimate_meteor_trail_length(
    focal_length_mm: float,
    exposure_time_sec: float,
    image_width_px: int,
    meteor_speed_factor: float = 3.0,
) -> float:
    """
    Estimate meteor trail length in pixels.

    Meteors move faster than stars, leaving longer trails even with same exposure time.

    Args:
        focal_length_mm: Focal length in 35mm equivalent (mm)
        exposure_time_sec: Exposure time in seconds
        image_width_px: Image width in pixels
        meteor_speed_factor: Meteor speed factor (how many times faster than stars, default 3x)

    Returns:
        Meteor trail length in pixels
    """
    # Calculate star trail length
    star_trail = estimate_star_trail_length(
        focal_length_mm, exposure_time_sec, image_width_px
    )

    # Meteors move faster than stars
    # Typically 2-5x speed (average 3x)
    meteor_trail = star_trail * meteor_speed_factor

    return meteor_trail


def optimize_min_line_score_npf(
    exif_data: Dict[str, Any], npf_metrics: Dict[str, Any]
) -> float:
    """
    Optimize meteor detection threshold based on NPF.

    Args:
        exif_data: EXIF information
        npf_metrics: NPF metrics

    Returns:
        Optimized min_line_score

    Algorithm:
        1. Calculate expected meteor trail length
        2. Set threshold considering detectability
        3. Integrated consideration of focal length and exposure time
    """
    focal_length = exif_data.get("focal_length_35mm")
    exposure_time = exif_data.get("exposure_time")
    image_width = exif_data.get("image_width")
    star_trail_px = npf_metrics.get("star_trail_px")

    # Default value
    min_score = 80.0

    if focal_length and exposure_time and image_width:
        # Estimate expected meteor trail length
        meteor_trail = estimate_meteor_trail_length(
            focal_length, exposure_time, image_width, meteor_speed_factor=3.0
        )

        # Set threshold to 50-70% of expected trail
        # This detects shorter meteors while excluding noise
        min_score = meteor_trail * 0.6
    elif star_trail_px:
        # Estimate from NPF metrics (meteors 3x faster than stars)
        meteor_trail = star_trail_px * 3.0
        min_score = meteor_trail * 0.6

    # Adjust for focal length
    if focal_length:
        if focal_length < 20:
            # Ultra-wide: trails shorter so lower threshold
            min_score *= 0.7
        elif focal_length < 35:
            # Wide angle: lower slightly
            min_score *= 0.85
        elif focal_length > 70:
            # Telephoto: raise threshold as trails become longer
            min_score *= 1.2

    # Adjust for exposure time
    if exposure_time:
        if exposure_time < 5:
            # Short exposure: shorter trails
            min_score *= 0.8
        elif exposure_time > 20:
            # Long exposure: longer trails
            min_score *= 1.1

    # Final range limit
    min_score = max(30.0, min(200.0, min_score))

    return min_score


def calculate_shooting_quality_score(
    exif_data: Dict[str, Any], npf_metrics: Dict[str, Any]
) -> Tuple[float, str]:
    """
    Calculate shooting condition quality score.

    Args:
        exif_data: EXIF information
        npf_metrics: NPF metrics

    Returns:
        (quality_score 0.0-1.0, level)
        level: "EXCELLENT", "GOOD", "FAIR", or "POOR"
    """
    score = 1.0

    # NPF compliance (most important)
    overshoot = npf_metrics.get("overshoot_factor", 1.0)
    if overshoot <= 1.0:
        npf_score = 1.0
    elif overshoot <= 1.5:
        npf_score = 0.8
    elif overshoot <= 2.5:
        npf_score = 0.5
    else:
        npf_score = 0.3

    score *= npf_score

    # ISO sensitivity (high ISO increases noise)
    iso_value = exif_data.get("iso")
    if iso_value:
        if iso_value <= 1600:
            iso_score = 1.0
        elif iso_value <= 3200:
            iso_score = 0.9
        elif iso_value <= 6400:
            iso_score = 0.7
        else:
            iso_score = 0.5
        score *= iso_score

    # Focal length (wide angle advantageous)
    focal_length = exif_data.get("focal_length_35mm")
    if focal_length:
        if focal_length <= 24:
            focal_score = 1.0
        elif focal_length <= 35:
            focal_score = 0.95
        elif focal_length <= 50:
            focal_score = 0.85
        else:
            focal_score = 0.7
        score *= focal_score

    # Determine level
    if score >= 0.8:
        level = "EXCELLENT"
    elif score >= 0.6:
        level = "GOOD"
    elif score >= 0.4:
        level = "FAIR"
    else:
        level = "POOR"

    return score, level


def optimize_params_with_npf(
    exif_data: Dict[str, Any],
    npf_metrics: Dict[str, Any],
    user_specified_diff_threshold: bool = False,
    user_specified_min_area: bool = False,
    user_specified_min_line_score: bool = False,
    current_diff_threshold: int = DEFAULT_DIFF_THRESHOLD,
    current_min_area: int = DEFAULT_MIN_AREA,
    current_min_line_score: float = DEFAULT_MIN_LINE_SCORE,
) -> Tuple[int, int, float, Dict[str, Any]]:
    """
    Optimize all parameters comprehensively based on NPF Rule.

    Args:
        exif_data: EXIF information
        npf_metrics: NPF metrics
        user_specified_*: UserSpecificationFlag
        current_*: Current values

    Returns:
        (diff_threshold, min_area, min_line_score, optimization_info)
    """
    optimization_info = {
        "quality_score": 0.0,
        "quality_level": "UNKNOWN",
        "adjustments": [],
    }

    # Shooting Condition Quality Score Calculation
    quality_score, quality_level = calculate_shooting_quality_score(
        exif_data, npf_metrics
    )
    optimization_info["quality_score"] = quality_score
    optimization_info["quality_level"] = quality_level

    # diff_threshold Optimization
    if not user_specified_diff_threshold:
        diff_threshold = optimize_diff_threshold_npf(exif_data, npf_metrics)
        optimization_info["adjustments"].append(
            f"diff_threshold: {current_diff_threshold} → {diff_threshold} (ISO/NPF-based)"
        )
    else:
        diff_threshold = current_diff_threshold

    # min_area Optimization
    if not user_specified_min_area:
        min_area = optimize_min_area_npf(exif_data, npf_metrics)
        optimization_info["adjustments"].append(
            f"min_area: {current_min_area} → {min_area} (star trail-based)"
        )
    else:
        min_area = current_min_area

    # min_line_score Optimization
    if not user_specified_min_line_score:
        min_line_score = optimize_min_line_score_npf(exif_data, npf_metrics)
        optimization_info["adjustments"].append(
            f"min_line_score: {current_min_line_score:.1f} → {min_line_score:.1f} (meteor trail-based)"
        )
    else:
        min_line_score = current_min_line_score

    return diff_threshold, min_area, min_line_score, optimization_info


# ==========================================
# Memory and Batch Size Functions
# ==========================================


def get_available_memory_bytes() -> Optional[int]:
    """Return available system memory in bytes (best-effort)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass

    return None


def estimate_batch_size(
    requested_batch_size: int,
    image_shape: Tuple[int, int],
    num_workers: int,
    safety_fraction: float = AUTO_BATCH_MEMORY_FRACTION,
    available_mem: Optional[int] = None,
) -> int:
    """
    Estimate a safe batch size based on available memory.

    The calculation uses the first loaded image to approximate memory usage
    of two RAW frames plus intermediate arrays within one worker process.
    """

    if available_mem is None:
        available_mem = get_available_memory_bytes()
    if available_mem is None:
        return requested_batch_size

    height, width = image_shape
    base_bytes = height * width * np.dtype(np.uint16).itemsize

    # Two RAW frames + diff buffer + mask + modest overhead for temporaries
    estimated_pair_bytes = int(
        base_bytes * 2 + base_bytes + (height * width) + base_bytes * 0.5
    )
    if estimated_pair_bytes <= 0:
        return requested_batch_size

    per_worker_budget = available_mem * safety_fraction / max(1, num_workers)
    max_pairs = max(1, int(per_worker_budget // estimated_pair_bytes))

    return max(1, min(requested_batch_size, max_pairs))


# ==========================================
# Progress and Hash Functions
# ==========================================


def compute_params_hash(params: Dict) -> str:
    """Create a stable hash from parameter dictionary."""
    # Convert NumPy types to native Python types for JSON serialization
    params_clean = {}
    for key, value in params.items():
        if isinstance(value, np.integer):
            params_clean[key] = int(value)
        elif isinstance(value, np.floating):
            params_clean[key] = float(value)
        elif isinstance(value, np.ndarray):
            params_clean[key] = value.tolist()
        elif isinstance(value, list):
            # Handle list of lists (like roi_polygon)
            params_clean[key] = [
                (
                    [int(x) if isinstance(x, np.integer) else x for x in item]
                    if isinstance(item, (list, np.ndarray))
                    else item
                )
                for item in value
            ]
        else:
            params_clean[key] = value

    params_json = json.dumps(params_clean, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(params_json.encode("utf-8")).hexdigest()


# ==========================================
# ROI Polygon Utilities
# ==========================================


def parse_roi_polygon_string(roi_str: str) -> List[List[int]]:
    """Parse --roi polygon format x1,y1;x2,y2;..."""

    segments = [
        seg.strip() for seg in roi_str.replace(" ", "").split(";") if seg.strip()
    ]
    if len(segments) < 3:
        raise ValueError(
            "ROI polygon must have at least 3 vertices in the format x1,y1;x2,y2;..."
        )

    polygon: List[List[int]] = []
    for seg in segments:
        try:
            x_str, y_str = seg.split(",")
            polygon.append([int(x_str), int(y_str)])
        except Exception as exc:
            raise ValueError(
                "ROI polygon must be specified as pairs like x1,y1;x2,y2;..."
            ) from exc

    return polygon


def format_polygon_string(polygon: List[List[int]]) -> str:
    """Format polygon vertices into "x1,y1;x2,y2;..." string."""

    return ";".join(f"{x},{y}" for x, y in polygon)


# ==========================================
# Display Functions
# ==========================================


def display_exif_info(
    exif_data: Dict[str, Any],
    focal_length_source: str = "EXIF",
    focal_factor: Optional[float] = None,
    npf_metrics: Optional[Dict[str, Any]] = None,
    locale: Optional[str] = None,
) -> None:
    """
    Display EXIF Information and NPF Rule Analysis (Formatted)

    Args:
        exif_data: Return value from extract_exif_metadata()
        focal_length_source: Source of focal length acquisition("EXIF", "CLI", "Unknown")
        focal_factor: Focal Length Equivalent Factor (if specified)
        npf_metrics: calculate_npf_metrics() return value (Options)
    """
    locale = _resolve_locale(locale)
    print(f"\n{'='*60}")
    print(get_message("ui.utils.exif.header", locale=locale))
    print(f"{'='*60}")

    if exif_data.get("camera_make") or exif_data.get("camera_model"):
        camera_str = f"{exif_data.get('camera_make', '')} {exif_data.get('camera_model', '')}".strip()
        if camera_str:
            print(
                get_message(
                    "ui.utils.exif.line.camera", locale=locale, value=camera_str
                )
            )

    if exif_data.get("lens_model"):
        print(
            get_message(
                "ui.utils.exif.line.lens",
                locale=locale,
                value=exif_data["lens_model"],
            )
        )

    # Focal length
    if exif_data.get("focal_length_35mm"):
        print(
            get_message(
                "ui.utils.exif.focal_length.equiv",
                locale=locale,
                value=exif_data["focal_length_35mm"],
                source=focal_length_source,
            )
        )
    elif exif_data.get("focal_length"):
        print(
            get_message(
                "ui.utils.exif.focal_length.actual",
                locale=locale,
                value=exif_data["focal_length"],
                source=focal_length_source,
            )
        )
        if focal_factor:
            equiv = exif_data["focal_length"] * focal_factor
            print(
                get_message(
                    "ui.utils.exif.focal_length.calculated",
                    locale=locale,
                    value=equiv,
                    factor=focal_factor,
                )
            )
        else:
            print(get_message("ui.utils.exif.focal_length.no_equiv", locale=locale))
    else:
        print(get_message("ui.utils.exif.focal_length.unavailable", locale=locale))

    # ISO sensitivity
    if exif_data.get("iso"):
        print(
            get_message("ui.utils.exif.line.iso", locale=locale, value=exif_data["iso"])
        )
    else:
        print(get_message("ui.utils.exif.line.iso_unavailable", locale=locale))

    # Exposure time
    if exif_data.get("exposure_time"):
        exp = exif_data["exposure_time"]
        if exp >= 1:
            print(
                get_message(
                    "ui.utils.exif.exposure.seconds_long", locale=locale, value=exp
                )
            )
        elif exp >= 0.1:
            print(
                get_message(
                    "ui.utils.exif.exposure.seconds_short", locale=locale, value=exp
                )
            )
        else:
            print(
                get_message(
                    "ui.utils.exif.exposure.fraction",
                    locale=locale,
                    denominator=int(1 / exp),
                )
            )
    else:
        print(get_message("ui.utils.exif.exposure.unavailable", locale=locale))

    # F-number (aperture)
    if exif_data.get("f_number"):
        print(
            get_message(
                "ui.utils.exif.line.aperture",
                locale=locale,
                value=exif_data["f_number"],
            )
        )

    # Image resolution
    if exif_data.get("image_width") and exif_data.get("image_height"):
        print(
            get_message(
                "ui.utils.exif.line.resolution",
                locale=locale,
                width=exif_data["image_width"],
                height=exif_data["image_height"],
            )
        )

    print(f"{'='*60}\n")

    # NPF Rule Analysis
    if npf_metrics and npf_metrics.get("npf_recommended_sec"):
        print(f"{'='*60}")
        print(get_message("ui.utils.npf.header", locale=locale))
        print(f"{'='*60}")

        # Pixel Pitch
        if npf_metrics.get("pixel_pitch_um"):
            pp = npf_metrics["pixel_pitch_um"]
            print(
                get_message(
                    "ui.utils.npf.pixel_pitch",
                    locale=locale,
                    value=pp,
                ),
                end="",
            )
            if npf_metrics.get("sensor_width_mm"):
                print(
                    get_message(
                        "ui.utils.npf.pixel_pitch_sensor",
                        locale=locale,
                        sensor_width=npf_metrics["sensor_width_mm"],
                    )
                )
            else:
                print(get_message("ui.utils.npf.pixel_pitch_default", locale=locale))

        # NPF recommended value
        npf_rec = npf_metrics["npf_recommended_sec"]
        print(
            get_message(
                "ui.utils.npf.recommended",
                locale=locale,
                value=npf_rec,
            )
        )

        # Actual exposure
        if exif_data.get("exposure_time"):
            actual_exp = exif_data["exposure_time"]
            print(
                get_message(
                    "ui.utils.npf.actual_exposure",
                    locale=locale,
                    value=actual_exp,
                ),
                end="",
            )

            # Compliance evaluation
            level = npf_metrics.get("compliance_level", "UNKNOWN")
            factor = npf_metrics.get("overshoot_factor", 0.0)

            if level == "OK":
                print(get_message("ui.utils.npf.compliance.ok", locale=locale))
            elif level == "WARNING":
                print(
                    get_message(
                        "ui.utils.npf.compliance.exceeded",
                        locale=locale,
                        factor=factor,
                    )
                )
            elif level == "CRITICAL":
                print(
                    get_message(
                        "ui.utils.npf.compliance.critical",
                        locale=locale,
                        factor=factor,
                    )
                )
            else:
                print()

        # Star trail estimation
        if npf_metrics.get("star_trail_px"):
            trail = npf_metrics["star_trail_px"]
            print(
                get_message(
                    "ui.utils.npf.star_trail",
                    locale=locale,
                    value=trail,
                )
            )

            # Impact determination
            if trail < 1.0:
                impact = get_message(
                    "ui.utils.npf.impact.minimal",
                    locale=locale,
                )
            elif trail < 2.0:
                impact = get_message(
                    "ui.utils.npf.impact.low",
                    locale=locale,
                )
            elif trail < 5.0:
                impact = get_message(
                    "ui.utils.npf.impact.moderate",
                    locale=locale,
                )
            else:
                impact = get_message(
                    "ui.utils.npf.impact.high",
                    locale=locale,
                )
            print(get_message("ui.utils.npf.impact.line", locale=locale, value=impact))

        # Data Completeness
        if not npf_metrics.get("has_complete_data"):
            print(
                "\n"
                + get_message(
                    "ui.utils.npf.incomplete_note",
                    locale=locale,
                )
            )

        print(f"{'='*60}\n")
