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
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

from .schema import (
    SENSOR_PRESETS,
    CROP_FACTORS,
    FISHEYE_PROJECTION_MODELS,
    DEFAULT_FISHEYE_MODEL,
    DEFAULT_PIXEL_PITCH_UM,
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_LINE_SCORE,
    AUTO_BATCH_MEMORY_FRACTION,
)


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
    key = focal_factor_str.upper().replace("-", "_").replace(" ", "_")
    return CROP_FACTORS.get(key)


def get_sensor_preset(sensor_type: str) -> Optional[Dict[str, Any]]:
    """
    Get sensor preset configuration by sensor type name.

    Args:
        sensor_type: Sensor type string (e.g., "MFT", "APS-C", "FF")

    Returns:
        Sensor preset dictionary or None if not found
    """
    if not sensor_type:
        return None

    key = sensor_type.upper().replace("-", "_").replace(" ", "_")
    return SENSOR_PRESETS.get(key)


def apply_sensor_preset(args, verbose: bool = False) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[Dict[str, Any]],
]:
    """
    Apply sensor preset values, with individual arguments taking priority.

    Args:
        args: Parsed argparse namespace
        verbose: If True, print which values are being used

    Returns:
        Tuple of (focal_factor, sensor_width, focal_length, pixel_pitch, preset)
    """
    focal_factor_value = None
    sensor_width_value = args.sensor_width
    focal_length_value = args.focal_length
    pixel_pitch_value = args.pixel_pitch

    if args.focal_factor:
        focal_factor_value = parse_focal_factor(args.focal_factor)

    preset = None
    if hasattr(args, "sensor_type") and args.sensor_type:
        preset = get_sensor_preset(args.sensor_type)
        if preset is None:
            return (
                focal_factor_value,
                sensor_width_value,
                focal_length_value,
                pixel_pitch_value,
                None,
            )

    if preset:
        preset_applied = []

        if focal_factor_value is None:
            focal_factor_value = preset.get("focal_factor")
            if focal_factor_value is not None:
                preset_applied.append(f"focal_factor={focal_factor_value}")

        if sensor_width_value is None:
            sensor_width_value = preset.get("sensor_width")
            if sensor_width_value is not None:
                preset_applied.append(f"sensor_width={sensor_width_value}mm")

        if pixel_pitch_value is None:
            pixel_pitch_value = preset.get("pixel_pitch")
            if pixel_pitch_value is not None:
                preset_applied.append(f"pixel_pitch={pixel_pitch_value}μm")

        if verbose and preset_applied:
            print(f"  Sensor preset [{args.sensor_type}]: {', '.join(preset_applied)}")

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
) -> None:
    """
    Validate that overridden values are not significantly different from preset.
    """
    if not preset or not args.sensor_type:
        return

    warnings = []

    if args.sensor_width is not None and sensor_width_value is not None:
        preset_width = preset.get("sensor_width")
        if preset_width:
            deviation_pct = abs(sensor_width_value - preset_width) / preset_width * 100
            if deviation_pct > 30.0:
                warnings.append(
                    f"⚠ Warning: --sensor-width {sensor_width_value}mm deviates "
                    f"{deviation_pct:.1f}% from --sensor-type {args.sensor_type} "
                    f"preset ({preset_width}mm)"
                )

    if args.pixel_pitch is not None and pixel_pitch_value is not None:
        preset_pitch = preset.get("pixel_pitch")
        if preset_pitch:
            deviation_pct = abs(pixel_pitch_value - preset_pitch) / preset_pitch * 100
            if deviation_pct > 50.0:
                warnings.append(
                    f"⚠ Warning: --pixel-pitch {pixel_pitch_value}μm deviates "
                    f"{deviation_pct:.1f}% from --sensor-type {args.sensor_type} "
                    f"preset ({preset_pitch}μm)"
                )

    if warnings:
        print(f"\n{'='*70}")
        for warning in warnings:
            print(warning)
        print(f"{'='*70}\n")


def list_sensor_types() -> None:
    """Display available sensor type presets and their configurations."""
    print(f"\n{'='*70}")
    print("Available Sensor Types (--sensor-type)")
    print(f"{'='*70}\n")

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
            print(f"  {sensor_type:12}  {preset['description']}")
            print(
                f"                  focal_factor={preset['focal_factor']}, "
                f"sensor_width={preset['sensor_width']}mm, "
                f"pixel_pitch={preset['pixel_pitch']}μm"
            )
            print()

    print(f"{'='*70}")
    print("Aliases:")
    print("  1-INCH, 1_INCH      → 1INCH")
    print("  APS-C, APS_C        → APSC")
    print("  APS-C_CANON         → APSC_CANON")
    print("  APS-H, APS_H        → APSH")
    print("  FULLFRAME           → FF")
    print("  MF44-33, MF44_33    → MF44X33")
    print("  MF54-40, MF54_40    → MF54X40")
    print(f"{'='*70}")
    print("\nUsage Examples:")
    print("  --sensor-type MFT")
    print("  --sensor-type APS-C")
    print("  --sensor-type FF --pixel-pitch 5.9   # Override pixel pitch")
    print("  --sensor-type MF44X33                # Fujifilm GFX / Pentax 645Z")
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

    Args:
        nominal_focal_length_mm: Nominal (center) focal length in mm
        normalized_radius: Distance from image center (0.0-1.0)
        projection_model: Fisheye projection model

    Returns:
        Effective focal length at the given position (mm)
    """
    if normalized_radius < 0 or normalized_radius > 1:
        normalized_radius = max(0.0, min(1.0, normalized_radius))

    if projection_model.upper() != "EQUISOLID":
        return nominal_focal_length_mm

    max_theta_rad = math.pi / 2
    theta_rad = normalized_radius * max_theta_rad
    compression_factor = math.cos(theta_rad / 2)
    effective_focal_length = nominal_focal_length_mm * compression_factor

    return effective_focal_length


def calculate_fisheye_edge_focal_length(
    nominal_focal_length_mm: float,
    projection_model: str = "EQUISOLID",
) -> float:
    """Calculate effective focal length at the edge/corner of fisheye image."""
    return calculate_fisheye_effective_focal_length(
        nominal_focal_length_mm,
        normalized_radius=1.0,
        projection_model=projection_model,
    )


def calculate_fisheye_trail_length_ratio(
    normalized_radius: float,
    projection_model: str = "EQUISOLID",
) -> float:
    """Calculate the ratio of star trail length at given position vs center."""
    if normalized_radius <= 0:
        return 1.0

    if projection_model.upper() != "EQUISOLID":
        return 1.0

    max_theta_rad = math.pi / 2
    theta_rad = normalized_radius * max_theta_rad
    compression_factor = math.cos(theta_rad / 2)

    if compression_factor > 0:
        return 1.0 / compression_factor
    else:
        return 1.0


def get_fisheye_max_trail_ratio(projection_model: str = "EQUISOLID") -> float:
    """Get the maximum trail length ratio (at image edge vs center)."""
    return calculate_fisheye_trail_length_ratio(1.0, projection_model)


def display_fisheye_info(
    nominal_focal_length_mm: float,
    projection_model: str = "EQUISOLID",
) -> None:
    """Display fisheye correction information."""
    edge_focal = calculate_fisheye_edge_focal_length(
        nominal_focal_length_mm, projection_model
    )
    max_ratio = get_fisheye_max_trail_ratio(projection_model)

    model_info = FISHEYE_PROJECTION_MODELS.get(
        projection_model.upper(), {"name": projection_model, "description": ""}
    )

    print(f"\n{'='*60}")
    print("Fisheye Correction")
    print(f"{'='*60}")
    print(f"  Projection model:   {model_info['name']}")
    print(f"  Nominal focal:      {nominal_focal_length_mm:.1f}mm (center)")
    print(f"  Effective focal:    {edge_focal:.1f}mm (edge)")
    print(f"  Trail length ratio: {max_ratio:.2f}× (edge vs center)")
    print(f"  NPF calculation:    Based on edge (worst case)")
    print(f"{'='*60}")


# ==========================================
# NPF Rule Functions
# ==========================================


def calculate_pixel_pitch(sensor_width_mm: float, image_width_px: int) -> float:
    """Calculate pixel pitch in micrometers (μm)."""
    return (sensor_width_mm * 1000.0) / image_width_px


def calculate_npf_rule(
    focal_length_mm: float, aperture: float, pixel_pitch_um: float
) -> float:
    """
    Calculate recommended maximum exposure time based on NPF Rule.

    NPF Rule formula:
        Exposure Time (seconds) = (35 × Aperture + 30 × Pixel Pitch) / Focal Length
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
    """Estimate star trail length in pixels during exposure."""
    if focal_length_mm <= 0 or exposure_time_sec <= 0 or image_width_px <= 0:
        return 0.0

    EARTH_ROTATION_DEG_PER_SEC = 15.0 / 3600.0
    star_movement_deg = EARTH_ROTATION_DEG_PER_SEC * exposure_time_sec
    declination_factor = math.cos(math.radians(declination_deg))
    star_movement_deg *= declination_factor

    sensor_width_35mm = 36.0
    fov_rad = 2 * math.atan(sensor_width_35mm / (2 * focal_length_mm))
    fov_deg = math.degrees(fov_rad)

    pixels_per_degree = image_width_px / fov_deg
    trail_length_px = star_movement_deg * pixels_per_degree

    return trail_length_px


def evaluate_npf_compliance(
    exposure_time_sec: float, npf_recommended_sec: float
) -> Tuple[str, float]:
    """Evaluate NPF Rule compliance."""
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
    """Calculate NPF Rule-related metrics from EXIF information."""
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

    if not focal_length or not aperture or not exposure_time:
        return result

    effective_focal_length = focal_length
    trail_length_ratio = 1.0

    if fisheye:
        effective_focal_length = calculate_fisheye_edge_focal_length(
            focal_length, fisheye_model
        )
        trail_length_ratio = get_fisheye_max_trail_ratio(fisheye_model)

    result["effective_focal_length"] = effective_focal_length
    result["trail_length_ratio"] = trail_length_ratio

    if pixel_pitch_um is not None:
        result["pixel_pitch_um"] = pixel_pitch_um
    elif sensor_width_mm and image_width:
        result["pixel_pitch_um"] = calculate_pixel_pitch(sensor_width_mm, image_width)
        result["sensor_width_mm"] = sensor_width_mm
    else:
        result["pixel_pitch_um"] = DEFAULT_PIXEL_PITCH_UM

    result["npf_recommended_sec"] = calculate_npf_rule(
        effective_focal_length, aperture, result["pixel_pitch_um"]
    )

    if image_width:
        center_trail = estimate_star_trail_length(
            focal_length, exposure_time, image_width
        )
        result["star_trail_px"] = center_trail * trail_length_ratio

    result["compliance_level"], result["overshoot_factor"] = evaluate_npf_compliance(
        exposure_time, result["npf_recommended_sec"]
    )

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
    """Optimize diff_threshold based on ISO sensitivity and NPF overshoot."""
    iso_value = exif_data.get("iso")
    exposure_time = exif_data.get("exposure_time")
    overshoot_factor = npf_metrics.get("overshoot_factor", 1.0)

    threshold = base_threshold

    if iso_value:
        iso_factor = math.log2(max(iso_value, 100) / 800.0) * 2.0
        threshold += iso_factor

    if exposure_time and exposure_time > 15.0:
        exp_factor = math.log2(exposure_time / 15.0) * 1.0
        threshold += exp_factor

    if overshoot_factor > 1.5:
        npf_factor = (overshoot_factor - 1.5) * 1.5
        threshold += npf_factor

    threshold = max(3, min(25, threshold))

    return int(round(threshold))


def optimize_min_area_npf(
    exif_data: Dict[str, Any], npf_metrics: Dict[str, Any]
) -> int:
    """Optimize min_area based on star trail length and focal length."""
    focal_length = exif_data.get("focal_length_35mm")
    star_trail_px = npf_metrics.get("star_trail_px")
    overshoot_factor = npf_metrics.get("overshoot_factor", 1.0)

    min_area = 10

    if star_trail_px:
        base_area = star_trail_px * 0.5
        min_area = max(3, int(base_area))

    if focal_length:
        if focal_length < 20:
            min_area = max(3, int(min_area * 0.7))
        elif focal_length < 35:
            min_area = max(4, int(min_area * 0.85))
        elif focal_length > 70:
            min_area = int(min_area * 1.3)

    if overshoot_factor > 2.0:
        min_area = int(min_area * 1.2)
    elif overshoot_factor < 0.8:
        min_area = max(3, int(min_area * 0.8))

    min_area = max(3, min(50, min_area))

    return min_area


def estimate_meteor_trail_length(
    focal_length_mm: float,
    exposure_time_sec: float,
    image_width_px: int,
    meteor_speed_factor: float = 3.0,
) -> float:
    """Estimate meteor trail length in pixels."""
    star_trail = estimate_star_trail_length(
        focal_length_mm, exposure_time_sec, image_width_px
    )
    meteor_trail = star_trail * meteor_speed_factor
    return meteor_trail


def optimize_min_line_score_npf(
    exif_data: Dict[str, Any], npf_metrics: Dict[str, Any]
) -> float:
    """Optimize meteor detection threshold based on NPF."""
    focal_length = exif_data.get("focal_length_35mm")
    exposure_time = exif_data.get("exposure_time")
    image_width = exif_data.get("image_width")
    star_trail_px = npf_metrics.get("star_trail_px")

    min_score = 80.0

    if focal_length and exposure_time and image_width:
        meteor_trail = estimate_meteor_trail_length(
            focal_length, exposure_time, image_width, meteor_speed_factor=3.0
        )
        min_score = meteor_trail * 0.6
    elif star_trail_px:
        meteor_trail = star_trail_px * 3.0
        min_score = meteor_trail * 0.6

    if focal_length:
        if focal_length < 20:
            min_score *= 0.7
        elif focal_length < 35:
            min_score *= 0.85
        elif focal_length > 70:
            min_score *= 1.2

    if exposure_time:
        if exposure_time < 5:
            min_score *= 0.8
        elif exposure_time > 20:
            min_score *= 1.1

    min_score = max(30.0, min(200.0, min_score))

    return min_score


def calculate_shooting_quality_score(
    exif_data: Dict[str, Any], npf_metrics: Dict[str, Any]
) -> Tuple[float, str]:
    """Calculate shooting condition quality score."""
    score = 1.0

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
    """Optimize all parameters comprehensively based on NPF Rule."""
    optimization_info = {
        "quality_score": 0.0,
        "quality_level": "UNKNOWN",
        "adjustments": [],
    }

    quality_score, quality_level = calculate_shooting_quality_score(
        exif_data, npf_metrics
    )
    optimization_info["quality_score"] = quality_score
    optimization_info["quality_level"] = quality_level

    if not user_specified_diff_threshold:
        diff_threshold = optimize_diff_threshold_npf(exif_data, npf_metrics)
        optimization_info["adjustments"].append(
            f"diff_threshold: {current_diff_threshold} → {diff_threshold} (ISO/NPF-based)"
        )
    else:
        diff_threshold = current_diff_threshold

    if not user_specified_min_area:
        min_area = optimize_min_area_npf(exif_data, npf_metrics)
        optimization_info["adjustments"].append(
            f"min_area: {current_min_area} → {min_area} (star trail-based)"
        )
    else:
        min_area = current_min_area

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
    """Estimate a safe batch size based on available memory."""
    if available_mem is None:
        available_mem = get_available_memory_bytes()
    if available_mem is None:
        return requested_batch_size

    height, width = image_shape
    base_bytes = height * width * np.dtype(np.uint16).itemsize

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
    params_clean = {}
    for key, value in params.items():
        if isinstance(value, np.integer):
            params_clean[key] = int(value)
        elif isinstance(value, np.floating):
            params_clean[key] = float(value)
        elif isinstance(value, np.ndarray):
            params_clean[key] = value.tolist()
        elif isinstance(value, list):
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
) -> None:
    """Display EXIF Information and NPF Rule Analysis (Formatted)."""
    print(f"\n{'='*60}")
    print("Camera Settings (EXIF Metadata)")
    print(f"{'='*60}")

    if exif_data.get("camera_make") or exif_data.get("camera_model"):
        camera_str = f"{exif_data.get('camera_make', '')} {exif_data.get('camera_model', '')}".strip()
        if camera_str:
            print(f"  Camera:           {camera_str}")

    if exif_data.get("lens_model"):
        print(f"  Lens:             {exif_data['lens_model']}")

    if exif_data.get("focal_length_35mm"):
        print(
            f"  Focal length:     {exif_data['focal_length_35mm']:.1f}mm (35mm equiv.) [{focal_length_source}]"
        )
    elif exif_data.get("focal_length"):
        print(
            f"  Focal length:     {exif_data['focal_length']:.1f}mm (actual) [{focal_length_source}]"
        )
        if focal_factor:
            equiv = exif_data["focal_length"] * focal_factor
            print(
                f"                    ~ {equiv:.1f}mm (35mm equiv., calculated with factor {focal_factor})"
            )
        else:
            print(f"                    ⚠ No 35mm equivalent found")
    else:
        print(f"  Focal length:     Not available")

    if exif_data.get("iso"):
        print(f"  ISO:              {exif_data['iso']}")
    else:
        print(f"  ISO:              Not available")

    if exif_data.get("exposure_time"):
        exp = exif_data["exposure_time"]
        if exp >= 1:
            print(f"  Exposure:         {exp:.1f}s")
        elif exp >= 0.1:
            print(f"  Exposure:         {exp:.2f}s")
        else:
            print(f"  Exposure:         1/{int(1/exp)}s")
    else:
        print(f"  Exposure:         Not available")

    if exif_data.get("f_number"):
        print(f"  Aperture:         f/{exif_data['f_number']:.1f}")

    if exif_data.get("image_width") and exif_data.get("image_height"):
        print(
            f"  Resolution:       {exif_data['image_width']}x{exif_data['image_height']} px"
        )

    print(f"{'='*60}\n")

    if npf_metrics and npf_metrics.get("npf_recommended_sec"):
        print(f"{'='*60}")
        print("NPF Rule Analysis")
        print(f"{'='*60}")

        if npf_metrics.get("pixel_pitch_um"):
            pp = npf_metrics["pixel_pitch_um"]
            print(f"  Pixel pitch:      {pp:.2f}μm", end="")
            if npf_metrics.get("sensor_width_mm"):
                print(f" (sensor: {npf_metrics['sensor_width_mm']:.1f}mm)")
            else:
                print(" (default)")

        npf_rec = npf_metrics["npf_recommended_sec"]
        print(f"  NPF recommended:  {npf_rec:.1f}s")

        if exif_data.get("exposure_time"):
            actual_exp = exif_data["exposure_time"]
            print(f"  Actual exposure:  {actual_exp:.1f}s", end="")

            level = npf_metrics.get("compliance_level", "UNKNOWN")
            factor = npf_metrics.get("overshoot_factor", 0.0)

            if level == "OK":
                print(" ✓ OK")
            elif level == "WARNING":
                print(f" ⚠  EXCEEDED ({factor:.2f}x)")
            elif level == "CRITICAL":
                print(f" ✗ CRITICAL ({factor:.2f}x)")
            else:
                print()

        if npf_metrics.get("star_trail_px"):
            trail = npf_metrics["star_trail_px"]
            print(f"  Star trail est.:  ~{trail:.1f} pixels")

            if trail < 1.0:
                impact = "MINIMAL"
            elif trail < 2.0:
                impact = "LOW"
            elif trail < 5.0:
                impact = "MODERATE"
            else:
                impact = "HIGH"
            print(f"  Impact:           {impact}")

        if not npf_metrics.get("has_complete_data"):
            print(f"\n  ⚠ Note: Incomplete data - using default/estimated values")

        print(f"{'='*60}\n")
