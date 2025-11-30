#!/usr/bin/env python

import rawpy
import numpy as np
import os
import glob
import cv2
import time
import shutil
import argparse
import math
import json
import hashlib
import sys
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Tuple, List, Optional, Dict

try:
    from PIL import Image
    from PIL.ExifTags import TAGS

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# ==========================================
# Default Settings
# ==========================================
VERSION = "1.5.1"

DEFAULT_PROGRESS_FILE = "progress.json"

DEFAULT_TARGET_FOLDER = "rawfiles"
DEFAULT_OUTPUT_FOLDER = "candidates"
DEFAULT_DEBUG_FOLDER = "debug_masks"

EXTENSIONS = ["*.ORF", "*.ARW", "*.CR2", "*.NEF", "*.DNG"]

DEFAULT_DIFF_THRESHOLD = 8
DEFAULT_MIN_AREA = 10
DEFAULT_MIN_ASPECT_RATIO = 3.0

DEFAULT_HOUGH_THRESHOLD = 10
DEFAULT_HOUGH_MIN_LINE_LENGTH = 15
DEFAULT_HOUGH_MAX_LINE_GAP = 5
DEFAULT_MIN_LINE_SCORE = 80.0

DEFAULT_ENABLE_ROI_SELECTION = True
DEFAULT_NUM_WORKERS = max(1, mp.cpu_count() - 1)
DEFAULT_BATCH_SIZE = 10  # Batch processing size
AUTO_BATCH_MEMORY_FRACTION = 0.6  # Portion of free RAM to use when auto-sizing batches

# NPF Rule related default values
# Default pixel pitch for fallback (μm)
DEFAULT_PIXEL_PITCH_UM = 4.0  # Typical value for APS-C/MFT cameras

# Sensor presets: unified configuration for each sensor type
# Each preset contains:
#   - focal_factor: Crop factor for 35mm equivalent conversion
#   - sensor_width: Sensor width in mm
#   - pixel_pitch: Typical pixel pitch in μm (None = calculate from resolution)
#   - description: Human-readable description
# Ordered by sensor size (smallest to largest)
SENSOR_PRESETS = {
    # 1-inch sensor (smallest)
    "1INCH": {
        "focal_factor": 2.7,
        "sensor_width": 13.2,
        "pixel_pitch": 2.4,  # Typical for 20MP 1-inch (e.g., Sony RX100)
        "description": "1-inch sensor (13.2×8.8mm)",
    },
    "1_INCH": {  # Alias for 1INCH
        "focal_factor": 2.7,
        "sensor_width": 13.2,
        "pixel_pitch": 2.4,
        "description": "1-inch sensor (13.2×8.8mm)",
    },
    # Micro Four Thirds
    "MFT": {
        "focal_factor": 2.0,
        "sensor_width": 17.3,
        "pixel_pitch": 3.7,  # Typical for 20MP MFT (e.g., OM-1, GH6)
        "description": "Micro Four Thirds (17.3×13mm)",
    },
    # APS-C (Sony/Nikon/Fuji)
    "APSC": {
        "focal_factor": 1.5,
        "sensor_width": 23.5,
        "pixel_pitch": 3.9,  # Typical for 26MP APS-C (e.g., Sony a6700, Fuji X-T5)
        "description": "APS-C Sony/Nikon/Fuji (23.5×15.6mm)",
    },
    "APS_C": {  # Alias for APSC
        "focal_factor": 1.5,
        "sensor_width": 23.5,
        "pixel_pitch": 3.9,
        "description": "APS-C Sony/Nikon/Fuji (23.5×15.6mm)",
    },
    # APS-C (Canon)
    "APSC_CANON": {
        "focal_factor": 1.6,
        "sensor_width": 22.3,
        "pixel_pitch": 3.2,  # Typical for 32MP Canon APS-C (e.g., R7)
        "description": "APS-C Canon (22.3×14.9mm)",
    },
    "APS_C_CANON": {  # Alias for APSC_CANON
        "focal_factor": 1.6,
        "sensor_width": 22.3,
        "pixel_pitch": 3.2,
        "description": "APS-C Canon (22.3×14.9mm)",
    },
    # APS-H
    "APSH": {
        "focal_factor": 1.3,
        "sensor_width": 27.9,
        "pixel_pitch": 5.7,  # Typical for 16MP APS-H (e.g., Canon 1D Mark IV)
        "description": "APS-H Canon (27.9×18.6mm)",
    },
    "APS_H": {  # Alias for APSH
        "focal_factor": 1.3,
        "sensor_width": 27.9,
        "pixel_pitch": 5.7,
        "description": "APS-H Canon (27.9×18.6mm)",
    },
    # Full Frame 35mm
    "FF": {
        "focal_factor": 1.0,
        "sensor_width": 36.0,
        "pixel_pitch": 4.3,  # Typical for 45-50MP FF (e.g., Sony a7RV, Canon R5)
        "description": "Full Frame 35mm (36×24mm)",
    },
    "FULLFRAME": {  # Alias for FF
        "focal_factor": 1.0,
        "sensor_width": 36.0,
        "pixel_pitch": 4.3,
        "description": "Full Frame 35mm (36×24mm)",
    },
    # Medium Format 44x33 (Fujifilm GFX, Pentax 645Z, Hasselblad X)
    "MF44X33": {
        "focal_factor": 0.79,
        "sensor_width": 43.8,
        "pixel_pitch": 3.76,  # Typical for 100MP (e.g., GFX100, X2D 100C)
        "description": "Medium Format 44×33 (43.8×32.9mm) - GFX/645Z/X2D",
    },
    "MF44_33": {  # Alias for MF44X33
        "focal_factor": 0.79,
        "sensor_width": 43.8,
        "pixel_pitch": 3.76,
        "description": "Medium Format 44×33 (43.8×32.9mm) - GFX/645Z/X2D",
    },
    # Medium Format 54x40 (Hasselblad H6D-100c)
    "MF54X40": {
        "focal_factor": 0.64,
        "sensor_width": 53.4,
        "pixel_pitch": 4.6,  # 100MP H6D-100c (11600×8700, 4.6μm)
        "description": "Medium Format 54×40 (53.4×40mm) - Hasselblad H6D-100c",
    },
    "MF54_40": {  # Alias for MF54X40
        "focal_factor": 0.64,
        "sensor_width": 53.4,
        "pixel_pitch": 4.6,
        "description": "Medium Format 54×40 (53.4×40mm) - Hasselblad H6D-100c",
    },
}

# Legacy compatibility: CROP_FACTORS dictionary for parse_focal_factor()
CROP_FACTORS = {key: preset["focal_factor"] for key, preset in SENSOR_PRESETS.items()}

# Legacy compatibility: DEFAULT_SENSOR_WIDTHS dictionary
DEFAULT_SENSOR_WIDTHS = {
    key: preset["sensor_width"] for key, preset in SENSOR_PRESETS.items()
}


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


def get_sensor_preset(sensor_type: str) -> Optional[Dict[str, any]]:
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
    args, verbose: bool = False
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
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
        Tuple of (focal_factor, sensor_width, focal_length, pixel_pitch)
        Each value is either from CLI argument, preset, or None.

    Examples:
        # --sensor-type MFT (no overrides)
        >>> apply_sensor_preset(args)  # args.sensor_type="MFT"
        (2.0, 17.3, None, 3.7)

        # --sensor-type MFT --sensor-width 18.0 (override sensor_width)
        >>> apply_sensor_preset(args)  # args.sensor_type="MFT", args.sensor_width=18.0
        (2.0, 18.0, None, 3.7)
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
            )

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
            print(f"  Sensor preset [{args.sensor_type}]: {', '.join(preset_applied)}")

    return (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
    )


def list_sensor_types() -> None:
    """
    Display available sensor type presets and their configurations.
    Ordered by sensor size (smallest to largest).
    """
    print(f"\n{'='*70}")
    print("Available Sensor Types (--sensor-type)")
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


def extract_exif_metadata(filepath: str) -> Dict[str, any]:
    """
    Extract EXIF information from RAW file

    Args:
        filepath: Path to RAW file

    Returns:
        Dictionary of EXIF information:
        {
            'focal_length': float,          # Actual focal length (mm)
            'focal_length_35mm': float,     # 35mm equivalent focal length (mm)
            'iso': int,                     # ISO sensitivity
            'exposure_time': float,         # Exposure time (seconds)
            'f_number': float,              # F-number (aperture)
            'camera_make': str,             # Camera manufacturer
            'camera_model': str,            # Camera model
            'lens_model': str,              # Lens model
            'image_width': int,             # Image width (px)
            'image_height': int,            # Image height (px)
        }
    """
    result = {
        "focal_length": None,
        "focal_length_35mm": None,
        "iso": None,
        "exposure_time": None,
        "f_number": None,
        "camera_make": None,
        "camera_model": None,
        "lens_model": None,
        "image_width": None,
        "image_height": None,
    }

    if not PILLOW_AVAILABLE:
        return result

    try:
        # Strategy 1: Extract EXIF from embedded thumbnail
        with rawpy.imread(filepath) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    from io import BytesIO

                    img = Image.open(BytesIO(thumb.data))
                    exif_data = img._getexif()

                    if exif_data:
                        exif_dict = {}
                        for tag_id, value in exif_data.items():
                            tag_name = TAGS.get(tag_id, tag_id)
                            exif_dict[tag_name] = value

                        # Focal length
                        if "FocalLength" in exif_dict:
                            focal = exif_dict["FocalLength"]
                            if isinstance(focal, tuple):
                                result["focal_length"] = float(focal[0]) / float(
                                    focal[1]
                                )
                            else:
                                result["focal_length"] = float(focal)

                        # 35mm equivalent focal length
                        if "FocalLengthIn35mmFilm" in exif_dict:
                            result["focal_length_35mm"] = float(
                                exif_dict["FocalLengthIn35mmFilm"]
                            )

                        # ISO sensitivity
                        if "ISOSpeedRatings" in exif_dict:
                            iso_value = exif_dict["ISOSpeedRatings"]
                            if isinstance(iso_value, (list, tuple)):
                                result["iso"] = int(iso_value[0])
                            else:
                                result["iso"] = int(iso_value)
                        elif "PhotographicSensitivity" in exif_dict:
                            result["iso"] = int(exif_dict["PhotographicSensitivity"])

                        # Exposure time
                        if "ExposureTime" in exif_dict:
                            exp = exif_dict["ExposureTime"]
                            if isinstance(exp, tuple):
                                result["exposure_time"] = float(exp[0]) / float(exp[1])
                            else:
                                result["exposure_time"] = float(exp)

                        # F-number (aperture)
                        if "FNumber" in exif_dict:
                            fnum = exif_dict["FNumber"]
                            if isinstance(fnum, tuple):
                                result["f_number"] = float(fnum[0]) / float(fnum[1])
                            else:
                                result["f_number"] = float(fnum)

                        # Camera information
                        if "Make" in exif_dict:
                            result["camera_make"] = str(exif_dict["Make"]).strip()
                        if "Model" in exif_dict:
                            result["camera_model"] = str(exif_dict["Model"]).strip()

                        # Lens information
                        if "LensModel" in exif_dict:
                            result["lens_model"] = str(exif_dict["LensModel"]).strip()
                        elif "LensSpecification" in exif_dict:
                            result["lens_model"] = str(exif_dict["LensSpecification"])

                        # Image resolution
                        if "ExifImageWidth" in exif_dict:
                            result["image_width"] = int(exif_dict["ExifImageWidth"])
                        elif "ImageWidth" in exif_dict:
                            result["image_width"] = int(exif_dict["ImageWidth"])

                        if "ExifImageHeight" in exif_dict:
                            result["image_height"] = int(exif_dict["ExifImageHeight"])
                        elif "ImageLength" in exif_dict:
                            result["image_height"] = int(exif_dict["ImageLength"])

            except Exception:
                pass

        # Strategy 2: Open RAW file directly with PIL (limited format support)
        if result["focal_length"] is None:
            try:
                img = Image.open(filepath)
                exif_data = img._getexif()
                if exif_data:
                    exif_dict = {
                        TAGS.get(tag_id, tag_id): value
                        for tag_id, value in exif_data.items()
                    }

                    if "FocalLength" in exif_dict and result["focal_length"] is None:
                        focal = exif_dict["FocalLength"]
                        result["focal_length"] = (
                            float(focal[0]) / float(focal[1])
                            if isinstance(focal, tuple)
                            else float(focal)
                        )

                    if (
                        "FocalLengthIn35mmFilm" in exif_dict
                        and result["focal_length_35mm"] is None
                    ):
                        result["focal_length_35mm"] = float(
                            exif_dict["FocalLengthIn35mmFilm"]
                        )
            except Exception:
                pass

        # Strategy 3: Get image dimensions directly from rawpy
        if result["image_width"] is None or result["image_height"] is None:
            try:
                with rawpy.imread(filepath) as raw:
                    sizes = raw.sizes
                    if result["image_width"] is None:
                        result["image_width"] = sizes.raw_width
                    if result["image_height"] is None:
                        result["image_height"] = sizes.raw_height
            except Exception:
                pass

    except Exception:
        pass

    return result


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
    sensor_width_35mm = 36.0  # 35mm film width
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
    exif_data: Dict[str, any],
    sensor_width_mm: Optional[float] = None,
    pixel_pitch_um: Optional[float] = None,
) -> Dict[str, any]:
    """
    Calculate NPF Rule-related metrics from EXIF information.

    Args:
        exif_data: Return value from extract_exif_metadata()
        sensor_width_mm: Sensor width in millimeters (None if cannot estimate)
        pixel_pitch_um: Pixel pitch in micrometers (μm) (calculated or default if None)

    Returns:
        Dictionary of NPF metrics:
        {
            'pixel_pitch_um': float,           # Pixel pitch (μm)
            'npf_recommended_sec': float,      # NPF recommended exposure time (seconds)
            'star_trail_px': float,            # Star trail length in pixels
            'compliance_level': str,           # "OK", "WARNING", "CRITICAL"
            'overshoot_factor': float,         # Overshoot factor
            'sensor_width_mm': float,          # Sensor width used
            'has_complete_data': bool,         # Whether complete data is available
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
    }

    focal_length = exif_data.get("focal_length_35mm")
    aperture = exif_data.get("f_number")
    exposure_time = exif_data.get("exposure_time")
    image_width = exif_data.get("image_width")

    # Check for required data
    if not focal_length or not aperture or not exposure_time:
        return result

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
    result["npf_recommended_sec"] = calculate_npf_rule(
        focal_length, aperture, result["pixel_pitch_um"]
    )

    # Estimate star trail length
    if image_width:
        result["star_trail_px"] = estimate_star_trail_length(
            focal_length, exposure_time, image_width
        )

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
    exif_data: Dict[str, any], npf_metrics: Dict[str, any], base_threshold: float = 5.0
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
    exif_data: Dict[str, any], npf_metrics: Dict[str, any]
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
    exif_data: Dict[str, any], npf_metrics: Dict[str, any]
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
    exif_data: Dict[str, any], npf_metrics: Dict[str, any]
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
    exif_data: Dict[str, any],
    npf_metrics: Dict[str, any],
    user_specified_diff_threshold: bool = False,
    user_specified_min_area: bool = False,
    user_specified_min_line_score: bool = False,
    current_diff_threshold: int = DEFAULT_DIFF_THRESHOLD,
    current_min_area: int = DEFAULT_MIN_AREA,
    current_min_line_score: float = DEFAULT_MIN_LINE_SCORE,
) -> Tuple[int, int, float, Dict[str, any]]:
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


def calculate_npf_metrics(
    exif_data: Dict[str, any],
    sensor_width_mm: Optional[float] = None,
    pixel_pitch_um: Optional[float] = None,
) -> Dict[str, any]:
    """
    Calculate NPF Rule-related metrics from EXIF information.

    Args:
        exif_data: Return value from extract_exif_metadata()
        sensor_width_mm: Sensor width in millimeters (None if cannot estimate)
        pixel_pitch_um: Pixel pitch in micrometers (μm) (calculated or default if None)

    Returns:
        Dictionary of NPF metrics:
        {
            'pixel_pitch_um': float,           # Pixel pitch (μm)
            'npf_recommended_sec': float,      # NPF recommended exposure time (seconds)
            'star_trail_px': float,            # Star trail length in pixels
            'compliance_level': str,           # "OK", "WARNING", "CRITICAL"
            'overshoot_factor': float,         # Overshoot factor
            'sensor_width_mm': float,          # Sensor width used
            'has_complete_data': bool,         # Whether complete data is available
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
    }

    focal_length = exif_data.get("focal_length_35mm")
    aperture = exif_data.get("f_number")
    exposure_time = exif_data.get("exposure_time")
    image_width = exif_data.get("image_width")

    # Check for required data
    if not focal_length or not aperture or not exposure_time:
        return result

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
    result["npf_recommended_sec"] = calculate_npf_rule(
        focal_length, aperture, result["pixel_pitch_um"]
    )

    # Estimate star trail length
    if image_width:
        result["star_trail_px"] = estimate_star_trail_length(
            focal_length, exposure_time, image_width
        )

    # Evaluate NPF compliance
    result["compliance_level"], result["overshoot_factor"] = evaluate_npf_compliance(
        exposure_time, result["npf_recommended_sec"]
    )

    # Check if complete data is available
    result["has_complete_data"] = all(
        [focal_length, aperture, exposure_time, result["pixel_pitch_um"], image_width]
    )

    return result


def load_and_bin_raw_fast(filepath: str) -> np.ndarray:
    """
    Load RAW file & 2x2 binning
    - Minimize memory allocation
    - Optimize casting operations
    """
    with rawpy.imread(filepath) as raw:
        bayer = raw.raw_image

        # Binning process (reduce memory copying with view operations)
        h, w = bayer.shape
        h_half, w_half = h // 2, w // 2

        # Calculate directly with uint16 (add incrementally to prevent overflow)
        result = np.empty((h_half, w_half), dtype=np.uint16)

        # Single operation calculation (directly without going through uint32)
        temp = bayer[0::2, 0::2].astype(np.uint32)
        temp += bayer[0::2, 1::2]
        temp += bayer[1::2, 0::2]
        temp += bayer[1::2, 1::2]
        result[:] = temp // 4

        return result


def compute_line_score_fast(mask: np.ndarray, hough_params: dict) -> Tuple[float, List]:
    """Line detection using Hough transform"""
    # Early return if few edges
    if np.count_nonzero(mask) < hough_params["min_line_length"]:
        return 0.0, []

    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 180,
        hough_params["threshold"],
        minLineLength=hough_params["min_line_length"],
        maxLineGap=hough_params["max_line_gap"],
    )

    if lines is None:
        return 0.0, []

    # Vectorize for speed
    lines_array = lines.reshape(-1, 4)
    dx = lines_array[:, 2] - lines_array[:, 0]
    dy = lines_array[:, 3] - lines_array[:, 1]
    lengths = np.sqrt(dx * dx + dy * dy)

    score = float(np.sum(lengths))
    line_segments = [
        (int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in lines_array
    ]

    return score, line_segments


def display_exif_info(
    exif_data: Dict[str, any],
    focal_length_source: str = "EXIF",
    focal_factor: Optional[float] = None,
    npf_metrics: Optional[Dict[str, any]] = None,
) -> None:
    """
    Display EXIF Information and NPF Rule Analysis (Formatted)

    Args:
        exif_data: Return value from extract_exif_metadata()
        focal_length_source: Source of focal length acquisition("EXIF", "CLI", "Unknown")
        focal_factor: Focal Length Equivalent Factor (if specified)
        npf_metrics: calculate_npf_metrics() return value (Options)
    """
    print(f"\n{'='*60}")
    print("Camera Settings (EXIF Metadata)")
    print(f"{'='*60}")

    if exif_data.get("camera_make") or exif_data.get("camera_model"):
        camera_str = f"{exif_data.get('camera_make', '')} {exif_data.get('camera_model', '')}".strip()
        if camera_str:
            print(f"  Camera:           {camera_str}")

    if exif_data.get("lens_model"):
        print(f"  Lens:             {exif_data['lens_model']}")

    # Focal length
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

    # ISO sensitivity
    if exif_data.get("iso"):
        print(f"  ISO:              {exif_data['iso']}")
    else:
        print(f"  ISO:              Not available")

    # Exposure time
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

    # F-number (aperture)
    if exif_data.get("f_number"):
        print(f"  Aperture:         f/{exif_data['f_number']:.1f}")

    # Image resolution
    if exif_data.get("image_width") and exif_data.get("image_height"):
        print(
            f"  Resolution:       {exif_data['image_width']}x{exif_data['image_height']} px"
        )

    print(f"{'='*60}\n")

    # NPF RuleAnalysis
    if npf_metrics and npf_metrics.get("npf_recommended_sec"):
        print(f"{'='*60}")
        print("NPF Rule Analysis")
        print(f"{'='*60}")

        # Pixel Pitch
        if npf_metrics.get("pixel_pitch_um"):
            pp = npf_metrics["pixel_pitch_um"]
            print(f"  Pixel pitch:      {pp:.2f}μm", end="")
            if npf_metrics.get("sensor_width_mm"):
                print(f" (sensor: {npf_metrics['sensor_width_mm']:.1f}mm)")
            else:
                print(" (default)")

        # NPFRecommended Value
        npf_rec = npf_metrics["npf_recommended_sec"]
        print(f"  NPF recommended:  {npf_rec:.1f}s")

        # Actual exposure
        if exif_data.get("exposure_time"):
            actual_exp = exif_data["exposure_time"]
            print(f"  Actual exposure:  {actual_exp:.1f}s", end="")

            # Compliance evaluation
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

        # Star trail estimation
        if npf_metrics.get("star_trail_px"):
            trail = npf_metrics["star_trail_px"]
            print(f"  Star trail est.:  ~{trail:.1f} pixels")

            # Impact determination
            if trail < 1.0:
                impact = "MINIMAL"
            elif trail < 2.0:
                impact = "LOW"
            elif trail < 5.0:
                impact = "MODERATE"
            else:
                impact = "HIGH"
            print(f"  Impact:           {impact}")

        # Data Completeness
        if not npf_metrics.get("has_complete_data"):
            print(f"\n  ⚠ Note: Incomplete data - using default/estimated values")

        print(f"{'='*60}\n")


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


def estimate_diff_threshold_from_samples(
    files: List[str], roi_mask: np.ndarray, sample_size: int = 5
) -> int:
    """
    Estimation using percentile-based approach

    Real-world sky brightness distributions are highly peaked, so
    percentile-based estimation is more appropriate than 3-sigma rule.

    Args:
        files: List of RAW file paths
        roi_mask: ROI mask to focus on sky area
        sample_size: Number of initial images to analyze

    Returns:
        Estimated diff_threshold value
    """
    print(f"\n{'='*50}")
    print(f"Auto-estimating diff_threshold from {sample_size} samples")
    print(f"Percentile-based approach")
    print(f"{'='*50}")

    sample_size = min(sample_size, len(files))
    if sample_size < 2:
        print("⚠ Not enough samples, using default")
        return DEFAULT_DIFF_THRESHOLD

    samples = []
    print(f"Loading samples... ", end="", flush=True)
    for i in range(sample_size):
        try:
            img = load_and_bin_raw_fast(files[i])
            samples.append(img)
        except Exception as exc:
            print(f"\n⚠ Warning: Failed to load sample {i}: {exc}")
            continue
    print(f"✓ Loaded {len(samples)} images")

    if len(samples) < 2:
        print("⚠ Not enough valid samples, using default")
        return DEFAULT_DIFF_THRESHOLD

    # Calculate frame-to-frame differences in ROI
    print("Analyzing frame-to-frame differences in ROI... ", end="", flush=True)
    diff_values = []
    for i in range(1, len(samples)):
        diff = cv2.absdiff(samples[i], samples[i - 1])
        roi_diff = diff[roi_mask == 255]
        diff_values.extend(roi_diff.flatten())
    print("✓")

    diff_array = np.array(diff_values, dtype=np.float32)

    # Calculate statistics
    mean_diff = np.mean(diff_array)
    std_diff = np.std(diff_array)
    median_diff = np.median(diff_array)

    # Percentiles (for peaked distributions)
    p90 = np.percentile(diff_array, 90)
    p95 = np.percentile(diff_array, 95)
    p98 = np.percentile(diff_array, 98)
    p99 = np.percentile(diff_array, 99)

    # Multiple estimation methods
    # Method 1: 98th percentile (works well for peaked distributions)
    method_1 = int(p98)

    # Method 2: Conservative sigma multiplier (3σ → 1.5σ for real sky data)
    method_2 = int(mean_diff + 1.5 * std_diff)

    # Method 3: Median-based (robust to outliers)
    method_3 = int(median_diff * 3.0)

    # Select the most sensitive (lowest) threshold
    estimated_threshold = min(method_1, method_2, method_3)

    # Clamp to reasonable range (adjusted based on real-world feedback)
    estimated_threshold = np.clip(estimated_threshold, 3, 18)

    print(f"\n{'─'*50}")
    print(f"ROI Difference Statistics (from {len(diff_values):,} pixels):")
    print(f"{'─'*50}")
    print(f"  Mean:         {mean_diff:.2f}")
    print(f"  Std Dev:      {std_diff:.2f}")
    print(f"  Median:       {median_diff:.2f}")
    print(f"  90th %ile:    {p90:.2f}")
    print(f"  95th %ile:    {p95:.2f}")
    print(f"  98th %ile:    {p98:.2f}")
    print(f"  99th %ile:    {p99:.2f}")
    print(f"{'─'*50}")
    print(f"Estimation methods:")
    print(f"  [1] 98th percentile:      {method_1}")
    print(f"  [2] Mean + 1.5σ:          {method_2}")
    print(f"  [3] Median × 3:           {method_3}")
    print(f"{'─'*50}")
    print(f"✓ Selected threshold: {estimated_threshold} (minimum)")
    print(f"{'='*50}\n")

    return estimated_threshold


def estimate_min_area_from_samples(
    files: List[str], roi_mask: np.ndarray, diff_threshold: int, sample_size: int = 3
) -> int:
    """
    v1.3.1: Improved min_area estimation with better star detection

    Args:
        files: List of RAW file paths
        roi_mask: ROI mask to focus on sky area
        diff_threshold: Threshold for star detection
        sample_size: Number of images to analyze

    Returns:
        Estimated min_area value
    """
    print(f"\n{'='*50}")
    print(f"Auto-estimating min_area from {sample_size} samples")
    print(f"Star size distribution analysis")
    print(f"{'='*50}")

    sample_size = min(sample_size, len(files))
    if sample_size < 1:
        print("⚠ Not enough samples, using default")
        return DEFAULT_MIN_AREA

    samples = []
    print(f"Loading samples... ", end="", flush=True)
    for i in range(sample_size):
        try:
            img = load_and_bin_raw_fast(files[i])
            samples.append(img)
        except Exception as exc:
            print(f"\n⚠ Warning: Failed to load sample {i}: {exc}")
            continue
    print(f"✓ Loaded {len(samples)} images")

    if not samples:
        print("⚠ No valid samples, using default")
        return DEFAULT_MIN_AREA

    print("Detecting stars in ROI... ", end="", flush=True)
    all_star_areas = []

    for img in samples:
        roi_pixels = img[roi_mask == 255]

        if len(roi_pixels) < 100:
            continue

        # v1.3.1: Use 98th percentile (brighter stars only, avoid noise)
        threshold = np.percentile(roi_pixels, 98)

        _, star_mask = cv2.threshold(img, int(threshold), 255, cv2.THRESH_BINARY)
        star_mask = cv2.bitwise_and(star_mask.astype(np.uint8), roi_mask)

        contours, _ = cv2.findContours(
            star_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # v1.3.1: Filter by area range to exclude noise and large artifacts
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 2.0 <= area <= 100.0:  # Exclude tiny noise and large artifacts
                all_star_areas.append(area)

    print(f"✓ Detected {len(all_star_areas)} stars")

    if len(all_star_areas) < 10:
        print("⚠ Not enough stars detected, using default")
        return DEFAULT_MIN_AREA

    star_areas = np.array(all_star_areas)
    median_star = np.median(star_areas)
    mean_star = np.mean(star_areas)
    p75_star = np.percentile(star_areas, 75)
    p90_star = np.percentile(star_areas, 90)

    # v1.3.1: Use 75th percentile × 2.0 for more robust estimation
    estimated_min_area = int(p75_star * 2.0)

    # Ensure minimum is at least default value
    estimated_min_area = max(estimated_min_area, DEFAULT_MIN_AREA)

    # Clamp to reasonable range
    estimated_min_area = np.clip(estimated_min_area, 8, 50)

    print(f"\n{'─'*50}")
    print(f"Star Size Statistics (from {len(all_star_areas)} stars):")
    print(f"{'─'*50}")
    print(f"  Median:       {median_star:.1f} pixels²")
    print(f"  Mean:         {mean_star:.1f} pixels²")
    print(f"  75th %ile:    {p75_star:.1f} pixels²")
    print(f"  90th %ile:    {p90_star:.1f} pixels²")
    print(f"{'─'*50}")
    print(f"✓ Estimated min_area: {estimated_min_area}")
    print(f"  → 75th percentile × 2.0 (robust to outliers)")
    print(f"{'='*50}\n")

    return estimated_min_area


def estimate_min_line_score_from_image(
    image_shape: Tuple[int, int], focal_length_mm: Optional[float] = None
) -> float:
    """
    v1.3.1: Fixed min_line_score estimation with corrected focal length logic

    Args:
        image_shape: (height, width) of image
        focal_length_mm: Focal length in mm (optional)

    Returns:
        Estimated min_line_score value
    """
    print(f"\n{'='*50}")
    print(f"Auto-estimating min_line_score from image geometry")
    print(f"{'='*50}")

    height, width = image_shape
    diagonal = np.sqrt(height**2 + width**2)

    # v1.3.1: Reduced base coefficient from 4% to 2.5% based on real data
    base_score = diagonal * 0.025

    if focal_length_mm:
        # v1.3.1: FIXED - Corrected focal length logic
        # Wide angle (14mm) → shorter trails → LOWER scores
        # Telephoto (50mm+) → longer relative trails → HIGHER scores
        # Factor calculation: focal_length / 50.0 (NOT 50.0 / focal_length)
        focal_factor = focal_length_mm / 50.0
        adjusted_score = base_score * focal_factor

        print(f"\n{'─'*50}")
        print(f"Image Geometry:")
        print(f"{'─'*50}")
        print(f"  Dimensions:   {width}×{height} pixels")
        print(f"  Diagonal:     {diagonal:.0f} pixels")
        print(f"  Focal length: {focal_length_mm}mm")
        print(f"  Focal factor: {focal_factor:.2f}×")
        print(f"  Base score:   {base_score:.1f}")
        print(f"  Adjusted:     {adjusted_score:.1f}")
    else:
        adjusted_score = base_score
        print(f"\n{'─'*50}")
        print(f"Image Geometry:")
        print(f"{'─'*50}")
        print(f"  Dimensions:   {width}×{height} pixels")
        print(f"  Diagonal:     {diagonal:.0f} pixels")
        print(f"  Base score:   {base_score:.1f}")
        print(f"  (No focal length provided)")

    # v1.3.1: Adjusted clamp range based on real meteor data
    estimated_score = np.clip(adjusted_score, 40.0, 150.0)

    print(f"{'─'*50}")
    print(f"✓ Estimated min_line_score: {estimated_score:.1f}")
    print(f"  → ~2.5% of image diagonal")
    print(f"{'='*50}\n")

    return estimated_score


def compute_params_hash(params: Dict) -> str:
    """Create a stable hash from parameter dictionary"""
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


def _init_worker_ignore_interrupt() -> None:
    """Ignore SIGINT in worker processes"""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def load_progress(progress_path: str) -> Optional[Dict]:
    """Load progress JSON if it exists"""
    if not os.path.exists(progress_path):
        return None

    try:
        with open(progress_path, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        print(f"Failed to read progress file {progress_path}: {exc}")
        return None


def save_progress(progress_path: str, progress_data: Dict) -> None:
    """Persist progress JSON to disk"""
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    progress_data.setdefault("created_at", now_iso)
    progress_data["last_updated"] = now_iso

    try:
        os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as fp:
            json.dump(progress_data, fp, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Failed to write progress file {progress_path}: {exc}")


def process_image_batch(
    batch_data: List[Tuple[str, str]], roi_mask: np.ndarray, params: dict
) -> List[Tuple]:
    """
    Process a batch of images (handle multiple pairs at once)

    Args:
        batch_data: List of [(curr_file, prev_file), ...]
        roi_mask: ROI mask
        params: Parameter dictionary

    Returns:
        List of processing results for each image
    """
    results = []

    # Pre-create kernel for morphology
    kernel = np.ones((3, 3), np.uint8)

    hough_params = {
        "threshold": params["hough_threshold"],
        "min_line_length": params["hough_min_line_length"],
        "max_line_gap": params["hough_max_line_gap"],
    }

    for curr_file, prev_file in batch_data:
        filename = os.path.basename(curr_file)

        try:
            # Load images
            curr_img = load_and_bin_raw_fast(curr_file)
            prev_img = load_and_bin_raw_fast(prev_file)

            # Calculate difference (save memory with in-place operation)
            diff = cv2.absdiff(curr_img, prev_img)

            # Binarize
            _, mask = cv2.threshold(
                diff, params["diff_threshold"], 255, cv2.THRESH_BINARY
            )
            mask = mask.astype(np.uint8)

            # Apply ROI
            cv2.bitwise_and(mask, roi_mask, dst=mask)

            # Noise removal
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Hough transform
            line_score, hough_lines = compute_line_score_fast(mask, hough_params)

            # Shape detection
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            is_meteor_candidate = False
            debug_img = None
            max_aspect_ratio = 0.0

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area > params["min_area"]:
                    rect = cv2.minAreaRect(cnt)
                    (w, h) = rect[1]

                    if w == 0 or h == 0:
                        continue

                    long_side = max(w, h)
                    short_side = min(w, h)
                    aspect_ratio = long_side / max(short_side, 1)
                    max_aspect_ratio = max(max_aspect_ratio, aspect_ratio)

                    if (
                        aspect_ratio > params["min_aspect_ratio"]
                        and line_score >= params["min_line_score"]
                    ):
                        is_meteor_candidate = True

                        if debug_img is None:
                            debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                            for x1, y1, x2, y2 in hough_lines:
                                cv2.line(
                                    debug_img, (x1, y1), (x2, y2), (0, 255, 255), 1
                                )

                        box = cv2.boxPoints(rect)
                        box = np.int64(box)
                        cv2.drawContours(debug_img, [box], 0, (0, 0, 255), 2)

            results.append(
                (
                    is_meteor_candidate,
                    filename,
                    curr_file,
                    line_score,
                    debug_img,
                    max_aspect_ratio,
                    len(hough_lines),
                )
            )

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            results.append((False, filename, curr_file, 0.0, None, 0.0, 0))

    return results


def select_roi(image_data):
    """Polygon ROI selection with vertex editing"""
    disp_img = image_data.astype(np.float32)
    disp_img = disp_img / np.max(disp_img)
    disp_img = (disp_img * 255).astype(np.uint8)

    h, w = disp_img.shape
    scale_factor = 1.0

    if w > 1200:
        scale_factor = 1200 / w
        disp_w = int(w * scale_factor)
        disp_h = int(h * scale_factor)
        disp_img_resized = cv2.resize(disp_img, (disp_w, disp_h))
    else:
        disp_img_resized = disp_img

    display_img = cv2.cvtColor(disp_img_resized, cv2.COLOR_GRAY2BGR)
    window_name = "Select Sky Area"

    print("\n--- ROI Selection Mode ---")
    print(
        "Left click: add vertex | Esc: delete last vertex | Close by clicking the start circle"
    )

    points: List[Tuple[int, int]] = []
    mouse_pos: Optional[Tuple[int, int]] = None
    polygon_closed = False
    closable_threshold = 12
    closable_radius = 6
    cancelled = False

    def draw_canvas():
        canvas = display_img.copy()

        if points:
            cv2.polylines(
                canvas, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 2
            )
            for px, py in points:
                cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)

        if mouse_pos and points:
            cv2.line(canvas, points[-1], mouse_pos, (255, 255, 0), 1)

        if len(points) >= 3:
            first = points[0]
            hover_distance = (
                math.hypot(mouse_pos[0] - first[0], mouse_pos[1] - first[1])
                if mouse_pos
                else None
            )
            if hover_distance is not None and hover_distance <= closable_threshold:
                cv2.circle(canvas, first, closable_radius, (0, 255, 255), -1)

        return canvas

    def on_mouse(event, x, y, *_):
        nonlocal mouse_pos, polygon_closed
        mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if (
                len(points) >= 3
                and math.hypot(x - points[0][0], y - points[0][1]) <= closable_threshold
            ):
                polygon_closed = True
            else:
                points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        canvas = draw_canvas()
        cv2.imshow(window_name, canvas)

        if polygon_closed:
            cv2.polylines(
                canvas, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2
            )
            cv2.imshow(window_name, canvas)
            cv2.waitKey(300)
            break

        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC: delete last vertex
            if points:
                points.pop()
        elif key == ord("q"):
            cancelled = True
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    if cancelled or len(points) < 3:
        return None

    points_scaled = [
        (int(px / scale_factor), int(py / scale_factor)) for px, py in points
    ]
    polygon = np.array(points_scaled, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    bounding_rect = cv2.boundingRect(polygon)

    return {"mask": mask, "polygon": polygon.tolist(), "bounding_rect": bounding_rect}


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
    """Format polygon vertices into "x1,y1;x2,y2;..." string"""

    return ";".join(f"{x},{y}" for x, y in polygon)


def collect_files(target_folder):
    """Collect RAW files from the specified folder

    Args:
        target_folder: Path to the folder to search for RAW files

    Returns:
        Sorted list of RAW file paths

    Raises:
        FileNotFoundError: If the directory doesn't exist
        NotADirectoryError: If the path is not a directory
        FileNotFoundError: If no RAW files are found in the directory
    """
    # Check if the directory exists
    if not os.path.exists(target_folder):
        raise FileNotFoundError(f"Directory does not exist: {target_folder}")

    # Check if the path is a directory
    if not os.path.isdir(target_folder):
        raise NotADirectoryError(f"Path is not a directory: {target_folder}")

    # Collect RAW files
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(target_folder, ext)))

    # Check if any RAW files were found
    if not files:
        raise FileNotFoundError(
            f"No RAW image files found in directory: {target_folder}\n"
            f"Supported formats: {', '.join(EXTENSIONS)}"
        )

    files.sort()
    return files


def validate_raw_file(index: int, raw_file: str):
    """Attempt to load a RAW file, returning any validation error."""

    try:
        load_and_bin_raw_fast(raw_file)
        return index, raw_file, None
    except Exception as exc:
        return index, raw_file, exc


def detect_meteors_advanced(
    target_folder=DEFAULT_TARGET_FOLDER,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    debug_folder=DEFAULT_DEBUG_FOLDER,
    diff_threshold=DEFAULT_DIFF_THRESHOLD,
    min_area=DEFAULT_MIN_AREA,
    min_aspect_ratio=DEFAULT_MIN_ASPECT_RATIO,
    hough_threshold=DEFAULT_HOUGH_THRESHOLD,
    hough_min_line_length=DEFAULT_HOUGH_MIN_LINE_LENGTH,
    hough_max_line_gap=DEFAULT_HOUGH_MAX_LINE_GAP,
    min_line_score=DEFAULT_MIN_LINE_SCORE,
    enable_roi_selection=DEFAULT_ENABLE_ROI_SELECTION,
    roi_polygon_cli=None,
    num_workers=DEFAULT_NUM_WORKERS,
    batch_size=DEFAULT_BATCH_SIZE,
    auto_batch_size=False,
    enable_parallel=True,
    profile=False,
    validate_raw=False,
    progress_file=DEFAULT_PROGRESS_FILE,
    resume=True,
    auto_params=False,
    user_specified_diff_threshold=False,
    user_specified_min_area=False,
    user_specified_min_line_score=False,
    focal_length_mm=None,
    focal_factor=None,
    sensor_width_mm=None,
    pixel_pitch_um=None,
    output_overwrite=False,
):
    """
    Main processing: detect meteor candidates from consecutive RAW images
    """
    timing = {}
    t_total = time.time()

    # Safety check: prevent overwriting source files
    target_fullpath = os.path.abspath(target_folder)
    output_fullpath = os.path.abspath(output_folder)

    if target_fullpath == output_fullpath:
        print(f"\n{'='*60}")
        print("⚠ ERROR: Target and output directories are the same!")
        print(f"{'='*60}")
        print(f"  Target:  {target_folder}")
        print(f"  Output:  {output_folder}")
        print(f"  Resolved to: {target_fullpath}")
        print(f"\nThis configuration would overwrite original RAW files.")
        print(f"Please specify a different output directory.")
        print(f"{'='*60}\n")
        return 0

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)

    print(f"Collecting RAW files from: {target_folder}")
    files = collect_files(target_folder)

    if len(files) < 2:
        print("Need at least 2 images. Exiting.")
        return 0

    print(f"Found {len(files)} files")

    # Load first image
    t_load = time.time()
    try:
        prev_img = load_and_bin_raw_fast(files[0])
    except Exception as exc:
        print(f"Failed to load first RAW file: {os.path.basename(files[0])} ({exc})")
        return 0

    if profile:
        timing["first_load"] = time.time() - t_load
    height, width = prev_img.shape

    # ROI setup
    roi_mask = np.full((height, width), 255, dtype=np.uint8)
    roi_polygon = None

    if roi_polygon_cli:
        print(
            f"ROI specified via command line: polygon={format_polygon_string(roi_polygon_cli)}"
        )
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [np.array(roi_polygon_cli, dtype=np.int32)], 255)
        roi_polygon = roi_polygon_cli
    elif enable_roi_selection:
        roi_selection = select_roi(prev_img)
        if roi_selection:
            roi_mask = roi_selection["mask"]
            roi_polygon = roi_selection["polygon"]
            print(f"ROI setup complete: polygon={format_polygon_string(roi_polygon)}")
        else:
            print("No ROI selected. Processing entire image.")
    else:
        print("Skipping ROI selection. Processing entire image.")

    # Auto-parameter estimation
    if auto_params:
        print(f"\n{'='*60}")
        print("Auto-params: NPF Rule-based Optimization")
        print(f"{'='*60}\n")

        # ========================================
        # Step 1: EXIF Information Extraction
        # ========================================
        exif_data = extract_exif_metadata(files[0])

        # Focal length acquisition (priority)
        focal_length_source = "Unknown"
        if focal_length_mm:
            # CLIarguments (--focal-length)has highest priority
            focal_length_source = "CLI (--focal-length)"
            exif_data["focal_length_35mm"] = focal_length_mm
        elif exif_data.get("focal_length_35mm"):
            # EXIF 35mmEquivalentFocal Length
            focal_length_mm = exif_data["focal_length_35mm"]
            focal_length_source = "EXIF"
        elif exif_data.get("focal_length") and focal_factor:
            # EXIF actual focal length + focal_factor calculation
            focal_length_mm = exif_data["focal_length"] * focal_factor
            exif_data["focal_length_35mm"] = focal_length_mm
            focal_length_source = f"Calculated (EXIF {exif_data['focal_length']:.1f}mm × factor {focal_factor})"
        elif exif_data.get("focal_length"):
            # EXIF Actual Focal Lengthonly (no 35mm equivalent)
            focal_length_mm = exif_data["focal_length"]
            focal_length_source = "EXIF (actual, no 35mm equiv.)"

        # ========================================
        # Step 2: NPF RuleAnalysis
        # ========================================
        npf_metrics = calculate_npf_metrics(
            exif_data, sensor_width_mm=sensor_width_mm, pixel_pitch_um=pixel_pitch_um
        )

        # Display EXIF Information and NPF Analysis
        display_exif_info(exif_data, focal_length_source, focal_factor, npf_metrics)

        # ========================================
        # Step 3: Display warnings
        # ========================================
        warnings = []
        if not focal_length_mm:
            warnings.append("Focal length not available from EXIF")
            warnings.append("  → Consider using --focal-length option")
        elif not exif_data.get("focal_length_35mm") and not focal_factor:
            warnings.append(
                f"35mm equivalent not found in EXIF (using actual: {focal_length_mm:.1f}mm)"
            )
            warnings.append(
                "  → For crop sensor cameras, use --focal-factor (e.g., --focal-factor MFT)"
            )

        if not exif_data.get("iso"):
            warnings.append("ISO value not available from EXIF")

        if not exif_data.get("exposure_time"):
            warnings.append("Exposure time not available from EXIF")

        if npf_metrics and not npf_metrics.get("has_complete_data"):
            if not sensor_width_mm and not pixel_pitch_um:
                warnings.append("Using default pixel pitch for NPF calculation")
                warnings.append(
                    "  → For better accuracy, use --sensor-width or --pixel-pitch"
                )

        if warnings:
            print(f"{'='*60}")
            print("⚠ Warnings:")
            for warning in warnings:
                if warning.startswith("  →"):
                    print(f"  {warning}")
                else:
                    print(f"  • {warning}")
            print(f"{'='*60}\n")

        # ========================================
        # Step 4: NPF-based parameter optimization
        # ========================================
        if npf_metrics and npf_metrics.get("npf_recommended_sec"):
            print(f"{'='*60}")
            print("Parameter Optimization (NPF Rule-based)")
            print(f"{'='*60}\n")

            # Integrated optimization
            diff_threshold, min_area, min_line_score, opt_info = (
                optimize_params_with_npf(
                    exif_data,
                    npf_metrics,
                    user_specified_diff_threshold=user_specified_diff_threshold,
                    user_specified_min_area=user_specified_min_area,
                    user_specified_min_line_score=user_specified_min_line_score,
                    current_diff_threshold=diff_threshold,
                    current_min_area=min_area,
                    current_min_line_score=min_line_score,
                )
            )

            # Display Shooting Condition Quality Score
            print(
                f"Shooting Quality Score: {opt_info['quality_score']:.2f} ({opt_info['quality_level']})"
            )

            # Parameter adjustment details
            if opt_info["adjustments"]:
                print(f"\nParameter Adjustments:")
                for adjustment in opt_info["adjustments"]:
                    print(f"  • {adjustment}")
            else:
                print(f"\nNo automatic adjustments (all parameters user-specified)")

            print(f"\n{'='*60}\n")
        else:
            # Fallback: legacy method
            print("⚠ Insufficient data for NPF-based optimization")
            print("  Falling back to legacy auto-params method\n")

            if not user_specified_diff_threshold:
                diff_threshold = estimate_diff_threshold_from_samples(
                    files, roi_mask, sample_size=5
                )
                print(f"→ Using sample-based diff_threshold: {diff_threshold}")
            else:
                print(f"→ Using user-specified diff_threshold: {diff_threshold}")

            if not user_specified_min_area:
                min_area = estimate_min_area_from_samples(
                    files, roi_mask, diff_threshold, sample_size=3
                )
                print(f"→ Using sample-based min_area: {min_area}")
            else:
                print(f"→ Using user-specified min_area: {min_area}")

            if not user_specified_min_line_score:
                min_line_score = estimate_min_line_score_from_image(
                    prev_img.shape, focal_length_mm
                )
                print(f"→ Using image-based min_line_score: {min_line_score:.1f}")
            else:
                print(f"→ Using user-specified min_line_score: {min_line_score}")

    params = {
        "diff_threshold": diff_threshold,
        "min_area": min_area,
        "min_aspect_ratio": min_aspect_ratio,
        "hough_threshold": hough_threshold,
        "hough_min_line_length": hough_min_line_length,
        "hough_max_line_gap": hough_max_line_gap,
        "min_line_score": min_line_score,
    }

    print(f"\n{'='*50}")
    print("Processing Parameters:")
    print(f"{'='*50}")
    print(f"  diff_threshold:        {diff_threshold}")
    print(f"  min_area:              {min_area}")
    print(f"  min_aspect_ratio:      {min_aspect_ratio}")
    print(f"  hough_threshold:       {hough_threshold}")
    print(f"  hough_min_line_length: {hough_min_line_length}")
    print(f"  hough_max_line_gap:    {hough_max_line_gap}")
    print(f"  min_line_score:        {min_line_score:.1f}")
    print(f"{'='*50}\n")

    # Progress tracking setup
    params_for_hash = params.copy()
    if roi_polygon:
        params_for_hash["roi_polygon"] = roi_polygon

    progress_data: Dict = {
        "version": VERSION,
        "params_hash": compute_params_hash(params_for_hash),
        "processed_files": [],
        "detected_files": [],
        "total_processed": 0,
        "total_detected": 0,
    }

    loaded_progress = load_progress(progress_file) if resume else None

    if loaded_progress:
        if loaded_progress.get("params_hash") == progress_data["params_hash"]:
            progress_data.update(
                {
                    key: loaded_progress.get(key, progress_data.get(key))
                    for key in [
                        "version",
                        "params_hash",
                        "processed_files",
                        "detected_files",
                        "total_processed",
                        "total_detected",
                        "created_at",
                        "last_updated",
                    ]
                }
            )
            print(
                f"Resuming from progress file: {progress_file} "
                f"(processed={progress_data['total_processed']}, "
                f"detected={progress_data['total_detected']})"
            )
        else:
            print(
                "Progress file exists but parameters differ. Starting with fresh progress."
            )

    existing_basenames = {os.path.basename(path) for path in files}
    progress_data["processed_files"] = [
        name
        for name in progress_data.get("processed_files", [])
        if name in existing_basenames
    ]
    progress_data["detected_files"] = [
        name
        for name in progress_data.get("detected_files", [])
        if name in existing_basenames
    ]

    processed_set = set(progress_data["processed_files"])
    detected_set = set(progress_data["detected_files"])

    progress_data["total_processed"] = len(processed_set)
    progress_data["total_detected"] = len(detected_set)

    save_progress(progress_file, progress_data)

    def record_result(filename: str, is_candidate: bool) -> None:
        processed_set.add(filename)
        if filename not in progress_data["processed_files"]:
            progress_data["processed_files"].append(filename)

        if is_candidate:
            detected_set.add(filename)
            if filename not in progress_data["detected_files"]:
                progress_data["detected_files"].append(filename)

        progress_data["total_processed"] = len(processed_set)
        progress_data["total_detected"] = len(detected_set)
        save_progress(progress_file, progress_data)

    image_pairs = [(files[i], files[i - 1]) for i in range(1, len(files))]
    image_pairs = [
        pair for pair in image_pairs if os.path.basename(pair[0]) not in processed_set
    ]

    resume_offset = len(processed_set)
    overall_total = resume_offset + len(image_pairs)

    print(f"Starting processing: {len(image_pairs)} image pairs")
    if enable_parallel:
        print(f"Parallel processing: {num_workers} workers, batch size: {batch_size}")

    detected_count = len(detected_set)
    t_process = time.time()

    try:
        if enable_parallel and num_workers > 1:
            # Split into batches
            batches = [
                image_pairs[i : i + batch_size]
                for i in range(0, len(image_pairs), batch_size)
            ]

            print(f"Number of batches: {len(batches)}")

            executor = ProcessPoolExecutor(
                max_workers=num_workers, initializer=_init_worker_ignore_interrupt
            )
            futures: List = []
            wait_for_tasks = True

            try:
                for batch in batches:
                    future = executor.submit(
                        process_image_batch, batch, roi_mask, params
                    )
                    futures.append(future)

                # Collect results
                processed = 0
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        for result in batch_results:
                            (
                                is_candidate,
                                filename,
                                filepath,
                                line_score,
                                debug_img,
                                aspect_ratio,
                                num_lines,
                            ) = result
                            processed += 1

                            if line_score > 0:
                                print(
                                    f"  [LINE] {filename}: score={line_score:.1f}, lines={num_lines}"
                                )

                            if is_candidate:
                                output_path = os.path.join(output_folder, filename)
                                # Check if file exists
                                if os.path.exists(output_path) and not output_overwrite:
                                    print(
                                        f"  [SKIP] {filename}: Already exists in output folder (use --output-overwrite to overwrite)"
                                    )
                                else:
                                    shutil.copy(filepath, output_path)
                                    if debug_img is not None:
                                        if roi_polygon:
                                            cv2.polylines(
                                                debug_img,
                                                [np.array(roi_polygon, dtype=np.int32)],
                                                True,
                                                (0, 255, 0),
                                                2,
                                            )
                                        cv2.imwrite(
                                            os.path.join(
                                                debug_folder, f"mask_{filename}.png"
                                            ),
                                            debug_img,
                                        )
                                    print(
                                        f"  [HIT] {filename}: Ratio={aspect_ratio:.2f}"
                                    )
                            else:
                                print(
                                    f"\rChecking... {resume_offset + processed}/{overall_total}",
                                    end="",
                                    flush=True,
                                )

                            record_result(filename, is_candidate)
                            detected_count = progress_data["total_detected"]

                    except Exception as e:
                        print(f"\nBatch processing error: {e}")
            except KeyboardInterrupt:
                print("\nInterrupted by user. Cancelling worker processes...")
                wait_for_tasks = False
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(
                    wait=wait_for_tasks, cancel_futures=not wait_for_tasks
                )
        else:
            # Sequential processing with progress display
            for idx, pair in enumerate(image_pairs):
                current_index = resume_offset + idx + 1
                current_file = os.path.basename(pair[0])
                progress_line_active = True

                print(
                    f"\rProcessing {current_index}/{overall_total}: {current_file}",
                    end="",
                    flush=True,
                )

                batch_results = process_image_batch([pair], roi_mask, params)

                for result in batch_results:
                    (
                        is_candidate,
                        filename,
                        filepath,
                        line_score,
                        debug_img,
                        aspect_ratio,
                        num_lines,
                    ) = result

                    if line_score > 0:
                        if progress_line_active:
                            print()
                            progress_line_active = False
                        print(
                            f"  [LINE] {filename}: score={line_score:.1f}, lines={num_lines}"
                        )

                    if is_candidate:
                        if progress_line_active:
                            print()
                            progress_line_active = False

                        output_path = os.path.join(output_folder, filename)
                        # Check if file exists
                        if os.path.exists(output_path) and not output_overwrite:
                            print(
                                f"  [SKIP] {filename}: Already exists in output folder (use --output-overwrite to overwrite)"
                            )
                        else:
                            shutil.copy(filepath, output_path)

                            if debug_img is not None:
                                if roi_polygon:
                                    cv2.polylines(
                                        debug_img,
                                        [np.array(roi_polygon, dtype=np.int32)],
                                        True,
                                        (0, 255, 0),
                                        2,
                                    )
                                cv2.imwrite(
                                    os.path.join(debug_folder, f"mask_{filename}.png"),
                                    debug_img,
                                )

                            print(f"  [HIT] {filename}: Ratio={aspect_ratio:.2f}")
                    else:
                        print(
                            f"\rChecking... {current_index}/{overall_total}",
                            end="",
                            flush=True,
                        )
                        progress_line_active = True

                    record_result(filename, is_candidate)
                    detected_count = progress_data["total_detected"]

    except KeyboardInterrupt:
        print(f"\nInterrupted by user. Progress saved to {progress_file}.")
        save_progress(progress_file, progress_data)
        return detected_count

    if profile:
        timing["processing"] = time.time() - t_process
        timing["total"] = time.time() - t_total

        print("\n\n=== Performance Profile ===")
        print(f"First image load: {timing['first_load']:.3f}s")
        print(f"Processing time: {timing['processing']:.3f}s")
        print(f"Total time: {timing['total']:.3f}s")
        print(f"Images processed: {len(image_pairs)}")
        print(f"Average per image: {timing['processing'] / len(image_pairs):.3f}s")

    print(f"\nComplete! {detected_count} candidates extracted")
    return detected_count


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Meteor detection tool with comprehensive auto-parameter estimation (v1.3.1)"
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    parser.add_argument(
        "-t", "--target", default=DEFAULT_TARGET_FOLDER, help="Input RAW image folder"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_FOLDER,
        help="Folder to copy detected candidate RAW files",
    )
    parser.add_argument(
        "--debug-dir",
        default=DEFAULT_DEBUG_FOLDER,
        help="Folder to save mask/debug images",
    )

    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=DEFAULT_DIFF_THRESHOLD,
        help=f"Threshold for difference binarization (default: {DEFAULT_DIFF_THRESHOLD})",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help=f"Minimum contour area (default: {DEFAULT_MIN_AREA})",
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=DEFAULT_MIN_ASPECT_RATIO,
        help=f"Minimum aspect ratio (default: {DEFAULT_MIN_ASPECT_RATIO})",
    )

    parser.add_argument(
        "--hough-threshold",
        type=int,
        default=DEFAULT_HOUGH_THRESHOLD,
        help=f"Hough line detection threshold (default: {DEFAULT_HOUGH_THRESHOLD})",
    )
    parser.add_argument(
        "--hough-min-line-length",
        type=int,
        default=DEFAULT_HOUGH_MIN_LINE_LENGTH,
        help=f"Minimum line length (default: {DEFAULT_HOUGH_MIN_LINE_LENGTH})",
    )
    parser.add_argument(
        "--hough-max-line-gap",
        type=int,
        default=DEFAULT_HOUGH_MAX_LINE_GAP,
        help=f"Maximum line gap (default: {DEFAULT_HOUGH_MAX_LINE_GAP})",
    )
    parser.add_argument(
        "--min-line-score",
        type=float,
        default=DEFAULT_MIN_LINE_SCORE,
        help=f"Minimum line score (default: {DEFAULT_MIN_LINE_SCORE})",
    )
    parser.add_argument("--no-roi", action="store_true", help="Skip ROI selection")
    parser.add_argument(
        "--roi", type=str, default=None, help='Specify ROI polygon as "x1,y1;x2,y2;..."'
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--auto-batch-size",
        action="store_true",
        help="Auto-adjust batch size for memory",
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument("--profile", action="store_true", help="Display timing profile")
    parser.add_argument(
        "--validate-raw", action="store_true", help="Validate RAW files first"
    )
    parser.add_argument(
        "--progress-file", default=DEFAULT_PROGRESS_FILE, help="Progress JSON file path"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Ignore existing progress"
    )
    parser.add_argument(
        "--remove-progress", action="store_true", help="Delete progress and exit"
    )

    # Auto-params (v1.4.0: NPF Rule support, v1.5.0: --sensor-type shortcut)
    parser.add_argument(
        "--auto-params",
        action="store_true",
        help="Auto-estimate diff_threshold, min_area, and min_line_score based on EXIF (v1.4.0 with NPF Rule)",
    )
    parser.add_argument(
        "--sensor-type",
        type=str,
        default=None,
        metavar="TYPE",
        help="Sensor type preset for NPF Rule parameters. "
        "Sets focal_factor, sensor_width, and pixel_pitch automatically. "
        "Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40. "
        "Individual options (--focal-factor, --sensor-width, --pixel-pitch) override preset values.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Focal length in mm (35mm equivalent; used for NPF Rule and parameter estimation). "
        "Overrides --sensor-type preset if specified.",
    )
    parser.add_argument(
        "--focal-factor",
        type=str,
        default=None,
        help="Crop factor for 35mm equivalent calculation. "
        "Use sensor type (MFT, APS-C, APS-H, etc.) or numeric value (e.g., 2.0, 1.5). "
        "Common values: MFT=2.0, APS-C=1.5, APS-C_CANON=1.6, APS-H=1.3, FF=1.0. "
        "Overrides --sensor-type preset if specified.",
    )
    parser.add_argument(
        "--sensor-width",
        type=float,
        default=None,
        help="Sensor width in mm (for NPF Rule calculation). "
        "Common values: MFT=17.3, APS-C=23.5, APS-C(Canon)=22.3, FF=36.0. "
        "Overrides --sensor-type preset if specified.",
    )
    parser.add_argument(
        "--pixel-pitch",
        type=float,
        default=None,
        help="Pixel pitch in micrometers (μm) for NPF Rule. "
        "If not specified, calculated from sensor width and image resolution, or uses default. "
        "Overrides --sensor-type preset if specified.",
    )
    parser.add_argument(
        "--list-sensor-types",
        action="store_true",
        help="Display available sensor type presets and exit",
    )
    parser.add_argument(
        "--show-exif",
        action="store_true",
        help="Display EXIF metadata and NPF Rule analysis from first RAW file and exit",
    )
    parser.add_argument(
        "--show-npf",
        action="store_true",
        help="Display NPF Rule analysis details (implies --show-exif)",
    )
    parser.add_argument(
        "--output-overwrite",
        action="store_true",
        help="Force overwrite existing files in output folder (default: skip existing files)",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.remove_progress:
        if os.path.exists(args.progress_file):
            os.remove(args.progress_file)
            print(f"Removed progress file: {args.progress_file}")
        else:
            print(f"Progress file not found: {args.progress_file}")
        return

    # --list-sensor-types: Display available sensor types and exit
    if args.list_sensor_types:
        list_sensor_types()
        return

    # --show-exif or --show-npf: Display EXIF info and NPF analysis then exit
    if args.show_exif or args.show_npf:
        print(f"\n{'='*60}")
        if args.show_npf:
            print("EXIF Metadata & NPF Rule Analysis")
        else:
            print("EXIF Metadata Viewer")
        print(f"{'='*60}\n")
        print(f"Target folder: {args.target}")

        # Validate --sensor-type if specified
        if args.sensor_type and get_sensor_preset(args.sensor_type) is None:
            print(f"⚠ Error: Invalid --sensor-type value: '{args.sensor_type}'")
            print(
                f"  Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40"
            )
            return

        # Apply sensor preset (with individual args taking priority)
        (
            focal_factor_value,
            sensor_width_value,
            focal_length_value,
            pixel_pitch_value,
        ) = apply_sensor_preset(args, verbose=True)

        # Validate focal_factor if specified directly
        if args.focal_factor and focal_factor_value is None:
            print(f"⚠ Error: Invalid --focal-factor value: '{args.focal_factor}'")
            print(f"  Valid values: MFT, APS-C, APS-H, FF, or numeric (e.g., 2.0)")
            return

        try:
            files = collect_files(args.target)
            if not files:
                print("⚠ No RAW files found in target folder.")
                return

            print(f"Found {len(files)} RAW files")
            print(f"Reading EXIF from first file: {os.path.basename(files[0])}\n")

            exif_data = extract_exif_metadata(files[0])

            # Focal length priority processing
            focal_length_source = "Unknown"
            if focal_length_value:
                focal_length_source = "CLI (--focal-length)"
                exif_data["focal_length_35mm"] = focal_length_value
            elif exif_data.get("focal_length_35mm"):
                focal_length_source = "EXIF"
            elif exif_data.get("focal_length") and focal_factor_value:
                if args.focal_factor:
                    focal_length_source = (
                        f"Calculated (--focal-factor {args.focal_factor})"
                    )
                else:
                    focal_length_source = (
                        f"Calculated (--sensor-type {args.sensor_type})"
                    )
                exif_data["focal_length_35mm"] = (
                    exif_data["focal_length"] * focal_factor_value
                )
            elif exif_data.get("focal_length"):
                focal_length_source = "EXIF (actual, no 35mm equiv.)"

            # NPF Rule Analysis (when --show-npf or sufficient information exists)
            npf_metrics = None
            if args.show_npf or (
                exif_data.get("focal_length_35mm") and exif_data.get("f_number")
            ):
                npf_metrics = calculate_npf_metrics(
                    exif_data,
                    sensor_width_mm=sensor_width_value,
                    pixel_pitch_um=pixel_pitch_value,
                )

            display_exif_info(
                exif_data, focal_length_source, focal_factor_value, npf_metrics
            )

            # Display warnings
            warnings = []
            if (
                not exif_data.get("focal_length")
                and not exif_data.get("focal_length_35mm")
                and not focal_length_value
            ):
                warnings.append("Focal length not available")
            elif (
                not exif_data.get("focal_length_35mm")
                and not focal_factor_value
                and not focal_length_value
            ):
                warnings.append(
                    f"35mm equivalent not found. Consider using --sensor-type or --focal-factor"
                )
            if not exif_data.get("iso"):
                warnings.append("ISO value not available")
            if not exif_data.get("exposure_time"):
                warnings.append("Exposure time not available")

            # NPF-related warnings
            if args.show_npf or npf_metrics:
                if not sensor_width_value and not exif_data.get("image_width"):
                    warnings.append(
                        "Sensor width not specified. Use --sensor-type or --sensor-width for accurate NPF calculation"
                    )
                if npf_metrics and not npf_metrics.get("has_complete_data"):
                    warnings.append("NPF calculation using default/estimated values")

            if warnings:
                print(f"{'='*60}")
                print("⚠ Warnings:")
                for warning in warnings:
                    print(f"  • {warning}")
                print(f"{'='*60}\n")

            # Display Usage Examples
            if args.show_npf:
                print(f"{'='*60}")
                print("Usage Examples:")
                print(f"{'='*60}")
                print("\nUse --sensor-type for easy setup (recommended):")
                print(f"  --sensor-type MFT           # Micro Four Thirds")
                print(f"  --sensor-type APS-C         # APS-C (Sony/Nikon/Fuji)")
                print(f"  --sensor-type APS-C_CANON   # APS-C (Canon)")
                print(f"  --sensor-type FF            # Full Frame")
                print("\nOr specify individual parameters (overrides --sensor-type):")
                print(f"  --sensor-width 17.3   # Sensor width in mm")
                print(f"  --pixel-pitch 3.7     # Pixel pitch in micrometers")
                print(f"  --focal-factor 2.0    # Crop factor")
                print(f"\n{'='*60}\n")
                if warnings:
                    print("⚠ Warnings:")
                    for warning in warnings:
                        print(f"  • {warning}")
                    print(f"{'='*60}\n")

        except FileNotFoundError as e:
            print(f"⚠ Error: {e}")
        except Exception as e:
            print(f"⚠ Error reading EXIF: {e}")

        return

    roi_polygon_cli = None
    enable_roi_selection = DEFAULT_ENABLE_ROI_SELECTION

    if args.roi is not None:
        roi_polygon_cli = parse_roi_polygon_string(args.roi)
        enable_roi_selection = False
    elif args.no_roi:
        enable_roi_selection = False

    # Determine user specifications
    user_specified_diff_threshold = "--diff-threshold" in sys.argv
    user_specified_min_area = "--min-area" in sys.argv
    user_specified_min_line_score = "--min-line-score" in sys.argv

    # Validate --sensor-type if specified
    if args.sensor_type and get_sensor_preset(args.sensor_type) is None:
        print(f"⚠ Error: Invalid --sensor-type value: '{args.sensor_type}'")
        print(
            f"  Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40"
        )
        return

    # Apply sensor preset (with individual args taking priority)
    focal_factor_value, sensor_width_value, focal_length_value, pixel_pitch_value = (
        apply_sensor_preset(args, verbose=False)
    )

    # Validate focal_factor if specified directly
    if args.focal_factor and focal_factor_value is None:
        print(f"⚠ Error: Invalid --focal-factor value: '{args.focal_factor}'")
        print(f"  Valid values: MFT, APS-C, APS-H, FF, or numeric (e.g., 2.0)")
        return

    detect_meteors_advanced(
        target_folder=args.target,
        output_folder=args.output,
        debug_folder=args.debug_dir,
        diff_threshold=args.diff_threshold,
        min_area=args.min_area,
        min_aspect_ratio=args.min_aspect_ratio,
        hough_threshold=args.hough_threshold,
        hough_min_line_length=args.hough_min_line_length,
        hough_max_line_gap=args.hough_max_line_gap,
        min_line_score=args.min_line_score,
        enable_roi_selection=enable_roi_selection,
        roi_polygon_cli=roi_polygon_cli,
        num_workers=args.workers,
        batch_size=args.batch_size,
        auto_batch_size=args.auto_batch_size,
        enable_parallel=not args.no_parallel,
        profile=args.profile,
        validate_raw=args.validate_raw,
        progress_file=args.progress_file,
        resume=not args.no_resume,
        auto_params=args.auto_params,
        user_specified_diff_threshold=user_specified_diff_threshold,
        user_specified_min_area=user_specified_min_area,
        user_specified_min_line_score=user_specified_min_line_score,
        focal_length_mm=focal_length_value,
        focal_factor=focal_factor_value,
        sensor_width_mm=sensor_width_value,
        pixel_pitch_um=pixel_pitch_value,
        output_overwrite=args.output_overwrite,
    )


if __name__ == "__main__":
    main()
