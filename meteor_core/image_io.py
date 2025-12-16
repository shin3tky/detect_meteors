#!/usr/bin/env python
#
# Detect Meteors CLI - Image I/O
# © 2025 Shinichi Morita (shin3tky)
#

"""
Image loading and EXIF extraction functions.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import rawpy

# Module-level logger
logger = logging.getLogger(__name__)

try:
    from PIL import Image
    from PIL.ExifTags import TAGS

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


def load_and_bin_raw_fast(filepath: str) -> np.ndarray:
    """
    Load RAW file & 2x2 binning.

    Args:
        filepath: Path to the RAW file

    Returns:
        Binned image as numpy array (uint16)

    Raises:
        MeteorUnsupportedFormatError: If the file format is not supported by rawpy.
        MeteorLoadError: If the file cannot be loaded due to I/O errors,
            data corruption, or other loading failures.

    Note:
        - Minimizes memory allocation
        - Optimizes casting operations
    """
    import os

    from .exceptions import MeteorLoadError, MeteorUnsupportedFormatError
    from .schema import EXTENSIONS

    logger.info("Loading RAW file: %s", filepath)

    # Check if file exists first (provides clearer error message)
    if not os.path.exists(filepath):
        logger.error(
            "File not found: %s",
            filepath,
            extra={"error_category": "file_not_found", "filepath": filepath},
        )
        raise MeteorLoadError(
            "File not found",
            filepath=filepath,
            context={"error_category": "file_not_found"},
        )

    # Check file permissions
    if not os.access(filepath, os.R_OK):
        logger.error(
            "Permission denied: %s",
            filepath,
            extra={"error_category": "permission_denied", "filepath": filepath},
        )
        raise MeteorLoadError(
            "Permission denied: cannot read file",
            filepath=filepath,
            context={"error_category": "permission_denied"},
        )

    # Get supported formats for error messages
    supported_formats = [ext.replace("*.", ".") for ext in EXTENSIONS]

    try:
        with rawpy.imread(filepath) as raw:
            bayer = raw.raw_image

            # Binning process (reduce memory copying with view operations)
            h, w = bayer.shape
            h_half, w_half = h // 2, w // 2

            logger.debug(
                "Processing image: original=%dx%d, binned=%dx%d",
                w,
                h,
                w_half,
                h_half,
            )

            # Calculate directly with uint16 (add incrementally to prevent overflow)
            result = np.empty((h_half, w_half), dtype=np.uint16)

            # Single operation calculation (directly without going through uint32)
            temp = bayer[0::2, 0::2].astype(np.uint32)
            temp += bayer[0::2, 1::2]
            temp += bayer[1::2, 0::2]
            temp += bayer[1::2, 1::2]
            result[:] = temp // 4

            logger.info("Successfully loaded and binned: %s", filepath)
            return result

    except rawpy.LibRawFileUnsupportedError as e:
        # File format is not supported by libraw
        _, ext = os.path.splitext(filepath)
        logger.error(
            "Unsupported file format '%s': %s",
            ext or "unknown",
            filepath,
            extra={
                "error_category": "unsupported_format",
                "filepath": filepath,
                "detected_format": ext,
                "original_error": str(e),
            },
        )
        raise MeteorUnsupportedFormatError(
            f"Unsupported RAW format: {ext or 'unknown'}",
            filepath=filepath,
            original_error=e,
            detected_format=ext.upper() if ext else None,
            supported_formats=supported_formats,
        ) from e

    except rawpy.LibRawIOError as e:
        # I/O error during reading
        logger.error(
            "I/O error reading file: %s - %s",
            filepath,
            e,
            extra={
                "error_category": "io_error",
                "filepath": filepath,
                "original_error": str(e),
            },
        )
        raise MeteorLoadError(
            "I/O error while reading RAW file",
            filepath=filepath,
            original_error=e,
            context={"error_category": "io_error"},
        ) from e

    except rawpy.LibRawDataError as e:
        # Data corruption or invalid data
        logger.error(
            "Data corruption in file: %s - %s",
            filepath,
            e,
            extra={
                "error_category": "data_corruption",
                "filepath": filepath,
                "original_error": str(e),
            },
        )
        raise MeteorLoadError(
            "RAW file data is corrupted or invalid",
            filepath=filepath,
            original_error=e,
            context={"error_category": "data_corruption"},
        ) from e

    except rawpy.LibRawUnsufficientMemoryError as e:
        # Out of memory
        logger.error(
            "Insufficient memory loading file: %s - %s",
            filepath,
            e,
            extra={
                "error_category": "memory_error",
                "filepath": filepath,
                "original_error": str(e),
            },
        )
        raise MeteorLoadError(
            "Insufficient memory to load RAW file",
            filepath=filepath,
            original_error=e,
            context={"error_category": "memory_error"},
        ) from e

    except rawpy.LibRawTooBigError as e:
        # File too large
        logger.error(
            "File too large: %s - %s",
            filepath,
            e,
            extra={
                "error_category": "file_too_large",
                "filepath": filepath,
                "original_error": str(e),
            },
        )
        raise MeteorLoadError(
            "RAW file is too large to process",
            filepath=filepath,
            original_error=e,
            context={"error_category": "file_too_large"},
        ) from e

    except rawpy.LibRawError as e:
        # Generic libraw error
        logger.error(
            "LibRaw error loading file: %s - %s",
            filepath,
            e,
            extra={
                "error_category": "libraw_error",
                "filepath": filepath,
                "original_error": str(e),
            },
        )
        raise MeteorLoadError(
            "Failed to load RAW file",
            filepath=filepath,
            original_error=e,
            context={"error_category": "libraw_error"},
        ) from e

    except MemoryError as e:
        # Python memory error during binning
        logger.error(
            "Out of memory during binning: %s - %s",
            filepath,
            e,
            extra={
                "error_category": "memory_error",
                "filepath": filepath,
                "stage": "binning",
                "original_error": str(e),
            },
        )
        raise MeteorLoadError(
            "Out of memory during image processing",
            filepath=filepath,
            original_error=e,
            context={"error_category": "memory_error", "stage": "binning"},
        ) from e

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(
            "Unexpected error loading file: %s - %s: %s",
            filepath,
            type(e).__name__,
            e,
            extra={
                "error_category": "unexpected",
                "filepath": filepath,
                "error_type": type(e).__name__,
                "original_error": str(e),
            },
            exc_info=True,  # Include traceback for unexpected errors
        )
        raise MeteorLoadError(
            f"Unexpected error loading RAW file: {type(e).__name__}",
            filepath=filepath,
            original_error=e,
            context={"error_category": "unexpected"},
        ) from e


# --- EXIF extraction helpers ---


def _exif_ratio_to_float(value: Any) -> Optional[float]:
    """Convert EXIF ratio tuple or scalar to float.

    Args:
        value: EXIF value (tuple of (numerator, denominator) or scalar)

    Returns:
        Float value, or None if conversion fails
    """
    if value is None:
        return None
    try:
        if isinstance(value, tuple):
            return float(value[0]) / float(value[1]) if value[1] != 0 else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _populate_exif_from_dict(result: Dict[str, Any], exif_dict: Dict[str, Any]) -> int:
    """Populate result dictionary from EXIF data dictionary.

    Args:
        result: Result dictionary to populate (modified in place)
        exif_dict: Dictionary of EXIF tag names to values

    Returns:
        Number of fields that were populated
    """
    initial_count = sum(1 for v in result.values() if v is not None)

    # Focal length
    if "FocalLength" in exif_dict and result["focal_length"] is None:
        result["focal_length"] = _exif_ratio_to_float(exif_dict["FocalLength"])

    # 35mm equivalent focal length
    if "FocalLengthIn35mmFilm" in exif_dict and result["focal_length_35mm"] is None:
        try:
            result["focal_length_35mm"] = float(exif_dict["FocalLengthIn35mmFilm"])
        except (TypeError, ValueError):
            pass

    # ISO sensitivity
    if result["iso"] is None:
        iso_value = exif_dict.get("ISOSpeedRatings") or exif_dict.get(
            "PhotographicSensitivity"
        )
        if iso_value is not None:
            try:
                if isinstance(iso_value, (list, tuple)):
                    result["iso"] = int(iso_value[0])
                else:
                    result["iso"] = int(iso_value)
            except (TypeError, ValueError, IndexError):
                pass

    # Exposure time
    if "ExposureTime" in exif_dict and result["exposure_time"] is None:
        result["exposure_time"] = _exif_ratio_to_float(exif_dict["ExposureTime"])

    # F-number (aperture)
    if "FNumber" in exif_dict and result["f_number"] is None:
        result["f_number"] = _exif_ratio_to_float(exif_dict["FNumber"])

    # Camera information
    if "Make" in exif_dict and result["camera_make"] is None:
        result["camera_make"] = str(exif_dict["Make"]).strip()
    if "Model" in exif_dict and result["camera_model"] is None:
        result["camera_model"] = str(exif_dict["Model"]).strip()

    # Lens information
    if result["lens_model"] is None:
        lens = exif_dict.get("LensModel") or exif_dict.get("LensSpecification")
        if lens is not None:
            result["lens_model"] = str(lens).strip()

    # Image resolution
    if result["image_width"] is None:
        width = exif_dict.get("ExifImageWidth") or exif_dict.get("ImageWidth")
        if width is not None:
            try:
                result["image_width"] = int(width)
            except (TypeError, ValueError):
                pass

    if result["image_height"] is None:
        height = exif_dict.get("ExifImageHeight") or exif_dict.get("ImageLength")
        if height is not None:
            try:
                result["image_height"] = int(height)
            except (TypeError, ValueError):
                pass

    final_count = sum(1 for v in result.values() if v is not None)
    return final_count - initial_count


def _exif_strategy_thumbnail(filepath: str, result: Dict[str, Any]) -> bool:
    """Strategy 1: Extract EXIF from embedded JPEG thumbnail.

    Args:
        filepath: Path to RAW file
        result: Result dictionary to populate (modified in place)

    Returns:
        True if any fields were extracted, False otherwise
    """
    logger.debug("Strategy 1: Extracting EXIF from embedded thumbnail")
    try:
        with rawpy.imread(filepath) as raw:
            thumb = raw.extract_thumb()
            if thumb.format != rawpy.ThumbFormat.JPEG:
                logger.debug(
                    "Strategy 1: Thumbnail is not JPEG (format=%s) for: %s",
                    thumb.format,
                    filepath,
                )
                return False

            from io import BytesIO

            img = Image.open(BytesIO(thumb.data))
            exif_data = img._getexif()

            if not exif_data:
                logger.debug("Strategy 1: Thumbnail EXIF is empty for: %s", filepath)
                return False

            exif_dict = {TAGS.get(tag_id, tag_id): v for tag_id, v in exif_data.items()}
            fields_added = _populate_exif_from_dict(result, exif_dict)
            logger.debug(
                "Strategy 1 succeeded: extracted %d fields from thumbnail", fields_added
            )
            return fields_added > 0

    except rawpy.LibRawNoThumbnailError:
        logger.debug("Strategy 1: No thumbnail available in RAW file: %s", filepath)
    except rawpy.LibRawUnsupportedThumbnailError:
        logger.debug("Strategy 1: Unsupported thumbnail format in: %s", filepath)
    except rawpy.LibRawError as e:
        logger.warning(
            "Strategy 1 failed: Cannot open RAW file %s: %s: %s",
            filepath,
            type(e).__name__,
            e,
        )
    except Exception as e:
        logger.debug("Strategy 1 failed for %s: %s: %s", filepath, type(e).__name__, e)
    return False


def _exif_strategy_pil_direct(filepath: str, result: Dict[str, Any]) -> bool:
    """Strategy 2: Open RAW file directly with PIL.

    Args:
        filepath: Path to RAW file
        result: Result dictionary to populate (modified in place)

    Returns:
        True if any fields were extracted, False otherwise
    """
    logger.debug("Strategy 2: Opening RAW file directly with PIL")
    try:
        img = Image.open(filepath)
        exif_data = img._getexif()
        if not exif_data:
            logger.debug("Strategy 2: No EXIF data from PIL for: %s", filepath)
            return False

        exif_dict = {TAGS.get(tag_id, tag_id): v for tag_id, v in exif_data.items()}
        fields_added = _populate_exif_from_dict(result, exif_dict)
        logger.debug(
            "Strategy 2 succeeded: added %d fields",
            fields_added,
        )
        return fields_added > 0

    except OSError as e:
        logger.debug("Strategy 2: PIL cannot open RAW format for %s: %s", filepath, e)
    except Exception as e:
        logger.debug("Strategy 2 failed for %s: %s: %s", filepath, type(e).__name__, e)
    return False


def _exif_strategy_rawpy_dimensions(filepath: str, result: Dict[str, Any]) -> bool:
    """Strategy 3: Get image dimensions from rawpy.

    Args:
        filepath: Path to RAW file
        result: Result dictionary to populate (modified in place)

    Returns:
        True if dimensions were extracted, False otherwise
    """
    logger.debug("Strategy 3: Getting dimensions from rawpy")
    try:
        with rawpy.imread(filepath) as raw:
            sizes = raw.sizes
            if result["image_width"] is None:
                result["image_width"] = sizes.raw_width
            if result["image_height"] is None:
                result["image_height"] = sizes.raw_height
            logger.debug(
                "Strategy 3 succeeded: %dx%d",
                result["image_width"],
                result["image_height"],
            )
            return True

    except rawpy.LibRawError as e:
        logger.warning(
            "Strategy 3 failed: Cannot read dimensions from %s: %s: %s",
            filepath,
            type(e).__name__,
            e,
        )
    except Exception as e:
        logger.warning(
            "Strategy 3 failed for %s: %s: %s", filepath, type(e).__name__, e
        )
    return False


def extract_exif_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract EXIF information from RAW file.

    This function uses multiple strategies to extract EXIF metadata:
    1. Extract from embedded JPEG thumbnail (most reliable for RAW files)
    2. Open RAW file directly with PIL (limited format support)
    3. Get image dimensions from rawpy

    The function is designed to never raise exceptions; any failures during
    extraction are logged and the function returns partial or empty results.

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

    Note:
        - Returns None for fields that could not be extracted
        - Does not raise exceptions; failures are logged at debug/warning level
    """
    logger.debug("Extracting EXIF metadata from: %s", filepath)

    result: Dict[str, Any] = {
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
        logger.debug("Pillow not available, skipping EXIF extraction for: %s", filepath)
        return result

    # Strategy 1: Extract EXIF from embedded thumbnail (most reliable)
    _exif_strategy_thumbnail(filepath, result)

    # Strategy 2: Try PIL direct open if we're missing focal length data
    if result["focal_length"] is None:
        _exif_strategy_pil_direct(filepath, result)

    # Strategy 3: Get dimensions from rawpy if still missing
    if result["image_width"] is None or result["image_height"] is None:
        _exif_strategy_rawpy_dimensions(filepath, result)

    # Log summary of extracted fields
    extracted_fields = [k for k, v in result.items() if v is not None]
    if extracted_fields:
        logger.debug(
            "EXIF extraction completed for %s: extracted fields: %s",
            filepath,
            ", ".join(extracted_fields),
        )
    else:
        logger.debug("EXIF extraction completed for %s: no fields extracted", filepath)

    return result
