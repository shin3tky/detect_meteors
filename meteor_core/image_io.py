#!/usr/bin/env python
#
# Detect Meteors CLI - Image I/O
# © 2025 Shinichi Morita (shin3tky)
#

"""
Image loading and EXIF extraction functions.
"""

import logging
from typing import Any, Dict

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

    logger.debug("Loading RAW file: %s", filepath)

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

            logger.debug("Successfully loaded and binned: %s", filepath)
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


def extract_exif_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract EXIF information from RAW file.

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
