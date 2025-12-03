"""Application runner for detect_meteors CLI."""

import os
import sys
from typing import Any, Dict, List

from detect_meteors_cli import (
    DEFAULT_ENABLE_ROI_SELECTION,
    DEFAULT_FISHEYE_MODEL,
    SENSOR_PRESETS,
    apply_sensor_preset,
    calculate_npf_metrics,
    collect_files,
    detect_meteors_advanced,
    extract_exif_metadata,
    get_sensor_preset,
    parse_roi_polygon_string,
    validate_sensor_overrides,
)

from detect_meteors import exif as exif_utils
from detect_meteors import npf


def _sensor_type_listing() -> Dict[str, Any]:
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

    listing: List[Dict[str, Any]] = []
    for sensor_type in primary_types:
        preset = SENSOR_PRESETS.get(sensor_type)
        if preset:
            listing.append(
                {
                    "type": sensor_type,
                    "description": preset["description"],
                    "focal_factor": preset["focal_factor"],
                    "sensor_width": preset["sensor_width"],
                    "pixel_pitch": preset["pixel_pitch"],
                }
            )

    aliases = {
        "1INCH": ["1-INCH", "1_INCH"],
        "APSC": ["APS-C", "APS_C"],
        "APSC_CANON": ["APS-C_CANON"],
        "APSH": ["APS-H", "APS_H"],
        "FF": ["FULLFRAME"],
        "MF44X33": ["MF44-33", "MF44_33"],
        "MF54X40": ["MF54-40", "MF54_40"],
    }

    return {"list": listing, "aliases": aliases}


def run(args):
    """Execute the application logic using parsed arguments."""

    if args.remove_progress:
        removed = False
        if os.path.exists(args.progress_file):
            os.remove(args.progress_file)
            removed = True
        return {
            "action": "remove_progress",
            "progress_file": args.progress_file,
            "removed": removed,
        }

    if args.list_sensor_types:
        return {"action": "list_sensor_types", "data": _sensor_type_listing()}

    roi_polygon_cli = None
    enable_roi_selection = DEFAULT_ENABLE_ROI_SELECTION

    if args.roi is not None:
        roi_polygon_cli = parse_roi_polygon_string(args.roi)
        enable_roi_selection = False
    elif args.no_roi:
        enable_roi_selection = False

    user_specified_diff_threshold = "--diff-threshold" in sys.argv
    user_specified_min_area = "--min-area" in sys.argv
    user_specified_min_line_score = "--min-line-score" in sys.argv

    if args.sensor_type and get_sensor_preset(args.sensor_type) is None:
        raise ValueError(
            f"Invalid --sensor-type value: '{args.sensor_type}'. "
            "Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40"
        )

    (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    ) = apply_sensor_preset(args, verbose=False)

    warnings = validate_sensor_overrides(
        args, preset, sensor_width_value, pixel_pitch_value, collect_only=True
    )

    if args.focal_factor and focal_factor_value is None:
        raise ValueError(
            f"Invalid --focal-factor value: '{args.focal_factor}'. "
            "Valid values: MFT, APS-C, APS-H, FF, or numeric (e.g., 2.0)"
        )

    if args.show_exif or args.show_npf:
        files = collect_files(args.target)
        if not files:
            raise FileNotFoundError("No RAW files found in target folder.")

        exif_data = extract_exif_metadata(files[0])

        focal_length_source = "Unknown"
        if focal_length_value:
            focal_length_source = "CLI (--focal-length)"
            exif_data["focal_length_35mm"] = focal_length_value
        elif exif_data.get("focal_length_35mm"):
            focal_length_source = "EXIF"
        elif exif_data.get("focal_length") and focal_factor_value:
            if args.focal_factor:
                focal_length_source = f"Calculated (--focal-factor {args.focal_factor})"
            else:
                focal_length_source = f"Calculated (--sensor-type {args.sensor_type})"
            exif_data["focal_length_35mm"] = exif_data["focal_length"] * focal_factor_value
        elif exif_data.get("focal_length"):
            focal_length_source = "EXIF (actual, no 35mm equiv.)"

        npf_metrics = None
        if args.show_npf or (
            exif_data.get("focal_length_35mm") and exif_data.get("f_number")
        ):
            npf_metrics = calculate_npf_metrics(
                exif_data,
                sensor_width_mm=sensor_width_value,
                pixel_pitch_um=pixel_pitch_value,
                fisheye=args.fisheye,
                fisheye_model=DEFAULT_FISHEYE_MODEL,
            )

        if args.auto_params and (
            exif_data.get("focal_length_35mm") is None
            or exif_data.get("exposure_time") is None
            or exif_data.get("f_number") is None
        ):
            warnings.append(
                "Auto-parameter estimation requires focal_length_35mm, exposure_time, and f_number"
            )
        if (
            exif_data.get("focal_length_35mm") is None
            and exif_data.get("focal_length") is None
            and args.focal_factor is None
            and args.sensor_type is None
        ):
            warnings.append(
                "35mm equivalent not found. Consider using --sensor-type or --focal-factor"
            )
        if not exif_data.get("iso"):
            warnings.append("ISO value not available")
        if not exif_data.get("exposure_time"):
            warnings.append("Exposure time not available")

        warnings.extend(
            npf.build_warnings(
                exif_data=exif_data,
                npf_metrics=npf_metrics,
                auto_params=args.auto_params,
                sensor_type=args.sensor_type,
                focal_factor_arg=args.focal_factor,
                sensor_width_value=sensor_width_value,
                show_npf=args.show_npf,
            )
        )

        fisheye_text = None
        if args.fisheye and exif_data.get("focal_length_35mm"):
            fisheye_text = exif_utils.format_fisheye_info(
                exif_data["focal_length_35mm"], DEFAULT_FISHEYE_MODEL
            )

        return {
            "action": "show_exif",
            "target": args.target,
            "files_found": len(files),
            "first_file": os.path.basename(files[0]),
            "exif_text": exif_utils.format_exif_info(
                exif_data,
                focal_length_source,
                focal_factor_value,
                npf_metrics,
            ),
            "fisheye_text": fisheye_text,
            "warnings": warnings,
            "show_usage_examples": args.show_npf,
        }

    detected_count = detect_meteors_advanced(
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
        fisheye=args.fisheye,
    )

    return {"action": "detect", "detected_count": detected_count, "warnings": warnings}
