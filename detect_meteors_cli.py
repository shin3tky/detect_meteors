#!/usr/bin/env python
#
# Detect Meteors CLI
# © 2025 Shinichi Morita (shin3tky)
#
# CLI entry point for meteor detection.
# This module handles argument parsing and delegates to meteor_core.
#

import os
import sys
import shlex
import argparse
import logging
from typing import Any, Dict, Optional, Tuple

from meteor_core import (
    VERSION,
    # Exceptions
    MeteorError,
    MeteorLoadError,
    MeteorValidationError,
    format_error_for_user,
    save_diagnostic_report,
    get_message,
    DEFAULT_TARGET_FOLDER,
    DEFAULT_OUTPUT_FOLDER,
    DEFAULT_DEBUG_FOLDER,
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_ASPECT_RATIO,
    DEFAULT_HOUGH_THRESHOLD,
    DEFAULT_HOUGH_MIN_LINE_LENGTH,
    DEFAULT_HOUGH_MAX_LINE_GAP,
    DEFAULT_MIN_LINE_SCORE,
    DEFAULT_ENABLE_ROI_SELECTION,
    DEFAULT_NUM_WORKERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_PROGRESS_FILE,
    DEFAULT_FISHEYE_MODEL,
    # Functions
    collect_files,
    extract_exif_metadata,
    load_and_bin_raw_fast,
    get_sensor_preset,
    apply_sensor_preset,
    validate_sensor_overrides,
    list_sensor_types,
    parse_roi_polygon_string,
    format_polygon_string,
    display_exif_info,
    display_fisheye_info,
    calculate_npf_metrics,
    optimize_params_with_npf,
    estimate_diff_threshold_from_samples,
    estimate_min_area_from_samples,
    estimate_min_line_score_from_image,
    select_roi,
    process_image_batch,
    compute_params_hash,
    ProgressManager,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    default_locale = os.environ.get("DETECT_METEORS_LOCALE", "en")
    parser = argparse.ArgumentParser(
        description="Meteor detection tool with comprehensive auto-parameter estimation"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Detect Meteors CLI (https://github.com/shin3tky/detect_meteors) {VERSION}",
    )

    parser.add_argument(
        "--locale",
        default=default_locale,
        help=("Locale code for CLI messages (default: DETECT_METEORS_LOCALE or 'en')."),
    )

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
        "--verbose",
        action="store_true",
        help="Show detailed diagnostic information on errors",
    )
    parser.add_argument(
        "--save-diagnostic",
        metavar="FILE",
        nargs="?",
        const="",
        type=str,
        default=None,
        help="Save diagnostic report to file on error (default: auto-generated name)",
    )
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

    # Auto-params
    parser.add_argument(
        "--auto-params",
        action="store_true",
        help="Auto-estimate diff_threshold, min_area, and min_line_score based on EXIF",
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
    parser.add_argument(
        "--fisheye",
        action="store_true",
        help="Enable fisheye lens correction. Applies equisolid angle projection compensation "
        "to account for varying effective focal length across the image. "
        "NPF calculation uses edge (worst case) effective focal length. "
        "Star trail estimation accounts for longer trails at image edges.",
    )

    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure logging for CLI execution.

    When verbose mode is enabled, DEBUG-level logs from input handling modules
    become visible to aid troubleshooting; otherwise keep the default warning
    noise level.
    """

    if not verbose:
        return

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )

    for logger_name in (
        "meteor_core.inputs",
        "meteor_core.image_io",
    ):
        logging.getLogger(logger_name).setLevel(logging.DEBUG)


def validate_and_apply_sensor_preset(args, verbose: bool = False) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[Dict[str, Any]],
]:
    """
    Validate and apply sensor preset settings.

    Consolidates the repeated sensor preset validation/application logic
    that was duplicated in handle_show_exif() and main().

    Args:
        args: Parsed argparse namespace
        verbose: If True, print which values are being used

    Returns:
        Tuple of (focal_factor, sensor_width, focal_length, pixel_pitch, preset)

    Raises:
        MeteorValidationError: If sensor validation fails.
    """
    # Validate --sensor-type
    if args.sensor_type and get_sensor_preset(args.sensor_type) is None:
        raise MeteorValidationError(
            f"⚠ Error: Invalid --sensor-type value: '{args.sensor_type}'\n"
            "  Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40",
            parameter_name="sensor_type",
            provided_value=args.sensor_type,
        )

    # Apply sensor preset
    (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    ) = apply_sensor_preset(args, verbose=verbose)

    # Validate sensor overrides
    validate_sensor_overrides(args, preset, sensor_width_value, pixel_pitch_value)

    # Validate focal_factor
    if args.focal_factor and focal_factor_value is None:
        raise MeteorValidationError(
            f"⚠ Error: Invalid --focal-factor value: '{args.focal_factor}'\n"
            "  Valid values: MFT, APS-C, APS-H, FF, or numeric (e.g., 2.0)",
            parameter_name="focal_factor",
            provided_value=args.focal_factor,
        )

    return (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    )


def prepare_exif_and_npf_analysis(
    filepath: str,
    focal_factor_value,
    focal_length_value,
    sensor_width_value,
    pixel_pitch_value,
    args_sensor_type,
    args_focal_factor,
    fisheye: bool = False,
    fisheye_model: str = DEFAULT_FISHEYE_MODEL,
):
    """
    Extract EXIF metadata and calculate NPF metrics.

    Consolidates the EXIF extraction and NPF calculation logic that was
    duplicated in handle_show_exif() and detect_meteors_advanced().

    Args:
        filepath: Path to the RAW file
        focal_factor_value: Parsed focal factor value
        focal_length_value: Parsed focal length value (from CLI)
        sensor_width_value: Sensor width in mm
        pixel_pitch_value: Pixel pitch in μm
        args_sensor_type: --sensor-type argument value
        args_focal_factor: --focal-factor argument value
        fisheye: Whether fisheye correction is enabled
        fisheye_model: Fisheye projection model

    Returns:
        Tuple of (exif_data, focal_length_source, npf_metrics)
    """
    exif_data = extract_exif_metadata(filepath)

    # Focal length priority
    focal_length_source = "Unknown"
    if focal_length_value:
        focal_length_source = "CLI (--focal-length)"
        exif_data["focal_length_35mm"] = focal_length_value
    elif exif_data.get("focal_length_35mm"):
        focal_length_source = "EXIF"
    elif exif_data.get("focal_length") and focal_factor_value:
        if args_focal_factor:
            focal_length_source = f"Calculated (--focal-factor {args_focal_factor})"
        else:
            focal_length_source = f"Calculated (--sensor-type {args_sensor_type})"
        exif_data["focal_length_35mm"] = exif_data["focal_length"] * focal_factor_value
    elif exif_data.get("focal_length"):
        focal_length_source = "EXIF (actual, no 35mm equiv.)"

    # NPF metrics calculation
    npf_metrics = None
    if exif_data.get("focal_length_35mm") and exif_data.get("f_number"):
        npf_metrics = calculate_npf_metrics(
            exif_data,
            sensor_width_mm=sensor_width_value,
            pixel_pitch_um=pixel_pitch_value,
            fisheye=fisheye,
            fisheye_model=fisheye_model,
        )

    return exif_data, focal_length_source, npf_metrics


def collect_exif_warnings(
    exif_data,
    focal_length_value,
    focal_factor_value,
    sensor_width_value,
    pixel_pitch_value,
    npf_metrics,
    check_npf: bool = True,
):
    """
    Collect warnings related to EXIF data and NPF calculation.

    Args:
        exif_data: EXIF metadata dictionary
        focal_length_value: CLI-specified focal length
        focal_factor_value: Parsed focal factor
        sensor_width_value: Sensor width in mm
        pixel_pitch_value: Pixel pitch in μm
        npf_metrics: NPF metrics dictionary
        check_npf: Whether to include NPF-related warnings

    Returns:
        List of warning strings
    """
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
            "35mm equivalent not found. Consider using --sensor-type or --focal-factor"
        )

    if not exif_data.get("iso"):
        warnings.append("ISO value not available")
    if not exif_data.get("exposure_time"):
        warnings.append("Exposure time not available")

    # NPF-related warnings
    if check_npf and npf_metrics:
        if not sensor_width_value and not exif_data.get("image_width"):
            warnings.append(
                "Sensor width not specified. Use --sensor-type or --sensor-width for accurate NPF calculation"
            )
        if not npf_metrics.get("has_complete_data"):
            warnings.append("NPF calculation using default/estimated values")

    return warnings


def print_warnings(warnings):
    """Print warnings in a formatted box."""
    if warnings:
        print(f"{'='*60}")
        print("⚠ Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
        print(f"{'='*60}\n")


def handle_show_exif(args) -> None:
    """Handle --show-exif / --show-npf commands.

    Raises:
        MeteorError: On any error during EXIF extraction or display.
    """
    print(f"\n{'='*60}")
    if args.show_npf:
        print("EXIF Metadata & NPF Rule Analysis")
    else:
        print("EXIF Metadata Viewer")
    print(f"{'='*60}\n")
    print(f"Target folder: {args.target}")

    # Validate and apply sensor preset
    (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    ) = validate_and_apply_sensor_preset(args, verbose=True)

    # Collect files (raises MeteorLoadError if directory doesn't exist)
    files = collect_files(args.target)
    if not files:
        raise MeteorLoadError(
            "No RAW files found in target folder",
            filepath=args.target,
            context={"error_category": "no_files"},
        )

    print(f"Found {len(files)} RAW files")
    print(f"Reading EXIF from first file: {os.path.basename(files[0])}\n")

    # Extract EXIF and calculate NPF metrics
    exif_data, focal_length_source, npf_metrics = prepare_exif_and_npf_analysis(
        filepath=files[0],
        focal_factor_value=focal_factor_value,
        focal_length_value=focal_length_value,
        sensor_width_value=sensor_width_value,
        pixel_pitch_value=pixel_pitch_value,
        args_sensor_type=args.sensor_type,
        args_focal_factor=args.focal_factor,
        fisheye=args.fisheye,
        fisheye_model=DEFAULT_FISHEYE_MODEL,
    )

    # For --show-npf, force NPF calculation even if data is incomplete
    if args.show_npf and npf_metrics is None:
        npf_metrics = calculate_npf_metrics(
            exif_data,
            sensor_width_mm=sensor_width_value,
            pixel_pitch_um=pixel_pitch_value,
            fisheye=args.fisheye,
            fisheye_model=DEFAULT_FISHEYE_MODEL,
        )

    # Display fisheye info
    if args.fisheye and exif_data.get("focal_length_35mm"):
        display_fisheye_info(exif_data["focal_length_35mm"], DEFAULT_FISHEYE_MODEL)

    display_exif_info(exif_data, focal_length_source, focal_factor_value, npf_metrics)

    # Collect and print warnings
    warnings = collect_exif_warnings(
        exif_data=exif_data,
        focal_length_value=focal_length_value,
        focal_factor_value=focal_factor_value,
        sensor_width_value=sensor_width_value,
        pixel_pitch_value=pixel_pitch_value,
        npf_metrics=npf_metrics,
        check_npf=(args.show_npf or npf_metrics is not None),
    )
    print_warnings(warnings)


def _init_worker_ignore_interrupt() -> None:
    """Ignore SIGINT in worker processes. Must be at module level for pickling."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _validate_directories(
    target_folder: str, output_folder: str, debug_folder: str
) -> None:
    """
    Validate and create directories for processing.

    Args:
        target_folder: Input folder with RAW files
        output_folder: Output folder for candidates
        debug_folder: Folder for debug images

    Raises:
        MeteorValidationError: If target and output directories are the same.
        MeteorLoadError: If directories cannot be created.
    """
    import os

    # Safety check: prevent overwriting source files
    target_fullpath = os.path.abspath(target_folder)
    output_fullpath = os.path.abspath(output_folder)

    if target_fullpath == output_fullpath:
        raise MeteorValidationError(
            "Target and output directories cannot be the same",
            parameter_name="output_folder",
            provided_value=output_folder,
            expected="a different directory than target_folder",
            context={
                "target_folder": target_folder,
                "output_folder": output_folder,
                "resolved_path": target_fullpath,
            },
        )

    try:
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(debug_folder, exist_ok=True)
    except OSError as e:
        raise MeteorLoadError(
            f"Failed to create output directory: {e}",
            filepath=output_folder if "output" in str(e).lower() else debug_folder,
            original_error=e,
            context={"error_category": "directory_creation"},
        ) from e


def _setup_roi(
    prev_img,
    roi_polygon_cli,
    enable_roi_selection: bool,
):
    """
    Set up ROI (Region of Interest) mask.

    Args:
        prev_img: First loaded image (numpy array)
        roi_polygon_cli: ROI polygon from CLI argument
        enable_roi_selection: Whether to enable interactive selection

    Returns:
        Tuple of (roi_mask, roi_polygon)
    """
    import cv2
    import numpy as np

    height, width = prev_img.shape
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

    return roi_mask, roi_polygon


def _run_auto_params(
    files,
    prev_img,
    roi_mask,
    diff_threshold: int,
    min_area: int,
    min_line_score: float,
    user_specified_diff_threshold: bool,
    user_specified_min_area: bool,
    user_specified_min_line_score: bool,
    focal_length_mm,
    focal_factor,
    sensor_width_mm,
    pixel_pitch_um,
    fisheye: bool,
):
    """
    Run automatic parameter estimation based on EXIF and NPF Rule.

    Args:
        files: List of RAW file paths
        prev_img: First loaded image
        roi_mask: ROI mask
        diff_threshold: Current diff_threshold value
        min_area: Current min_area value
        min_line_score: Current min_line_score value
        user_specified_*: Flags for user-specified values
        focal_length_mm: Focal length from CLI
        focal_factor: Focal factor value
        sensor_width_mm: Sensor width in mm
        pixel_pitch_um: Pixel pitch in μm
        fisheye: Whether fisheye correction is enabled

    Returns:
        Tuple of (diff_threshold, min_area, min_line_score, focal_length_mm)
    """
    print(f"\n{'='*60}")
    print("Auto-params: NPF Rule-based Optimization")
    print(f"{'='*60}\n")

    # Step 1: EXIF Information Extraction
    exif_data = extract_exif_metadata(files[0])

    # Focal length acquisition (priority)
    focal_length_source = "Unknown"
    if focal_length_mm:
        focal_length_source = "CLI (--focal-length)"
        exif_data["focal_length_35mm"] = focal_length_mm
    elif exif_data.get("focal_length_35mm"):
        focal_length_mm = exif_data["focal_length_35mm"]
        focal_length_source = "EXIF"
    elif exif_data.get("focal_length") and focal_factor:
        focal_length_mm = exif_data["focal_length"] * focal_factor
        exif_data["focal_length_35mm"] = focal_length_mm
        focal_length_source = f"Calculated (EXIF {exif_data['focal_length']:.1f}mm × factor {focal_factor})"
    elif exif_data.get("focal_length"):
        focal_length_mm = exif_data["focal_length"]
        focal_length_source = "EXIF (actual, no 35mm equiv.)"

    # Step 2: NPF Rule Analysis
    npf_metrics = calculate_npf_metrics(
        exif_data,
        sensor_width_mm=sensor_width_mm,
        pixel_pitch_um=pixel_pitch_um,
        fisheye=fisheye,
        fisheye_model=DEFAULT_FISHEYE_MODEL,
    )

    # Display fisheye info if enabled
    if fisheye and exif_data.get("focal_length_35mm"):
        display_fisheye_info(exif_data["focal_length_35mm"], DEFAULT_FISHEYE_MODEL)

    # Display EXIF Information and NPF Analysis
    display_exif_info(exif_data, focal_length_source, focal_factor, npf_metrics)

    # Step 3: Display warnings
    _display_auto_params_warnings(
        exif_data,
        focal_length_mm,
        focal_factor,
        sensor_width_mm,
        pixel_pitch_um,
        npf_metrics,
    )

    # Step 4: NPF-based parameter optimization
    if npf_metrics and npf_metrics.get("npf_recommended_sec"):
        diff_threshold, min_area, min_line_score = _optimize_with_npf(
            exif_data,
            npf_metrics,
            diff_threshold,
            min_area,
            min_line_score,
            user_specified_diff_threshold,
            user_specified_min_area,
            user_specified_min_line_score,
        )
    else:
        diff_threshold, min_area, min_line_score = _optimize_with_legacy(
            files,
            prev_img,
            roi_mask,
            diff_threshold,
            min_area,
            min_line_score,
            user_specified_diff_threshold,
            user_specified_min_area,
            user_specified_min_line_score,
            focal_length_mm,
        )

    return diff_threshold, min_area, min_line_score, focal_length_mm


def _display_auto_params_warnings(
    exif_data,
    focal_length_mm,
    focal_factor,
    sensor_width_mm,
    pixel_pitch_um,
    npf_metrics,
):
    """Display warnings for auto-params mode."""
    warnings = []

    if not focal_length_mm:
        warnings.append("Focal length not available from EXIF")
        warnings.append("  → Consider using --focal-length option")
    elif not exif_data.get("focal_length_35mm") and not focal_factor:
        warnings.append(
            f"35mm equivalent not found in EXIF (using actual: {focal_length_mm:.1f}mm)"
        )
        warnings.append("  → For crop sensor cameras, use --focal-factor")

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


def _optimize_with_npf(
    exif_data,
    npf_metrics,
    diff_threshold,
    min_area,
    min_line_score,
    user_specified_diff_threshold,
    user_specified_min_area,
    user_specified_min_line_score,
):
    """Optimize parameters using NPF Rule."""
    print(f"{'='*60}")
    print("Parameter Optimization (NPF Rule-based)")
    print(f"{'='*60}\n")

    diff_threshold, min_area, min_line_score, opt_info = optimize_params_with_npf(
        exif_data,
        npf_metrics,
        user_specified_diff_threshold=user_specified_diff_threshold,
        user_specified_min_area=user_specified_min_area,
        user_specified_min_line_score=user_specified_min_line_score,
        current_diff_threshold=diff_threshold,
        current_min_area=min_area,
        current_min_line_score=min_line_score,
    )

    print(
        f"Shooting Quality Score: {opt_info['quality_score']:.2f} ({opt_info['quality_level']})"
    )

    if opt_info["adjustments"]:
        print("\nParameter Adjustments:")
        for adjustment in opt_info["adjustments"]:
            print(f"  • {adjustment}")
    else:
        print("\nNo automatic adjustments (all parameters user-specified)")

    print(f"\n{'='*60}\n")

    return diff_threshold, min_area, min_line_score


def _optimize_with_legacy(
    files,
    prev_img,
    roi_mask,
    diff_threshold,
    min_area,
    min_line_score,
    user_specified_diff_threshold,
    user_specified_min_area,
    user_specified_min_line_score,
    focal_length_mm,
):
    """Fallback to legacy parameter estimation method."""
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

    return diff_threshold, min_area, min_line_score


def _print_processing_params(processing_params):
    """Print processing parameters in a formatted box."""
    print(f"\n{'='*50}")
    print("Processing Parameters:")
    print(f"{'='*50}")
    print(f"  diff_threshold:        {processing_params['diff_threshold']}")
    print(f"  min_area:              {processing_params['min_area']}")
    print(f"  min_aspect_ratio:      {processing_params['min_aspect_ratio']}")
    print(f"  hough_threshold:       {processing_params['hough_threshold']}")
    print(f"  hough_min_line_length: {processing_params['hough_min_line_length']}")
    print(f"  hough_max_line_gap:    {processing_params['hough_max_line_gap']}")
    print(f"  min_line_score:        {processing_params['min_line_score']:.1f}")
    print(f"{'='*50}\n")


def _save_candidate_file(
    filepath: str,
    filename: str,
    output_folder: str,
    debug_folder: str,
    debug_img,
    roi_polygon,
    output_overwrite: bool,
) -> bool:
    """
    Save a candidate file and its debug image.

    Args:
        filepath: Source file path
        filename: Output filename
        output_folder: Destination folder for RAW file
        debug_folder: Destination folder for debug image
        debug_img: Debug visualization image (or None)
        roi_polygon: ROI polygon to draw on debug image
        output_overwrite: Whether to overwrite existing files

    Returns:
        True if file was saved, False if skipped
    """
    import os
    import shutil
    import cv2
    import numpy as np

    output_path = os.path.join(output_folder, filename)

    if os.path.exists(output_path) and not output_overwrite:
        return False

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

    return True


def _run_parallel_detection(
    image_pairs,
    roi_mask,
    processing_params,
    roi_polygon,
    output_folder,
    debug_folder,
    output_overwrite,
    num_workers,
    batch_size,
    resume_offset,
    overall_total,
    record_result_callback,
):
    """
    Run detection in parallel using ProcessPoolExecutor.

    Args:
        image_pairs: List of (current_file, previous_file) tuples
        roi_mask: ROI mask
        processing_params: Detection parameters
        roi_polygon: ROI polygon for debug visualization
        output_folder: Output folder for candidates
        debug_folder: Folder for debug images
        output_overwrite: Whether to overwrite existing files
        num_workers: Number of parallel workers
        batch_size: Batch size for processing
        resume_offset: Offset for progress display
        overall_total: Total number of files for progress display
        record_result_callback: Callback to record results (filename, is_candidate, score, lines, ratio)

    Returns:
        Number of detected candidates
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    batches = [
        image_pairs[i : i + batch_size] for i in range(0, len(image_pairs), batch_size)
    ]
    print(f"Number of batches: {len(batches)}")

    executor = ProcessPoolExecutor(
        max_workers=num_workers, initializer=_init_worker_ignore_interrupt
    )
    futures = []
    wait_for_tasks = True
    detected_count = 0

    try:
        for batch in batches:
            future = executor.submit(
                process_image_batch, batch, roi_mask, processing_params
            )
            futures.append(future)

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
                        saved = _save_candidate_file(
                            filepath,
                            filename,
                            output_folder,
                            debug_folder,
                            debug_img,
                            roi_polygon,
                            output_overwrite,
                        )
                        if saved:
                            print(f"  [HIT] {filename}: Ratio={aspect_ratio:.2f}")
                        else:
                            print(
                                f"  [SKIP] {filename}: Already exists in output folder"
                            )
                    else:
                        print(
                            f"\rChecking... {resume_offset + processed}/{overall_total}",
                            end="",
                            flush=True,
                        )

                    detected_count = record_result_callback(
                        filename, is_candidate, line_score, num_lines, aspect_ratio
                    )

            except Exception as e:
                print(f"\nBatch processing error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cancelling worker processes...")
        wait_for_tasks = False
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=wait_for_tasks, cancel_futures=not wait_for_tasks)

    return detected_count


def _run_sequential_detection(
    image_pairs,
    roi_mask,
    processing_params,
    roi_polygon,
    output_folder,
    debug_folder,
    output_overwrite,
    resume_offset,
    overall_total,
    record_result_callback,
):
    """
    Run detection sequentially (single-threaded).

    Args:
        image_pairs: List of (current_file, previous_file) tuples
        roi_mask: ROI mask
        processing_params: Detection parameters
        roi_polygon: ROI polygon for debug visualization
        output_folder: Output folder for candidates
        debug_folder: Folder for debug images
        output_overwrite: Whether to overwrite existing files
        resume_offset: Offset for progress display
        overall_total: Total number of files for progress display
        record_result_callback: Callback to record results (filename, is_candidate, score, lines, ratio)

    Returns:
        Number of detected candidates
    """
    detected_count = 0

    for idx, pair in enumerate(image_pairs):
        current_index = resume_offset + idx + 1
        current_file = os.path.basename(pair[0])
        progress_line_active = True

        print(
            f"\rProcessing {current_index}/{overall_total}: {current_file}",
            end="",
            flush=True,
        )

        batch_results = process_image_batch([pair], roi_mask, processing_params)

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
                print(f"  [LINE] {filename}: score={line_score:.1f}, lines={num_lines}")

            if is_candidate:
                if progress_line_active:
                    print()
                    progress_line_active = False

                saved = _save_candidate_file(
                    filepath,
                    filename,
                    output_folder,
                    debug_folder,
                    debug_img,
                    roi_polygon,
                    output_overwrite,
                )
                if saved:
                    print(f"  [HIT] {filename}: Ratio={aspect_ratio:.2f}")
                else:
                    print(
                        f"  [SKIP] {filename}: Already exists in output folder (use --output-overwrite to overwrite)"
                    )
            else:
                print(
                    f"\rChecking... {current_index}/{overall_total}",
                    end="",
                    flush=True,
                )
                progress_line_active = True

            detected_count = record_result_callback(
                filename, is_candidate, line_score, num_lines, aspect_ratio
            )

    return detected_count


def detect_meteors_advanced(
    target_folder: str,
    output_folder: str,
    debug_folder: str,
    diff_threshold: int,
    min_area: int,
    min_aspect_ratio: float,
    hough_threshold: int,
    hough_min_line_length: int,
    hough_max_line_gap: int,
    min_line_score: float,
    enable_roi_selection: bool,
    roi_polygon_cli,
    num_workers: int,
    batch_size: int,
    auto_batch_size: bool,
    enable_parallel: bool,
    profile: bool,
    validate_raw: bool,
    progress_file: str,
    resume: bool,
    auto_params: bool,
    user_specified_diff_threshold: bool,
    user_specified_min_area: bool,
    user_specified_min_line_score: bool,
    focal_length_mm,
    focal_factor,
    sensor_width_mm,
    pixel_pitch_um,
    output_overwrite: bool,
    fisheye: bool,
    cli_param_string: str,
    locale: str = "en",
) -> int:
    """
    Main processing: detect meteor candidates from consecutive RAW images.

    This is the main entry point that orchestrates the detection workflow.
    """
    import time

    timing = {}
    t_total = time.time()

    # Validate and create directories (raises MeteorError on failure)
    _validate_directories(target_folder, output_folder, debug_folder)

    # Collect files (raises MeteorLoadError if directory doesn't exist or is empty)
    print(f"Collecting RAW files from: {target_folder}")
    files = collect_files(target_folder)

    if len(files) < 2:
        raise MeteorValidationError(
            "Need at least 2 images for meteor detection",
            parameter_name="target_folder",
            provided_value=target_folder,
            expected="directory with at least 2 RAW files",
            context={"files_found": len(files)},
        )

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

    # ROI setup
    roi_mask, roi_polygon = _setup_roi(prev_img, roi_polygon_cli, enable_roi_selection)

    # Auto-parameter estimation
    if auto_params:
        diff_threshold, min_area, min_line_score, focal_length_mm = _run_auto_params(
            files=files,
            prev_img=prev_img,
            roi_mask=roi_mask,
            diff_threshold=diff_threshold,
            min_area=min_area,
            min_line_score=min_line_score,
            user_specified_diff_threshold=user_specified_diff_threshold,
            user_specified_min_area=user_specified_min_area,
            user_specified_min_line_score=user_specified_min_line_score,
            focal_length_mm=focal_length_mm,
            focal_factor=focal_factor,
            sensor_width_mm=sensor_width_mm,
            pixel_pitch_um=pixel_pitch_um,
            fisheye=fisheye,
        )

    # Build processing parameters
    processing_params = {
        "diff_threshold": diff_threshold,
        "min_area": min_area,
        "min_aspect_ratio": min_aspect_ratio,
        "hough_threshold": hough_threshold,
        "hough_min_line_length": hough_min_line_length,
        "hough_max_line_gap": hough_max_line_gap,
        "min_line_score": min_line_score,
    }

    _print_processing_params(processing_params)

    # Progress tracking setup
    params_for_hash = processing_params.copy()
    if roi_polygon:
        params_for_hash["roi_polygon"] = roi_polygon

    progress_manager = ProgressManager(progress_file)
    loaded_progress = progress_manager.load() if resume else False

    params_hash = compute_params_hash(params_for_hash)
    if loaded_progress and progress_manager.get_params_hash() == params_hash:
        print(
            f"Resuming from progress file: {progress_file} "
            f"(processed={progress_manager.get_total_processed()}, "
            f"detected={progress_manager.get_total_detected()})"
        )
    else:
        if loaded_progress:
            print(
                "Progress file exists but parameters differ. Starting with fresh progress."
            )
        progress_manager.reset()

    progress_manager.set_params_hash(params_hash)
    progress_manager.set_params(cli_param_string)
    progress_manager.set_roi(roi_polygon or "full_image")
    progress_manager.set_processing_params(processing_params)

    existing_basenames = {os.path.basename(path) for path in files}
    progress_manager.filter_existing_files(existing_basenames)
    progress_manager.save()

    processed_set = progress_manager.processed_set
    detected_set = progress_manager.detected_set

    def record_result(
        filename: str,
        is_candidate: bool,
        score: float = 0.0,
        lines: int = 0,
        ratio: float = 0.0,
    ) -> int:
        """Record result and return current detected count."""
        return progress_manager.record_result(
            filename, is_candidate, score, lines, ratio
        )

    # Build image pairs and filter already processed
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
            detected_count = _run_parallel_detection(
                image_pairs=image_pairs,
                roi_mask=roi_mask,
                processing_params=processing_params,
                roi_polygon=roi_polygon,
                output_folder=output_folder,
                debug_folder=debug_folder,
                output_overwrite=output_overwrite,
                num_workers=num_workers,
                batch_size=batch_size,
                resume_offset=resume_offset,
                overall_total=overall_total,
                record_result_callback=record_result,
            )
        else:
            detected_count = _run_sequential_detection(
                image_pairs=image_pairs,
                roi_mask=roi_mask,
                processing_params=processing_params,
                roi_polygon=roi_polygon,
                output_folder=output_folder,
                debug_folder=debug_folder,
                output_overwrite=output_overwrite,
                resume_offset=resume_offset,
                overall_total=overall_total,
                record_result_callback=record_result,
            )
    except KeyboardInterrupt:
        print(
            "\n"
            + get_message(
                "interrupt.progress",
                locale=locale,
                progress_file=progress_file,
            )
        )
        progress_manager.save()
        return detected_count

    # Performance profiling
    if profile:
        timing["processing"] = time.time() - t_process
        timing["total"] = time.time() - t_total

        print("\n\n=== Performance Profile ===")
        print(f"First image load: {timing['first_load']:.3f}s")
        print(f"Processing time: {timing['processing']:.3f}s")
        print(f"Total time: {timing['total']:.3f}s")
        print(f"Images processed: {len(image_pairs)}")
        if image_pairs:
            print(f"Average per image: {timing['processing'] / len(image_pairs):.3f}s")

    print(f"\nComplete! {detected_count} candidates extracted")
    return detected_count


def main():
    """Main entry point for the CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()
    cli_param_string = shlex.join(sys.argv[1:])

    args.locale = args.locale or os.environ.get("DETECT_METEORS_LOCALE", "en")
    locale = args.locale

    _configure_logging(args.verbose)

    try:
        if args.remove_progress:
            if os.path.exists(args.progress_file):
                os.remove(args.progress_file)
                print(
                    get_message(
                        "progress.removed",
                        locale=locale,
                        path=args.progress_file,
                    )
                )
            else:
                print(
                    get_message(
                        "progress.not_found",
                        locale=locale,
                        path=args.progress_file,
                    )
                )
            return

        # --list-sensor-types: Display available sensor types and exit
        if args.list_sensor_types:
            list_sensor_types()
            return

        # --show-exif or --show-npf: Display EXIF info and NPF analysis then exit
        if args.show_exif or args.show_npf:
            handle_show_exif(args)
            return

        _run_main(args, cli_param_string)
    except MeteorError as e:
        # Handle meteor_core exceptions with user-friendly output
        print(
            format_error_for_user(e, verbose=args.verbose, locale=locale),
            file=sys.stderr,
        )

        # Save diagnostic report if requested
        if args.save_diagnostic is not None:
            # Use provided path or auto-generate
            diag_path = args.save_diagnostic if args.save_diagnostic else None
            saved_path = save_diagnostic_report(e, diag_path)
            print(
                get_message("diagnostic.report.saved", locale=locale, path=saved_path),
                file=sys.stderr,
            )
        elif not args.verbose:
            # Hint about diagnostic options
            print(
                get_message("diagnostic.hint.save", locale=locale),
                file=sys.stderr,
            )

        sys.exit(1)
    except KeyboardInterrupt:
        print(
            "\n" + get_message("interrupt.generic", locale=locale),
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception as e:
        # Unexpected errors - show traceback in verbose mode
        if args.verbose:
            import traceback

            traceback.print_exc()
        else:
            print(
                "\n"
                + get_message(
                    "error.unexpected",
                    locale=locale,
                    error_type=type(e).__name__,
                    error_message=e,
                ),
                file=sys.stderr,
            )
            print(
                get_message("error.unexpected.hint", locale=locale),
                file=sys.stderr,
            )
        sys.exit(1)


def _run_main(args, cli_param_string: str):
    """Run the main detection logic.

    This function contains the core logic extracted from main() to allow
    proper exception handling at the top level.

    Args:
        args: Parsed command line arguments.
        cli_param_string: Original CLI parameter string.

    Raises:
        MeteorError: On any meteor_core related errors.
    """
    locale = getattr(args, "locale", "en")
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

    # Validate and apply sensor preset
    (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    ) = validate_and_apply_sensor_preset(args, verbose=False)

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
        fisheye=args.fisheye,
        cli_param_string=cli_param_string,
        locale=locale,
    )


if __name__ == "__main__":
    main()
