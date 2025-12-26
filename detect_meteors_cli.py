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
import json
import shlex
import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from meteor_core import (
    VERSION,
    # Exceptions
    MeteorError,
    MeteorLoadError,
    MeteorValidationError,
    MeteorConfigError,
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
    PipelineConfig,
    HookConfig,
    DetectionParams,
    normalize_hook_configs,
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
    DetectionResult,
    MeteorDetectionPipeline,
    load_pipeline_config,
    SUPPORTED_CONFIG_EXTENSIONS,
)
from meteor_core.utils import _display_width, _pad_label


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
        "--config",
        type=str,
        default=None,
        help=(
            "Path to pipeline configuration file (YAML/JSON). "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_CONFIG_EXTENSIONS))}."
        ),
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
        help="Folder to save mask/debug images (used with --debug-image)",
    )
    debug_image_group = parser.add_mutually_exclusive_group()
    debug_image_group.add_argument(
        "--debug-image",
        dest="debug_image",
        action="store_true",
        help="Enable saving mask/debug images to --debug-dir",
    )
    debug_image_group.add_argument(
        "--no-debug-image",
        dest="debug_image",
        action="store_false",
        help="Disable saving mask/debug images to --debug-dir (default)",
    )
    parser.set_defaults(debug_image=False)

    parser.add_argument(
        "--input-loader",
        type=str,
        default=None,
        help="Input loader plugin name (overrides config file if provided)",
    )
    parser.add_argument(
        "--input-loader-config",
        type=str,
        default=None,
        help="Input loader config (JSON/YAML string or file path)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=None,
        help="Detector plugin name (overrides config file if provided)",
    )
    parser.add_argument(
        "--detector-config",
        type=str,
        default=None,
        help="Detector config (JSON/YAML string or file path)",
    )
    parser.add_argument(
        "--output-handler",
        type=str,
        default=None,
        help="Output handler plugin name (overrides config file if provided)",
    )
    parser.add_argument(
        "--output-handler-config",
        type=str,
        default=None,
        help="Output handler config (JSON/YAML string or file path)",
    )
    parser.add_argument(
        "--hooks",
        type=str,
        default=None,
        help="Comma-separated hook plugin names (execution order)",
    )
    parser.add_argument(
        "--hook-config",
        type=str,
        default=None,
        help="Hook config (JSON/YAML list or mapping, or file path)",
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
    auto_batch_group = parser.add_mutually_exclusive_group()
    auto_batch_group.add_argument(
        "--auto-batch-size",
        dest="auto_batch_size",
        action="store_true",
        help="Auto-adjust batch size for memory",
    )
    auto_batch_group.add_argument(
        "--no-auto-batch-size",
        dest="auto_batch_size",
        action="store_false",
        help="Disable auto-adjusted batch size",
    )
    parser.set_defaults(auto_batch_size=None)
    parallel_group = parser.add_mutually_exclusive_group()
    parallel_group.add_argument(
        "--parallel",
        dest="enable_parallel",
        action="store_true",
        help="Enable parallel processing",
    )
    parallel_group.add_argument(
        "--no-parallel",
        dest="enable_parallel",
        action="store_false",
        help="Disable parallel processing",
    )
    parser.set_defaults(enable_parallel=True)
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


def _flag_present(*flags: str) -> bool:
    """Return True if any of the CLI flags were provided."""
    return any(flag in sys.argv for flag in flags)


def _parse_config_payload(value: Optional[str], arg_name: str) -> Any:
    if value is None:
        return None
    if value == "":
        return {}

    path = Path(value)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_CONFIG_EXTENSIONS:
            raise MeteorConfigError(
                "Unsupported configuration file format",
                filepath=str(path),
                config_key=arg_name,
                context={"supported_extensions": sorted(SUPPORTED_CONFIG_EXTENSIONS)},
            )
        if suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise MeteorConfigError(
                    "Invalid JSON configuration",
                    filepath=str(path),
                    config_key=arg_name,
                    original_error=exc,
                ) from exc
        else:
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise MeteorConfigError(
                    "Invalid YAML configuration",
                    filepath=str(path),
                    config_key=arg_name,
                    original_error=exc,
                ) from exc
        return data if data is not None else {}

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise MeteorConfigError(
                "Invalid configuration string",
                config_key=arg_name,
                original_error=exc,
            ) from exc


def _parse_plugin_config(
    value: Optional[str], arg_name: str
) -> Optional[Dict[str, Any]]:
    """Parse plugin configuration from a JSON/YAML string or file path."""
    data = _parse_config_payload(value, arg_name)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise MeteorConfigError(
            "Plugin configuration must be a mapping",
            config_key=arg_name,
            context={"provided_type": type(data).__name__},
        )
    return data


def _parse_hook_config(value: Optional[str], arg_name: str) -> Any:
    data = _parse_config_payload(value, arg_name)
    if data is None:
        return None
    if not isinstance(data, (list, dict)):
        raise MeteorConfigError(
            "Hook configuration must be a list or mapping",
            config_key=arg_name,
            context={"provided_type": type(data).__name__},
        )
    return data


def _warn_legacy_param_flags() -> None:
    legacy_flags = {
        "--diff-threshold",
        "--min-area",
        "--min-aspect-ratio",
        "--hough-threshold",
        "--hough-min-line-length",
        "--hough-max-line-gap",
        "--min-line-score",
    }
    used = sorted(flag for flag in legacy_flags if flag in sys.argv)
    if used:
        warnings.warn(
            "Legacy CLI detection parameter flags are deprecated and will be "
            "replaced by pipeline configuration files. "
            f"Flags detected: {', '.join(used)}",
            UserWarning,
            stacklevel=2,
        )


def _resolve_value(arg_value, base_value, *flags: str):
    if flags and _flag_present(*flags):
        return arg_value
    return base_value


def _parse_hook_names(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    return names


def _build_hook_configs(
    hook_names: Optional[List[str]],
    hook_config_data: Optional[Any],
) -> Optional[List[HookConfig]]:
    if hook_names is None and hook_config_data is None:
        return None
    configs = normalize_hook_configs(hook_config_data)
    config_map = {config.name: config.config for config in configs or []}
    if hook_names is not None:
        return [
            HookConfig(name=name, config=config_map.get(name)) for name in hook_names
        ]
    return configs


def _build_pipeline_config(args) -> PipelineConfig:
    base_config = load_pipeline_config(args.config) if args.config else None
    base_params = base_config.params if base_config else None

    params = DetectionParams(
        diff_threshold=_resolve_value(
            args.diff_threshold,
            base_params.diff_threshold if base_params else args.diff_threshold,
            "--diff-threshold",
        ),
        min_area=_resolve_value(
            args.min_area,
            base_params.min_area if base_params else args.min_area,
            "--min-area",
        ),
        min_aspect_ratio=_resolve_value(
            args.min_aspect_ratio,
            base_params.min_aspect_ratio if base_params else args.min_aspect_ratio,
            "--min-aspect-ratio",
        ),
        hough_threshold=_resolve_value(
            args.hough_threshold,
            base_params.hough_threshold if base_params else args.hough_threshold,
            "--hough-threshold",
        ),
        hough_min_line_length=_resolve_value(
            args.hough_min_line_length,
            base_params.hough_min_line_length
            if base_params
            else args.hough_min_line_length,
            "--hough-min-line-length",
        ),
        hough_max_line_gap=_resolve_value(
            args.hough_max_line_gap,
            base_params.hough_max_line_gap if base_params else args.hough_max_line_gap,
            "--hough-max-line-gap",
        ),
        min_line_score=_resolve_value(
            args.min_line_score,
            base_params.min_line_score if base_params else args.min_line_score,
            "--min-line-score",
        ),
    )

    input_loader_config = (
        _parse_plugin_config(args.input_loader_config, "input_loader_config")
        if args.input_loader_config is not None
        else (base_config.input_loader_config if base_config else None)
    )
    detector_config = (
        _parse_plugin_config(args.detector_config, "detector_config")
        if args.detector_config is not None
        else (base_config.detector_config if base_config else None)
    )
    output_handler_config = (
        _parse_plugin_config(args.output_handler_config, "output_handler_config")
        if args.output_handler_config is not None
        else (base_config.output_handler_config if base_config else None)
    )

    target_folder = _resolve_value(
        args.target,
        base_config.target_folder if base_config else args.target,
        "-t",
        "--target",
    )
    output_folder = _resolve_value(
        args.output,
        base_config.output_folder if base_config else args.output,
        "-o",
        "--output",
    )
    debug_folder = _resolve_value(
        args.debug_dir,
        base_config.debug_folder if base_config else args.debug_dir,
        "--debug-dir",
    )
    num_workers = _resolve_value(
        args.workers,
        base_config.num_workers if base_config else args.workers,
        "--workers",
    )
    batch_size = _resolve_value(
        args.batch_size,
        base_config.batch_size if base_config else args.batch_size,
        "--batch-size",
    )
    auto_batch_size = _resolve_value(
        args.auto_batch_size,
        base_config.auto_batch_size if base_config else args.auto_batch_size,
        "--auto-batch-size",
        "--no-auto-batch-size",
    )
    enable_parallel = _resolve_value(
        args.enable_parallel,
        base_config.enable_parallel if base_config else args.enable_parallel,
        "--parallel",
        "--no-parallel",
    )
    progress_file = _resolve_value(
        args.progress_file,
        base_config.progress_file if base_config else args.progress_file,
        "--progress-file",
    )
    output_overwrite = _resolve_value(
        args.output_overwrite,
        base_config.output_overwrite if base_config else args.output_overwrite,
        "--output-overwrite",
    )
    hook_names = _parse_hook_names(args.hooks)
    hook_config_data = (
        _parse_hook_config(args.hook_config, "hook_config")
        if args.hook_config is not None
        else None
    )
    hooks = (
        _build_hook_configs(hook_names, hook_config_data)
        if hook_names is not None or hook_config_data is not None
        else (base_config.hooks if base_config else None)
    )

    pipeline_config = PipelineConfig(
        target_folder=target_folder,
        output_folder=output_folder,
        debug_folder=debug_folder,
        params=params,
        num_workers=num_workers,
        batch_size=batch_size,
        auto_batch_size=auto_batch_size,
        enable_parallel=enable_parallel,
        progress_file=progress_file,
        output_overwrite=output_overwrite,
        input_loader_name=args.input_loader
        if args.input_loader is not None
        else (base_config.input_loader_name if base_config else None),
        input_loader_config=input_loader_config,
        detector_name=args.detector
        if args.detector is not None
        else (base_config.detector_name if base_config else None),
        detector_config=detector_config,
        output_handler_name=args.output_handler
        if args.output_handler is not None
        else (base_config.output_handler_name if base_config else None),
        output_handler_config=output_handler_config,
        hooks=hooks,
    )
    return pipeline_config


def validate_and_apply_sensor_preset(
    args, verbose: bool = False, locale: Optional[str] = None
) -> Tuple[
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
    ) = apply_sensor_preset(args, verbose=verbose, locale=locale)

    # Validate sensor overrides
    validate_sensor_overrides(
        args, preset, sensor_width_value, pixel_pitch_value, locale=locale
    )

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
    locale: Optional[str] = None,
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
    focal_length_source = get_message("ui.cli.exif.focal_source.unknown", locale=locale)
    if focal_length_value:
        focal_length_source = get_message(
            "ui.cli.exif.focal_source.cli",
            locale=locale,
        )
        exif_data["focal_length_35mm"] = focal_length_value
    elif exif_data.get("focal_length_35mm"):
        focal_length_source = get_message(
            "ui.cli.exif.focal_source.exif",
            locale=locale,
        )
    elif exif_data.get("focal_length") and focal_factor_value:
        if args_focal_factor:
            focal_length_source = get_message(
                "ui.cli.exif.focal_source.calculated_factor",
                locale=locale,
                factor=args_focal_factor,
            )
        else:
            focal_length_source = get_message(
                "ui.cli.exif.focal_source.calculated_sensor",
                locale=locale,
                sensor_type=args_sensor_type,
            )
        exif_data["focal_length_35mm"] = exif_data["focal_length"] * focal_factor_value
    elif exif_data.get("focal_length"):
        focal_length_source = get_message(
            "ui.cli.exif.focal_source.exif_actual",
            locale=locale,
        )

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
    locale: Optional[str] = None,
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
        warnings.append(
            get_message("ui.cli.exif.warning.focal_length_unavailable", locale=locale)
        )
    elif (
        not exif_data.get("focal_length_35mm")
        and not focal_factor_value
        and not focal_length_value
    ):
        warnings.append(
            get_message(
                "ui.cli.exif.warning.no_equiv",
                locale=locale,
            )
        )

    if not exif_data.get("iso"):
        warnings.append(
            get_message("ui.cli.exif.warning.iso_unavailable", locale=locale)
        )
    if not exif_data.get("exposure_time"):
        warnings.append(
            get_message("ui.cli.exif.warning.exposure_unavailable", locale=locale)
        )

    # NPF-related warnings
    if check_npf and npf_metrics:
        if not sensor_width_value and not exif_data.get("image_width"):
            warnings.append(
                get_message("ui.cli.exif.warning.sensor_width_missing", locale=locale)
            )
        if not npf_metrics.get("has_complete_data"):
            warnings.append(
                get_message("ui.cli.exif.warning.npf_incomplete", locale=locale)
            )

    return warnings


def print_warnings(warnings, locale: Optional[str] = None):
    """Print warnings in a formatted box."""
    if warnings:
        print(f"{'=' * 60}")
        print(get_message("ui.cli.warnings.header", locale=locale))
        for warning in warnings:
            print(f"  • {warning}")
        print(f"{'=' * 60}\n")


def handle_show_exif(args) -> None:
    """Handle --show-exif / --show-npf commands.

    Raises:
        MeteorError: On any error during EXIF extraction or display.
    """
    print(f"\n{'=' * 60}")
    if args.show_npf:
        print(get_message("ui.cli.exif.header_npf", locale=args.locale))
    else:
        print(get_message("ui.cli.exif.header", locale=args.locale))
    print(f"{'=' * 60}\n")
    print(
        get_message("ui.cli.exif.target_folder", locale=args.locale, path=args.target)
    )

    # Validate and apply sensor preset
    (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    ) = validate_and_apply_sensor_preset(args, verbose=True, locale=args.locale)

    # Collect files (raises MeteorLoadError if directory doesn't exist)
    files = collect_files(args.target)
    if not files:
        raise MeteorLoadError(
            "No RAW files found in target folder",
            filepath=args.target,
            context={"error_category": "no_files"},
        )

    print(
        get_message(
            "ui.cli.exif.found_raw_files",
            locale=args.locale,
            count=len(files),
        )
    )
    print(
        get_message(
            "ui.cli.exif.reading_first_file",
            locale=args.locale,
            filename=os.path.basename(files[0]),
        )
        + "\n"
    )

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
        locale=args.locale,
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
        display_fisheye_info(
            exif_data["focal_length_35mm"], DEFAULT_FISHEYE_MODEL, locale=args.locale
        )

    display_exif_info(
        exif_data,
        focal_length_source,
        focal_factor_value,
        npf_metrics,
        locale=args.locale,
    )

    # Collect and print warnings
    warnings = collect_exif_warnings(
        exif_data=exif_data,
        focal_length_value=focal_length_value,
        focal_factor_value=focal_factor_value,
        sensor_width_value=sensor_width_value,
        pixel_pitch_value=pixel_pitch_value,
        npf_metrics=npf_metrics,
        check_npf=(args.show_npf or npf_metrics is not None),
        locale=args.locale,
    )
    print_warnings(warnings, locale=args.locale)


def _init_worker_ignore_interrupt() -> None:
    """Ignore SIGINT in worker processes. Must be at module level for pickling."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _validate_directories(
    target_folder: str,
    output_folder: str,
    debug_folder: str,
    debug_image_enabled: bool,
) -> None:
    """
    Validate and create directories for processing.

    Args:
        target_folder: Input folder with RAW files
        output_folder: Output folder for candidates
        debug_folder: Folder for debug images
        debug_image_enabled: Whether to create the debug image folder

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
        if debug_image_enabled:
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
    locale: Optional[str] = None,
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
            get_message(
                "ui.cli.roi.cli_specified",
                locale=locale,
                polygon=format_polygon_string(roi_polygon_cli),
            )
        )
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [np.array(roi_polygon_cli, dtype=np.int32)], 255)
        roi_polygon = roi_polygon_cli
    elif enable_roi_selection:
        roi_selection = select_roi(prev_img, locale=locale)
        if roi_selection:
            roi_mask = roi_selection["mask"]
            roi_polygon = roi_selection["polygon"]
            print(
                get_message(
                    "ui.cli.roi.setup_complete",
                    locale=locale,
                    polygon=format_polygon_string(roi_polygon),
                )
            )
        else:
            print(get_message("ui.cli.roi.none_selected", locale=locale))
    else:
        print(get_message("ui.cli.roi.skipped", locale=locale))

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
    locale: Optional[str] = None,
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
    print(f"\n{'=' * 60}")
    print(get_message("ui.cli.auto_params.header", locale=locale))
    print(f"{'=' * 60}\n")

    # Step 1: EXIF Information Extraction
    exif_data = extract_exif_metadata(files[0])

    # Focal length acquisition (priority)
    focal_length_source = get_message("ui.cli.exif.focal_source.unknown", locale=locale)
    if focal_length_mm:
        focal_length_source = get_message(
            "ui.cli.exif.focal_source.cli",
            locale=locale,
        )
        exif_data["focal_length_35mm"] = focal_length_mm
    elif exif_data.get("focal_length_35mm"):
        focal_length_mm = exif_data["focal_length_35mm"]
        focal_length_source = get_message(
            "ui.cli.exif.focal_source.exif",
            locale=locale,
        )
    elif exif_data.get("focal_length") and focal_factor:
        focal_length_mm = exif_data["focal_length"] * focal_factor
        exif_data["focal_length_35mm"] = focal_length_mm
        focal_length_source = get_message(
            "ui.cli.exif.focal_source.calculated_exif",
            locale=locale,
            focal_length=exif_data["focal_length"],
            factor=focal_factor,
        )
    elif exif_data.get("focal_length"):
        focal_length_mm = exif_data["focal_length"]
        focal_length_source = get_message(
            "ui.cli.exif.focal_source.exif_actual",
            locale=locale,
        )

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
        display_fisheye_info(
            exif_data["focal_length_35mm"],
            DEFAULT_FISHEYE_MODEL,
            locale=locale,
        )

    # Display EXIF Information and NPF Analysis
    display_exif_info(
        exif_data,
        focal_length_source,
        focal_factor,
        npf_metrics,
        locale=locale,
    )

    # Step 3: Display warnings
    _display_auto_params_warnings(
        exif_data,
        focal_length_mm,
        focal_factor,
        sensor_width_mm,
        pixel_pitch_um,
        npf_metrics,
        locale=locale,
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
            locale=locale,
        )

    return diff_threshold, min_area, min_line_score, focal_length_mm


def _display_auto_params_warnings(
    exif_data,
    focal_length_mm,
    focal_factor,
    sensor_width_mm,
    pixel_pitch_um,
    npf_metrics,
    locale: Optional[str] = None,
):
    """Display warnings for auto-params mode."""
    warnings = []

    if not focal_length_mm:
        warnings.append(
            get_message(
                "ui.cli.auto_params.warning.focal_length_missing",
                locale=locale,
            )
        )
        warnings.append(
            get_message(
                "ui.cli.auto_params.warning.focal_length_hint",
                locale=locale,
            )
        )
    elif not exif_data.get("focal_length_35mm") and not focal_factor:
        warnings.append(
            get_message(
                "ui.cli.auto_params.warning.no_equiv",
                locale=locale,
                focal_length=focal_length_mm,
            )
        )
        warnings.append(
            get_message(
                "ui.cli.auto_params.warning.no_equiv_hint",
                locale=locale,
            )
        )

    if not exif_data.get("iso"):
        warnings.append(
            get_message(
                "ui.cli.auto_params.warning.iso_missing",
                locale=locale,
            )
        )

    if not exif_data.get("exposure_time"):
        warnings.append(
            get_message(
                "ui.cli.auto_params.warning.exposure_missing",
                locale=locale,
            )
        )

    if npf_metrics and not npf_metrics.get("has_complete_data"):
        if not sensor_width_mm and not pixel_pitch_um:
            warnings.append(
                get_message(
                    "ui.cli.auto_params.warning.default_pixel_pitch",
                    locale=locale,
                )
            )
            warnings.append(
                get_message(
                    "ui.cli.auto_params.warning.pixel_pitch_hint",
                    locale=locale,
                )
            )

    if warnings:
        print(f"{'=' * 60}")
        print(get_message("ui.cli.warnings.header", locale=locale))
        for warning in warnings:
            if warning.startswith("  →"):
                print(f"  {warning}")
            else:
                print(f"  • {warning}")
        print(f"{'=' * 60}\n")


def _optimize_with_npf(
    exif_data,
    npf_metrics,
    diff_threshold,
    min_area,
    min_line_score,
    user_specified_diff_threshold,
    user_specified_min_area,
    user_specified_min_line_score,
    locale: Optional[str] = None,
):
    """Optimize parameters using NPF Rule."""
    print(f"{'=' * 60}")
    print(get_message("ui.cli.auto_params.optimize.header", locale=locale))
    print(f"{'=' * 60}\n")

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
        get_message(
            "ui.cli.auto_params.optimize.quality_score",
            locale=locale,
            score=opt_info["quality_score"],
            level=opt_info["quality_level"],
        )
    )

    if opt_info["adjustments"]:
        print(
            "\n"
            + get_message(
                "ui.cli.auto_params.optimize.adjustments.header", locale=locale
            )
        )
        for adjustment in opt_info["adjustments"]:
            print(
                get_message(
                    "ui.cli.auto_params.optimize.adjustments.item",
                    locale=locale,
                    text=adjustment,
                )
            )
    else:
        print(
            "\n"
            + get_message(
                "ui.cli.auto_params.optimize.adjustments.none",
                locale=locale,
            )
        )

    print(f"\n{'=' * 60}\n")

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
    locale: Optional[str] = None,
):
    """Fallback to legacy parameter estimation method."""
    print(get_message("ui.cli.auto_params.legacy.insufficient", locale=locale))
    print(get_message("ui.cli.auto_params.legacy.fallback", locale=locale) + "\n")

    if not user_specified_diff_threshold:
        diff_threshold = estimate_diff_threshold_from_samples(
            files, roi_mask, sample_size=5, locale=locale
        )
        print(
            get_message(
                "ui.cli.auto_params.legacy.diff_threshold.sample",
                locale=locale,
                value=diff_threshold,
            )
        )
    else:
        print(
            get_message(
                "ui.cli.auto_params.legacy.diff_threshold.user",
                locale=locale,
                value=diff_threshold,
            )
        )

    if not user_specified_min_area:
        min_area = estimate_min_area_from_samples(
            files, roi_mask, diff_threshold, sample_size=3, locale=locale
        )
        print(
            get_message(
                "ui.cli.auto_params.legacy.min_area.sample",
                locale=locale,
                value=min_area,
            )
        )
    else:
        print(
            get_message(
                "ui.cli.auto_params.legacy.min_area.user",
                locale=locale,
                value=min_area,
            )
        )

    if not user_specified_min_line_score:
        min_line_score = estimate_min_line_score_from_image(
            prev_img.shape, focal_length_mm, locale=locale
        )
        print(
            get_message(
                "ui.cli.auto_params.legacy.min_line_score.image",
                locale=locale,
                value=min_line_score,
            )
        )
    else:
        print(
            get_message(
                "ui.cli.auto_params.legacy.min_line_score.user",
                locale=locale,
                value=min_line_score,
            )
        )

    return diff_threshold, min_area, min_line_score


def _print_processing_params(processing_params, locale: Optional[str] = None):
    """Print processing parameters in a formatted box."""
    print(f"\n{'=' * 50}")
    print(get_message("ui.cli.processing_params.header", locale=locale))
    print(f"{'=' * 50}")
    labels = {
        "diff_threshold": get_message(
            "ui.cli.processing_params.label.diff_threshold", locale=locale
        ),
        "min_area": get_message(
            "ui.cli.processing_params.label.min_area", locale=locale
        ),
        "min_aspect_ratio": get_message(
            "ui.cli.processing_params.label.min_aspect_ratio", locale=locale
        ),
        "hough_threshold": get_message(
            "ui.cli.processing_params.label.hough_threshold", locale=locale
        ),
        "hough_min_line_length": get_message(
            "ui.cli.processing_params.label.hough_min_line_length", locale=locale
        ),
        "hough_max_line_gap": get_message(
            "ui.cli.processing_params.label.hough_max_line_gap", locale=locale
        ),
        "min_line_score": get_message(
            "ui.cli.processing_params.label.min_line_score", locale=locale
        ),
    }
    label_width = max(_display_width(label) for label in labels.values())

    def format_line(label_key: str, value: str) -> str:
        return get_message(
            "ui.cli.processing_params.line",
            locale=locale,
            label=_pad_label(labels[label_key], label_width),
            value=value,
        )

    print(format_line("diff_threshold", str(processing_params["diff_threshold"])))
    print(format_line("min_area", str(processing_params["min_area"])))
    print(format_line("min_aspect_ratio", str(processing_params["min_aspect_ratio"])))
    print(format_line("hough_threshold", str(processing_params["hough_threshold"])))
    print(
        format_line(
            "hough_min_line_length",
            str(processing_params["hough_min_line_length"]),
        )
    )
    print(
        format_line(
            "hough_max_line_gap",
            str(processing_params["hough_max_line_gap"]),
        )
    )
    print(format_line("min_line_score", f"{processing_params['min_line_score']:.1f}"))
    print(f"{'=' * 50}\n")


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


def _extract_frame_indices(detection_context_payload):
    ctx_frame_index = None
    ctx_prev_frame_index = None
    if detection_context_payload and isinstance(detection_context_payload, dict):
        metadata = detection_context_payload.get("metadata", {})
        ctx_frame_index = metadata.get("frame_index")
        ctx_prev_frame_index = metadata.get("prev_frame_index")
    return ctx_frame_index, ctx_prev_frame_index


def _run_parallel_detection(
    image_pairs,
    roi_mask,
    processing_params,
    roi_polygon,
    output_folder,
    debug_folder,
    output_overwrite,
    debug_image_enabled,
    num_workers,
    batch_size,
    resume_offset,
    overall_total,
    record_result_callback,
    locale: Optional[str] = None,
):
    """
    Run detection in parallel using ProcessPoolExecutor.

    Args:
        image_pairs: List of (frame_index, current_file, previous_file) tuples
        roi_mask: ROI mask
        processing_params: Detection parameters
        roi_polygon: ROI polygon for debug visualization
        output_folder: Output folder for candidates
        debug_folder: Folder for debug images
        output_overwrite: Whether to overwrite existing files
        debug_image_enabled: Whether to save debug images to disk
        num_workers: Number of parallel workers
        batch_size: Batch size for processing
        resume_offset: Offset for progress display
        overall_total: Total number of files for progress display
        record_result_callback: Callback to record results (filename, is_candidate, score, lines, ratio, detection_result)

    Returns:
        Number of detected candidates
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    batches = [
        image_pairs[i : i + batch_size] for i in range(0, len(image_pairs), batch_size)
    ]
    print(
        get_message(
            "ui.cli.processing.batches",
            locale=locale,
            count=len(batches),
        )
    )

    executor = ProcessPoolExecutor(
        max_workers=num_workers, initializer=_init_worker_ignore_interrupt
    )
    futures = []
    wait_for_tasks = True
    detected_count = 0

    try:
        for batch_idx, batch in enumerate(batches, 1):
            print(
                get_message(
                    "ui.cli.processing.submitting_batches",
                    locale=locale,
                    current=batch_idx,
                    total=len(batches),
                ),
                end="\r",
                flush=True,
            )
            future = executor.submit(
                process_image_batch,
                batch,
                roi_mask,
                processing_params,
                None,
                None,
                debug_image_enabled,
            )
            futures.append(future)

        # Clear the submitting line
        print()

        processed = 0
        completed_batches = 0
        total_batches = len(batches)

        # Show initial batch progress (0 completed)
        print(
            get_message(
                "ui.cli.processing.batch_progress",
                locale=locale,
                completed=0,
                total=total_batches,
            ),
            end="\r",
            flush=True,
        )

        for future in as_completed(futures):
            completed_batches += 1
            try:
                batch_results = future.result()

                # Show batch progress when a batch completes
                print(
                    get_message(
                        "ui.cli.processing.batch_progress",
                        locale=locale,
                        completed=completed_batches,
                        total=total_batches,
                    ),
                    end="\r",
                    flush=True,
                )

                for result in batch_results:
                    (
                        is_candidate,
                        filename,
                        filepath,
                        line_score,
                        debug_img,
                        aspect_ratio,
                        num_lines,
                        detection_result,
                        _detection_context,
                    ) = result
                    processed += 1

                    # Always show progress first
                    print(
                        get_message(
                            "ui.cli.processing.checking",
                            locale=locale,
                            current=resume_offset + processed,
                            total=overall_total,
                        ),
                        end="\r",
                        flush=True,
                    )

                    if line_score > 0:
                        print()  # Move to new line before [LINE]
                        print(
                            get_message(
                                "ui.cli.processing.line",
                                locale=locale,
                                filename=filename,
                                score=line_score,
                                lines=num_lines,
                            )
                        )

                    if is_candidate:
                        if line_score <= 0:
                            print()  # Move to new line if [LINE] wasn't printed
                        if not debug_image_enabled:
                            debug_img = None
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
                            print(
                                get_message(
                                    "ui.cli.processing.hit",
                                    locale=locale,
                                    filename=filename,
                                    ratio=aspect_ratio,
                                )
                            )
                        else:
                            print(
                                get_message(
                                    "ui.cli.processing.skip",
                                    locale=locale,
                                    filename=filename,
                                )
                            )

                    # Extract frame indices from detection context
                    ctx_frame_index, ctx_prev_frame_index = _extract_frame_indices(
                        _detection_context
                    )

                    detected_count = record_result_callback(
                        filename,
                        is_candidate,
                        line_score,
                        num_lines,
                        aspect_ratio,
                        detection_result,
                        ctx_frame_index,
                        ctx_prev_frame_index,
                    )

            except Exception as e:
                print(
                    "\n"
                    + get_message(
                        "ui.cli.processing.batch_error",
                        locale=locale,
                        error=e,
                    )
                )
    except KeyboardInterrupt:
        print("\n" + get_message("ui.cli.processing.interrupt_cancel", locale=locale))
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
    debug_image_enabled,
    resume_offset,
    overall_total,
    record_result_callback,
    locale: Optional[str] = None,
):
    """
    Run detection sequentially (single-threaded).

    Args:
        image_pairs: List of (frame_index, current_file, previous_file) tuples
        roi_mask: ROI mask
        processing_params: Detection parameters
        roi_polygon: ROI polygon for debug visualization
        output_folder: Output folder for candidates
        debug_folder: Folder for debug images
        output_overwrite: Whether to overwrite existing files
        debug_image_enabled: Whether to save debug images to disk
        resume_offset: Offset for progress display
        overall_total: Total number of files for progress display
        record_result_callback: Callback to record results (filename, is_candidate, score, lines, ratio, detection_result)

    Returns:
        Number of detected candidates
    """
    detected_count = 0

    for idx, pair in enumerate(image_pairs):
        current_index = resume_offset + idx + 1
        current_file = os.path.basename(pair[1])  # pair = (frame_index, curr, prev)
        progress_line_active = True

        print(
            get_message(
                "ui.cli.processing.processing_file",
                locale=locale,
                current=current_index,
                total=overall_total,
                filename=current_file,
            ),
            end="",
            flush=True,
        )

        batch_results = process_image_batch(
            [pair],
            roi_mask,
            processing_params,
            None,
            None,
            debug_image_enabled,
        )

        for result in batch_results:
            (
                is_candidate,
                filename,
                filepath,
                line_score,
                debug_img,
                aspect_ratio,
                num_lines,
                detection_result,
                _detection_context,
            ) = result

            if line_score > 0:
                if progress_line_active:
                    print()
                    progress_line_active = False
                print(
                    get_message(
                        "ui.cli.processing.line",
                        locale=locale,
                        filename=filename,
                        score=line_score,
                        lines=num_lines,
                    )
                )

            if is_candidate:
                if progress_line_active:
                    print()
                    progress_line_active = False

                if not debug_image_enabled:
                    debug_img = None
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
                    print(
                        get_message(
                            "ui.cli.processing.hit",
                            locale=locale,
                            filename=filename,
                            ratio=aspect_ratio,
                        )
                    )
                else:
                    print(
                        get_message(
                            "ui.cli.processing.skip_with_overwrite",
                            locale=locale,
                            filename=filename,
                        )
                    )
            else:
                print(
                    get_message(
                        "ui.cli.processing.checking",
                        locale=locale,
                        current=current_index,
                        total=overall_total,
                    ),
                    end="\r",
                    flush=True,
                )
                progress_line_active = True

            # Extract frame indices from detection context
            ctx_frame_index, ctx_prev_frame_index = _extract_frame_indices(
                _detection_context
            )

            detected_count = record_result_callback(
                filename,
                is_candidate,
                line_score,
                num_lines,
                aspect_ratio,
                detection_result,
                ctx_frame_index,
                ctx_prev_frame_index,
            )

    return detected_count


def detect_meteors_advanced(
    target_folder: str,
    output_folder: str,
    debug_folder: str,
    debug_image_enabled: bool,
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
    _validate_directories(
        target_folder, output_folder, debug_folder, debug_image_enabled
    )

    # Collect files (raises MeteorLoadError if directory doesn't exist or is empty)
    print(
        get_message(
            "ui.cli.collecting",
            locale=locale,
            path=target_folder,
        )
    )
    files = collect_files(target_folder)

    if len(files) < 2:
        raise MeteorValidationError(
            "Need at least 2 images for meteor detection",
            parameter_name="target_folder",
            provided_value=target_folder,
            expected="directory with at least 2 RAW files",
            context={"files_found": len(files)},
        )

    print(get_message("ui.cli.found_files", locale=locale, count=len(files)))

    # Load first image
    t_load = time.time()
    try:
        prev_img = load_and_bin_raw_fast(files[0])
    except Exception as exc:
        print(
            get_message(
                "ui.cli.load_first_failed",
                locale=locale,
                filename=os.path.basename(files[0]),
                error=exc,
            )
        )
        return 0

    if profile:
        timing["first_load"] = time.time() - t_load

    # ROI setup
    roi_mask, roi_polygon = _setup_roi(
        prev_img, roi_polygon_cli, enable_roi_selection, locale=locale
    )

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
            locale=locale,
        )

    # Build processing parameters.
    # NOTE: These are still constructed as a flat dict for backward compatibility.
    # The pipeline wraps them into namespaced runtime_params for detectors.
    processing_params = {
        "diff_threshold": diff_threshold,
        "min_area": min_area,
        "min_aspect_ratio": min_aspect_ratio,
        "hough_threshold": hough_threshold,
        "hough_min_line_length": hough_min_line_length,
        "hough_max_line_gap": hough_max_line_gap,
        "min_line_score": min_line_score,
    }

    _print_processing_params(processing_params, locale=locale)

    # Progress tracking setup
    params_for_hash = processing_params.copy()
    if roi_polygon:
        params_for_hash["roi_polygon"] = roi_polygon

    progress_manager = ProgressManager(progress_file)
    loaded_progress = progress_manager.load() if resume else False

    params_hash = compute_params_hash(params_for_hash)
    if loaded_progress and progress_manager.get_params_hash() == params_hash:
        print(
            get_message(
                "ui.cli.progress.resuming",
                locale=locale,
                path=progress_file,
                processed=progress_manager.get_total_processed(),
                detected=progress_manager.get_total_detected(),
            )
        )
    else:
        if loaded_progress:
            print(get_message("ui.cli.progress.param_mismatch", locale=locale))
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
        detection_result: Optional[DetectionResult] = None,
        frame_index: Optional[int] = None,
        prev_frame_index: Optional[int] = None,
    ) -> int:
        """Record result and return current detected count."""
        return progress_manager.record_result(
            filename,
            is_candidate,
            score,
            lines,
            ratio,
            frame_index=frame_index,
            prev_frame_index=prev_frame_index,
        )

    # Build image pairs with frame index and filter already processed
    # (frame_index, curr_file, prev_file)
    image_pairs = [(i, files[i], files[i - 1]) for i in range(1, len(files))]
    image_pairs = [
        pair for pair in image_pairs if os.path.basename(pair[1]) not in processed_set
    ]

    resume_offset = len(processed_set)
    overall_total = resume_offset + len(image_pairs)

    print(
        get_message(
            "ui.cli.processing.start",
            locale=locale,
            count=len(image_pairs),
        )
    )
    if enable_parallel:
        print(
            get_message(
                "ui.cli.processing.parallel",
                locale=locale,
                workers=num_workers,
                batch_size=batch_size,
            )
        )

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
                debug_image_enabled=debug_image_enabled,
                num_workers=num_workers,
                batch_size=batch_size,
                resume_offset=resume_offset,
                overall_total=overall_total,
                record_result_callback=record_result,
                locale=locale,
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
                debug_image_enabled=debug_image_enabled,
                resume_offset=resume_offset,
                overall_total=overall_total,
                record_result_callback=record_result,
                locale=locale,
            )
    except KeyboardInterrupt:
        print(
            "\n"
            + get_message(
                "ui.interrupt.progress",
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

        print("\n\n" + get_message("ui.profile.header", locale=locale))
        print(
            get_message(
                "ui.profile.first_load",
                locale=locale,
                value=timing["first_load"],
            )
        )
        print(
            get_message(
                "ui.profile.processing_time",
                locale=locale,
                value=timing["processing"],
            )
        )
        print(
            get_message(
                "ui.profile.total_time",
                locale=locale,
                value=timing["total"],
            )
        )
        print(
            get_message(
                "ui.profile.images_processed",
                locale=locale,
                count=len(image_pairs),
            )
        )
        if image_pairs:
            print(
                get_message(
                    "ui.profile.average_per_image",
                    locale=locale,
                    value=timing["processing"] / len(image_pairs),
                )
            )

    summary_message = get_message("ui.run.summary", locale=locale, count=detected_count)
    print(f"\n{summary_message}")
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
                        "ui.progress.removed",
                        locale=locale,
                        path=args.progress_file,
                    )
                )
            else:
                print(
                    get_message(
                        "ui.progress.not_found",
                        locale=locale,
                        path=args.progress_file,
                    )
                )
            return

        # --list-sensor-types: Display available sensor types and exit
        if args.list_sensor_types:
            list_sensor_types(locale=locale)
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
            saved_path = save_diagnostic_report(e, diag_path, locale=locale)
            print(
                get_message(
                    "ui.diagnostic.report.saved", locale=locale, path=saved_path
                ),
                file=sys.stderr,
            )
        elif not args.verbose:
            # Hint about diagnostic options
            print(
                get_message("ui.diagnostic.hint.save", locale=locale),
                file=sys.stderr,
            )

        sys.exit(1)
    except KeyboardInterrupt:
        print(
            "\n" + get_message("ui.interrupt.generic", locale=locale),
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
                    "ui.error.unexpected",
                    locale=locale,
                    error_type=type(e).__name__,
                    error_message=e,
                ),
                file=sys.stderr,
            )
            print(
                get_message("ui.error.unexpected.hint", locale=locale),
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

    _warn_legacy_param_flags()

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
    ) = validate_and_apply_sensor_preset(args, verbose=False, locale=locale)

    pipeline_config = _build_pipeline_config(args)

    if args.auto_params:
        files = collect_files(pipeline_config.target_folder)
        if len(files) < 2:
            raise MeteorValidationError(
                "Need at least 2 images for meteor detection",
                parameter_name="target_folder",
                provided_value=pipeline_config.target_folder,
                expected="directory with at least 2 RAW files",
                context={"files_found": len(files)},
            )
        prev_img = load_and_bin_raw_fast(files[0])
        roi_mask, roi_polygon = _setup_roi(
            prev_img, roi_polygon_cli, enable_roi_selection, locale=locale
        )
        (
            diff_threshold,
            min_area,
            min_line_score,
            _focal_length_value,
        ) = _run_auto_params(
            files=files,
            prev_img=prev_img,
            roi_mask=roi_mask,
            diff_threshold=pipeline_config.params.diff_threshold,
            min_area=pipeline_config.params.min_area,
            min_line_score=pipeline_config.params.min_line_score,
            user_specified_diff_threshold=user_specified_diff_threshold,
            user_specified_min_area=user_specified_min_area,
            user_specified_min_line_score=user_specified_min_line_score,
            focal_length_mm=focal_length_value,
            focal_factor=focal_factor_value,
            sensor_width_mm=sensor_width_value,
            pixel_pitch_um=pixel_pitch_value,
            fisheye=args.fisheye,
            locale=locale,
        )
        pipeline_config.params = DetectionParams(
            diff_threshold=diff_threshold,
            min_area=min_area,
            min_aspect_ratio=pipeline_config.params.min_aspect_ratio,
            hough_threshold=pipeline_config.params.hough_threshold,
            hough_min_line_length=pipeline_config.params.hough_min_line_length,
            hough_max_line_gap=pipeline_config.params.hough_max_line_gap,
            min_line_score=min_line_score,
        )
        roi_polygon_cli = roi_polygon
        enable_roi_selection = False

    _print_processing_params(pipeline_config.params.to_dict(), locale=locale)

    pipeline = MeteorDetectionPipeline(pipeline_config)
    pipeline.run(
        enable_roi_selection=enable_roi_selection,
        roi_polygon_cli=roi_polygon_cli,
        resume=not args.no_resume,
        profile=args.profile,
        debug_image_enabled=args.debug_image,
    )


if __name__ == "__main__":
    main()
