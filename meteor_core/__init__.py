#!/usr/bin/env python
#
# Detect Meteors CLI - Core Package
# © 2025 Shinichi Morita (shin3tky)
#

"""
Meteor detection core library.

This package provides the core functionality for meteor detection:
- schema: Data structures and constants
- image_io: Image loading and EXIF extraction
- roi_selector: ROI selection GUI
- utils: Utility functions
- detectors: Detection algorithms
- outputs: Output handling
- pipeline: Processing orchestration
"""

from .schema import (
    VERSION,
    EXTENSIONS,
    SENSOR_PRESETS,
    CROP_FACTORS,
    DEFAULT_SENSOR_WIDTHS,
    FISHEYE_PROJECTION_MODELS,
    DEFAULT_FISHEYE_MODEL,
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
    DEFAULT_TARGET_FOLDER,
    DEFAULT_OUTPUT_FOLDER,
    DEFAULT_DEBUG_FOLDER,
    DEFAULT_PIXEL_PITCH_UM,
    AUTO_BATCH_MEMORY_FRACTION,
    HoughParams,
    DetectionParams,
    ExifData,
    NPFMetrics,
    DetectionResult,
    ROIData,
    ProgressData,
    OptimizationInfo,
)

from .image_io import (
    load_and_bin_raw_fast,
    extract_exif_metadata,
)

from .inputs.raw import RawImageLoader, RawLoaderConfig, create_raw_loader

from .roi_selector import (
    select_roi,
    create_roi_mask_from_polygon,
    create_full_roi_mask,
)

from .utils import (
    parse_focal_factor,
    get_sensor_preset,
    apply_sensor_preset,
    validate_sensor_overrides,
    list_sensor_types,
    calculate_fisheye_effective_focal_length,
    calculate_fisheye_edge_focal_length,
    calculate_fisheye_trail_length_ratio,
    get_fisheye_max_trail_ratio,
    display_fisheye_info,
    calculate_pixel_pitch,
    calculate_npf_rule,
    estimate_star_trail_length,
    evaluate_npf_compliance,
    calculate_npf_metrics,
    optimize_diff_threshold_npf,
    optimize_min_area_npf,
    estimate_meteor_trail_length,
    optimize_min_line_score_npf,
    calculate_shooting_quality_score,
    optimize_params_with_npf,
    get_available_memory_bytes,
    estimate_batch_size,
    compute_params_hash,
    parse_roi_polygon_string,
    format_polygon_string,
    display_exif_info,
)

from .detectors import (
    BaseDetector,
    HoughDetector,
    compute_line_score_fast,
)

from .outputs import (
    OutputHandler,
    OutputWriter,
    ProgressManager,
    load_progress,
    save_progress,
)

from .pipeline import (
    collect_files,
    validate_raw_file,
    process_image_batch,
    estimate_diff_threshold_from_samples,
    estimate_min_area_from_samples,
    estimate_min_line_score_from_image,
    MeteorDetectionPipeline,
)

__all__ = [
    # Version
    "VERSION",
    # Constants
    "EXTENSIONS",
    "SENSOR_PRESETS",
    "CROP_FACTORS",
    "DEFAULT_SENSOR_WIDTHS",
    "FISHEYE_PROJECTION_MODELS",
    "DEFAULT_FISHEYE_MODEL",
    "DEFAULT_DIFF_THRESHOLD",
    "DEFAULT_MIN_AREA",
    "DEFAULT_MIN_ASPECT_RATIO",
    "DEFAULT_HOUGH_THRESHOLD",
    "DEFAULT_HOUGH_MIN_LINE_LENGTH",
    "DEFAULT_HOUGH_MAX_LINE_GAP",
    "DEFAULT_MIN_LINE_SCORE",
    "DEFAULT_ENABLE_ROI_SELECTION",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PROGRESS_FILE",
    "DEFAULT_TARGET_FOLDER",
    "DEFAULT_OUTPUT_FOLDER",
    "DEFAULT_DEBUG_FOLDER",
    "DEFAULT_PIXEL_PITCH_UM",
    "AUTO_BATCH_MEMORY_FRACTION",
    # Data Classes
    "HoughParams",
    "DetectionParams",
    "ExifData",
    "NPFMetrics",
    "DetectionResult",
    "ROIData",
    "ProgressData",
    "OptimizationInfo",
    # Image I/O
    "load_and_bin_raw_fast",
    "extract_exif_metadata",
    "RawImageLoader",
    "RawLoaderConfig",
    "create_raw_loader",
    # ROI
    "select_roi",
    "create_roi_mask_from_polygon",
    "create_full_roi_mask",
    # Utils
    "parse_focal_factor",
    "get_sensor_preset",
    "apply_sensor_preset",
    "validate_sensor_overrides",
    "list_sensor_types",
    "calculate_fisheye_effective_focal_length",
    "calculate_fisheye_edge_focal_length",
    "calculate_fisheye_trail_length_ratio",
    "get_fisheye_max_trail_ratio",
    "display_fisheye_info",
    "calculate_pixel_pitch",
    "calculate_npf_rule",
    "estimate_star_trail_length",
    "evaluate_npf_compliance",
    "calculate_npf_metrics",
    "optimize_diff_threshold_npf",
    "optimize_min_area_npf",
    "estimate_meteor_trail_length",
    "optimize_min_line_score_npf",
    "calculate_shooting_quality_score",
    "optimize_params_with_npf",
    "get_available_memory_bytes",
    "estimate_batch_size",
    "compute_params_hash",
    "parse_roi_polygon_string",
    "format_polygon_string",
    "display_exif_info",
    # Detectors
    "BaseDetector",
    "HoughDetector",
    "compute_line_score_fast",
    # Outputs
    "OutputHandler",
    "OutputWriter",
    "ProgressManager",
    "load_progress",
    "save_progress",
    # Pipeline
    "collect_files",
    "validate_raw_file",
    "process_image_batch",
    "estimate_diff_threshold_from_samples",
    "estimate_min_area_from_samples",
    "estimate_min_line_score_from_image",
    "MeteorDetectionPipeline",
]
