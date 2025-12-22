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

Logging:
    This library uses Python's standard logging module. By default, a NullHandler
    is attached to prevent "No handler found" warnings. To see log output, configure
    logging in your application:

    Example:
        >>> import logging
        >>> logging.basicConfig(level=logging.DEBUG)

    Or attach a handler to the 'meteor_core' logger:

        >>> import logging
        >>> logger = logging.getLogger('meteor_core')
        >>> logger.addHandler(logging.StreamHandler())
        >>> logger.setLevel(logging.DEBUG)
"""

import logging

# Configure library-level logger with NullHandler to prevent
# "No handler found" warnings when the library is used without
# explicit logging configuration.
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

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
    DEFAULT_LOADER_NAME,
    DEFAULT_DETECTOR_NAME,
    DEFAULT_OUTPUT_HANDLER_NAME,
    DEFAULT_TARGET_FOLDER,
    DEFAULT_OUTPUT_FOLDER,
    DEFAULT_DEBUG_FOLDER,
    DEFAULT_PIXEL_PITCH_UM,
    AUTO_BATCH_MEMORY_FRACTION,
    HoughParams,
    DetectionParams,
    DetectionContext,
    PipelineConfig,
    ExifData,
    NPFMetrics,
    DetectionResult,
    ROIData,
    ProgressData,
    OptimizationInfo,
)

from .exceptions import (
    DiagnosticInfo,
    MeteorError,
    MeteorLoadError,
    MeteorUnsupportedFormatError,
    MeteorValidationError,
    MeteorConfigError,
    format_error_for_user,
    save_diagnostic_report,
    create_diagnostic_from_exception,
)
from .i18n import get_message

from .image_io import (
    load_and_bin_raw_fast,
    extract_exif_metadata,
)

from .inputs.raw import RawImageLoader, RawLoaderConfig, create_raw_loader

from .inputs.base import (
    BaseInputLoader,
    BaseMetadataExtractor,
    DataclassInputLoader,
    PydanticInputLoader,
    supports_metadata_extraction,
    _is_valid_input_loader,
)

from .inputs.registry import LoaderRegistry
from .inputs.discovery import PLUGIN_DIR, PLUGIN_GROUP

# Deprecated: use LoaderRegistry.discover() instead
from .inputs.discovery import discover_loaders

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
    ensure_numpy,
    ensure_tensor,
)

from .detectors import (
    BaseDetector,
    HoughDetector,
    compute_line_score_fast,
    DetectorRegistry,
)

# Deprecated: use DetectorRegistry.discover() instead
from .detectors import discover_detectors

from .outputs import (
    BaseOutputHandler,
    DataclassOutputHandler,
    _is_valid_output_handler,
    FileOutputConfig,
    FileOutputHandler,
    create_file_handler,
    OutputHandlerRegistry,
    ProgressManager,
    load_progress,
    save_progress,
    # Backward compatibility (deprecated)
    OutputWriter,
)

from .outputs.discovery import PLUGIN_DIR as OUTPUT_PLUGIN_DIR
from .outputs.discovery import PLUGIN_GROUP as OUTPUT_PLUGIN_GROUP

# Deprecated: use OutputHandlerRegistry.discover() instead
from .outputs import discover_handlers

from .pipeline import (
    collect_files,
    validate_raw_file,
    process_image_batch,
    estimate_diff_threshold_from_samples,
    estimate_min_area_from_samples,
    estimate_min_line_score_from_image,
    DetectionPipeline,
    MeteorDetectionPipeline,
    DefaultPipelineClass,
    create_default_pipeline,
    # Backward compatibility (deprecated)
    DefaultPipelineClass as DefaultPipeline,
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
    "DEFAULT_LOADER_NAME",
    "DEFAULT_DETECTOR_NAME",
    "DEFAULT_OUTPUT_HANDLER_NAME",
    "DEFAULT_TARGET_FOLDER",
    "DEFAULT_OUTPUT_FOLDER",
    "DEFAULT_DEBUG_FOLDER",
    "DEFAULT_PIXEL_PITCH_UM",
    "AUTO_BATCH_MEMORY_FRACTION",
    # Data Classes
    "HoughParams",
    "DetectionParams",
    "DetectionContext",
    "PipelineConfig",
    "ExifData",
    "NPFMetrics",
    "DetectionResult",
    "ROIData",
    "ProgressData",
    "OptimizationInfo",
    # Exceptions
    "DiagnosticInfo",
    "MeteorError",
    "MeteorLoadError",
    "MeteorUnsupportedFormatError",
    "MeteorValidationError",
    "MeteorConfigError",
    "format_error_for_user",
    "save_diagnostic_report",
    "create_diagnostic_from_exception",
    "get_message",
    # Image I/O
    "load_and_bin_raw_fast",
    "extract_exif_metadata",
    # Input Loaders - Abstract Base Classes
    "BaseInputLoader",
    "BaseMetadataExtractor",
    "DataclassInputLoader",
    "PydanticInputLoader",
    "supports_metadata_extraction",
    "_is_valid_input_loader",
    # Input Loaders - RAW
    "RawImageLoader",
    "RawLoaderConfig",
    "create_raw_loader",
    # Input Loaders - Registry (recommended)
    "LoaderRegistry",
    # Input Loaders - Discovery constants
    "PLUGIN_DIR",
    "PLUGIN_GROUP",
    # Input Loaders - Discovery function (deprecated)
    "discover_loaders",
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
    "ensure_numpy",
    "ensure_tensor",
    # Detectors
    "BaseDetector",
    "HoughDetector",
    "compute_line_score_fast",
    # Detectors - Registry (recommended)
    "DetectorRegistry",
    # Detectors - Discovery function (deprecated)
    "discover_detectors",
    # Outputs - Base classes
    "BaseOutputHandler",
    "DataclassOutputHandler",
    "_is_valid_output_handler",
    # Outputs - File handler (default implementation)
    "FileOutputConfig",
    "FileOutputHandler",
    "create_file_handler",
    # Outputs - Registry (recommended)
    "OutputHandlerRegistry",
    # Outputs - Discovery constants
    "OUTPUT_PLUGIN_DIR",
    "OUTPUT_PLUGIN_GROUP",
    # Outputs - Discovery function (deprecated)
    "discover_handlers",
    # Outputs - Progress tracking
    "ProgressManager",
    "load_progress",
    "save_progress",
    # Outputs - Backward compatibility (deprecated)
    "OutputWriter",
    # Pipeline
    "collect_files",
    "validate_raw_file",
    "process_image_batch",
    "estimate_diff_threshold_from_samples",
    "estimate_min_area_from_samples",
    "estimate_min_line_score_from_image",
    "DetectionPipeline",
    "MeteorDetectionPipeline",
    "DefaultPipelineClass",
    "create_default_pipeline",
    "DefaultPipeline",  # Backward compatibility (deprecated)
]
