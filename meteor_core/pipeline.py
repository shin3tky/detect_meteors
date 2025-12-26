#!/usr/bin/env python
#
# Detect Meteors CLI - Pipeline
# © 2025 Shinichi Morita (shin3tky)
#

"""
Processing pipeline for meteor detection.
Handles batch processing, multiprocessing, and orchestration.
"""

import glob
import logging
import os
import signal
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    overload,
)

import cv2
import numpy as np

from .exceptions import MeteorConfigError, MeteorError, MeteorLoadError
from .i18n import DEFAULT_LOCALE, get_message
from .schema import (
    EXTENSIONS,
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_MIN_AREA,
    DEFAULT_DETECTOR_NAME,
    DEFAULT_NUM_WORKERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_PROGRESS_FILE,
    DetectionContext,
    DetectionParams,
    DetectionResult,
    InputContext,
    OutputResult,
    PipelineConfig,
    RuntimeParams,
    RUNTIME_PARAMS_SCHEMA_VERSION,
    normalize_detection_context,
    normalize_detection_result,
    normalize_input_context,
    normalize_output_result,
)
from .image_io import extract_exif_metadata
from .inputs import BaseInputLoader, LoaderRegistry
from .inputs.base import supports_metadata_extraction
from .hooks import HookRegistry
from .roi_selector import select_roi, create_roi_mask_from_polygon, create_full_roi_mask
from .detectors import BaseDetector, DetectorRegistry
from .utils import _display_width, _pad_label
from .outputs import (
    BaseOutputHandler,
    OutputHandlerRegistry,
    ProgressManager,
)
from .utils import (
    compute_params_hash,
    format_polygon_string,
)

# Module-level logger
logger = logging.getLogger(__name__)

# Scaling factor for uint16 normalization.
_UINT16_MAX = float(np.iinfo(np.uint16).max)


def _normalize_detection_context(context: DetectionContext) -> DetectionContext:
    try:
        return normalize_detection_context(context)
    except ValueError as exc:
        raise MeteorConfigError(str(exc)) from exc


def _normalize_input_context(
    context: InputContext | Any, filepath: Optional[str] = None
) -> InputContext:
    if not isinstance(context, InputContext):
        return InputContext(
            image_data=context,
            filepath=filepath or "",
        )
    try:
        return normalize_input_context(context)
    except ValueError as exc:
        raise MeteorConfigError(str(exc)) from exc


def _normalize_detection_result(result: DetectionResult) -> DetectionResult:
    try:
        return normalize_detection_result(result)
    except ValueError as exc:
        raise MeteorConfigError(str(exc)) from exc


def _normalize_output_result(result: OutputResult | bool) -> OutputResult:
    if isinstance(result, bool):
        warnings.warn(
            "Output handler returned a legacy boolean instead of OutputResult. "
            "Wrap legacy output handlers to return OutputResult to avoid this warning.",
            DeprecationWarning,
            stacklevel=3,
        )
        return OutputResult(
            saved=result,
            output_path=None,
            debug_path=None,
            handler_info={"legacy_return": True},
        )
    try:
        return normalize_output_result(result)
    except ValueError as exc:
        raise MeteorConfigError(str(exc)) from exc


# Lazy initialization to avoid circular imports
_DEFAULT_INPUT_LOADER: Optional[BaseInputLoader] = None


def _resolve_locale(locale: Optional[str]) -> str:
    if locale:
        return locale
    return os.environ.get("DETECT_METEORS_LOCALE", DEFAULT_LOCALE)


def _get_default_input_loader() -> BaseInputLoader:
    """Get default input loader with lazy initialization."""
    global _DEFAULT_INPUT_LOADER
    if _DEFAULT_INPUT_LOADER is None:
        _DEFAULT_INPUT_LOADER = LoaderRegistry.create_default()
    return _DEFAULT_INPUT_LOADER


# Lazy initialization for default detector
_DEFAULT_DETECTOR: Optional[BaseDetector] = None


def _get_default_detector() -> BaseDetector:
    """Get default detector with lazy initialization."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = DetectorRegistry.create_default()
    return _DEFAULT_DETECTOR


def _build_runtime_params(
    params: Dict[str, Any], detector: BaseDetector
) -> Dict[str, Any]:
    detector_name = getattr(detector, "plugin_name", "") or DEFAULT_DETECTOR_NAME
    return RuntimeParams(
        schema_version=RUNTIME_PARAMS_SCHEMA_VERSION,
        global_params=params,
        detector={detector_name: params},
    ).to_dict(include_schema_version=False)


def _is_normalized_image(image: Any) -> bool:
    """Return True when image values are normalized to [0, 1] float range."""
    if not isinstance(image, np.ndarray):
        return False
    if not np.issubdtype(image.dtype, np.floating):
        return False
    try:
        max_value = float(np.nanmax(image))
    except (TypeError, ValueError):
        return False
    return max_value <= 1.0


def _loader_normalizes(input_loader: Optional[BaseInputLoader]) -> bool:
    """Return True if the loader config indicates normalized output."""
    if input_loader is None:
        return False
    config = getattr(input_loader, "config", None)
    if config is None:
        return False
    normalize_value = getattr(config, "normalize", None)
    if normalize_value is None and isinstance(config, dict):
        normalize_value = config.get("normalize")
    return bool(normalize_value)


def _apply_normalized_diff_threshold(
    params: Dict[str, Any], normalized_input: bool
) -> Dict[str, Any]:
    """Scale diff_threshold when inputs are normalized to [0, 1]."""
    if not normalized_input:
        return params
    diff_threshold = params.get("diff_threshold")
    if diff_threshold is None:
        return params
    try:
        diff_value = float(diff_threshold)
    except (TypeError, ValueError):
        return params
    if diff_value <= 1:
        return params
    scaled = diff_value / _UINT16_MAX
    updated = dict(params)
    updated["diff_threshold"] = scaled
    logger.debug(
        "Input normalization detected; scaled diff_threshold from %s to %.6f",
        diff_threshold,
        scaled,
    )
    return updated


def _ensure_frame_role(context: InputContext, role: str) -> InputContext:
    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    if metadata.get("frame_role") == role:
        return context
    updated_metadata = dict(metadata)
    updated_metadata["frame_role"] = role
    return InputContext(
        image_data=context.image_data,
        filepath=context.filepath,
        metadata=updated_metadata,
        loader_info=context.loader_info,
        schema_version=context.schema_version,
    )


def _apply_image_loaded_hooks(
    context: InputContext, hooks: List[Any], role: str
) -> InputContext:
    context = _ensure_frame_role(context, role)
    for hook in hooks:
        try:
            context = _normalize_input_context(
                hook.on_image_loaded(context),
                filepath=context.filepath,
            )
        except Exception as exc:
            logger.warning(
                "on_image_loaded hook failed for %s (%s): %s",
                context.filepath,
                role,
                exc,
            )
        else:
            context = _ensure_frame_role(context, role)
    return context


def _resolve_detector(
    detector: Optional[BaseDetector] = None,
    detector_name: Optional[str] = None,
    detector_config: Optional[Dict[str, Any]] = None,
) -> BaseDetector:
    """Resolve detector instance from various input combinations.

    Priority order:
    1. Explicit detector instance (if provided)
    2. detector_name lookup via DetectorRegistry (creates new instance)
    3. Default detector (HoughDetector)

    Args:
        detector: Pre-initialized BaseDetector instance.
        detector_name: Name of detector to use (e.g., "hough").
        detector_config: Configuration dict for the detector (currently unused,
            reserved for future detector configuration support).

    Returns:
        BaseDetector instance ready for use.

    Raises:
        MeteorConfigError: If detector_name is not found in available detectors.

    Example:
        >>> det = _resolve_detector()  # Returns default HoughDetector
        >>> det = _resolve_detector(detector_name="hough")  # Returns new HoughDetector
        >>> det = _resolve_detector(detector=my_detector)  # Returns my_detector
    """
    if detector is not None:
        logger.debug("Using provided detector instance: %s", type(detector).__name__)
        return detector

    if detector_name:
        logger.debug("Resolving detector by name: %s", detector_name)
        try:
            return DetectorRegistry.create(detector_name, detector_config)
        except KeyError as e:
            available = DetectorRegistry.list_available()
            logger.error(
                "Detector '%s' not found. Available: %s",
                detector_name,
                available,
            )
            raise MeteorConfigError(
                f"Unknown detector: '{detector_name}'",
                config_key="detector_name",
                plugin_name=detector_name,
                original_error=e,
                context={"available_detectors": available},
            ) from e

    logger.debug("Using default detector")
    return _get_default_detector()


def _resolve_input_loader(
    input_loader: Optional[BaseInputLoader] = None,
    loader_name: Optional[str] = None,
    loader_config: Optional[Dict[str, Any]] = None,
) -> BaseInputLoader:
    """Resolve input loader instance from various input combinations.

    Priority order:
    1. Explicit loader instance (if provided)
    2. loader_name lookup via LoaderRegistry (creates new instance)
    3. Default loader (raw)

    Args:
        input_loader: Pre-initialized BaseInputLoader instance.
        loader_name: Name of loader to use (e.g., "raw").
        loader_config: Configuration dict or instance for the loader.

    Returns:
        BaseInputLoader instance ready for use.

    Raises:
        MeteorConfigError: If loader_name is not found in available loaders.
    """
    if input_loader is not None:
        logger.debug("Using provided loader instance: %s", type(input_loader).__name__)
        return input_loader

    if loader_name:
        logger.debug("Resolving loader by name: %s", loader_name)
        try:
            return LoaderRegistry.create(loader_name, loader_config)
        except KeyError as e:
            available = list(LoaderRegistry.list_loaders().keys())
            logger.error(
                "Loader '%s' not found. Available: %s",
                loader_name,
                available,
            )
            raise MeteorConfigError(
                f"Unknown input loader: '{loader_name}'",
                config_key="loader_name",
                plugin_name=loader_name,
                original_error=e,
                context={"available_loaders": available},
            ) from e

    logger.debug("Using default input loader")
    return _get_default_input_loader()


def _resolve_output_handler(
    output_handler: Optional[BaseOutputHandler] = None,
    handler_name: Optional[str] = None,
    handler_config: Optional[Dict[str, Any]] = None,
    *,
    # Fallback values for default handler creation
    fallback_output_folder: Optional[str] = None,
    fallback_debug_folder: Optional[str] = None,
    fallback_output_overwrite: bool = False,
) -> BaseOutputHandler:
    """Resolve output handler instance from various input combinations.

    Priority order:
    1. Explicit handler instance (if provided)
    2. handler_name lookup via OutputHandlerRegistry (creates new instance)
    3. Default handler (FileOutputHandler) with fallback config

    Args:
        output_handler: Pre-initialized BaseOutputHandler instance.
        handler_name: Name of handler to use (e.g., "file").
        handler_config: Configuration dict or instance for the handler.
        fallback_output_folder: Output folder for default handler creation.
        fallback_debug_folder: Debug folder for default handler creation.
        fallback_output_overwrite: Overwrite flag for default handler creation.

    Returns:
        BaseOutputHandler instance ready for use.

    Raises:
        MeteorConfigError: If handler_name is not found in available handlers,
            or if default handler is needed but fallback folders not provided.

    Example:
        >>> handler = _resolve_output_handler()  # Raises MeteorConfigError (no fallback)
        >>> handler = _resolve_output_handler(handler_name="file", handler_config={...})
        >>> handler = _resolve_output_handler(output_handler=my_handler)
        >>> handler = _resolve_output_handler(
        ...     fallback_output_folder="./out",
        ...     fallback_debug_folder="./debug",
        ... )
    """
    if output_handler is not None:
        logger.debug(
            "Using provided output handler instance: %s",
            type(output_handler).__name__,
        )
        return output_handler

    if handler_name:
        logger.debug("Resolving output handler by name: %s", handler_name)
        try:
            return OutputHandlerRegistry.create(handler_name, handler_config)
        except KeyError as e:
            available = OutputHandlerRegistry.list_available()
            logger.error(
                "Output handler '%s' not found. Available: %s",
                handler_name,
                available,
            )
            raise MeteorConfigError(
                f"Unknown output handler: '{handler_name}'",
                config_key="handler_name",
                plugin_name=handler_name,
                original_error=e,
                context={"available_handlers": available},
            ) from e

    # Create default handler with fallback config
    if fallback_output_folder and fallback_debug_folder:
        logger.debug(
            "Creating default output handler: output=%s, debug=%s",
            fallback_output_folder,
            fallback_debug_folder,
        )
        return OutputHandlerRegistry.create_default(
            output_folder=fallback_output_folder,
            debug_folder=fallback_debug_folder,
            output_overwrite=fallback_output_overwrite,
        )

    logger.error(
        "Cannot resolve output handler: no handler or fallback config provided"
    )
    raise MeteorConfigError(
        "Cannot resolve output handler: provide output_handler, handler_name, "
        "or fallback folder configuration (fallback_output_folder, fallback_debug_folder).",
        context={
            "fallback_output_folder": fallback_output_folder,
            "fallback_debug_folder": fallback_debug_folder,
        },
    )


def _extract_metadata_from_loader(
    loader: BaseInputLoader, filepath: str
) -> Dict[str, Any]:
    """Extract metadata using the loader if it supports it, otherwise use default."""
    if supports_metadata_extraction(loader):
        return loader.extract_metadata(filepath)  # type: ignore[union-attr]
    return extract_exif_metadata(filepath)


class DetectionPipeline(Protocol):
    """Protocol describing the interface for meteor detection pipelines.

    This protocol defines the contract that all detection pipeline implementations
    must follow. It uses properties to allow both attribute access and method
    calls to work correctly with isinstance checks.

    Implementers should provide:
        - target_folder property: Input folder path
        - output_folder property: Output folder path
        - debug_folder property: Debug output folder path
        - params property: Detection parameters
        - extract_metadata method: Metadata extraction
        - run method: Main execution
    """

    @property
    def config(self) -> PipelineConfig:
        """Pipeline configuration."""
        ...

    @property
    def target_folder(self) -> str:
        """Input folder containing RAW files."""
        ...

    @property
    def output_folder(self) -> str:
        """Output folder for detected candidates."""
        ...

    @property
    def debug_folder(self) -> str:
        """Folder for debug images."""
        ...

    @property
    def params(self) -> DetectionParams:
        """Detection parameters."""
        ...

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata for the provided file using the configured loader.

        Args:
            filepath: Path to the file to extract metadata from.

        Returns:
            Dictionary containing file metadata.
        """
        ...

    def run(
        self,
        enable_roi_selection: bool = True,
        roi_polygon_cli: Optional[List[List[int]]] = None,
        resume: bool = True,
        profile: bool = False,
        debug_image_enabled: bool = True,
    ) -> int:
        """Run the detection pipeline and return number of detected candidates.

        Args:
            enable_roi_selection: Whether to enable interactive ROI selection.
            roi_polygon_cli: Pre-defined ROI polygon from command line.
            resume: Whether to resume from previous progress.
            profile: Whether to collect performance metrics.
            debug_image_enabled: Whether to include debug images in results.

        Returns:
            Number of detected meteor candidates.
        """
        ...


def _init_worker_ignore_interrupt() -> None:
    """Ignore SIGINT in worker processes."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def collect_files(target_folder: str) -> List[str]:
    """
    Collect RAW files from the specified folder.

    Args:
        target_folder: Path to the folder to search for RAW files

    Returns:
        Sorted list of RAW file paths

    Raises:
        MeteorLoadError: If the directory doesn't exist, is not a directory,
            or no RAW files are found.
    """
    logger.info("Collecting RAW files from: %s", target_folder)

    # Check if the directory exists
    if not os.path.exists(target_folder):
        logger.error("Directory does not exist: %s", target_folder)
        raise MeteorLoadError(
            "Directory does not exist",
            filepath=target_folder,
            context={"error_category": "directory_not_found"},
        )

    # Check if the path is a directory
    if not os.path.isdir(target_folder):
        logger.error("Path is not a directory: %s", target_folder)
        raise MeteorLoadError(
            "Path is not a directory",
            filepath=target_folder,
            context={"error_category": "not_a_directory"},
        )

    # Collect RAW files
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(target_folder, ext)))

    # Check if any RAW files were found
    if not files:
        logger.warning(
            "No RAW files found in directory: %s (supported: %s)",
            target_folder,
            ", ".join(EXTENSIONS),
        )
        raise MeteorLoadError(
            "No RAW image files found in directory",
            filepath=target_folder,
            context={
                "error_category": "no_files_found",
                "supported_formats": ", ".join(EXTENSIONS),
            },
        )

    files.sort()
    logger.info("Found %d RAW files in %s", len(files), target_folder)
    return files


def validate_raw_file(
    index: int, raw_file: str, input_loader: Optional[BaseInputLoader] = None
) -> Tuple[int, str, Optional[Exception]]:
    """
    Attempt to load a RAW file, returning any validation error.

    Args:
        index: File index for reference
        raw_file: Path to RAW file
        input_loader: Optional input loader instance

    Returns:
        Tuple of (index, filepath, error or None).
        If successful, error is None. If failed, error is a MeteorError subclass.
    """
    loader = _resolve_input_loader(input_loader)

    logger.debug("Validating RAW file [%d]: %s", index, raw_file)

    try:
        context = loader.load(raw_file)
        _normalize_input_context(context, filepath=raw_file)
        logger.debug("Validation successful [%d]: %s", index, raw_file)
        return index, raw_file, None
    except MeteorError as exc:
        # Already a MeteorError, pass through
        logger.warning(
            "Validation failed [%d]: %s - %s",
            index,
            raw_file,
            exc.message,
        )
        return index, raw_file, exc
    except Exception as exc:
        # Wrap unexpected exceptions
        logger.warning(
            "Validation failed [%d]: %s - %s: %s",
            index,
            raw_file,
            type(exc).__name__,
            exc,
        )
        wrapped = MeteorLoadError(
            f"Validation failed: {type(exc).__name__}",
            filepath=raw_file,
            original_error=exc,
            context={"error_category": "validation_failed", "index": index},
        )
        return index, raw_file, wrapped


def process_image_batch(
    batch_data: List[Tuple[int, str, str]],
    roi_mask: np.ndarray,
    params: dict,
    input_loader: Optional[BaseInputLoader] = None,
    detector: Optional[BaseDetector] = None,
    debug_image_enabled: bool = True,
) -> List[
    Tuple[
        bool,
        str,
        str,
        float,
        Optional[Any],
        float,
        int,
        Optional[DetectionResult],
        Optional[Dict[str, Any]],
    ]
]:
    """
    Process a batch of images (handle multiple pairs at once).

    Args:
        batch_data: List of [(frame_index, curr_file, prev_file), ...]
        roi_mask: ROI mask
        params: Parameter dictionary
        input_loader: Optional input loader instance
        detector: Optional detector instance (defaults to HoughDetector)
        debug_image_enabled: Whether to include debug images in results

    Returns:
        List of processing results for each image:
        [
            (
                is_candidate,
                filename,
                filepath,
                line_score,
                debug_img,
                aspect_ratio,
                num_lines,
                detection_result,
                detection_context_payload,
            ),
            ...
        ]

    Note:
        Errors during processing are logged but do not stop batch processing.
        Failed images are returned with is_candidate=False.
    """
    results = []

    loader = _resolve_input_loader(input_loader)
    det = _resolve_detector(detector=detector)
    runtime_params = _build_runtime_params(params, det)
    hooks = HookRegistry.create_all()

    logger.debug("Processing batch of %d image pairs", len(batch_data))

    for frame_index, curr_file, prev_file in batch_data:
        filename = os.path.basename(curr_file)
        prev_frame_index = frame_index - 1

        try:
            # Load images
            curr_context = _normalize_input_context(
                loader.load(curr_file),
                filepath=curr_file,
            )
            prev_context = _normalize_input_context(
                loader.load(prev_file),
                filepath=prev_file,
            )
            curr_context = _apply_image_loaded_hooks(
                curr_context,
                hooks,
                role="current",
            )
            prev_context = _apply_image_loaded_hooks(
                prev_context,
                hooks,
                role="previous",
            )

            # Delegate detection to the detector
            try:
                metadata = {
                    "current": curr_context.metadata,
                    "previous": prev_context.metadata,
                    "frame_index": frame_index,
                    "prev_frame_index": prev_frame_index,
                }
            except Exception as exc:
                logger.warning("Metadata extraction failed for %s: %s", filename, exc)
                metadata = {
                    "current": {},
                    "previous": {},
                    "frame_index": frame_index,
                    "prev_frame_index": prev_frame_index,
                }
            context = DetectionContext(
                current_image=curr_context.image_data,
                previous_image=prev_context.image_data,
                roi_mask=roi_mask,
                runtime_params=runtime_params,
                metadata=metadata,
            )
            context = _normalize_detection_context(context)
            result = _normalize_detection_result(det.detect(context))
            context_payload = context.to_dict()
            debug_image = result.debug_image if result.is_candidate else None
            if not result.is_candidate or not debug_image_enabled:
                result.debug_image = None
                if not debug_image_enabled:
                    debug_image = None

            results.append(
                (
                    result.is_candidate,
                    filename,
                    curr_file,
                    result.score,
                    debug_image,
                    result.aspect_ratio,
                    len(result.lines),
                    result,
                    context_payload,
                )
            )

            if result.is_candidate:
                logger.debug(
                    "Candidate detected: %s (score=%.2f, lines=%d)",
                    filename,
                    result.score,
                    len(result.lines),
                )

        except MeteorError as e:
            logger.error(
                "Error processing %s: %s",
                filename,
                e.message,
                extra={
                    "filepath": curr_file,
                    "error_type": type(e).__name__,
                    "error_category": e.context.get("error_category"),
                },
            )
            results.append((False, filename, curr_file, 0.0, None, 0.0, 0, None, None))

        except Exception as e:
            logger.error(
                "Unexpected error processing %s: %s: %s",
                filename,
                type(e).__name__,
                e,
                exc_info=True,
                extra={"filepath": curr_file, "error_type": type(e).__name__},
            )
            results.append((False, filename, curr_file, 0.0, None, 0.0, 0, None, None))

    return results


def _extract_frame_indices(
    detection_context_payload: Optional[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int]]:
    ctx_frame_index = None
    ctx_prev_frame_index = None
    if detection_context_payload and isinstance(detection_context_payload, dict):
        metadata = detection_context_payload.get("metadata", {})
        ctx_frame_index = metadata.get("frame_index")
        ctx_prev_frame_index = metadata.get("prev_frame_index")
    return ctx_frame_index, ctx_prev_frame_index


def estimate_diff_threshold_from_samples(
    files: List[str],
    roi_mask: np.ndarray,
    sample_size: int = 5,
    input_loader: Optional[BaseInputLoader] = None,
    locale: Optional[str] = None,
) -> int:
    """
    Estimation using percentile-based approach.

    Real-world sky brightness distributions are highly peaked, so
    percentile-based estimation is more appropriate than 3-sigma rule.

    Args:
        files: List of RAW file paths
        roi_mask: ROI mask to focus on sky area
        sample_size: Number of initial images to analyze

    Returns:
        Estimated diff_threshold value
    """
    locale = _resolve_locale(locale)
    print(f"\n{'=' * 50}")
    print(
        get_message(
            "ui.pipeline.estimate.diff_threshold.header",
            locale=locale,
            sample_size=sample_size,
        )
    )
    print(get_message("ui.pipeline.estimate.diff_threshold.subtitle", locale=locale))
    print(f"{'=' * 50}")

    sample_size = min(sample_size, len(files))
    if sample_size < 2:
        print(
            get_message(
                "ui.pipeline.estimate.diff_threshold.too_few_samples", locale=locale
            )
        )
        return DEFAULT_DIFF_THRESHOLD

    samples = []
    print(
        get_message("ui.pipeline.estimate.loading_samples", locale=locale),
        end="",
        flush=True,
    )
    loader = _resolve_input_loader(input_loader)

    for i in range(sample_size):
        try:
            context = _normalize_input_context(
                loader.load(files[i]),
                filepath=files[i],
            )
            samples.append(context.image_data)
        except Exception as exc:
            print(
                get_message(
                    "ui.pipeline.estimate.sample_load_failed",
                    locale=locale,
                    index=i,
                    error=exc,
                )
            )
            continue
    print(
        get_message(
            "ui.pipeline.estimate.samples_loaded",
            locale=locale,
            count=len(samples),
        )
    )

    if len(samples) < 2:
        print(
            get_message("ui.pipeline.estimate.not_enough_valid_samples", locale=locale)
        )
        return DEFAULT_DIFF_THRESHOLD

    # Calculate frame-to-frame differences in ROI
    print(
        get_message("ui.pipeline.estimate.diff_threshold.analyzing", locale=locale),
        end="",
        flush=True,
    )
    diff_values = []
    for i in range(1, len(samples)):
        diff = cv2.absdiff(samples[i], samples[i - 1])
        roi_diff = diff[roi_mask == 255]
        diff_values.extend(roi_diff.flatten())
    print(get_message("ui.pipeline.estimate.done", locale=locale))

    diff_array = np.array(diff_values, dtype=np.float32)

    # Calculate statistics
    mean_diff = np.mean(diff_array)
    std_diff = np.std(diff_array)
    median_diff = np.median(diff_array)

    # Percentiles
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

    print(f"\n{'─' * 50}")
    print(
        get_message(
            "ui.pipeline.estimate.diff_threshold.stats.header",
            locale=locale,
            pixel_count=len(diff_values),
        )
    )
    print(f"{'─' * 50}")
    stats_labels = {
        "mean": get_message(
            "ui.pipeline.estimate.diff_threshold.stats.label.mean", locale=locale
        ),
        "std_dev": get_message(
            "ui.pipeline.estimate.diff_threshold.stats.label.std_dev", locale=locale
        ),
        "median": get_message(
            "ui.pipeline.estimate.diff_threshold.stats.label.median", locale=locale
        ),
        "p90": get_message(
            "ui.pipeline.estimate.diff_threshold.stats.label.p90", locale=locale
        ),
        "p95": get_message(
            "ui.pipeline.estimate.diff_threshold.stats.label.p95", locale=locale
        ),
        "p98": get_message(
            "ui.pipeline.estimate.diff_threshold.stats.label.p98", locale=locale
        ),
        "p99": get_message(
            "ui.pipeline.estimate.diff_threshold.stats.label.p99", locale=locale
        ),
    }
    stats_width = max(_display_width(label) for label in stats_labels.values())

    def format_stat(label_key: str, value: float) -> str:
        return f"  {_pad_label(stats_labels[label_key], stats_width)} {value:.2f}"

    print(format_stat("mean", mean_diff))
    print(format_stat("std_dev", std_diff))
    print(format_stat("median", median_diff))
    print(format_stat("p90", p90))
    print(format_stat("p95", p95))
    print(format_stat("p98", p98))
    print(format_stat("p99", p99))
    print(f"{'─' * 50}")
    print(
        get_message("ui.pipeline.estimate.diff_threshold.methods.header", locale=locale)
    )
    method_labels = {
        "p98": get_message(
            "ui.pipeline.estimate.diff_threshold.methods.label.p98", locale=locale
        ),
        "mean_sigma": get_message(
            "ui.pipeline.estimate.diff_threshold.methods.label.mean_sigma",
            locale=locale,
        ),
        "median": get_message(
            "ui.pipeline.estimate.diff_threshold.methods.label.median", locale=locale
        ),
    }
    method_width = max(_display_width(label) for label in method_labels.values())

    def format_method(label_key: str, value: int) -> str:
        return f"  {_pad_label(method_labels[label_key], method_width)} {value}"

    print(format_method("p98", method_1))
    print(format_method("mean_sigma", method_2))
    print(format_method("median", method_3))
    print(f"{'─' * 50}")
    print(
        get_message(
            "ui.pipeline.estimate.diff_threshold.selected",
            locale=locale,
            value=estimated_threshold,
        )
    )
    print(f"{'=' * 50}\n")

    return estimated_threshold


def estimate_min_area_from_samples(
    files: List[str],
    roi_mask: np.ndarray,
    diff_threshold: int,
    sample_size: int = 3,
    input_loader: Optional[BaseInputLoader] = None,
    locale: Optional[str] = None,
) -> int:
    """
    Improved min_area estimation with better star detection.

    Args:
        files: List of RAW file paths
        roi_mask: ROI mask to focus on sky area
        diff_threshold: Threshold for star detection
        sample_size: Number of images to analyze

    Returns:
        Estimated min_area value
    """
    locale = _resolve_locale(locale)
    print(f"\n{'=' * 50}")
    print(
        get_message(
            "ui.pipeline.estimate.min_area.header",
            locale=locale,
            sample_size=sample_size,
        )
    )
    print(get_message("ui.pipeline.estimate.min_area.subtitle", locale=locale))
    print(f"{'=' * 50}")

    sample_size = min(sample_size, len(files))
    if sample_size < 1:
        print(
            get_message("ui.pipeline.estimate.min_area.too_few_samples", locale=locale)
        )
        return DEFAULT_MIN_AREA

    samples = []
    print(
        get_message("ui.pipeline.estimate.loading_samples", locale=locale),
        end="",
        flush=True,
    )
    loader = _resolve_input_loader(input_loader)

    for i in range(sample_size):
        try:
            context = _normalize_input_context(
                loader.load(files[i]),
                filepath=files[i],
            )
            samples.append(context.image_data)
        except Exception as exc:
            print(
                get_message(
                    "ui.pipeline.estimate.sample_load_failed",
                    locale=locale,
                    index=i,
                    error=exc,
                )
            )
            continue
    print(
        get_message(
            "ui.pipeline.estimate.samples_loaded",
            locale=locale,
            count=len(samples),
        )
    )

    if not samples:
        print(get_message("ui.pipeline.estimate.no_valid_samples", locale=locale))
        return DEFAULT_MIN_AREA

    print(
        get_message("ui.pipeline.estimate.min_area.detecting_stars", locale=locale),
        end="",
        flush=True,
    )
    all_star_areas = []

    for img in samples:
        roi_pixels = img[roi_mask == 255]

        if len(roi_pixels) < 100:
            continue

        # Use 98th percentile (brighter stars only, avoid noise)
        threshold = np.percentile(roi_pixels, 98)

        _, star_mask = cv2.threshold(img, int(threshold), 255, cv2.THRESH_BINARY)
        star_mask = cv2.bitwise_and(star_mask.astype(np.uint8), roi_mask)

        contours, _ = cv2.findContours(
            star_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by area range to exclude noise and large artifacts
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 2.0 <= area <= 100.0:  # Exclude tiny noise and large artifacts
                all_star_areas.append(area)

    print(
        get_message(
            "ui.pipeline.estimate.min_area.detected_stars",
            locale=locale,
            count=len(all_star_areas),
        )
    )

    if len(all_star_areas) < 10:
        print(get_message("ui.pipeline.estimate.min_area.too_few_stars", locale=locale))
        return DEFAULT_MIN_AREA

    star_areas = np.array(all_star_areas)
    median_star = np.median(star_areas)
    mean_star = np.mean(star_areas)
    p75_star = np.percentile(star_areas, 75)
    p90_star = np.percentile(star_areas, 90)

    # Use 75th percentile × 2.0 for more robust estimation
    estimated_min_area = int(p75_star * 2.0)

    # Ensure minimum is at least default value
    estimated_min_area = max(estimated_min_area, DEFAULT_MIN_AREA)

    # Clamp to reasonable range
    estimated_min_area = np.clip(estimated_min_area, 8, 50)

    print(f"\n{'─' * 50}")
    print(
        get_message(
            "ui.pipeline.estimate.min_area.stats.header",
            locale=locale,
            count=len(all_star_areas),
        )
    )
    print(f"{'─' * 50}")
    stats_labels = {
        "median": get_message(
            "ui.pipeline.estimate.min_area.stats.label.median", locale=locale
        ),
        "mean": get_message(
            "ui.pipeline.estimate.min_area.stats.label.mean", locale=locale
        ),
        "p75": get_message(
            "ui.pipeline.estimate.min_area.stats.label.p75", locale=locale
        ),
        "p90": get_message(
            "ui.pipeline.estimate.min_area.stats.label.p90", locale=locale
        ),
    }
    stats_width = max(_display_width(label) for label in stats_labels.values())

    def format_stat(label_key: str, value: float) -> str:
        return (
            f"  {_pad_label(stats_labels[label_key], stats_width)} "
            f"{get_message('ui.pipeline.estimate.min_area.stats.value.pixels', locale=locale, value=value)}"
        )

    print(format_stat("median", median_star))
    print(format_stat("mean", mean_star))
    print(format_stat("p75", p75_star))
    print(format_stat("p90", p90_star))
    print(f"{'─' * 50}")
    print(
        get_message(
            "ui.pipeline.estimate.min_area.estimated",
            locale=locale,
            value=estimated_min_area,
        )
    )
    print(get_message("ui.pipeline.estimate.min_area.method_note", locale=locale))
    print(f"{'=' * 50}\n")

    return estimated_min_area


def estimate_min_line_score_from_image(
    image_shape: Tuple[int, int],
    focal_length_mm: Optional[float] = None,
    locale: Optional[str] = None,
) -> float:
    """
    Fixed min_line_score estimation with corrected focal length logic.

    Args:
        image_shape: (height, width) of image
        focal_length_mm: Focal length in mm (optional)

    Returns:
        Estimated min_line_score value
    """
    locale = _resolve_locale(locale)
    print(f"\n{'=' * 50}")
    print(get_message("ui.pipeline.estimate.min_line_score.header", locale=locale))
    print(f"{'=' * 50}")

    height, width = image_shape
    diagonal = np.sqrt(height**2 + width**2)

    # Reduced base coefficient from 4% to 2.5% based on real data
    base_score = diagonal * 0.025

    if focal_length_mm:
        focal_factor = focal_length_mm / 50.0
        adjusted_score = base_score * focal_factor

        print(f"\n{'─' * 50}")
        print(
            get_message(
                "ui.pipeline.estimate.min_line_score.geometry.header", locale=locale
            )
        )
        print(f"{'─' * 50}")
        geometry_labels = {
            "dimensions": get_message(
                "ui.pipeline.estimate.min_line_score.geometry.label.dimensions",
                locale=locale,
            ),
            "diagonal": get_message(
                "ui.pipeline.estimate.min_line_score.geometry.label.diagonal",
                locale=locale,
            ),
            "focal_length": get_message(
                "ui.pipeline.estimate.min_line_score.geometry.label.focal_length",
                locale=locale,
            ),
            "focal_factor": get_message(
                "ui.pipeline.estimate.min_line_score.geometry.label.focal_factor",
                locale=locale,
            ),
            "base_score": get_message(
                "ui.pipeline.estimate.min_line_score.geometry.label.base_score",
                locale=locale,
            ),
            "adjusted_score": get_message(
                "ui.pipeline.estimate.min_line_score.geometry.label.adjusted_score",
                locale=locale,
            ),
        }
        geometry_width = max(
            _display_width(label) for label in geometry_labels.values()
        )

        def format_geometry(label_key: str, value: str) -> str:
            return f"  {_pad_label(geometry_labels[label_key], geometry_width)} {value}"

        print(
            format_geometry(
                "dimensions",
                get_message(
                    "ui.pipeline.estimate.min_line_score.geometry.value.dimensions",
                    locale=locale,
                    width=width,
                    height=height,
                ),
            )
        )
        print(
            format_geometry(
                "diagonal",
                get_message(
                    "ui.pipeline.estimate.min_line_score.geometry.value.diagonal",
                    locale=locale,
                    value=diagonal,
                ),
            )
        )
        print(
            format_geometry(
                "focal_length",
                get_message(
                    "ui.pipeline.estimate.min_line_score.geometry.value.focal_length",
                    locale=locale,
                    value=focal_length_mm,
                ),
            )
        )
        print(
            format_geometry(
                "focal_factor",
                get_message(
                    "ui.pipeline.estimate.min_line_score.geometry.value.focal_factor",
                    locale=locale,
                    value=focal_factor,
                ),
            )
        )
        print(
            format_geometry(
                "base_score",
                get_message(
                    "ui.pipeline.estimate.min_line_score.geometry.value.base_score",
                    locale=locale,
                    value=base_score,
                ),
            )
        )
        print(
            format_geometry(
                "adjusted_score",
                get_message(
                    "ui.pipeline.estimate.min_line_score.geometry.value.adjusted_score",
                    locale=locale,
                    value=adjusted_score,
                ),
            )
        )
    else:
        adjusted_score = base_score
        print(f"\n{'─' * 50}")
        print(
            get_message(
                "ui.pipeline.estimate.min_line_score.geometry.header", locale=locale
            )
        )
        print(f"{'─' * 50}")
        geometry_labels = {
            "dimensions": get_message(
                "ui.pipeline.estimate.min_line_score.geometry.label.dimensions",
                locale=locale,
            ),
            "diagonal": get_message(
                "ui.pipeline.estimate.min_line_score.geometry.label.diagonal",
                locale=locale,
            ),
            "base_score": get_message(
                "ui.pipeline.estimate.min_line_score.geometry.label.base_score",
                locale=locale,
            ),
        }
        geometry_width = max(
            _display_width(label) for label in geometry_labels.values()
        )

        def format_geometry(label_key: str, value: str) -> str:
            return f"  {_pad_label(geometry_labels[label_key], geometry_width)} {value}"

        print(
            format_geometry(
                "dimensions",
                get_message(
                    "ui.pipeline.estimate.min_line_score.geometry.value.dimensions",
                    locale=locale,
                    width=width,
                    height=height,
                ),
            )
        )
        print(
            format_geometry(
                "diagonal",
                get_message(
                    "ui.pipeline.estimate.min_line_score.geometry.value.diagonal",
                    locale=locale,
                    value=diagonal,
                ),
            )
        )
        print(
            format_geometry(
                "base_score",
                get_message(
                    "ui.pipeline.estimate.min_line_score.geometry.value.base_score",
                    locale=locale,
                    value=base_score,
                ),
            )
        )
        print(
            get_message(
                "ui.pipeline.estimate.min_line_score.geometry.no_focal", locale=locale
            )
        )

    # Adjusted clamp range based on real meteor data
    estimated_score = np.clip(adjusted_score, 40.0, 150.0)

    print(f"{'─' * 50}")
    print(
        get_message(
            "ui.pipeline.estimate.min_line_score.estimated",
            locale=locale,
            value=estimated_score,
        )
    )
    print(get_message("ui.pipeline.estimate.min_line_score.method_note", locale=locale))
    print(f"{'=' * 50}\n")

    return estimated_score


class MeteorDetectionPipeline:
    """
    Main pipeline class for meteor detection processing.

    Orchestrates the entire detection workflow including:
    - File collection and validation
    - ROI selection
    - Parameter optimization
    - Batch processing with multiprocessing
    - Progress tracking and resumption

    This class implements the DetectionPipeline protocol.

    There are two ways to initialize this class:

    1. New API (recommended): Pass a PipelineConfig object
       >>> config = PipelineConfig(
       ...     target_folder="./raw",
       ...     output_folder="./candidates",
       ...     debug_folder="./debug",
       ...     params=DetectionParams(),
       ... )
       >>> pipeline = MeteorDetectionPipeline(config)

    2. Legacy API (backward compatible): Pass individual arguments
       >>> pipeline = MeteorDetectionPipeline(
       ...     target_folder="./raw",
       ...     output_folder="./candidates",
       ...     debug_folder="./debug",
       ...     params=DetectionParams(),
       ... )
    """

    @overload
    def __init__(
        self,
        config: PipelineConfig,
        *,
        input_loader: Optional[BaseInputLoader] = None,
        input_loader_name: Optional[str] = None,
        input_loader_config: Optional[Dict[str, Any]] = None,
        output_handler: Optional[BaseOutputHandler] = None,
        detector: Optional[BaseDetector] = None,
    ) -> None:
        """Initialize with PipelineConfig (new API)."""
        ...

    @overload
    def __init__(
        self,
        target_folder: str,
        output_folder: str,
        debug_folder: str,
        params: DetectionParams,
        num_workers: int = ...,
        batch_size: int = ...,
        auto_batch_size: bool = ...,
        enable_parallel: bool = ...,
        progress_file: str = ...,
        output_overwrite: bool = ...,
        input_loader: Optional[BaseInputLoader] = ...,
        input_loader_name: Optional[str] = ...,
        input_loader_config: Optional[Dict[str, Any]] = ...,
        output_handler: Optional[BaseOutputHandler] = ...,
        detector: Optional[BaseDetector] = ...,
    ) -> None:
        """Initialize with individual arguments (legacy API)."""
        ...

    def __init__(
        self,
        config_or_target: Union[PipelineConfig, str],
        output_folder: Optional[str] = None,
        debug_folder: Optional[str] = None,
        params: Optional[DetectionParams] = None,
        num_workers: int = DEFAULT_NUM_WORKERS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        auto_batch_size: bool = False,
        enable_parallel: bool = True,
        progress_file: str = DEFAULT_PROGRESS_FILE,
        output_overwrite: bool = False,
        input_loader: Optional[BaseInputLoader] = None,
        input_loader_name: Optional[str] = None,
        input_loader_config: Optional[Dict[str, Any]] = None,
        output_handler: Optional[BaseOutputHandler] = None,
        detector: Optional[BaseDetector] = None,
    ):
        """
        Initialize the detection pipeline.

        Can be called in two ways:

        1. New API with PipelineConfig:
           MeteorDetectionPipeline(config, input_loader=..., output_handler=...)

        2. Legacy API with individual arguments:
           MeteorDetectionPipeline(target_folder, output_folder, debug_folder, params, ...)

        Args:
            config_or_target: Either a PipelineConfig object (new API) or
                target_folder string (legacy API)
            output_folder: Output folder (legacy API only, ignored if config provided)
            debug_folder: Debug folder (legacy API only, ignored if config provided)
            params: Detection parameters (legacy API only, ignored if config provided)
            num_workers: Number of parallel workers (legacy API only)
            batch_size: Batch size for processing (legacy API only)
            auto_batch_size: Auto-calculate batch size (legacy API only)
            enable_parallel: Enable parallel processing (legacy API only)
            progress_file: Progress tracking file path (legacy API only)
            output_overwrite: Overwrite existing files (legacy API only)
            input_loader: Pre-initialized BaseInputLoader instance
            input_loader_name: Name of loader plugin to use
            input_loader_config: Configuration for the loader
            output_handler: Custom BaseOutputHandler implementation
            detector: Custom BaseDetector implementation (defaults to HoughDetector)

        Raises:
            MeteorConfigError: If configuration is invalid or required parameters
                are missing.
        """
        logger.debug(
            "Initializing MeteorDetectionPipeline with %s",
            type(config_or_target).__name__,
        )

        # Determine which API is being used
        if isinstance(config_or_target, PipelineConfig):
            # New API: PipelineConfig provided
            self._config = config_or_target
            logger.debug("Using PipelineConfig API")
        elif isinstance(config_or_target, str):
            # Legacy API: individual arguments
            if output_folder is None or debug_folder is None or params is None:
                logger.error(
                    "Legacy API requires output_folder, debug_folder, and params"
                )
                raise MeteorConfigError(
                    "When using legacy API, output_folder, debug_folder, and params "
                    "are required. Consider using PipelineConfig instead.",
                    context={
                        "output_folder": output_folder,
                        "debug_folder": debug_folder,
                        "params_provided": params is not None,
                    },
                )

            # Issue deprecation warning for legacy API
            warnings.warn(
                "Passing individual arguments to MeteorDetectionPipeline is deprecated. "
                "Use PipelineConfig instead: "
                "MeteorDetectionPipeline(PipelineConfig(target_folder=..., ...))",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.debug("Using legacy API (deprecated)")

            self._config = PipelineConfig(
                target_folder=config_or_target,
                output_folder=output_folder,
                debug_folder=debug_folder,
                params=params,
                num_workers=num_workers,
                batch_size=batch_size,
                auto_batch_size=auto_batch_size,
                enable_parallel=enable_parallel,
                progress_file=progress_file,
                output_overwrite=output_overwrite,
            )
        else:
            logger.error(
                "Invalid first argument type: %s", type(config_or_target).__name__
            )
            raise MeteorConfigError(
                f"First argument must be PipelineConfig or str (target_folder), "
                f"got {type(config_or_target).__name__}",
                context={"provided_type": type(config_or_target).__name__},
            )

        # Store loader configuration (explicit args take priority over config)
        self.input_loader = input_loader
        self.input_loader_name = input_loader_name or self._config.input_loader_name
        self.input_loader_config = (
            input_loader_config or self._config.input_loader_config
        )

        # Resolve detector: explicit argument > config > default
        # Note: We store the resolved detector, not the raw argument
        self.detector = _resolve_detector(
            detector=detector,
            detector_name=self._config.detector_name,
            detector_config=self._config.detector_config,
        )

        # Initialize output handler
        self.output_handler = _resolve_output_handler(
            output_handler=output_handler,
            handler_name=self._config.output_handler_name,
            handler_config=self._config.output_handler_config,
            fallback_output_folder=self._config.output_folder,
            fallback_debug_folder=self._config.debug_folder,
            fallback_output_overwrite=self._config.output_overwrite,
        )
        self.progress_manager = ProgressManager(self._config.progress_file)

        logger.info(
            "Pipeline initialized: target=%s, workers=%d, batch_size=%d",
            self._config.target_folder,
            self._config.num_workers,
            self._config.batch_size,
        )

    @property
    def config(self) -> PipelineConfig:
        """Pipeline configuration."""
        return self._config

    @property
    def target_folder(self) -> str:
        """Input folder containing RAW files."""
        return self._config.target_folder

    @property
    def output_folder(self) -> str:
        """Output folder for detected candidates."""
        return self._config.output_folder

    @property
    def debug_folder(self) -> str:
        """Folder for debug images."""
        return self._config.debug_folder

    @property
    def params(self) -> DetectionParams:
        """Detection parameters."""
        return self._config.params

    @property
    def num_workers(self) -> int:
        """Number of parallel workers."""
        return self._config.num_workers

    @property
    def batch_size(self) -> int:
        """Batch size for processing."""
        return self._config.batch_size

    @property
    def auto_batch_size(self) -> bool:
        """Whether to auto-adjust batch size."""
        return self._config.auto_batch_size

    @property
    def enable_parallel(self) -> bool:
        """Whether parallel processing is enabled."""
        return self._config.enable_parallel

    @property
    def progress_file(self) -> str:
        """Path to progress tracking file."""
        return self._config.progress_file

    @property
    def output_overwrite(self) -> bool:
        """Whether to overwrite existing files."""
        return self._config.output_overwrite

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata for the provided file using the configured loader."""

        loader = _resolve_input_loader(
            self.input_loader, self.input_loader_name, self.input_loader_config
        )
        return _extract_metadata_from_loader(loader, filepath)

    def run(
        self,
        enable_roi_selection: bool = True,
        roi_polygon_cli: Optional[List[List[int]]] = None,
        resume: bool = True,
        profile: bool = False,
        debug_image_enabled: bool = True,
    ) -> int:
        """
        Run the full detection pipeline.

        Args:
            enable_roi_selection: Whether to enable interactive ROI selection
            roi_polygon_cli: Pre-defined ROI polygon from CLI
            resume: Whether to resume from previous progress
            profile: Whether to collect performance metrics

        Returns:
            Number of detected candidates
        """
        locale = _resolve_locale(None)
        timing = {}
        t_total = time.time()
        input_loader = _resolve_input_loader(
            self.input_loader, self.input_loader_name, self.input_loader_config
        )

        # Safety check
        target_fullpath = os.path.abspath(self._config.target_folder)
        output_fullpath = os.path.abspath(self._config.output_folder)

        if target_fullpath == output_fullpath:
            print(f"\n{'=' * 60}")
            print(get_message("ui.pipeline.error.same_directories", locale=locale))
            print(f"{'=' * 60}")
            return 0

        # Collect files
        print(
            get_message(
                "ui.pipeline.collecting",
                locale=locale,
                path=self._config.target_folder,
            )
        )
        files = collect_files(self._config.target_folder)

        hooks = HookRegistry.create_all()
        normalized_files = [os.path.normpath(os.path.abspath(path)) for path in files]
        filtered_files = []
        for filepath in normalized_files:
            keep = True
            for hook in hooks:
                try:
                    if not hook.on_file_found(filepath):
                        keep = False
                        break
                except Exception as exc:
                    logger.warning(
                        "on_file_found hook failed for %s: %s",
                        filepath,
                        exc,
                    )
            if keep:
                filtered_files.append(filepath)
        files = filtered_files

        if len(files) < 2:
            print(get_message("ui.pipeline.need_two_images", locale=locale))
            return 0

        print(get_message("ui.pipeline.found_files", locale=locale, count=len(files)))

        # Load first image
        t_load = time.time()
        try:
            prev_context = _normalize_input_context(
                input_loader.load(files[0]),
                filepath=files[0],
            )
        except Exception as exc:
            print(
                get_message(
                    "ui.pipeline.load_first_failed",
                    locale=locale,
                    filename=os.path.basename(files[0]),
                    error=exc,
                )
            )
            return 0

        if profile:
            timing["first_load"] = time.time() - t_load
        prev_img = prev_context.image_data
        height, width = prev_img.shape

        # ROI setup
        roi_mask = create_full_roi_mask((height, width))
        roi_polygon = None

        if roi_polygon_cli:
            print(
                get_message(
                    "ui.pipeline.roi.cli_specified",
                    locale=locale,
                    polygon=format_polygon_string(roi_polygon_cli),
                )
            )
            roi_mask = create_roi_mask_from_polygon(roi_polygon_cli, (height, width))
            roi_polygon = roi_polygon_cli
        elif enable_roi_selection:
            roi_selection = select_roi(prev_img, locale=locale)
            if roi_selection:
                roi_mask = roi_selection["mask"]
                roi_polygon = roi_selection["polygon"]
                print(
                    get_message(
                        "ui.pipeline.roi.setup_complete",
                        locale=locale,
                        polygon=format_polygon_string(roi_polygon),
                    )
                )
            else:
                print(get_message("ui.pipeline.roi.none_selected", locale=locale))
        else:
            print(get_message("ui.pipeline.roi.skipped", locale=locale))

        # Get params dict (unscaled for progress tracking)
        params = self._config.params.to_dict()
        normalized_input = _loader_normalizes(input_loader) or _is_normalized_image(
            prev_context.image_data
        )
        runtime_params = _apply_normalized_diff_threshold(params, normalized_input)

        # Progress tracking setup
        params_for_hash = params.copy()
        if roi_polygon:
            params_for_hash["roi_polygon"] = roi_polygon

        self.progress_manager.set_params(params)
        self.progress_manager.set_roi(roi_polygon or "full_image")
        self.progress_manager.set_processing_params(params)
        params_hash = compute_params_hash(params_for_hash)
        self.progress_manager.set_params_hash(params_hash)

        if resume:
            loaded = self.progress_manager.load()
            if loaded and self.progress_manager.get_params_hash() == params_hash:
                print(
                    get_message(
                        "ui.pipeline.progress.resuming",
                        locale=locale,
                        path=self._config.progress_file,
                        processed=self.progress_manager.get_total_processed(),
                        detected=self.progress_manager.get_total_detected(),
                    )
                )
            elif loaded:
                print(get_message("ui.pipeline.progress.param_mismatch", locale=locale))
                self.progress_manager.reset()
                self.progress_manager.set_params_hash(params_hash)

        # Filter existing files
        existing_basenames = {os.path.basename(path) for path in files}
        self.progress_manager.filter_existing_files(existing_basenames)
        self.progress_manager.save()

        # Build image pairs
        image_pairs = [(i, files[i], files[i - 1]) for i in range(1, len(files))]
        image_pairs = [
            pair
            for pair in image_pairs
            if not self.progress_manager.is_processed(os.path.basename(pair[1]))
        ]

        resume_offset = self.progress_manager.get_total_processed()
        overall_total = resume_offset + len(image_pairs)

        print(
            get_message(
                "ui.pipeline.processing.start",
                locale=locale,
                count=len(image_pairs),
            )
        )
        if self._config.enable_parallel:
            print(
                get_message(
                    "ui.pipeline.processing.parallel",
                    locale=locale,
                    workers=self._config.num_workers,
                    batch_size=self._config.batch_size,
                )
            )

        detected_count = self.progress_manager.get_total_detected()
        t_process = time.time()

        try:
            if self._config.enable_parallel and self._config.num_workers > 1:
                detected_count = self._process_parallel(
                    image_pairs,
                    roi_mask,
                    runtime_params,
                    roi_polygon,
                    resume_offset,
                    overall_total,
                    input_loader,
                    self.detector,
                    debug_image_enabled,
                    locale=locale,
                )
            else:
                detected_count = self._process_sequential(
                    image_pairs,
                    roi_mask,
                    runtime_params,
                    roi_polygon,
                    resume_offset,
                    overall_total,
                    input_loader,
                    self.detector,
                    debug_image_enabled,
                    locale=locale,
                )
        except KeyboardInterrupt:
            print(
                "\n"
                + get_message(
                    "ui.pipeline.interrupt.saved",
                    locale=locale,
                    path=self._config.progress_file,
                )
            )
            self.progress_manager.save()
            return self.progress_manager.get_total_detected()

        processing_elapsed = time.time() - t_process
        total_processed = self.progress_manager.get_total_processed()
        total_detected = self.progress_manager.get_total_detected()
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "on_pipeline_complete elapsed=%.2fs processed=%s detected=%s",
                    processing_elapsed,
                    total_processed,
                    total_detected,
                )
            self.output_handler.on_pipeline_complete(
                total_processed,
                total_detected,
                processing_elapsed,
            )
        except Exception as exc:
            logger.warning("on_pipeline_complete hook failed: %s", exc)

        if profile:
            timing["processing"] = processing_elapsed
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

        print(
            "\n"
            + get_message(
                "ui.pipeline.summary",
                locale=locale,
                count=detected_count,
            )
        )
        return detected_count

    def _process_parallel(
        self,
        image_pairs: List[Tuple[int, str, str]],
        roi_mask: np.ndarray,
        params: Dict[str, Any],
        roi_polygon: Optional[List[List[int]]],
        resume_offset: int,
        overall_total: int,
        input_loader: Optional[BaseInputLoader],
        detector: Optional[BaseDetector],
        debug_image_enabled: bool,
        locale: Optional[str] = None,
    ) -> int:
        """Process images in parallel using ProcessPoolExecutor."""
        locale = _resolve_locale(locale)
        start_time = time.time()
        batches = [
            image_pairs[i : i + self._config.batch_size]
            for i in range(0, len(image_pairs), self._config.batch_size)
        ]

        print(
            get_message(
                "ui.pipeline.processing.batches",
                locale=locale,
                count=len(batches),
            )
        )

        executor = ProcessPoolExecutor(
            max_workers=self._config.num_workers,
            initializer=_init_worker_ignore_interrupt,
        )
        futures: List = []
        wait_for_tasks = True

        try:
            for batch_idx, batch in enumerate(batches, 1):
                print(
                    get_message(
                        "ui.pipeline.processing.submitting_batches",
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
                    params,
                    input_loader,
                    detector,
                    debug_image_enabled,
                )
                futures.append(future)

            # Clear the submitting line
            print()

            # Collect results
            processed = 0
            completed_batches = 0
            total_batches = len(batches)

            # Show initial batch progress (0 completed)
            print(
                get_message(
                    "ui.pipeline.processing.batch_progress",
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
                            "ui.pipeline.processing.batch_progress",
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
                            detection_context_payload,
                        ) = result
                        processed += 1

                        # Always show progress first
                        print(
                            get_message(
                                "ui.pipeline.processing.checking",
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
                                    "ui.pipeline.processing.line",
                                    locale=locale,
                                    filename=filename,
                                    score=line_score,
                                    lines=num_lines,
                                )
                            )

                        if detection_result and detection_context_payload:
                            try:
                                self.output_handler.on_detection_result(
                                    detection_context_payload,
                                    detection_result,
                                    filepath,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "on_detection_result hook failed for %s: %s",
                                    filename,
                                    exc,
                                )

                        if is_candidate:
                            if line_score <= 0:
                                print()  # Move to new line if [LINE] wasn't printed
                            output_result = self.output_handler.save_candidate(
                                filepath, filename, debug_img, roi_polygon
                            )
                            output_result = _normalize_output_result(output_result)
                            if output_result.saved:
                                print(
                                    get_message(
                                        "ui.pipeline.processing.hit",
                                        locale=locale,
                                        filename=filename,
                                        ratio=aspect_ratio,
                                    )
                                )
                            else:
                                print(
                                    get_message(
                                        "ui.pipeline.processing.skip",
                                        locale=locale,
                                        filename=filename,
                                    )
                                )
                            try:
                                self.output_handler.on_candidate_detected(
                                    filename,
                                    output_result.saved,
                                    score=line_score,
                                    aspect_ratio=aspect_ratio,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "on_candidate_detected hook failed for %s: %s",
                                    filename,
                                    exc,
                                )

                        # Extract frame indices from detection context
                        ctx_frame_index, ctx_prev_frame_index = _extract_frame_indices(
                            detection_context_payload
                        )

                        self.progress_manager.record_result(
                            filename,
                            is_candidate,
                            line_score,
                            num_lines,
                            aspect_ratio,
                            frame_index=ctx_frame_index,
                            prev_frame_index=ctx_prev_frame_index,
                        )

                    try:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "on_batch_complete elapsed=%.2fs processed=%s detected=%s batch_size=%s",
                                time.time() - start_time,
                                self.progress_manager.get_total_processed(),
                                self.progress_manager.get_total_detected(),
                                len(batch_results),
                            )
                        self.output_handler.on_batch_complete(
                            self.progress_manager.get_total_processed(),
                            self.progress_manager.get_total_detected(),
                            len(batch_results),
                        )
                    except Exception as exc:
                        logger.warning(
                            "on_batch_complete hook failed for batch %s: %s",
                            completed_batches,
                            exc,
                        )
                except Exception as e:
                    print(
                        "\n"
                        + get_message(
                            "ui.pipeline.processing.batch_error",
                            locale=locale,
                            error=e,
                        )
                    )
        except KeyboardInterrupt:
            print(
                "\n"
                + get_message(
                    "ui.pipeline.processing.interrupt_cancel",
                    locale=locale,
                )
            )
            wait_for_tasks = False
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=wait_for_tasks, cancel_futures=not wait_for_tasks)

        return self.progress_manager.get_total_detected()

    def _process_sequential(
        self,
        image_pairs: List[Tuple[int, str, str]],
        roi_mask: np.ndarray,
        params: Dict[str, Any],
        roi_polygon: Optional[List[List[int]]],
        resume_offset: int,
        overall_total: int,
        input_loader: Optional[BaseInputLoader],
        detector: Optional[BaseDetector],
        debug_image_enabled: bool,
        locale: Optional[str] = None,
    ) -> int:
        """Process images sequentially."""
        locale = _resolve_locale(locale)
        start_time = time.time()
        for idx, pair in enumerate(image_pairs):
            current_index = resume_offset + idx + 1
            current_file = os.path.basename(pair[1])

            print(
                get_message(
                    "ui.pipeline.processing.processing_file",
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
                params,
                input_loader,
                detector,
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
                    detection_context_payload,
                ) = result

                if line_score > 0:
                    print()
                    print(
                        get_message(
                            "ui.pipeline.processing.line",
                            locale=locale,
                            filename=filename,
                            score=line_score,
                            lines=num_lines,
                        )
                    )

                if detection_result and detection_context_payload:
                    try:
                        self.output_handler.on_detection_result(
                            detection_context_payload,
                            detection_result,
                            filepath,
                        )
                    except Exception as exc:
                        logger.warning(
                            "on_detection_result hook failed for %s: %s",
                            filename,
                            exc,
                        )

                if is_candidate:
                    print()
                    output_result = self.output_handler.save_candidate(
                        filepath, filename, debug_img, roi_polygon
                    )
                    output_result = _normalize_output_result(output_result)
                    if output_result.saved:
                        print(
                            get_message(
                                "ui.pipeline.processing.hit",
                                locale=locale,
                                filename=filename,
                                ratio=aspect_ratio,
                            )
                        )
                    else:
                        print(
                            get_message(
                                "ui.pipeline.processing.skip",
                                locale=locale,
                                filename=filename,
                            )
                        )
                    try:
                        self.output_handler.on_candidate_detected(
                            filename,
                            output_result.saved,
                            score=line_score,
                            aspect_ratio=aspect_ratio,
                        )
                    except Exception as exc:
                        logger.warning(
                            "on_candidate_detected hook failed for %s: %s",
                            filename,
                            exc,
                        )

                # Extract frame indices from detection context
                ctx_frame_index, ctx_prev_frame_index = _extract_frame_indices(
                    detection_context_payload
                )

                self.progress_manager.record_result(
                    filename,
                    is_candidate,
                    line_score,
                    num_lines,
                    aspect_ratio,
                    frame_index=ctx_frame_index,
                    prev_frame_index=ctx_prev_frame_index,
                )

            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "on_batch_complete elapsed=%.2fs processed=%s detected=%s batch_size=%s",
                        time.time() - start_time,
                        self.progress_manager.get_total_processed(),
                        self.progress_manager.get_total_detected(),
                        len(batch_results),
                    )
                self.output_handler.on_batch_complete(
                    self.progress_manager.get_total_processed(),
                    self.progress_manager.get_total_detected(),
                    len(batch_results),
                )
            except Exception as exc:
                logger.warning(
                    "on_batch_complete hook failed for %s: %s",
                    current_file,
                    exc,
                )

        return self.progress_manager.get_total_detected()


# Type alias for the default pipeline class
# Note: This is Type[...], not an instance - use it to instantiate pipelines
DefaultPipelineClass: Type[MeteorDetectionPipeline] = MeteorDetectionPipeline


def create_default_pipeline(
    config: Optional[PipelineConfig] = None,
    *,
    # Legacy API support
    target_folder: Optional[str] = None,
    output_folder: Optional[str] = None,
    debug_folder: Optional[str] = None,
    params: Optional[DetectionParams] = None,
    **kwargs: Any,
) -> MeteorDetectionPipeline:
    """Factory function to create the default detection pipeline.

    This function provides a convenient way to create a MeteorDetectionPipeline.

    Can be called in two ways:

    1. New API with PipelineConfig (recommended):
       >>> config = PipelineConfig(
       ...     target_folder="./raw",
       ...     output_folder="./candidates",
       ...     debug_folder="./debug",
       ...     params=DetectionParams(),
       ... )
       >>> pipeline = create_default_pipeline(config)

    2. Legacy API with individual arguments:
       >>> pipeline = create_default_pipeline(
       ...     target_folder="./raw",
       ...     output_folder="./candidates",
       ...     debug_folder="./debug",
       ...     params=DetectionParams(),
       ... )

    Args:
        config: PipelineConfig object (new API)
        target_folder: Input folder (legacy API)
        output_folder: Output folder (legacy API)
        debug_folder: Debug folder (legacy API)
        params: Detection parameters (legacy API)
        **kwargs: Additional arguments passed to MeteorDetectionPipeline

    Returns:
        Configured MeteorDetectionPipeline instance.

    Example:
        >>> from meteor_core import create_default_pipeline, DetectionParams
        >>> pipeline = create_default_pipeline(
        ...     target_folder="./raw",
        ...     output_folder="./candidates",
        ...     debug_folder="./debug",
        ...     params=DetectionParams(),
        ... )
        >>> detected = pipeline.run()
    """
    if config is not None:
        return MeteorDetectionPipeline(config, **kwargs)

    if target_folder is not None:
        # Legacy API
        if output_folder is None or debug_folder is None or params is None:
            raise TypeError(
                "When using legacy API, output_folder, debug_folder, and params "
                "are required. Consider using PipelineConfig instead."
            )

        pipeline_config = PipelineConfig(
            target_folder=target_folder,
            output_folder=output_folder,
            debug_folder=debug_folder,
            params=params,
            **{
                k: v
                for k, v in kwargs.items()
                if k in PipelineConfig.__dataclass_fields__
            },
        )

        loader_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in PipelineConfig.__dataclass_fields__
        }

        return MeteorDetectionPipeline(pipeline_config, **loader_kwargs)

    raise TypeError(
        "Either config or target_folder must be provided. "
        "Use PipelineConfig for the recommended API."
    )


# Backward compatibility alias (deprecated, use DefaultPipelineClass or create_default_pipeline)
DefaultPipeline = DefaultPipelineClass
