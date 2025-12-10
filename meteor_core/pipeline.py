#!/usr/bin/env python
#
# Detect Meteors CLI - Pipeline
# © 2025 Shinichi Morita (shin3tky)
#

"""
Processing pipeline for meteor detection.
Handles batch processing, multiprocessing, and orchestration.
"""

import os
import glob
import time
import signal
import warnings
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Any, Protocol, Type, Union, overload

from dataclasses import is_dataclass

from .schema import (
    EXTENSIONS,
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_MIN_AREA,
    DEFAULT_NUM_WORKERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_PROGRESS_FILE,
    DetectionParams,
    PipelineConfig,
)
from .image_io import extract_exif_metadata
from .inputs import BaseInputLoader, create_raw_loader, discover_input_loaders
from .inputs.base import supports_metadata_extraction
from .roi_selector import select_roi, create_roi_mask_from_polygon, create_full_roi_mask
from .detectors import compute_line_score_fast
from .outputs import (
    BaseOutputHandler,
    OutputWriter,
    ProgressManager,
)
from .utils import (
    compute_params_hash,
    format_polygon_string,
)


_DEFAULT_INPUT_LOADER = create_raw_loader()


def _coerce_loader_config(loader_cls, loader_config: Optional[Dict[str, Any]]):
    config_type = getattr(loader_cls, "ConfigType", None)

    if loader_config is None:
        if config_type is not None:
            try:
                return config_type()
            except Exception:
                return None
        return None

    if config_type is None:
        return loader_config

    if isinstance(loader_config, config_type):
        return loader_config

    if is_dataclass(config_type) and isinstance(loader_config, dict):
        return config_type(**loader_config)

    if hasattr(config_type, "model_validate") and isinstance(loader_config, dict):
        return config_type.model_validate(loader_config)

    if hasattr(config_type, "parse_obj") and isinstance(loader_config, dict):
        return config_type.parse_obj(loader_config)

    return loader_config


def _resolve_input_loader(
    input_loader: Optional[BaseInputLoader] = None,
    loader_name: Optional[str] = None,
    loader_config: Optional[Dict[str, Any]] = None,
) -> BaseInputLoader:
    if input_loader is not None:
        return input_loader

    if loader_name:
        available_loaders = discover_input_loaders()
        loader_cls = available_loaders.get(loader_name)
        if loader_cls is None:
            raise ValueError(
                f"Unknown input loader '{loader_name}'. "
                f"Available: {', '.join(sorted(available_loaders)) or 'none'}"
            )

        coerced_config = _coerce_loader_config(loader_cls, loader_config)
        return loader_cls(coerced_config)

    return _DEFAULT_INPUT_LOADER


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
    ) -> int:
        """Run the detection pipeline and return number of detected candidates.

        Args:
            enable_roi_selection: Whether to enable interactive ROI selection.
            roi_polygon_cli: Pre-defined ROI polygon from command line.
            resume: Whether to resume from previous progress.
            profile: Whether to collect performance metrics.

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
        FileNotFoundError: If the directory doesn't exist or no RAW files found
        NotADirectoryError: If the path is not a directory
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


def validate_raw_file(
    index: int, raw_file: str, input_loader: Optional[BaseInputLoader] = None
) -> Tuple[int, str, Optional[Exception]]:
    """
    Attempt to load a RAW file, returning any validation error.

    Args:
        index: File index for reference
        raw_file: Path to RAW file

    Returns:
        Tuple of (index, filepath, error or None)
    """
    loader = _resolve_input_loader(input_loader)

    try:
        loader.load(raw_file)
        return index, raw_file, None
    except Exception as exc:
        return index, raw_file, exc


def process_image_batch(
    batch_data: List[Tuple[str, str]],
    roi_mask: np.ndarray,
    params: dict,
    input_loader: Optional[BaseInputLoader] = None,
) -> List[Tuple]:
    """
    Process a batch of images (handle multiple pairs at once).

    Args:
        batch_data: List of [(curr_file, prev_file), ...]
        roi_mask: ROI mask
        params: Parameter dictionary

    Returns:
        List of processing results for each image:
        [(is_candidate, filename, filepath, line_score, debug_img, aspect_ratio, num_lines), ...]
    """
    results = []

    # Pre-create kernel for morphology
    kernel = np.ones((3, 3), np.uint8)

    hough_params = {
        "threshold": params["hough_threshold"],
        "min_line_length": params["hough_min_line_length"],
        "max_line_gap": params["hough_max_line_gap"],
    }

    loader = _resolve_input_loader(input_loader)

    for curr_file, prev_file in batch_data:
        filename = os.path.basename(curr_file)

        try:
            # Load images
            curr_img = loader.load(curr_file)
            prev_img = loader.load(prev_file)

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


def estimate_diff_threshold_from_samples(
    files: List[str],
    roi_mask: np.ndarray,
    sample_size: int = 5,
    input_loader: Optional[BaseInputLoader] = None,
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
    print(f"\n{'='*50}")
    print(f"Auto-estimating diff_threshold from {sample_size} samples")
    print("Percentile-based approach")
    print(f"{'='*50}")

    sample_size = min(sample_size, len(files))
    if sample_size < 2:
        print("⚠ Not enough samples, using default")
        return DEFAULT_DIFF_THRESHOLD

    samples = []
    print("Loading samples... ", end="", flush=True)
    loader = _resolve_input_loader(input_loader)

    for i in range(sample_size):
        try:
            img = loader.load(files[i])
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
    print("Estimation methods:")
    print(f"  [1] 98th percentile:      {method_1}")
    print(f"  [2] Mean + 1.5σ:          {method_2}")
    print(f"  [3] Median × 3:           {method_3}")
    print(f"{'─'*50}")
    print(f"✓ Selected threshold: {estimated_threshold} (minimum)")
    print(f"{'='*50}\n")

    return estimated_threshold


def estimate_min_area_from_samples(
    files: List[str],
    roi_mask: np.ndarray,
    diff_threshold: int,
    sample_size: int = 3,
    input_loader: Optional[BaseInputLoader] = None,
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
    print(f"\n{'='*50}")
    print(f"Auto-estimating min_area from {sample_size} samples")
    print("Star size distribution analysis")
    print(f"{'='*50}")

    sample_size = min(sample_size, len(files))
    if sample_size < 1:
        print("⚠ Not enough samples, using default")
        return DEFAULT_MIN_AREA

    samples = []
    print("Loading samples... ", end="", flush=True)
    loader = _resolve_input_loader(input_loader)

    for i in range(sample_size):
        try:
            img = loader.load(files[i])
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

    print(f"✓ Detected {len(all_star_areas)} stars")

    if len(all_star_areas) < 10:
        print("⚠ Not enough stars detected, using default")
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

    print(f"\n{'─'*50}")
    print(f"Star Size Statistics (from {len(all_star_areas)} stars):")
    print(f"{'─'*50}")
    print(f"  Median:       {median_star:.1f} pixels²")
    print(f"  Mean:         {mean_star:.1f} pixels²")
    print(f"  75th %ile:    {p75_star:.1f} pixels²")
    print(f"  90th %ile:    {p90_star:.1f} pixels²")
    print(f"{'─'*50}")
    print(f"✓ Estimated min_area: {estimated_min_area}")
    print("  → 75th percentile × 2.0 (robust to outliers)")
    print(f"{'='*50}\n")

    return estimated_min_area


def estimate_min_line_score_from_image(
    image_shape: Tuple[int, int], focal_length_mm: Optional[float] = None
) -> float:
    """
    Fixed min_line_score estimation with corrected focal length logic.

    Args:
        image_shape: (height, width) of image
        focal_length_mm: Focal length in mm (optional)

    Returns:
        Estimated min_line_score value
    """
    print(f"\n{'='*50}")
    print("Auto-estimating min_line_score from image geometry")
    print(f"{'='*50}")

    height, width = image_shape
    diagonal = np.sqrt(height**2 + width**2)

    # Reduced base coefficient from 4% to 2.5% based on real data
    base_score = diagonal * 0.025

    if focal_length_mm:
        focal_factor = focal_length_mm / 50.0
        adjusted_score = base_score * focal_factor

        print(f"\n{'─'*50}")
        print("Image Geometry:")
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
        print("Image Geometry:")
        print(f"{'─'*50}")
        print(f"  Dimensions:   {width}×{height} pixels")
        print(f"  Diagonal:     {diagonal:.0f} pixels")
        print(f"  Base score:   {base_score:.1f}")
        print("  (No focal length provided)")

    # Adjusted clamp range based on real meteor data
    estimated_score = np.clip(adjusted_score, 40.0, 150.0)

    print(f"{'─'*50}")
    print(f"✓ Estimated min_line_score: {estimated_score:.1f}")
    print("  → ~2.5% of image diagonal")
    print(f"{'='*50}\n")

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
        """
        # Determine which API is being used
        if isinstance(config_or_target, PipelineConfig):
            # New API: PipelineConfig provided
            self._config = config_or_target
        elif isinstance(config_or_target, str):
            # Legacy API: individual arguments
            if output_folder is None or debug_folder is None or params is None:
                raise TypeError(
                    "When using legacy API, output_folder, debug_folder, and params "
                    "are required. Consider using PipelineConfig instead."
                )

            # Issue deprecation warning for legacy API
            warnings.warn(
                "Passing individual arguments to MeteorDetectionPipeline is deprecated. "
                "Use PipelineConfig instead: "
                "MeteorDetectionPipeline(PipelineConfig(target_folder=..., ...))",
                DeprecationWarning,
                stacklevel=2,
            )

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
            raise TypeError(
                f"First argument must be PipelineConfig or str (target_folder), "
                f"got {type(config_or_target).__name__}"
            )

        # Store loader configuration
        self.input_loader = input_loader
        self.input_loader_name = input_loader_name
        self.input_loader_config = input_loader_config

        # Initialize output handler
        self.output_writer: BaseOutputHandler = output_handler or OutputWriter(
            self._config.output_folder,
            self._config.debug_folder,
            self._config.progress_file,
            self._config.output_overwrite,
        )
        self.progress_manager = ProgressManager(self._config.progress_file)

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
        timing = {}
        t_total = time.time()
        input_loader = _resolve_input_loader(
            self.input_loader, self.input_loader_name, self.input_loader_config
        )

        # Safety check
        target_fullpath = os.path.abspath(self._config.target_folder)
        output_fullpath = os.path.abspath(self._config.output_folder)

        if target_fullpath == output_fullpath:
            print(f"\n{'='*60}")
            print("⚠ ERROR: Target and output directories are the same!")
            print(f"{'='*60}")
            return 0

        # Collect files
        print(f"Collecting RAW files from: {self._config.target_folder}")
        files = collect_files(self._config.target_folder)

        if len(files) < 2:
            print("Need at least 2 images. Exiting.")
            return 0

        print(f"Found {len(files)} files")

        # Load first image
        t_load = time.time()
        try:
            prev_img = input_loader.load(files[0])
        except Exception as exc:
            print(
                f"Failed to load first RAW file: {os.path.basename(files[0])} ({exc})"
            )
            return 0

        if profile:
            timing["first_load"] = time.time() - t_load
        height, width = prev_img.shape

        # ROI setup
        roi_mask = create_full_roi_mask((height, width))
        roi_polygon = None

        if roi_polygon_cli:
            print(
                f"ROI specified via command line: polygon={format_polygon_string(roi_polygon_cli)}"
            )
            roi_mask = create_roi_mask_from_polygon(roi_polygon_cli, (height, width))
            roi_polygon = roi_polygon_cli
        elif enable_roi_selection:
            roi_selection = select_roi(prev_img)
            if roi_selection:
                roi_mask = roi_selection["mask"]
                roi_polygon = roi_selection["polygon"]
                print(
                    f"ROI setup complete: polygon={format_polygon_string(roi_polygon)}"
                )
            else:
                print("No ROI selected. Processing entire image.")
        else:
            print("Skipping ROI selection. Processing entire image.")

        # Get params dict
        params = self._config.params.to_dict()

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
                    f"Resuming from progress file: {self._config.progress_file} "
                    f"(processed={self.progress_manager.get_total_processed()}, "
                    f"detected={self.progress_manager.get_total_detected()})"
                )
            elif loaded:
                print("Progress file exists but parameters differ. Starting fresh.")
                self.progress_manager.reset()
                self.progress_manager.set_params_hash(params_hash)

        # Filter existing files
        existing_basenames = {os.path.basename(path) for path in files}
        self.progress_manager.filter_existing_files(existing_basenames)
        self.progress_manager.save()

        # Build image pairs
        image_pairs = [(files[i], files[i - 1]) for i in range(1, len(files))]
        image_pairs = [
            pair
            for pair in image_pairs
            if not self.progress_manager.is_processed(os.path.basename(pair[0]))
        ]

        resume_offset = self.progress_manager.get_total_processed()
        overall_total = resume_offset + len(image_pairs)

        print(f"Starting processing: {len(image_pairs)} image pairs")
        if self._config.enable_parallel:
            print(
                f"Parallel processing: {self._config.num_workers} workers, "
                f"batch size: {self._config.batch_size}"
            )

        detected_count = self.progress_manager.get_total_detected()
        t_process = time.time()

        try:
            if self._config.enable_parallel and self._config.num_workers > 1:
                detected_count = self._process_parallel(
                    image_pairs,
                    roi_mask,
                    params,
                    roi_polygon,
                    resume_offset,
                    overall_total,
                    input_loader,
                )
            else:
                detected_count = self._process_sequential(
                    image_pairs,
                    roi_mask,
                    params,
                    roi_polygon,
                    resume_offset,
                    overall_total,
                    input_loader,
                )
        except KeyboardInterrupt:
            print(
                f"\nInterrupted by user. Progress saved to {self._config.progress_file}."
            )
            self.progress_manager.save()
            return self.progress_manager.get_total_detected()

        if profile:
            timing["processing"] = time.time() - t_process
            timing["total"] = time.time() - t_total

            print("\n\n=== Performance Profile ===")
            print(f"First image load: {timing['first_load']:.3f}s")
            print(f"Processing time: {timing['processing']:.3f}s")
            print(f"Total time: {timing['total']:.3f}s")
            print(f"Images processed: {len(image_pairs)}")
            if image_pairs:
                print(
                    f"Average per image: {timing['processing'] / len(image_pairs):.3f}s"
                )

        print(f"\nComplete! {detected_count} candidates extracted")
        return detected_count

    def _process_parallel(
        self,
        image_pairs: List[Tuple[str, str]],
        roi_mask: np.ndarray,
        params: Dict[str, Any],
        roi_polygon: Optional[List[List[int]]],
        resume_offset: int,
        overall_total: int,
        input_loader: Optional[BaseInputLoader],
    ) -> int:
        """Process images in parallel using ProcessPoolExecutor."""
        batches = [
            image_pairs[i : i + self._config.batch_size]
            for i in range(0, len(image_pairs), self._config.batch_size)
        ]

        print(f"Number of batches: {len(batches)}")

        executor = ProcessPoolExecutor(
            max_workers=self._config.num_workers,
            initializer=_init_worker_ignore_interrupt,
        )
        futures: List = []
        wait_for_tasks = True

        try:
            for batch in batches:
                future = executor.submit(
                    process_image_batch, batch, roi_mask, params, input_loader
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
                            saved = self.output_writer.save_candidate(
                                filepath, filename, debug_img, roi_polygon
                            )
                            if saved:
                                print(f"  [HIT] {filename}: Ratio={aspect_ratio:.2f}")
                            else:
                                print(f"  [SKIP] {filename}: Already exists")
                        else:
                            print(
                                f"\rChecking... {resume_offset + processed}/{overall_total}",
                                end="",
                                flush=True,
                            )

                        self.progress_manager.record_result(filename, is_candidate)

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

        return self.progress_manager.get_total_detected()

    def _process_sequential(
        self,
        image_pairs: List[Tuple[str, str]],
        roi_mask: np.ndarray,
        params: Dict[str, Any],
        roi_polygon: Optional[List[List[int]]],
        resume_offset: int,
        overall_total: int,
        input_loader: Optional[BaseInputLoader],
    ) -> int:
        """Process images sequentially."""
        for idx, pair in enumerate(image_pairs):
            current_index = resume_offset + idx + 1
            current_file = os.path.basename(pair[0])

            print(
                f"\rProcessing {current_index}/{overall_total}: {current_file}",
                end="",
                flush=True,
            )

            batch_results = process_image_batch([pair], roi_mask, params, input_loader)

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
                    print()
                    print(
                        f"  [LINE] {filename}: score={line_score:.1f}, lines={num_lines}"
                    )

                if is_candidate:
                    print()
                    saved = self.output_writer.save_candidate(
                        filepath, filename, debug_img, roi_polygon
                    )
                    if saved:
                        print(f"  [HIT] {filename}: Ratio={aspect_ratio:.2f}")
                    else:
                        print(f"  [SKIP] {filename}: Already exists")

                self.progress_manager.record_result(filename, is_candidate)

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
