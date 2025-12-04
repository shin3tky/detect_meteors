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
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Any

from .schema import (
    EXTENSIONS,
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_LINE_SCORE,
    DEFAULT_FISHEYE_MODEL,
    DetectionParams,
)
from .image_io import load_and_bin_raw_fast, extract_exif_metadata
from .roi_selector import select_roi, create_roi_mask_from_polygon, create_full_roi_mask
from .detectors import HoughDetector, compute_line_score_fast
from .outputs import OutputWriter, ProgressManager, load_progress, save_progress
from .utils import (
    compute_params_hash,
    format_polygon_string,
    display_exif_info,
    display_fisheye_info,
    calculate_npf_metrics,
    optimize_params_with_npf,
    estimate_batch_size,
)


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
    index: int, raw_file: str
) -> Tuple[int, str, Optional[Exception]]:
    """
    Attempt to load a RAW file, returning any validation error.

    Args:
        index: File index for reference
        raw_file: Path to RAW file

    Returns:
        Tuple of (index, filepath, error or None)
    """
    try:
        load_and_bin_raw_fast(raw_file)
        return index, raw_file, None
    except Exception as exc:
        return index, raw_file, exc


def process_image_batch(
    batch_data: List[Tuple[str, str]], roi_mask: np.ndarray, params: dict
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


def estimate_diff_threshold_from_samples(
    files: List[str], roi_mask: np.ndarray, sample_size: int = 5
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
    print(f"  → 75th percentile × 2.0 (robust to outliers)")
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
    print(f"Auto-estimating min_line_score from image geometry")
    print(f"{'='*50}")

    height, width = image_shape
    diagonal = np.sqrt(height**2 + width**2)

    # Reduced base coefficient from 4% to 2.5% based on real data
    base_score = diagonal * 0.025

    if focal_length_mm:
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

    # Adjusted clamp range based on real meteor data
    estimated_score = np.clip(adjusted_score, 40.0, 150.0)

    print(f"{'─'*50}")
    print(f"✓ Estimated min_line_score: {estimated_score:.1f}")
    print(f"  → ~2.5% of image diagonal")
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
    """

    def __init__(
        self,
        target_folder: str,
        output_folder: str,
        debug_folder: str,
        params: DetectionParams,
        num_workers: int = 1,
        batch_size: int = 10,
        auto_batch_size: bool = False,
        enable_parallel: bool = True,
        progress_file: str = "progress.json",
        output_overwrite: bool = False,
    ):
        """
        Initialize the detection pipeline.

        Args:
            target_folder: Input folder containing RAW files
            output_folder: Output folder for detected candidates
            debug_folder: Folder for debug images
            params: Detection parameters
            num_workers: Number of parallel workers
            batch_size: Batch size for processing
            auto_batch_size: Whether to auto-calculate batch size
            enable_parallel: Whether to enable parallel processing
            progress_file: Path to progress tracking file
            output_overwrite: Whether to overwrite existing output files
        """
        self.target_folder = target_folder
        self.output_folder = output_folder
        self.debug_folder = debug_folder
        self.params = params
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.auto_batch_size = auto_batch_size
        self.enable_parallel = enable_parallel
        self.progress_file = progress_file
        self.output_overwrite = output_overwrite

        self.output_writer = OutputWriter(
            output_folder, debug_folder, progress_file, output_overwrite
        )
        self.progress_manager = ProgressManager(progress_file)

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

        # Safety check
        target_fullpath = os.path.abspath(self.target_folder)
        output_fullpath = os.path.abspath(self.output_folder)

        if target_fullpath == output_fullpath:
            print(f"\n{'='*60}")
            print("⚠ ERROR: Target and output directories are the same!")
            print(f"{'='*60}")
            return 0

        # Collect files
        print(f"Collecting RAW files from: {self.target_folder}")
        files = collect_files(self.target_folder)

        if len(files) < 2:
            print("Need at least 2 images. Exiting.")
            return 0

        print(f"Found {len(files)} files")

        # Load first image
        t_load = time.time()
        try:
            prev_img = load_and_bin_raw_fast(files[0])
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
        params = self.params.to_dict()

        # Progress tracking setup
        params_for_hash = params.copy()
        if roi_polygon:
            params_for_hash["roi_polygon"] = roi_polygon

        params_hash = compute_params_hash(params_for_hash)
        self.progress_manager.set_params_hash(params_hash)

        if resume:
            loaded = self.progress_manager.load()
            if loaded and self.progress_manager.get_params_hash() == params_hash:
                print(
                    f"Resuming from progress file: {self.progress_file} "
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
        if self.enable_parallel:
            print(
                f"Parallel processing: {self.num_workers} workers, batch size: {self.batch_size}"
            )

        detected_count = self.progress_manager.get_total_detected()
        t_process = time.time()

        try:
            if self.enable_parallel and self.num_workers > 1:
                detected_count = self._process_parallel(
                    image_pairs,
                    roi_mask,
                    params,
                    roi_polygon,
                    resume_offset,
                    overall_total,
                )
            else:
                detected_count = self._process_sequential(
                    image_pairs,
                    roi_mask,
                    params,
                    roi_polygon,
                    resume_offset,
                    overall_total,
                )
        except KeyboardInterrupt:
            print(f"\nInterrupted by user. Progress saved to {self.progress_file}.")
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
    ) -> int:
        """Process images in parallel using ProcessPoolExecutor."""
        batches = [
            image_pairs[i : i + self.batch_size]
            for i in range(0, len(image_pairs), self.batch_size)
        ]

        print(f"Number of batches: {len(batches)}")

        executor = ProcessPoolExecutor(
            max_workers=self.num_workers, initializer=_init_worker_ignore_interrupt
        )
        futures: List = []
        wait_for_tasks = True

        try:
            for batch in batches:
                future = executor.submit(process_image_batch, batch, roi_mask, params)
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
