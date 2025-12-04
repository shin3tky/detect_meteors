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
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from meteor_core import (
    VERSION,
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
    DetectionParams,
    # Functions
    collect_files,
    validate_raw_file,
    extract_exif_metadata,
    load_and_bin_raw_fast,
    parse_focal_factor,
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
    create_roi_mask_from_polygon,
    create_full_roi_mask,
    process_image_batch,
    compute_params_hash,
    OutputWriter,
    ProgressManager,
    load_progress,
    save_progress,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Meteor detection tool with comprehensive auto-parameter estimation (v1.3.1)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Detect Meteors CLI (https://github.com/shin3tky/detect_meteors) {VERSION}",
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
        help="Auto-estimate diff_threshold, min_area, and min_line_score based on EXIF (v1.4.0 with NPF Rule)",
    )
    parser.add_argument(
        "--sensor-type",
        type=str,
        default=None,
        metavar="TYPE",
        help="Sensor type preset for NPF Rule parameters. "
        "Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Focal length in mm (35mm equivalent).",
    )
    parser.add_argument(
        "--focal-factor",
        type=str,
        default=None,
        help="Crop factor for 35mm equivalent calculation.",
    )
    parser.add_argument(
        "--sensor-width",
        type=float,
        default=None,
        help="Sensor width in mm (for NPF Rule calculation).",
    )
    parser.add_argument(
        "--pixel-pitch",
        type=float,
        default=None,
        help="Pixel pitch in micrometers (μm) for NPF Rule.",
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
        help="Force overwrite existing files in output folder",
    )
    parser.add_argument(
        "--fisheye",
        action="store_true",
        help="Enable fisheye lens correction.",
    )

    return parser


def handle_show_exif(args) -> None:
    """Handle --show-exif / --show-npf commands."""
    print(f"\n{'='*60}")
    if args.show_npf:
        print("EXIF Metadata & NPF Rule Analysis")
    else:
        print("EXIF Metadata Viewer")
    print(f"{'='*60}\n")
    print(f"Target folder: {args.target}")

    # Validate --sensor-type
    if args.sensor_type and get_sensor_preset(args.sensor_type) is None:
        print(f"⚠ Error: Invalid --sensor-type value: '{args.sensor_type}'")
        return

    # Apply sensor preset
    (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    ) = apply_sensor_preset(args, verbose=True)

    # Validate sensor overrides
    validate_sensor_overrides(args, preset, sensor_width_value, pixel_pitch_value)

    # Validate focal_factor
    if args.focal_factor and focal_factor_value is None:
        print(f"⚠ Error: Invalid --focal-factor value: '{args.focal_factor}'")
        return

    try:
        files = collect_files(args.target)
        if not files:
            print("⚠ No RAW files found in target folder.")
            return

        print(f"Found {len(files)} RAW files")
        print(f"Reading EXIF from first file: {os.path.basename(files[0])}\n")

        exif_data = extract_exif_metadata(files[0])

        # Focal length priority
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
            exif_data["focal_length_35mm"] = (
                exif_data["focal_length"] * focal_factor_value
            )
        elif exif_data.get("focal_length"):
            focal_length_source = "EXIF (actual, no 35mm equiv.)"

        # NPF metrics
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

        # Display fisheye info
        if args.fisheye and exif_data.get("focal_length_35mm"):
            display_fisheye_info(exif_data["focal_length_35mm"], DEFAULT_FISHEYE_MODEL)

        display_exif_info(
            exif_data, focal_length_source, focal_factor_value, npf_metrics
        )

        # Warnings
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

        if args.show_npf or npf_metrics:
            if not sensor_width_value and not exif_data.get("image_width"):
                warnings.append("Sensor width not specified")
            if npf_metrics and not npf_metrics.get("has_complete_data"):
                warnings.append("NPF calculation using default/estimated values")

        if warnings:
            print(f"{'='*60}")
            print("⚠ Warnings:")
            for warning in warnings:
                print(f"  • {warning}")
            print(f"{'='*60}\n")

    except FileNotFoundError as e:
        print(f"⚠ Error: {e}")
    except Exception as e:
        print(f"⚠ Error reading EXIF: {e}")


def _init_worker_ignore_interrupt() -> None:
    """Ignore SIGINT in worker processes. Must be at module level for pickling."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


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
) -> int:
    """
    Main processing: detect meteor candidates from consecutive RAW images.

    This is the main entry point that orchestrates the detection workflow.
    """
    import time
    import shutil
    import cv2
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor, as_completed

    timing = {}
    t_total = time.time()

    # Safety check
    target_fullpath = os.path.abspath(target_folder)
    output_fullpath = os.path.abspath(output_folder)

    if target_fullpath == output_fullpath:
        print(f"\n{'='*60}")
        print("⚠ ERROR: Target and output directories are the same!")
        print(f"{'='*60}")
        return 0

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)

    print(f"Collecting RAW files from: {target_folder}")
    files = collect_files(target_folder)

    if len(files) < 2:
        print("Need at least 2 images. Exiting.")
        return 0

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
    height, width = prev_img.shape

    # ROI setup
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

    # Auto-parameter estimation
    if auto_params:
        print(f"\n{'='*60}")
        print("Auto-params: NPF Rule-based Optimization")
        print(f"{'='*60}\n")

        exif_data = extract_exif_metadata(files[0])

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

        npf_metrics = calculate_npf_metrics(
            exif_data,
            sensor_width_mm=sensor_width_mm,
            pixel_pitch_um=pixel_pitch_um,
            fisheye=fisheye,
            fisheye_model=DEFAULT_FISHEYE_MODEL,
        )

        if fisheye and exif_data.get("focal_length_35mm"):
            display_fisheye_info(exif_data["focal_length_35mm"], DEFAULT_FISHEYE_MODEL)

        display_exif_info(exif_data, focal_length_source, focal_factor, npf_metrics)

        # Warnings
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

        # NPF-based parameter optimization
        if npf_metrics and npf_metrics.get("npf_recommended_sec"):
            print(f"{'='*60}")
            print("Parameter Optimization (NPF Rule-based)")
            print(f"{'='*60}\n")

            diff_threshold, min_area, min_line_score, opt_info = (
                optimize_params_with_npf(
                    exif_data,
                    npf_metrics,
                    user_specified_diff_threshold=user_specified_diff_threshold,
                    user_specified_min_area=user_specified_min_area,
                    user_specified_min_line_score=user_specified_min_line_score,
                    current_diff_threshold=diff_threshold,
                    current_min_area=min_area,
                    current_min_line_score=min_line_score,
                )
            )

            print(
                f"Shooting Quality Score: {opt_info['quality_score']:.2f} ({opt_info['quality_level']})"
            )

            if opt_info["adjustments"]:
                print(f"\nParameter Adjustments:")
                for adjustment in opt_info["adjustments"]:
                    print(f"  • {adjustment}")
            else:
                print(f"\nNo automatic adjustments (all parameters user-specified)")

            print(f"\n{'='*60}\n")
        else:
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

    params = {
        "diff_threshold": diff_threshold,
        "min_area": min_area,
        "min_aspect_ratio": min_aspect_ratio,
        "hough_threshold": hough_threshold,
        "hough_min_line_length": hough_min_line_length,
        "hough_max_line_gap": hough_max_line_gap,
        "min_line_score": min_line_score,
    }

    print(f"\n{'='*50}")
    print("Processing Parameters:")
    print(f"{'='*50}")
    print(f"  diff_threshold:        {diff_threshold}")
    print(f"  min_area:              {min_area}")
    print(f"  min_aspect_ratio:      {min_aspect_ratio}")
    print(f"  hough_threshold:       {hough_threshold}")
    print(f"  hough_min_line_length: {hough_min_line_length}")
    print(f"  hough_max_line_gap:    {hough_max_line_gap}")
    print(f"  min_line_score:        {min_line_score:.1f}")
    print(f"{'='*50}\n")

    # Progress tracking
    params_for_hash = params.copy()
    if roi_polygon:
        params_for_hash["roi_polygon"] = roi_polygon

    progress_data = {
        "version": VERSION,
        "params_hash": compute_params_hash(params_for_hash),
        "processed_files": [],
        "detected_files": [],
        "total_processed": 0,
        "total_detected": 0,
    }

    loaded_progress = load_progress(progress_file) if resume else None

    if loaded_progress:
        if loaded_progress.get("params_hash") == progress_data["params_hash"]:
            progress_data.update(
                {
                    key: loaded_progress.get(key, progress_data.get(key))
                    for key in [
                        "version",
                        "params_hash",
                        "processed_files",
                        "detected_files",
                        "total_processed",
                        "total_detected",
                        "created_at",
                        "last_updated",
                    ]
                }
            )
            print(
                f"Resuming from progress file: {progress_file} "
                f"(processed={progress_data['total_processed']}, "
                f"detected={progress_data['total_detected']})"
            )
        else:
            print(
                "Progress file exists but parameters differ. Starting with fresh progress."
            )

    existing_basenames = {os.path.basename(path) for path in files}
    progress_data["processed_files"] = [
        name
        for name in progress_data.get("processed_files", [])
        if name in existing_basenames
    ]
    progress_data["detected_files"] = [
        name
        for name in progress_data.get("detected_files", [])
        if name in existing_basenames
    ]

    processed_set = set(progress_data["processed_files"])
    detected_set = set(progress_data["detected_files"])

    progress_data["total_processed"] = len(processed_set)
    progress_data["total_detected"] = len(detected_set)

    save_progress(progress_file, progress_data)

    def record_result(filename: str, is_candidate: bool) -> None:
        processed_set.add(filename)
        if filename not in progress_data["processed_files"]:
            progress_data["processed_files"].append(filename)

        if is_candidate:
            detected_set.add(filename)
            if filename not in progress_data["detected_files"]:
                progress_data["detected_files"].append(filename)

        progress_data["total_processed"] = len(processed_set)
        progress_data["total_detected"] = len(detected_set)
        save_progress(progress_file, progress_data)

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
            batches = [
                image_pairs[i : i + batch_size]
                for i in range(0, len(image_pairs), batch_size)
            ]
            print(f"Number of batches: {len(batches)}")

            executor = ProcessPoolExecutor(
                max_workers=num_workers, initializer=_init_worker_ignore_interrupt
            )
            futures = []
            wait_for_tasks = True

            try:
                for batch in batches:
                    future = executor.submit(
                        process_image_batch, batch, roi_mask, params
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
                                output_path = os.path.join(output_folder, filename)
                                if os.path.exists(output_path) and not output_overwrite:
                                    print(
                                        f"  [SKIP] {filename}: Already exists in output folder"
                                    )
                                else:
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
                                            os.path.join(
                                                debug_folder, f"mask_{filename}.png"
                                            ),
                                            debug_img,
                                        )
                                    print(
                                        f"  [HIT] {filename}: Ratio={aspect_ratio:.2f}"
                                    )
                            else:
                                print(
                                    f"\rChecking... {resume_offset + processed}/{overall_total}",
                                    end="",
                                    flush=True,
                                )

                            record_result(filename, is_candidate)
                            detected_count = progress_data["total_detected"]

                    except Exception as e:
                        print(f"\nBatch processing error: {e}")
            except KeyboardInterrupt:
                print("\nInterrupted by user. Cancelling worker processes...")
                wait_for_tasks = False
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(
                    wait=wait_for_tasks, cancel_futures=not wait_for_tasks
                )
        else:
            # Sequential processing
            for idx, pair in enumerate(image_pairs):
                current_index = resume_offset + idx + 1
                current_file = os.path.basename(pair[0])
                progress_line_active = True

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
                        if progress_line_active:
                            print()
                            progress_line_active = False
                        print(
                            f"  [LINE] {filename}: score={line_score:.1f}, lines={num_lines}"
                        )

                    if is_candidate:
                        if progress_line_active:
                            print()
                            progress_line_active = False

                        output_path = os.path.join(output_folder, filename)
                        if os.path.exists(output_path) and not output_overwrite:
                            print(
                                f"  [SKIP] {filename}: Already exists in output folder"
                            )
                        else:
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
                            print(f"  [HIT] {filename}: Ratio={aspect_ratio:.2f}")
                    else:
                        print(
                            f"\rChecking... {current_index}/{overall_total}",
                            end="",
                            flush=True,
                        )
                        progress_line_active = True

                    record_result(filename, is_candidate)
                    detected_count = progress_data["total_detected"]

    except KeyboardInterrupt:
        print(f"\nInterrupted by user. Progress saved to {progress_file}.")
        save_progress(progress_file, progress_data)
        return detected_count

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

    if args.remove_progress:
        if os.path.exists(args.progress_file):
            os.remove(args.progress_file)
            print(f"Removed progress file: {args.progress_file}")
        else:
            print(f"Progress file not found: {args.progress_file}")
        return

    if args.list_sensor_types:
        list_sensor_types()
        return

    if args.show_exif or args.show_npf:
        handle_show_exif(args)
        return

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

    # Validate --sensor-type
    if args.sensor_type and get_sensor_preset(args.sensor_type) is None:
        print(f"⚠ Error: Invalid --sensor-type value: '{args.sensor_type}'")
        print(
            f"  Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40"
        )
        return

    # Apply sensor preset
    (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    ) = apply_sensor_preset(args, verbose=False)

    # Validate sensor overrides
    validate_sensor_overrides(args, preset, sensor_width_value, pixel_pitch_value)

    # Validate focal_factor
    if args.focal_factor and focal_factor_value is None:
        print(f"⚠ Error: Invalid --focal-factor value: '{args.focal_factor}'")
        print(f"  Valid values: MFT, APS-C, APS-H, FF, or numeric (e.g., 2.0)")
        return

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
    )


if __name__ == "__main__":
    main()
