#!/usr/bin/env python

import rawpy
import numpy as np
import os
import glob
import cv2
import time
import shutil
import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Tuple, List, Optional

# ==========================================
# Default Settings
# ==========================================
VERSION = "1.0.2"

DEFAULT_TARGET_FOLDER = "examples"
DEFAULT_OUTPUT_FOLDER = "candidates"
DEFAULT_DEBUG_FOLDER = "debug_masks"

EXTENSIONS = ["*.ORF", "*.ARW", "*.CR2", "*.NEF", "*.DNG"]

DEFAULT_DIFF_THRESHOLD = 8
DEFAULT_MIN_AREA = 10
DEFAULT_MIN_ASPECT_RATIO = 3.0

DEFAULT_HOUGH_THRESHOLD = 10
DEFAULT_HOUGH_MIN_LINE_LENGTH = 15
DEFAULT_HOUGH_MAX_LINE_GAP = 5
DEFAULT_MIN_LINE_SCORE = 80.0

DEFAULT_ENABLE_ROI_SELECTION = True
DEFAULT_NUM_WORKERS = max(1, mp.cpu_count() - 1)
DEFAULT_BATCH_SIZE = 10  # Batch processing size
AUTO_BATCH_MEMORY_FRACTION = 0.6  # Portion of free RAM to use when auto-sizing batches


def load_and_bin_raw_fast(filepath: str) -> np.ndarray:
    """
    Load RAW file & 2x2 binning
    - Minimize memory allocation
    - Optimize casting operations
    """
    with rawpy.imread(filepath) as raw:
        bayer = raw.raw_image

        # Binning process (reduce memory copying with view operations)
        h, w = bayer.shape
        h_half, w_half = h // 2, w // 2

        # Calculate directly with uint16 (add incrementally to prevent overflow)
        result = np.empty((h_half, w_half), dtype=np.uint16)

        # Single operation calculation (directly without going through uint32)
        temp = bayer[0::2, 0::2].astype(np.uint32)
        temp += bayer[0::2, 1::2]
        temp += bayer[1::2, 0::2]
        temp += bayer[1::2, 1::2]
        result[:] = temp // 4

        return result


def compute_line_score_fast(mask: np.ndarray, hough_params: dict) -> Tuple[float, List]:
    """Line detection using Hough transform"""
    # Early return if few edges
    if np.count_nonzero(mask) < hough_params["min_line_length"]:
        return 0.0, []

    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 180,
        hough_params["threshold"],
        minLineLength=hough_params["min_line_length"],
        maxLineGap=hough_params["max_line_gap"],
    )

    if lines is None:
        return 0.0, []

    # Vectorize for speed
    lines_array = lines.reshape(-1, 4)
    dx = lines_array[:, 2] - lines_array[:, 0]
    dy = lines_array[:, 3] - lines_array[:, 1]
    lengths = np.sqrt(dx * dx + dy * dy)

    score = float(np.sum(lengths))
    line_segments = [
        (int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in lines_array
    ]

    return score, line_segments


def get_available_memory_bytes() -> Optional[int]:
    """Return available system memory in bytes (best-effort)."""

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass

    return None


def estimate_batch_size(
    requested_batch_size: int,
    image_shape: Tuple[int, int],
    num_workers: int,
    safety_fraction: float = AUTO_BATCH_MEMORY_FRACTION,
    available_mem: Optional[int] = None,
) -> int:
    """
    Estimate a safe batch size based on available memory.

    The calculation uses the first loaded image to approximate memory usage
    of two RAW frames plus intermediate arrays within one worker process.
    """

    if available_mem is None:
        available_mem = get_available_memory_bytes()
    if available_mem is None:
        return requested_batch_size

    height, width = image_shape
    base_bytes = height * width * np.dtype(np.uint16).itemsize

    # Two RAW frames + diff buffer + mask + modest overhead for temporaries
    estimated_pair_bytes = int(base_bytes * 2 + base_bytes + (height * width) + base_bytes * 0.5)
    if estimated_pair_bytes <= 0:
        return requested_batch_size

    per_worker_budget = available_mem * safety_fraction / max(1, num_workers)
    max_pairs = max(1, int(per_worker_budget // estimated_pair_bytes))

    return max(1, min(requested_batch_size, max_pairs))


def process_image_batch(
    batch_data: List[Tuple[str, str]], roi_mask: np.ndarray, params: dict
) -> List[Tuple]:
    """
    Process a batch of images (handle multiple pairs at once)

    Args:
        batch_data: List of [(curr_file, prev_file), ...]
        roi_mask: ROI mask
        params: Parameter dictionary

    Returns:
        List of processing results for each image
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


def select_roi(image_data):
    """Polygon ROI selection with vertex editing and close hint."""

    disp_img = image_data.astype(np.float32)
    disp_img = disp_img / np.max(disp_img)
    disp_img = (disp_img * 255).astype(np.uint8)

    h, w = disp_img.shape
    scale_factor = 1.0

    if w > 1200:
        scale_factor = 1200 / w
        disp_w = int(w * scale_factor)
        disp_h = int(h * scale_factor)
        disp_img_resized = cv2.resize(disp_img, (disp_w, disp_h))
    else:
        disp_img_resized = disp_img

    display_img = cv2.cvtColor(disp_img_resized, cv2.COLOR_GRAY2BGR)
    window_name = "Select Sky Area"

    print("\n--- ROI Selection Mode ---")
    print(
        "Left click: add vertex | Esc: delete last vertex | Close by clicking the start circle"
    )

    points: List[Tuple[int, int]] = []
    mouse_pos: Optional[Tuple[int, int]] = None
    polygon_closed = False
    closable_threshold = 12
    closable_radius = 6
    cancelled = False

    def draw_canvas():
        canvas = display_img.copy()

        if points:
            cv2.polylines(
                canvas, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 2
            )
            for px, py in points:
                cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)

        if mouse_pos and points:
            cv2.line(canvas, points[-1], mouse_pos, (255, 255, 0), 1)

        if len(points) >= 3:
            first = points[0]
            hover_distance = (
                math.hypot(mouse_pos[0] - first[0], mouse_pos[1] - first[1])
                if mouse_pos
                else None
            )
            if hover_distance is not None and hover_distance <= closable_threshold:
                cv2.circle(canvas, first, closable_radius, (0, 255, 255), -1)

        return canvas

    def on_mouse(event, x, y, *_):
        nonlocal mouse_pos, polygon_closed
        mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if (
                len(points) >= 3
                and math.hypot(x - points[0][0], y - points[0][1]) <= closable_threshold
            ):
                polygon_closed = True
            else:
                points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        canvas = draw_canvas()
        cv2.imshow(window_name, canvas)

        if polygon_closed:
            cv2.polylines(
                canvas, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2
            )
            cv2.imshow(window_name, canvas)
            cv2.waitKey(300)
            break

        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC: delete last vertex
            if points:
                points.pop()
        elif key == ord("q"):
            cancelled = True
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    if cancelled or len(points) < 3:
        return None

    points_scaled = [
        (int(px / scale_factor), int(py / scale_factor)) for px, py in points
    ]
    polygon = np.array(points_scaled, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    bounding_rect = cv2.boundingRect(polygon)

    return {"mask": mask, "polygon": polygon.tolist(), "bounding_rect": bounding_rect}


def parse_roi_polygon_string(roi_str: str) -> List[List[int]]:
    """Parse --roi polygon format x1,y1;x2,y2;..."""

    segments = [
        seg.strip() for seg in roi_str.replace(" ", "").split(";") if seg.strip()
    ]
    if len(segments) < 3:
        raise ValueError(
            "ROI polygon must have at least 3 vertices in the format x1,y1;x2,y2;..."
        )

    polygon: List[List[int]] = []
    for seg in segments:
        try:
            x_str, y_str = seg.split(",")
            polygon.append([int(x_str), int(y_str)])
        except Exception as exc:
            raise ValueError(
                "ROI polygon must be specified as pairs like x1,y1;x2,y2;..."
            ) from exc

    return polygon


def format_polygon_string(polygon: List[List[int]]) -> str:
    """Format polygon vertices into "x1,y1;x2,y2;..." string"""

    return ";".join(f"{x},{y}" for x, y in polygon)


def collect_files(target_folder):
    """Collect RAW files"""
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(target_folder, ext)))
    if not files:
        raise FileNotFoundError(f"No RAW image files found in folder: {target_folder}")
    files.sort()
    return files


def detect_meteors_advanced(
    target_folder=DEFAULT_TARGET_FOLDER,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    debug_folder=DEFAULT_DEBUG_FOLDER,
    diff_threshold=DEFAULT_DIFF_THRESHOLD,
    min_area=DEFAULT_MIN_AREA,
    min_aspect_ratio=DEFAULT_MIN_ASPECT_RATIO,
    hough_threshold=DEFAULT_HOUGH_THRESHOLD,
    hough_min_line_length=DEFAULT_HOUGH_MIN_LINE_LENGTH,
    hough_max_line_gap=DEFAULT_HOUGH_MAX_LINE_GAP,
    min_line_score=DEFAULT_MIN_LINE_SCORE,
    enable_roi_selection=DEFAULT_ENABLE_ROI_SELECTION,
    roi_polygon_cli=None,
    num_workers=DEFAULT_NUM_WORKERS,
    batch_size=DEFAULT_BATCH_SIZE,
    auto_batch_size=False,
    enable_parallel=True,
    profile=False,
):
    """
    Main processing: detect meteor candidates from consecutive RAW images
    """
    timing = {}
    t_total = time.time()

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)

    print(f"Collecting RAW files from: {target_folder}")
    files = collect_files(target_folder)

    if len(files) < 2:
        print("Need at least 2 images. Exiting.")
        return 0

    print(f"Found {len(files)} files")

    # Validate files and load the first usable image
    t_load = time.time()
    valid_files: List[str] = []
    prev_img: Optional[np.ndarray] = None
    total_files = len(files)

    print("Validating files...")
    last_progress_report = t_load

    for idx, raw_file in enumerate(files, start=1):
        try:
            img = load_and_bin_raw_fast(raw_file)
        except Exception as exc:
            print(
                f"Skipping corrupted RAW file: {os.path.basename(raw_file)} ({exc})"
            )
            continue

        if prev_img is None:
            prev_img = img
            if profile:
                timing["first_load"] = time.time() - t_load

        valid_files.append(raw_file)

        if time.time() - last_progress_report >= 1.0 or idx == total_files:
            suffix = "" if idx == total_files else "\r"
            print(
                f"Validated {idx}/{total_files} files", end=suffix, flush=True
            )
            last_progress_report = time.time()

    print()

    if prev_img is None or len(valid_files) < 2:
        print("Need at least 2 valid images. Exiting.")
        return 0

    files = valid_files
    height, width = prev_img.shape

    # ROI setup
    roi_mask = np.full((height, width), 255, dtype=np.uint8)
    roi_polygon = None

    if roi_polygon_cli:
        print(
            "ROI specified via command line: "
            f"polygon={format_polygon_string(roi_polygon_cli)}"
        )
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [np.array(roi_polygon_cli, dtype=np.int32)], 255)
        roi_polygon = roi_polygon_cli
    elif enable_roi_selection:
        roi_selection = select_roi(prev_img)
        if roi_selection:
            roi_mask = roi_selection["mask"]
            roi_polygon = roi_selection["polygon"]
            print(
                "ROI setup complete: " f"polygon={format_polygon_string(roi_polygon)}"
            )
        else:
            print("No ROI selected. Processing entire image.")
    else:
        print("Skipping ROI selection. Processing entire image.")

    # Prepare parameters
    params = {
        "diff_threshold": diff_threshold,
        "min_area": min_area,
        "min_aspect_ratio": min_aspect_ratio,
        "hough_threshold": hough_threshold,
        "hough_min_line_length": hough_min_line_length,
        "hough_max_line_gap": hough_max_line_gap,
        "min_line_score": min_line_score,
    }

    print(f"Starting processing: {len(files)} images")
    if enable_parallel:
        print(f"Parallel processing: {num_workers} workers, batch size: {batch_size}")

    if auto_batch_size:
        available_mem = get_available_memory_bytes()
        if available_mem is None:
            print(
                "Auto batch sizing: unable to read system memory; "
                "keeping requested batch size"
            )
        else:
            adjusted_batch_size = estimate_batch_size(
                batch_size, prev_img.shape, num_workers, available_mem=available_mem
            )
            if adjusted_batch_size != batch_size:
                print(
                    "Auto batch sizing: adjusting batch size "
                    f"from {batch_size} to {adjusted_batch_size} "
                    f"(free memory {available_mem / (1024 ** 3):.2f} GB)"
                )
                batch_size = adjusted_batch_size
            else:
                print("Auto batch sizing: requested batch size fits available memory")

    detected_count = 0
    t_process = time.time()

    # Create image pairs
    image_pairs = [(files[i], files[i - 1]) for i in range(1, len(files))]

    if enable_parallel and num_workers > 1:
        # Split into batches
        batches = [
            image_pairs[i : i + batch_size]
            for i in range(0, len(image_pairs), batch_size)
        ]

        print(f"Number of batches: {len(batches)}")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []

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
                            detected_count += 1
                            shutil.copy(filepath, os.path.join(output_folder, filename))

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
                                f"\rChecking... {processed}/{len(image_pairs)}",
                                end="",
                                flush=True,
                            )

                except Exception as e:
                    print(f"\nBatch processing error: {e}")

    else:
        # Sequential processing
        batch_results = process_image_batch(image_pairs, roi_mask, params)

        for idx, result in enumerate(batch_results):
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
                print(f"  [LINE] {filename}: score={line_score:.1f}, lines={num_lines}")

            if is_candidate:
                detected_count += 1
                shutil.copy(filepath, os.path.join(output_folder, filename))

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
                        os.path.join(debug_folder, f"mask_{filename}.png"), debug_img
                    )

                print(f"  [HIT] {filename}: Ratio={aspect_ratio:.2f}")
            else:
                print(f"\rChecking... {idx + 1}/{len(image_pairs)}", end="", flush=True)

    if profile:
        timing["processing"] = time.time() - t_process
        timing["total"] = time.time() - t_total

        print("\n\n=== Performance Profile ===")
        print(f"First image load: {timing['first_load']:.3f}s")
        print(f"Processing time: {timing['processing']:.3f}s")
        print(f"Total time: {timing['total']:.3f}s")
        print(f"Images processed: {len(files) - 1}")
        print(f"Average per image: {timing['processing'] / (len(files) - 1):.3f}s")
        print(
            f"Speedup potential: {(len(files) - 1) * 0.5 / timing['processing']:.2f}x"
        )

    print(f"\nComplete! {detected_count} candidates extracted")
    return detected_count


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Tool to detect meteor candidates from sequential RAW images"
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

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
        help=f"Minimum aspect ratio (long side/short side) (default: {DEFAULT_MIN_ASPECT_RATIO})",
    )

    parser.add_argument(
        "--hough-threshold",
        type=int,
        default=DEFAULT_HOUGH_THRESHOLD,
        help=(
            "Accumulator threshold for Hough line detection "
            f"(default: {DEFAULT_HOUGH_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--hough-min-line-length",
        type=int,
        default=DEFAULT_HOUGH_MIN_LINE_LENGTH,
        help=(
            "Minimum length of a line to detect with Hough transform "
            f"(default: {DEFAULT_HOUGH_MIN_LINE_LENGTH})"
        ),
    )
    parser.add_argument(
        "--hough-max-line-gap",
        type=int,
        default=DEFAULT_HOUGH_MAX_LINE_GAP,
        help=(
            "Maximum allowed gap between points on the same Hough line "
            f"(default: {DEFAULT_HOUGH_MAX_LINE_GAP})"
        ),
    )
    parser.add_argument(
        "--min-line-score",
        type=float,
        default=DEFAULT_MIN_LINE_SCORE,
        help=(
            "Minimum total line length score required to mark a meteor candidate "
            f"(default: {DEFAULT_MIN_LINE_SCORE})"
        ),
    )

    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Skip ROI selection and process entire image",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help='Specify ROI polygon as "x1,y1;x2,y2;..." (requires at least 3 vertices)',
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel processing workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch processing size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--auto-batch-size",
        action="store_true",
        help=(
            "Automatically reduce batch size to fit available memory "
            f"(uses {int(AUTO_BATCH_MEMORY_FRACTION * 100)}% of free RAM)"
        ),
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Display execution time profiling"
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    roi_polygon_cli = None
    enable_roi_selection = DEFAULT_ENABLE_ROI_SELECTION

    if args.roi is not None:
        roi_polygon_cli = parse_roi_polygon_string(args.roi)
        enable_roi_selection = False
    elif args.no_roi:
        enable_roi_selection = False

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
    )


if __name__ == "__main__":
    main()
