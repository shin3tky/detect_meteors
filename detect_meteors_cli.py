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
DEFAULT_TARGET_FOLDER = "examples"
DEFAULT_OUTPUT_FOLDER = "candidates"
DEFAULT_DEBUG_FOLDER = "debug_masks"

EXTENSIONS = ["*.ORF", "*.ARW", "*.CR2", "*.NEF", "*.DNG"]

DEFAULT_DIFF_THRESHOLD = 8
DEFAULT_MIN_AREA = 10
DEFAULT_MIN_ASPECT_RATIO = 3.0

HOUGH_THRESHOLD = 10
HOUGH_MIN_LINE_LENGTH = 15
HOUGH_MAX_LINE_GAP = 5
MIN_LINE_SCORE = 80.0

DEFAULT_ENABLE_ROI_SELECTION = True
DEFAULT_NUM_WORKERS = max(1, mp.cpu_count() - 1)
DEFAULT_BATCH_SIZE = 10  # Batch processing size


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
    """Function to select processing region (ROI) with mouse"""
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

    print("\n--- ROI Selection Mode ---")
    print(
        "1. In the displayed window, drag with your mouse to outline the 'sky area (where you want to search for meteors)'."
    )
    print("2. Press Enter to confirm, or Esc to cancel.\n")

    cv2.namedWindow("Select Sky Area", cv2.WINDOW_NORMAL)
    cv2.imshow("Select Sky Area", disp_img_resized)

    r = cv2.selectROI(
        "Select Sky Area", disp_img_resized, showCrosshair=True, fromCenter=False
    )
    cv2.destroyWindow("Select Sky Area")
    cv2.waitKey(1)

    if r == (0, 0, 0, 0):
        return None

    x = int(r[0] / scale_factor)
    y = int(r[1] / scale_factor)
    w = int(r[2] / scale_factor)
    h = int(r[3] / scale_factor)

    return (x, y, w, h)


def parse_roi_string(roi_str):
    """Parse --roi "x,y,w,h" format"""
    parts = roi_str.split(",")
    if len(parts) != 4:
        raise ValueError("ROI must be specified with 4 integers in the format x,y,w,h")
    x, y, w, h = map(int, parts)
    return (x, y, w, h)


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
    enable_roi_selection=DEFAULT_ENABLE_ROI_SELECTION,
    roi_rect_cli=None,
    num_workers=DEFAULT_NUM_WORKERS,
    batch_size=DEFAULT_BATCH_SIZE,
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

    # Load first image
    t_load = time.time()
    prev_img = load_and_bin_raw_fast(files[0])
    if profile:
        timing["first_load"] = time.time() - t_load

    height, width = prev_img.shape

    # ROI setup
    roi_mask = np.full((height, width), 255, dtype=np.uint8)
    roi_rect = None

    if roi_rect_cli:
        x, y, w, h = roi_rect_cli
        print(f"ROI specified via command line: x={x}, y={y}, w={w}, h={h}")
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[y : y + h, x : x + w] = 255
        roi_rect = roi_rect_cli
    elif enable_roi_selection:
        roi_rect = select_roi(prev_img)
        if roi_rect:
            x, y, w, h = roi_rect
            print(f"ROI setup complete: x={x}, y={y}, w={w}, h={h}")
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            roi_mask[y : y + h, x : x + w] = 255
        else:
            print("No ROI selected. Processing entire image.")
    else:
        print("Skipping ROI selection. Processing entire image.")

    # Prepare parameters
    params = {
        "diff_threshold": diff_threshold,
        "min_area": min_area,
        "min_aspect_ratio": min_aspect_ratio,
        "hough_threshold": HOUGH_THRESHOLD,
        "hough_min_line_length": HOUGH_MIN_LINE_LENGTH,
        "hough_max_line_gap": HOUGH_MAX_LINE_GAP,
        "min_line_score": MIN_LINE_SCORE,
    }

    print(f"Starting processing: {len(files)} images")
    if enable_parallel:
        print(f"Parallel processing: {num_workers} workers, batch size: {batch_size}")

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
                                if roi_rect:
                                    x, y, w, h = roi_rect
                                    cv2.rectangle(
                                        debug_img,
                                        (x, y),
                                        (x + w, y + h),
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
                    if roi_rect:
                        x, y, w, h = roi_rect
                        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

    parser.add_argument("--hough-threshold", type=int, default=HOUGH_THRESHOLD)
    parser.add_argument(
        "--hough-min-line-length", type=int, default=HOUGH_MIN_LINE_LENGTH
    )
    parser.add_argument("--hough-max-line-gap", type=int, default=HOUGH_MAX_LINE_GAP)
    parser.add_argument("--min-line-score", type=float, default=MIN_LINE_SCORE)

    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Skip ROI selection and process entire image",
    )
    parser.add_argument(
        "--roi", type=str, default=None, help='Specify ROI directly in "x,y,w,h" format'
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
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Display execution time profiling"
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    global HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP, MIN_LINE_SCORE
    HOUGH_THRESHOLD = args.hough_threshold
    HOUGH_MIN_LINE_LENGTH = args.hough_min_line_length
    HOUGH_MAX_LINE_GAP = args.hough_max_line_gap
    MIN_LINE_SCORE = args.min_line_score

    roi_rect_cli = None
    enable_roi_selection = DEFAULT_ENABLE_ROI_SELECTION

    if args.roi is not None:
        roi_rect_cli = parse_roi_string(args.roi)
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
        enable_roi_selection=enable_roi_selection,
        roi_rect_cli=roi_rect_cli,
        num_workers=args.workers,
        batch_size=args.batch_size,
        enable_parallel=not args.no_parallel,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
