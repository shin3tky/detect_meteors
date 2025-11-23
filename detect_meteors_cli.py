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
import json
import hashlib
import sys
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Tuple, List, Optional, Dict

# ==========================================
# Default Settings
# ==========================================
VERSION = "1.3.1"

DEFAULT_PROGRESS_FILE = "progress.json"

DEFAULT_TARGET_FOLDER = "rawfiles"
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
    estimated_pair_bytes = int(
        base_bytes * 2 + base_bytes + (height * width) + base_bytes * 0.5
    )
    if estimated_pair_bytes <= 0:
        return requested_batch_size

    per_worker_budget = available_mem * safety_fraction / max(1, num_workers)
    max_pairs = max(1, int(per_worker_budget // estimated_pair_bytes))

    return max(1, min(requested_batch_size, max_pairs))


def estimate_diff_threshold_from_samples(
    files: List[str], roi_mask: np.ndarray, sample_size: int = 5
) -> int:
    """
    Estimation using percentile-based approach

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

    # Percentiles (for peaked distributions)
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
    v1.3.1: Improved min_area estimation with better star detection

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

        # v1.3.1: Use 98th percentile (brighter stars only, avoid noise)
        threshold = np.percentile(roi_pixels, 98)

        _, star_mask = cv2.threshold(img, int(threshold), 255, cv2.THRESH_BINARY)
        star_mask = cv2.bitwise_and(star_mask.astype(np.uint8), roi_mask)

        contours, _ = cv2.findContours(
            star_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # v1.3.1: Filter by area range to exclude noise and large artifacts
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

    # v1.3.1: Use 75th percentile × 2.0 for more robust estimation
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
    v1.3.1: Fixed min_line_score estimation with corrected focal length logic

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

    # v1.3.1: Reduced base coefficient from 4% to 2.5% based on real data
    base_score = diagonal * 0.025

    if focal_length_mm:
        # v1.3.1: FIXED - Corrected focal length logic
        # Wide angle (14mm) → shorter trails → LOWER scores
        # Telephoto (50mm+) → longer relative trails → HIGHER scores
        # Factor calculation: focal_length / 50.0 (NOT 50.0 / focal_length)
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

    # v1.3.1: Adjusted clamp range based on real meteor data
    estimated_score = np.clip(adjusted_score, 40.0, 150.0)

    print(f"{'─'*50}")
    print(f"✓ Estimated min_line_score: {estimated_score:.1f}")
    print(f"  → ~2.5% of image diagonal")
    print(f"{'='*50}\n")

    return estimated_score


def compute_params_hash(params: Dict) -> str:
    """Create a stable hash from parameter dictionary"""
    # Convert NumPy types to native Python types for JSON serialization
    params_clean = {}
    for key, value in params.items():
        if isinstance(value, np.integer):
            params_clean[key] = int(value)
        elif isinstance(value, np.floating):
            params_clean[key] = float(value)
        elif isinstance(value, np.ndarray):
            params_clean[key] = value.tolist()
        elif isinstance(value, list):
            # Handle list of lists (like roi_polygon)
            params_clean[key] = [
                (
                    [int(x) if isinstance(x, np.integer) else x for x in item]
                    if isinstance(item, (list, np.ndarray))
                    else item
                )
                for item in value
            ]
        else:
            params_clean[key] = value

    params_json = json.dumps(params_clean, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(params_json.encode("utf-8")).hexdigest()


def _init_worker_ignore_interrupt() -> None:
    """Ignore SIGINT in worker processes"""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def load_progress(progress_path: str) -> Optional[Dict]:
    """Load progress JSON if it exists"""
    if not os.path.exists(progress_path):
        return None

    try:
        with open(progress_path, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        print(f"Failed to read progress file {progress_path}: {exc}")
        return None


def save_progress(progress_path: str, progress_data: Dict) -> None:
    """Persist progress JSON to disk"""
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    progress_data.setdefault("created_at", now_iso)
    progress_data["last_updated"] = now_iso

    try:
        os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as fp:
            json.dump(progress_data, fp, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Failed to write progress file {progress_path}: {exc}")


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
    """Polygon ROI selection with vertex editing"""
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


def validate_raw_file(index: int, raw_file: str):
    """Attempt to load a RAW file, returning any validation error."""

    try:
        load_and_bin_raw_fast(raw_file)
        return index, raw_file, None
    except Exception as exc:
        return index, raw_file, exc


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
    validate_raw=False,
    progress_file=DEFAULT_PROGRESS_FILE,
    resume=True,
    auto_params=False,
    user_specified_diff_threshold=False,
    user_specified_min_area=False,
    user_specified_min_line_score=False,
    focal_length_mm=None,
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
        # Phase 1: diff_threshold
        if not user_specified_diff_threshold:
            diff_threshold = estimate_diff_threshold_from_samples(
                files, roi_mask, sample_size=5
            )
            print(f"→ Using auto-estimated diff_threshold: {diff_threshold}")
        else:
            print(f"→ Using user-specified diff_threshold: {diff_threshold}")

        # Phase 2.1: min_area
        if not user_specified_min_area:
            min_area = estimate_min_area_from_samples(
                files, roi_mask, diff_threshold, sample_size=3
            )
            print(f"→ Using auto-estimated min_area: {min_area}")
        else:
            print(f"→ Using user-specified min_area: {min_area}")

        # Phase 2.2: min_line_score
        if not user_specified_min_line_score:
            min_line_score = estimate_min_line_score_from_image(
                prev_img.shape, focal_length_mm
            )
            print(f"→ Using auto-estimated min_line_score: {min_line_score:.1f}")
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

    # Progress tracking setup
    params_for_hash = params.copy()
    if roi_polygon:
        params_for_hash["roi_polygon"] = roi_polygon

    progress_data: Dict = {
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
            # Split into batches
            batches = [
                image_pairs[i : i + batch_size]
                for i in range(0, len(image_pairs), batch_size)
            ]

            print(f"Number of batches: {len(batches)}")

            executor = ProcessPoolExecutor(
                max_workers=num_workers, initializer=_init_worker_ignore_interrupt
            )
            futures: List = []
            wait_for_tasks = True

            try:
                for batch in batches:
                    future = executor.submit(
                        process_image_batch, batch, roi_mask, params
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
                                shutil.copy(
                                    filepath, os.path.join(output_folder, filename)
                                )
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
                                print(f"  [HIT] {filename}: Ratio={aspect_ratio:.2f}")
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
                    print(
                        f"  [LINE] {filename}: score={line_score:.1f}, lines={num_lines}"
                    )

                if is_candidate:
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
                        f"\rChecking... {resume_offset + idx + 1}/{overall_total}",
                        end="",
                        flush=True,
                    )

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
        print(f"Average per image: {timing['processing'] / len(image_pairs):.3f}s")

    print(f"\nComplete! {detected_count} candidates extracted")
    return detected_count


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Meteor detection tool with comprehensive auto-parameter estimation (v1.3.1)"
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

    # Auto-params (v1.3.1: fixed focal length logic)
    parser.add_argument(
        "--auto-params",
        action="store_true",
        help="Auto-estimate diff_threshold, min_area, and min_line_score (v1.3.1)",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Focal length in mm (35mm equivalent; used for min_line_score estimation)",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.remove_progress:
        if os.path.exists(args.progress_file):
            os.remove(args.progress_file)
            print(f"Removed progress file: {args.progress_file}")
        else:
            print(f"Progress file not found: {args.progress_file}")
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
        focal_length_mm=args.focal_length,
    )


if __name__ == "__main__":
    main()
