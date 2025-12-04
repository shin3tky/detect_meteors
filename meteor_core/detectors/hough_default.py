#!/usr/bin/env python
#
# Detect Meteors CLI - Hough-based Detector
# © 2025 Shinichi Morita (shin3tky)
#

"""
Default meteor detector using Hough line transform.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from .base import BaseDetector


class HoughDetector(BaseDetector):
    """
    Meteor detector using Hough line transform for linear feature detection.

    This is the default detector that implements the original detection algorithm:
    1. Compute frame difference
    2. Binarize with threshold
    3. Apply morphological operations for noise reduction
    4. Detect lines using probabilistic Hough transform
    5. Analyze contour shapes for aspect ratio filtering
    """

    name: str = "HoughDetector"
    version: str = "1.0.0"

    def __init__(self):
        """Initialize the Hough detector."""
        # Pre-create morphology kernel (reused across detections)
        self._kernel = np.ones((3, 3), np.uint8)

    def detect(
        self,
        current_image: np.ndarray,
        previous_image: np.ndarray,
        roi_mask: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[
        bool, float, List[Tuple[int, int, int, int]], float, Optional[np.ndarray]
    ]:
        """
        Detect meteor candidates by comparing two consecutive frames.

        Args:
            current_image: Current frame (uint16 grayscale)
            previous_image: Previous frame (uint16 grayscale)
            roi_mask: Binary mask for region of interest (uint8)
            params: Detection parameters dictionary

        Returns:
            Tuple of (is_candidate, line_score, line_segments, max_aspect_ratio, debug_image)
        """
        # Calculate difference
        diff = cv2.absdiff(current_image, previous_image)

        # Binarize
        _, mask = cv2.threshold(diff, params["diff_threshold"], 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        # Apply ROI
        cv2.bitwise_and(mask, roi_mask, dst=mask)

        # Noise removal with morphological opening
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        # Hough transform
        hough_params = {
            "threshold": params["hough_threshold"],
            "min_line_length": params["hough_min_line_length"],
            "max_line_gap": params["hough_max_line_gap"],
        }
        line_score, hough_lines = self.compute_line_score(mask, hough_params)

        # Shape detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_candidate = False
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
                    is_candidate = True

                    if debug_img is None:
                        debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        for x1, y1, x2, y2 in hough_lines:
                            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 1)

                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    cv2.drawContours(debug_img, [box], 0, (0, 0, 255), 2)

        return is_candidate, line_score, hough_lines, max_aspect_ratio, debug_img

    def compute_line_score(
        self, mask: np.ndarray, hough_params: Dict[str, int]
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """
        Line detection using probabilistic Hough transform.

        Args:
            mask: Binary mask of detected changes (uint8)
            hough_params: Dictionary with Hough transform parameters

        Returns:
            Tuple of (score, line_segments)
        """
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


# Standalone function for backward compatibility
def compute_line_score_fast(
    mask: np.ndarray, hough_params: Dict[str, int]
) -> Tuple[float, List[Tuple[int, int, int, int]]]:
    """
    Line detection using Hough transform (standalone function).

    This function is kept for backward compatibility.
    Consider using HoughDetector.compute_line_score() instead.
    """
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
