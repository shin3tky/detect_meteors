#!/usr/bin/env python
#
# Detect Meteors CLI - Hough-based Detector
# © 2025 Shinichi Morita (shin3tky)
#

"""
Default meteor detector using Hough line transform.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np

from .base import BaseDetector
from meteor_core.schema import DetectionContext, DetectionResult


logger = logging.getLogger(__name__)


@dataclass
class HoughDetectorConfig:
    """Default configuration placeholder for :class:`HoughDetector`."""

    # The default Hough detector currently relies on runtime parameters provided
    # by the pipeline, so no fields are necessary. This dataclass exists to
    # satisfy registry expectations for a zero-argument ``ConfigType`` that
    # yields a complete default configuration.
    pass


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

    plugin_name: str = "hough"
    name: str = "HoughDetector"
    version: str = "1.0.0"

    ConfigType = HoughDetectorConfig

    def __init__(self, config: Optional[HoughDetectorConfig] = None):
        """Initialize the Hough detector.

        Args:
            config: Optional configuration to align with registry expectations.
                Currently unused because the default detector is parameter-free.
        """
        self.config = config or HoughDetectorConfig()
        # Pre-create morphology kernel (reused across detections)
        self._kernel = np.ones((3, 3), np.uint8)
        logger.debug("HoughDetector initialized with config: %s", self.config)

    def detect(
        self,
        context: DetectionContext,
    ) -> DetectionResult:
        """
        Detect meteor candidates by comparing two consecutive frames.

        Args:
            context: DetectionContext containing image data, ROI mask,
                runtime parameters, and metadata.

        Returns:
            DetectionResult with detection outcome and diagnostics.
        """
        current_image = context.current_image
        previous_image = context.previous_image
        roi_mask = context.roi_mask
        params = context.runtime_params

        if current_image.shape != previous_image.shape:
            logger.error("Current and previous images must share the same shape.")
            raise ValueError("current_image and previous_image must be the same size")

        if roi_mask.shape != current_image.shape:
            logger.error(
                "ROI mask shape %s does not match image shape %s.",
                roi_mask.shape,
                current_image.shape,
            )
            raise ValueError("roi_mask must match the shape of the input images")

        if not self.validate_params(params):
            raise ValueError("Invalid detection parameters supplied to HoughDetector")

        hough_runtime = params.get("hough", {})
        if not isinstance(hough_runtime, dict):
            logger.error("Hough runtime params must be a dictionary.")
            raise ValueError("Invalid detection parameters supplied to HoughDetector")

        required_keys = [
            "diff_threshold",
            "min_area",
            "min_aspect_ratio",
            "min_line_score",
        ]
        missing = [key for key in required_keys if key not in params]
        if missing:
            logger.error("Missing required detection parameters: %s", missing)
            raise ValueError("Invalid detection parameters supplied to HoughDetector")

        def _get_hough_param(key: str, fallback_key: str) -> Any:
            if key in params:
                return params[key]
            return hough_runtime.get(fallback_key)

        hough_threshold = _get_hough_param("hough_threshold", "threshold")
        hough_min_line_length = _get_hough_param(
            "hough_min_line_length", "min_line_length"
        )
        hough_max_line_gap = _get_hough_param("hough_max_line_gap", "max_line_gap")
        if None in (hough_threshold, hough_min_line_length, hough_max_line_gap):
            logger.error("Missing required Hough detection parameters.")
            raise ValueError("Invalid detection parameters supplied to HoughDetector")

        try:
            # Calculate difference
            diff = cv2.absdiff(current_image, previous_image)

            # Binarize
            _, mask = cv2.threshold(
                diff, params["diff_threshold"], 255, cv2.THRESH_BINARY
            )
            mask = mask.astype(np.uint8)

            # Apply ROI
            cv2.bitwise_and(mask, roi_mask, dst=mask)

            # Noise removal with morphological opening
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

            # Hough transform
            hough_params = {
                "threshold": hough_threshold,
                "min_line_length": hough_min_line_length,
                "max_line_gap": hough_max_line_gap,
            }
            line_score, hough_lines = self.compute_line_score(mask, hough_params)

            # Shape detection
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            is_candidate = False
            debug_img = None
            max_aspect_ratio = 0.0

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area > params["min_area"]:
                    rect = cv2.minAreaRect(cnt)
                    (w, h) = rect[1]

                    if w == 0 or h == 0:
                        logger.debug("Skipping contour with zero width/height.")
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
                                cv2.line(
                                    debug_img, (x1, y1), (x2, y2), (0, 255, 255), 1
                                )

                        box = cv2.boxPoints(rect)
                        box = np.int64(box)
                        cv2.drawContours(debug_img, [box], 0, (0, 0, 255), 2)

            logger.debug(
                "Detection finished: candidate=%s, line_score=%.2f, max_aspect_ratio=%.2f",
                is_candidate,
                line_score,
                max_aspect_ratio,
            )
            return DetectionResult(
                is_candidate=is_candidate,
                score=line_score,
                lines=hough_lines,
                aspect_ratio=max_aspect_ratio,
                debug_image=debug_img,
                extras={"contour_count": len(contours)},
            )
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("Error during HoughDetector detection: %s", exc)
            raise

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
