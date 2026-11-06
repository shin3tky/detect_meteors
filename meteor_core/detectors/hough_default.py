#!/usr/bin/env python
#
# Detect Meteors CLI - Hough-based Detector
# © 2025 Shinichi Morita (shin3tky)
#

"""
Default meteor detector using Hough line transform.
"""

import logging
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np

from .base import BaseDetector
from .hough_analysis import CandidateThresholds, ContourMeasurement, HoughAnalysis
from meteor_core.schema import DetectionContext, DetectionResult
from meteor_core.utils import ensure_numpy


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
        """Measure the image pair and apply the existing v1.x candidate rule."""
        start_time = time.perf_counter()
        analysis = self.analyze(context)
        decision = analysis.evaluate()
        debug_image = analysis.render_debug(decision) if decision.is_candidate else None
        metrics = {
            "duration_ms": (time.perf_counter() - start_time) * 1000.0,
            "num_contours": len(analysis.contours),
            "mask_area": int(np.count_nonzero(analysis.mask)),
            "hough_votes": len(analysis.lines),
        }
        logger.debug(
            "Detection finished: candidate=%s, line_score=%.2f, max_aspect_ratio=%.2f",
            decision.is_candidate,
            analysis.score,
            decision.max_aspect_ratio,
        )
        return DetectionResult(
            is_candidate=decision.is_candidate,
            score=analysis.score,
            lines=analysis.lines,
            aspect_ratio=decision.max_aspect_ratio,
            debug_image=debug_image,
            extras={},
            metrics=metrics,
        )

    def analyze(self, context: DetectionContext) -> HoughAnalysis:
        """
        Measure an image pair independently of candidate thresholds.

        Args:
            context: DetectionContext containing image data, ROI mask,
                runtime parameters, and metadata.

        Returns:
            HoughAnalysis with the mask, line evidence, and contour geometry.
            The caller may re-evaluate area, ratio, and score thresholds without
            recomputing image measurements. This is a built-in Hough extension;
            the BaseDetector plugin contract remains unchanged.
        """
        current_image = ensure_numpy(context.current_image)
        previous_image = ensure_numpy(context.previous_image)
        roi_mask = ensure_numpy(context.roi_mask)
        global_params, detector_params = self.split_runtime_params(
            context.runtime_params
        )
        params = {**global_params, **detector_params}

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

            measurements = tuple(
                ContourMeasurement.from_contour(contour) for contour in contours
            )
            return HoughAnalysis(
                mask=mask,
                score=line_score,
                lines=hough_lines,
                contours=measurements,
                thresholds=CandidateThresholds.from_params(params),
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
        return compute_line_score_fast(mask, hough_params)


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
