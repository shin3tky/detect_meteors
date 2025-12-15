"""Lightweight configurable detector example for validation and tests."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import DataclassDetector


@dataclass
class SimpleThresholdConfig:
    """Configuration for :class:`SimpleThresholdDetector`."""

    diff_threshold: int = 5
    min_area: int = 1


class SimpleThresholdDetector(DataclassDetector[SimpleThresholdConfig]):
    """Minimal detector that thresholds frame differences using a config."""

    # This detector intentionally keeps the algorithmic footprint tiny so it can be
    # used in tests and examples without heavy dependencies. The pipeline is:
    # 1. Compute absolute difference between the current and previous frames. This
    #    removes static background and highlights motion.
    # 2. Apply the ROI mask to ignore pixels outside the region of interest.
    # 3. Threshold the masked difference image using ``diff_threshold`` to create a
    #    binary map of candidate pixels.
    # 4. Extract external contours from the binary map and keep only those whose
    #    area exceeds ``min_area``. Each accepted contour becomes a candidate
    #    bounding box. Aspect ratios are tracked to give an approximate shape
    #    score (long thin streaks vs. blobs).
    #
    # The detector returns ``is_candidate`` as soon as any contour passes the area
    # check; it does not attempt temporal linking or advanced filtering. The
    # ``line_score`` is simply the sum of non-zero pixels in the thresholded mask,
    # which keeps scoring deterministic and easy to reason about for regression
    # tests. If you extend this class, prefer adding optional processing steps that
    # can be toggled via the config so the default behavior remains stable for
    # unit tests.

    plugin_name: str = "simple_threshold"
    name: str = "SimpleThresholdDetector"
    version: str = "1.0.0"
    ConfigType = SimpleThresholdConfig

    def __init__(self, config: SimpleThresholdConfig) -> None:
        super().__init__(config)

    def detect(
        self,
        current_image: np.ndarray,
        previous_image: np.ndarray,
        roi_mask: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[
        bool, float, List[Tuple[int, int, int, int]], float, Optional[np.ndarray]
    ]:
        # Frame inputs are expected to be single-channel uint8 images. Mixed
        # channel types will still work because OpenCV handles per-channel
        # subtraction, but tests keep images grayscale so the summed pixel
        # count below behaves deterministically.
        diff = cv2.absdiff(current_image, previous_image)

        # Mask out irrelevant pixels early to avoid spurious contours along ROI
        # edges; we re-use the mask for both inputs to keep the binary map
        # strictly limited to the region of interest.
        masked = cv2.bitwise_and(diff, diff, mask=roi_mask)
        _, binary = cv2.threshold(
            masked, self.config.diff_threshold, 255, cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        segments: List[Tuple[int, int, int, int]] = []
        max_aspect_ratio = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area:
                # Skip speckle noise and tiny flashes to reduce false positives
                # when validating the detector with small test frames.
                continue
            x, y, w, h = cv2.boundingRect(contour)
            segments.append((x, y, x + w, y + h))
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            max_aspect_ratio = max(max_aspect_ratio, aspect_ratio)

        is_candidate = len(segments) > 0
        # Use the raw sum to keep scoring monotonic with respect to pixel count
        # rather than contour count; this mirrors a naive energy estimate and
        # remains stable even if contour extraction changes slightly.
        line_score = float(np.sum(binary))
        return is_candidate, line_score, segments, max_aspect_ratio, None

    def compute_line_score(
        self, mask: np.ndarray, hough_params: Dict[str, int]
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        # This detector does not compute separate line scores; reuse detect results.
        _, line_score, segments, _, _ = self.detect(mask, mask, mask, hough_params)
        return line_score, segments


__all__ = ["SimpleThresholdDetector", "SimpleThresholdConfig"]
