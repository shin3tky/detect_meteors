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
        diff = cv2.absdiff(current_image, previous_image)
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
                continue
            x, y, w, h = cv2.boundingRect(contour)
            segments.append((x, y, x + w, y + h))
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            max_aspect_ratio = max(max_aspect_ratio, aspect_ratio)

        is_candidate = len(segments) > 0
        line_score = float(np.sum(binary))
        return is_candidate, line_score, segments, max_aspect_ratio, None

    def compute_line_score(
        self, mask: np.ndarray, hough_params: Dict[str, int]
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        # This detector does not compute separate line scores; reuse detect results.
        _, line_score, segments, _, _ = self.detect(mask, mask, mask, hough_params)
        return line_score, segments


__all__ = ["SimpleThresholdDetector", "SimpleThresholdConfig"]
