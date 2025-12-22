"""Lightweight configurable detector example for validation and tests."""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .base import DataclassDetector
from meteor_core.schema import DetectionContext, DetectionResult
from meteor_core.utils import ensure_numpy


logger = logging.getLogger(__name__)


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
        logger.debug("SimpleThresholdDetector initialized with config: %s", self.config)

    def detect(
        self,
        context: DetectionContext,
    ) -> DetectionResult:
        current_image = ensure_numpy(context.current_image)
        previous_image = ensure_numpy(context.previous_image)
        roi_mask = ensure_numpy(context.roi_mask)

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

        start_time = time.perf_counter()
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
        try:
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

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            mask_area = int(np.count_nonzero(binary))
            metrics = {
                "duration_ms": duration_ms,
                "num_contours": len(contours),
                "mask_area": mask_area,
                "hough_votes": 0,
            }

            logger.debug(
                "SimpleThreshold detection finished: candidate=%s, line_score=%.2f, max_aspect_ratio=%.2f",
                is_candidate,
                line_score,
                max_aspect_ratio,
            )
            return DetectionResult(
                is_candidate=is_candidate,
                score=line_score,
                lines=segments,
                aspect_ratio=max_aspect_ratio,
                debug_image=None,
                extras={},
                metrics=metrics,
            )
        except Exception as exc:  # pragma: no cover - guard against OpenCV errors
            logger.exception("Error during SimpleThreshold detection: %s", exc)
            raise

    def compute_line_score(
        self, mask: np.ndarray, hough_params: Dict[str, int]
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        # This detector does not compute separate line scores; reuse detect results.
        runtime_params = self.build_runtime_params(hough_params)
        context = DetectionContext(
            current_image=mask,
            previous_image=mask,
            roi_mask=mask,
            runtime_params=runtime_params,
            metadata={},
        )
        result = self.detect(context)
        return result.score, result.lines


__all__ = ["SimpleThresholdDetector", "SimpleThresholdConfig"]
