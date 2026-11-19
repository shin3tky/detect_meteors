"""Opt-in line segment detector with two interchangeable filters."""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import cv2
import numpy as np

from meteor_core.detectors import DataclassDetector
from meteor_core.schema import DetectionContext, DetectionResult
from trail_measurement import difference_of_gaussians, measure_segment, pair_residual


@dataclass(frozen=True)
class LsdConfig:
    preprocess: str = "dog"
    response_clip_dn: float = 70.0
    min_length_px: float = 15.0
    min_contrast: float = 15.0
    background_offset_px: int = 6
    max_report_segments: int = 128

    def __post_init__(self) -> None:
        if self.preprocess not in ("dog", "tophat"):
            raise ValueError("preprocess must be 'dog' or 'tophat'")
        for name, minimum in (
            ("response_clip_dn", 1.0),
            ("min_length_px", 1.0),
            ("min_contrast", 0.0),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or value < minimum
            ):
                raise ValueError(f"{name} must be a finite number >= {minimum}")
        for name, minimum in (("background_offset_px", 2), ("max_report_segments", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


def _filter_residual(residual, preprocess):
    if preprocess == "dog":
        return difference_of_gaussians(residual)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(residual, cv2.MORPH_TOPHAT, kernel).astype(np.float32)


class LsdDetector(DataclassDetector[LsdConfig]):
    """Find line segments on a filtered residual and score them on the original."""

    plugin_name = "lsd"
    name = "Experimental line segment detector"
    version = "0.1.0"
    ConfigType = LsdConfig

    def compute_line_score(self, mask, hough_params):
        """The legacy scoring call has no intensity residual for a profile."""
        return 0.0, []

    def detect(self, context: DetectionContext) -> DetectionResult:
        started = perf_counter()
        residual, roi = pair_residual(context)
        response = _filter_residual(residual, self.config.preprocess)
        roi_pixels = int(np.count_nonzero(roi))
        clipped = int(np.count_nonzero((response > self.config.response_clip_dn) & roi))
        np.clip(response, 0, self.config.response_clip_dn, out=response)
        response *= 255.0 / self.config.response_clip_dn
        gray = response.astype(np.uint8)

        # Detect on the full residual to avoid introducing edges at ROI boundaries.
        # Check each measured segment against the ROI before scoring its profile.
        lines, _, _, _ = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)
        segments = (
            []
            if lines is None
            else [
                measure_segment(
                    line,
                    residual,
                    roi,
                    min_length_px=self.config.min_length_px,
                    min_contrast=self.config.min_contrast,
                    background_offset_px=self.config.background_offset_px,
                )
                for line in lines.reshape(-1, 4)
            ]
        )
        segments.sort(
            key=lambda item: (item["accepted"], item["length_px"]), reverse=True
        )
        accepted = [item for item in segments if item["accepted"]]
        # Keep full subpixel geometry for accepted segments, plus bounded diagnostics.
        report_count = max(len(accepted), self.config.max_report_segments)
        return DetectionResult(
            is_candidate=bool(accepted),
            score=max((item["contrast"] for item in accepted), default=0.0),
            lines=[
                tuple(round(value) for value in item["line_xyxy"]) for item in accepted
            ],
            aspect_ratio=0.0,
            debug_image=None,
            extras={
                "lsd": {
                    "preprocess": self.config.preprocess,
                    "response_clip_dn": self.config.response_clip_dn,
                    "clipped_fraction": clipped / roi_pixels if roi_pixels else 0.0,
                    "segments": segments[:report_count],
                    "segments_omitted": max(0, len(segments) - report_count),
                    "accepted_segments": len(accepted),
                }
            },
            metrics={
                "duration_ms": (perf_counter() - started) * 1000.0,
                "num_segments": len(segments),
            },
        )
