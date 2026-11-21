"""Experimental detector exposing the full multiscale Fast Radon results."""

from dataclasses import dataclass
from time import perf_counter

import cv2

from meteor_core.detectors import DataclassDetector
from meteor_core.detectors.frt_projection import FrtProjectionConfig, project_segments
from meteor_core.schema import DetectionContext, DetectionResult
from trail_measurement import pair_images, signed_segment_result


@dataclass(frozen=True)
class FrtConfig(FrtProjectionConfig):
    max_report_segments: int = 1024

    def __post_init__(self):
        super().__post_init__()
        if type(self.max_report_segments) is not int or self.max_report_segments < 1:
            raise ValueError("max_report_segments must be an integer >= 1")


class FrtDetector(DataclassDetector[FrtConfig]):
    """Search all signed residuals with dyadic shift-and-add line integrals."""

    plugin_name = "frt"
    name = "Experimental full-ROI Fast Radon detector"
    version = "0.1.0"
    ConfigType = FrtConfig

    def compute_line_score(self, mask, hough_params):
        return 0.0, []

    def detect(self, context: DetectionContext) -> DetectionResult:
        started = perf_counter()
        current, previous, roi = pair_images(context)
        signed = cv2.subtract(current, previous, dtype=cv2.CV_32F)
        segments, projected = project_segments(signed, roi, self.config)
        return signed_segment_result(
            self.plugin_name,
            segments,
            started,
            self.config.max_report_segments,
            projection_cells=projected,
            min_snr=self.config.min_snr,
        )
