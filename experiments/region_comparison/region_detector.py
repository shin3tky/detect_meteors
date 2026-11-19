"""Opt-in experiment using connected regions and cross-trail contrast."""

from dataclasses import dataclass
from math import ceil, floor, isfinite
from time import perf_counter

import cv2
import numpy as np

from meteor_core.detectors import DataclassDetector
from meteor_core.schema import DetectionContext, DetectionResult
from trail_measurement import pair_residual, profile_contrast


@dataclass(frozen=True)
class RegionConfig:
    diff_threshold: int = 9
    min_area_px: int = 24
    min_length_px: float = 15.0
    min_aspect_ratio: float = 3.0
    min_contrast: float = 15.0
    background_offset_px: int = 6
    max_report_regions: int = 128

    def __post_init__(self) -> None:
        for name, minimum in (
            ("diff_threshold", 0),
            ("min_area_px", 2),
            ("background_offset_px", 2),
            ("max_report_regions", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, minimum in (
            ("min_length_px", 1.0),
            ("min_aspect_ratio", 1.0),
            ("min_contrast", 0.0),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or value < minimum
            ):
                raise ValueError(f"{name} must be a finite number >= {minimum}")


def _measure_region(labels, label, stats, residual, roi, config):
    x, y, width, height, area = map(int, stats)
    rows, columns = np.nonzero(labels[y : y + height, x : x + width] == label)
    points = np.column_stack((columns + x, rows + y)).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    direction = np.array((vx, vy), dtype=np.float64)
    normal = np.array((-vy, vx), dtype=np.float64)
    origin = np.array((x0, y0), dtype=np.float64)
    centered = points - origin
    along = centered @ direction
    across = centered @ normal
    start, end = float(along.min()), float(along.max())
    length = end - start
    cross_width = max(1.0, float(np.ptp(across)))
    ratio = length / cross_width

    # Consecutive one-pixel bins close to the axis define its supported intervals.
    bins = np.unique(np.rint(along[np.abs(across) <= 1.5]).astype(np.intp))
    bins = bins[(bins >= floor(start)) & (bins <= ceil(end))]
    spans = np.split(bins, np.flatnonzero(np.diff(bins) > 1) + 1)
    support = [
        [
            (origin + max(start, span[0] - 0.5) * direction).tolist(),
            (origin + min(end, span[-1] + 0.5) * direction).tolist(),
        ]
        for span in spans
        if span.size and min(end, span[-1] + 0.5) > max(start, span[0] - 0.5)
    ]
    contrast = profile_contrast(
        residual, roi, origin, direction, normal, bins, config.background_offset_px
    )
    gates = {
        "length": length >= config.min_length_px,
        "aspect_ratio": ratio >= config.min_aspect_ratio,
        "contrast": contrast is not None and contrast >= config.min_contrast,
    }
    return {
        "bbox_xywh": [x, y, width, height],
        "area_px": area,
        "length_px": length,
        "width_px": cross_width,
        "aspect_ratio": ratio,
        "contrast": contrast,
        "axis_support": support,
        "gates": gates,
        "accepted": all(gates.values()),
    }


class RegionDetector(DataclassDetector[RegionConfig]):
    """Find elongated components in an absolute frame difference."""

    plugin_name = "region"
    name = "Experimental region detector"
    version = "0.1.0"
    ConfigType = RegionConfig

    def compute_line_score(self, mask, hough_params):
        """Satisfy the existing detector interface for algorithms without lines."""
        return 0.0, []

    def detect(self, context: DetectionContext) -> DetectionResult:
        started = perf_counter()
        residual, roi = pair_residual(context)
        foreground = ((residual > self.config.diff_threshold) & roi).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
            foreground, 8, cv2.CV_32S, cv2.CCL_SAUF
        )
        component_stats = stats[1:]
        widths = component_stats[:, cv2.CC_STAT_WIDTH] - 1
        heights = component_stats[:, cv2.CC_STAT_HEIGHT] - 1
        measurable = (
            component_stats[:, cv2.CC_STAT_AREA] >= self.config.min_area_px
        ) & (np.hypot(widths, heights) >= self.config.min_length_px)
        regions = [
            _measure_region(
                labels, index + 1, stats[index + 1], residual, roi, self.config
            )
            for index in np.flatnonzero(measurable).tolist()
        ]
        accepted = [region for region in regions if region["accepted"]]
        regions.sort(
            key=lambda region: (region["accepted"], region["area_px"]), reverse=True
        )
        return DetectionResult(
            is_candidate=bool(accepted),
            score=max((region["contrast"] for region in accepted), default=0.0),
            lines=[],
            aspect_ratio=max(
                (region["aspect_ratio"] for region in regions), default=0.0
            ),
            debug_image=None,
            extras={
                "region": {
                    "regions": regions[: self.config.max_report_regions],
                    "regions_omitted": max(
                        0, len(regions) - self.config.max_report_regions
                    ),
                    "components_below_area_or_bbox_length": int(
                        count - 1 - len(regions)
                    ),
                    "accepted_regions": len(accepted),
                }
            },
            metrics={
                "duration_ms": (perf_counter() - started) * 1000.0,
                "num_components": count - 1,
            },
        )
