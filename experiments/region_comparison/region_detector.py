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
    enable_ransac: bool = False
    ransac_seed: int = 0
    ransac_max_regions: int = 64
    ransac_max_trials: int = 64
    ransac_max_points: int = 1024
    ransac_distance_px: float = 2.0
    ransac_min_inlier_fraction: float = 0.20

    def __post_init__(self) -> None:
        if type(self.enable_ransac) is not bool:
            raise ValueError("enable_ransac must be a boolean")
        for name, minimum in (
            ("diff_threshold", 0),
            ("min_area_px", 2),
            ("background_offset_px", 2),
            ("max_report_regions", 1),
            ("ransac_seed", 0),
            ("ransac_max_regions", 1),
            ("ransac_max_trials", 1),
            ("ransac_max_points", 2),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, minimum in (
            ("min_length_px", 1.0),
            ("min_aspect_ratio", 1.0),
            ("min_contrast", 0.0),
            ("ransac_distance_px", 0.0),
            ("ransac_min_inlier_fraction", 0.0),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or value < minimum
            ):
                raise ValueError(f"{name} must be a finite number >= {minimum}")
        if self.ransac_distance_px == 0:
            raise ValueError("ransac_distance_px must be positive")
        if not 0 < self.ransac_min_inlier_fraction <= 1:
            raise ValueError("ransac_min_inlier_fraction must be in (0, 1]")


def _component_points(labels, label, stats):
    x, y, width, height = map(int, stats[:4])
    rows, columns = np.nonzero(labels[y : y + height, x : x + width] == label)
    return np.column_stack((columns + x, rows + y)).astype(np.float32)


def _fit_axis(points):
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array((x0, y0), dtype=np.float64), np.array((vx, vy), dtype=np.float64)


def _measure_axis(points, origin, direction, residual, roi, config):
    normal = np.array((-direction[1], direction[0]))
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
        "length_px": length,
        "width_px": cross_width,
        "aspect_ratio": ratio,
        "contrast": contrast,
        "axis_support": support,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def _measure_region(labels, label, stats, residual, roi, config):
    x, y, width, height, area = map(int, stats)
    points = _component_points(labels, label, stats)
    return {
        "bbox_xywh": [x, y, width, height],
        "area_px": area,
        **_measure_axis(points, *_fit_axis(points), residual, roi, config),
    }


def _axis_distances(points, origin, direction):
    offsets = points - origin
    return np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])


def _ransac_axis(points, config, rng):
    """Choose a two-point hypothesis on a bounded sample, then refit its inliers."""
    sample = points
    if len(points) > config.ransac_max_points:
        sample = points[
            rng.choice(len(points), config.ransac_max_points, replace=False)
        ]
    first = rng.integers(len(sample), size=config.ransac_max_trials)
    second = rng.integers(len(sample) - 1, size=config.ransac_max_trials)
    second += second >= first
    best_key = (-1, float("-inf"))
    best_axis = None
    for start, finish in zip(sample[first], sample[second], strict=True):
        direction = finish.astype(np.float64) - start
        direction /= np.linalg.norm(direction)
        distances = _axis_distances(sample, start, direction)
        inliers = distances <= config.ransac_distance_px
        key = (int(np.count_nonzero(inliers)), -float(distances[inliers].sum()))
        if key > best_key:
            best_key, best_axis = key, (start, direction)

    # Every sample pair is distinct because component points are unique pixels.
    assert best_axis is not None
    inliers = points[_axis_distances(points, *best_axis) <= config.ransac_distance_px]
    if len(inliers) < 2:
        return *best_axis, inliers
    origin, direction = _fit_axis(inliers)
    inliers = points[
        _axis_distances(points, origin, direction) <= config.ransac_distance_px
    ]
    return origin, direction, inliers


def _retry_rejected_regions(
    regions, component_indices, labels, stats, residual, roi, config
):
    """Try the largest rejected regions first and retain the original accepted fits."""
    rejected = sorted(
        (index for index, region in enumerate(regions) if not region["accepted"]),
        key=lambda index: regions[index]["area_px"],
        reverse=True,
    )
    attempts = rejected[: config.ransac_max_regions]
    rng = np.random.default_rng(config.ransac_seed)
    recovered = 0
    for index in attempts:
        label = component_indices[index] + 1
        region = regions[index]
        points = _component_points(labels, label, stats[label])
        origin, direction, inliers = _ransac_axis(points, config, rng)
        count = len(inliers)
        fraction = count / len(points)
        enough_inliers = (
            count >= config.min_area_px
            and fraction >= config.ransac_min_inlier_fraction
        )
        refit = (
            _measure_axis(inliers, origin, direction, residual, roi, config)
            if enough_inliers
            else None
        )
        accepted = refit is not None and refit["accepted"]
        region["ransac"] = {
            "inlier_count": count,
            "inlier_fraction": fraction,
            "enough_inliers": enough_inliers,
            "accepted": accepted,
            "original": {
                key: region[key] for key in ("aspect_ratio", "contrast", "gates")
            },
            "refit": refit,
        }
        if accepted and refit is not None:
            region.update(refit)
            recovered += 1
    return {
        "seed": config.ransac_seed,
        "regions_attempted": len(attempts),
        "regions_skipped": len(rejected) - len(attempts),
        "regions_recovered": recovered,
    }


class RegionDetector(DataclassDetector[RegionConfig]):
    """Find elongated components in an absolute frame difference."""

    plugin_name = "region"
    name = "Experimental region detector"
    version = "0.2.0"
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
        component_indices = np.flatnonzero(measurable).tolist()
        regions = [
            _measure_region(
                labels, index + 1, stats[index + 1], residual, roi, self.config
            )
            for index in component_indices
        ]
        ransac_diagnostics = {}
        if self.config.enable_ransac:
            ransac_diagnostics["ransac"] = _retry_rejected_regions(
                regions, component_indices, labels, stats, residual, roi, self.config
            )
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
                    **ransac_diagnostics,
                }
            },
            metrics={
                "duration_ms": (perf_counter() - started) * 1000.0,
                "num_components": count - 1,
            },
        )
