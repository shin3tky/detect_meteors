"""Opt-in experiment using connected regions and cross-trail contrast."""

from dataclasses import dataclass
from math import ceil, dist, floor, isfinite, log1p
from time import perf_counter

import cv2
import numpy as np

from meteor_core.detectors import DataclassDetector
from meteor_core.schema import DetectionContext, DetectionResult
from trail_measurement import pair_images, profile_contrast


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
    ransac_priority: str = "area"
    ransac_seed: int = 0
    ransac_max_regions: int = 64
    ransac_max_trials: int = 64
    ransac_max_points: int = 1024
    ransac_distance_px: float = 2.0
    ransac_min_inlier_fraction: float = 0.20
    ransac_min_contrast: float | None = None
    ransac_min_contiguous_support_px: float | None = None
    reject_negative_axis_fraction: float | None = None
    min_positive_axis_fraction: float | None = None

    def __post_init__(self) -> None:
        if type(self.enable_ransac) is not bool:
            raise ValueError("enable_ransac must be a boolean")
        if self.ransac_priority not in ("area", "area_contrast_aspect"):
            raise ValueError("ransac_priority must be 'area' or 'area_contrast_aspect'")
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
        contrast = self.ransac_min_contrast
        if contrast is not None and (
            type(contrast) not in (int, float) or not isfinite(contrast) or contrast < 0
        ):
            raise ValueError("ransac_min_contrast must be a finite number >= 0 or null")
        support = self.ransac_min_contiguous_support_px
        if support is not None and (
            type(support) not in (int, float) or not isfinite(support) or support <= 0
        ):
            raise ValueError(
                "ransac_min_contiguous_support_px must be a finite number > 0 or null"
            )
        for name in ("reject_negative_axis_fraction", "min_positive_axis_fraction"):
            fraction = getattr(self, name)
            if fraction is not None and (
                type(fraction) not in (int, float) or not 0 < fraction <= 1
            ):
                raise ValueError(f"{name} must be in (0, 1] or null")


def _component_points(labels, label, stats):
    x, y, width, height = map(int, stats[:4])
    rows, columns = np.nonzero(labels[y : y + height, x : x + width] == label)
    return np.column_stack((columns + x, rows + y)).astype(np.float32)


def _fit_axis(points):
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array((x0, y0), dtype=np.float64), np.array((vx, vy), dtype=np.float64)


def _measure_axis(points, origin, direction, residual, roi, config, *, refit=False):
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
    }
    if refit and config.ransac_min_contrast is not None:
        gates["ransac_contrast"] = (
            contrast is not None and contrast >= config.ransac_min_contrast
        )
    else:
        gates["contrast"] = contrast is not None and contrast >= config.min_contrast
    measurement = {
        "length_px": length,
        "width_px": cross_width,
        "aspect_ratio": ratio,
        "contrast": contrast,
        "axis_support": support,
        "gates": gates,
    }
    min_support = config.ransac_min_contiguous_support_px
    if refit and min_support is not None:
        max_support = max((dist(start, end) for start, end in support), default=0.0)
        measurement["max_contiguous_support_px"] = max_support
        gates["contiguous_support"] = max_support >= min_support
    measurement["accepted"] = all(gates.values())
    return measurement


def _measure_region(labels, label, stats, residual, roi, config):
    x, y, width, height, area = map(int, stats)
    points = _component_points(labels, label, stats)
    return {
        "bbox_xywh": [x, y, width, height],
        "area_px": area,
        **_measure_axis(points, *_fit_axis(points), residual, roi, config),
    }


def _signed_axis_samples(spans, current, previous, roi):
    """Sample unique in-ROI pixels along the supported portions of an axis."""
    if not spans:
        return np.empty(0, dtype=np.int32)
    samples = [
        np.linspace(start, end, ceil(np.linalg.norm(np.subtract(end, start))) + 1)
        for start, end in spans
    ]
    coordinates = np.unique(np.rint(np.concatenate(samples)).astype(np.intp), axis=0)
    x, y = coordinates.T
    height, width = roi.shape
    within = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x[within], y[within]
    in_roi = roi[y, x]
    x, y = x[in_roi], y[in_roi]
    return np.subtract(current[y, x], previous[y, x], dtype=np.int32)


def _filter_signed_axes(
    regions, current, previous, roi, *, negative_limit, positive_min
):
    measured = rejected = 0
    for region in regions:
        if not region["accepted"]:
            continue
        values = _signed_axis_samples(region["axis_support"], current, previous, roi)
        negative_count = int(np.count_nonzero(values < 0))
        negative_fraction = negative_count / values.size if values.size else None
        evidence = {
            "sample_count": values.size,
            "negative_count": negative_count,
            "negative_fraction": negative_fraction,
            "median_dn": float(np.median(values)) if values.size else None,
        }
        # Keep the nested RANSAC gates as the evidence before this final filter.
        gates = dict(region["gates"])
        if negative_limit is not None:
            gates["signed_difference"] = (
                negative_fraction is None or negative_fraction < negative_limit
            )
        if positive_min is not None:
            positive_count = int(np.count_nonzero(values > 0))
            positive_fraction = positive_count / values.size if values.size else None
            evidence.update(
                positive_count=positive_count, positive_fraction=positive_fraction
            )
            gates["positive_axis_fraction"] = (
                positive_fraction is not None and positive_fraction >= positive_min
            )
        region["signed_difference"] = evidence
        region["gates"] = gates
        region["accepted"] = all(gates.values())
        measured += bool(values.size)
        rejected += not region["accepted"]
    diagnostics = {
        "regions_measured": measured,
        "regions_rejected": rejected,
    }
    if negative_limit is not None:
        diagnostics["negative_fraction_limit"] = negative_limit
    if positive_min is not None:
        diagnostics["positive_fraction_min"] = positive_min
    return diagnostics


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


def _ransac_priority_score(region, config):
    area = region["area_px"]
    if config.ransac_priority == "area":
        return area

    # Temper the influence of size and rank on measurements already available
    # from the ordinary fit. The ratios stop gaining weight at the existing gates.
    size_weight = log1p(area / config.min_area_px)
    contrast_weight = min(
        1.0, max(0.0, region["contrast"] or 0.0) / max(1.0, config.min_contrast)
    )
    aspect_weight = min(1.0, region["aspect_ratio"] / config.min_aspect_ratio)
    return size_weight * contrast_weight * aspect_weight


def _retry_rejected_regions(
    regions, component_indices, labels, stats, residual, roi, config
):
    """Re-evaluate a bounded subset of rejects and retain the original accepted fits."""
    rejected = [
        (_ransac_priority_score(region, config), region["area_px"], index)
        for index, region in enumerate(regions)
        if not region["accepted"]
    ]
    rejected.sort(key=lambda entry: entry[:2], reverse=True)
    attempts = rejected[: config.ransac_max_regions]
    rng = np.random.default_rng(config.ransac_seed)
    recovered = 0
    for rank, (priority_score, _, index) in enumerate(attempts, 1):
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
            _measure_axis(inliers, origin, direction, residual, roi, config, refit=True)
            if enough_inliers
            else None
        )
        accepted = refit is not None and refit["accepted"]
        region["ransac"] = {
            "priority_rank": rank,
            "priority_score": priority_score,
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
    diagnostics = {
        "priority": config.ransac_priority,
        "seed": config.ransac_seed,
        "regions_attempted": len(attempts),
        "regions_skipped": len(rejected) - len(attempts),
        "regions_recovered": recovered,
    }
    if config.ransac_min_contrast is not None:
        diagnostics["min_contrast"] = config.ransac_min_contrast
    if config.ransac_min_contiguous_support_px is not None:
        diagnostics["min_contiguous_support_px"] = (
            config.ransac_min_contiguous_support_px
        )
    return diagnostics


class RegionDetector(DataclassDetector[RegionConfig]):
    """Find elongated components in an absolute frame difference."""

    plugin_name = "region"
    name = "Experimental region detector"
    version = "0.7.0"
    ConfigType = RegionConfig

    def compute_line_score(self, mask, hough_params):
        """Satisfy the existing detector interface for algorithms without lines."""
        return 0.0, []

    def detect(self, context: DetectionContext) -> DetectionResult:
        started = perf_counter()
        current, previous, roi = pair_images(context)
        residual = cv2.absdiff(current, previous)
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
        diagnostics = {}
        if self.config.enable_ransac:
            diagnostics["ransac"] = _retry_rejected_regions(
                regions, component_indices, labels, stats, residual, roi, self.config
            )
        negative_fraction_limit = self.config.reject_negative_axis_fraction
        positive_fraction_min = self.config.min_positive_axis_fraction
        if negative_fraction_limit is not None or positive_fraction_min is not None:
            diagnostics["signed_difference"] = _filter_signed_axes(
                regions,
                current,
                previous,
                roi,
                negative_limit=negative_fraction_limit,
                positive_min=positive_fraction_min,
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
                    **diagnostics,
                }
            },
            metrics={
                "duration_ms": (perf_counter() - started) * 1000.0,
                "num_components": count - 1,
            },
        )
