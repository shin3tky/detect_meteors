"""Experimental bright-ridge detector using Gaussian derivatives."""

from dataclasses import dataclass
from math import ceil, isfinite
from time import perf_counter

import cv2
import numpy as np

from meteor_core.detectors import DataclassDetector
from meteor_core.schema import DetectionContext, DetectionResult
from trail_measurement import measure_segment, pair_images, signed_segment_result


@dataclass(frozen=True)
class RidgeConfig:
    scales_px: tuple[float, ...] = (1.0, 1.6)
    min_response: float = 1.5
    min_length_px: float = 15.0
    min_contrast: float = 3.0
    max_report_segments: int = 4096

    def __post_init__(self):
        if (
            not isinstance(self.scales_px, (list, tuple))
            or not self.scales_px
            or any(
                type(scale) not in (float, int)
                or not isfinite(scale)
                or not 0.7 <= scale <= 8
                for scale in self.scales_px
            )
        ):
            raise ValueError("scales_px must contain Gaussian scales in [0.7, 8]")
        object.__setattr__(self, "scales_px", tuple(self.scales_px))
        for name, minimum in (
            ("min_response", 0.01),
            ("min_length_px", 2),
            ("min_contrast", 0),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not isfinite(value)
                or value < minimum
            ):
                raise ValueError(f"{name} must be a finite number >= {minimum}")
        if type(self.max_report_segments) is not int or self.max_report_segments < 1:
            raise ValueError("max_report_segments must be a positive integer")


def _derivative_kernels(sigma):
    radius = ceil(4 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    smooth = np.exp(-0.5 * (x / sigma) ** 2)
    smooth /= smooth.sum()
    # sepFilter2D correlates with the kernel, so the first derivative has +x.
    first = x * smooth / sigma**2
    second = (x**2 - sigma**2) * smooth / sigma**4
    second -= second.sum() * smooth
    return tuple(kernel.astype(np.float32) for kernel in (smooth, first, second))


def _center_points(signed, roi, scales, min_response):
    """Locate a transverse maximum using a local quadratic expansion."""
    strength = np.zeros_like(signed)
    offset_x = np.zeros_like(signed)
    offset_y = np.zeros_like(signed)
    for sigma in scales:
        smooth, first, second = _derivative_kernels(sigma)
        xx = cv2.sepFilter2D(signed, -1, second, smooth)
        yy = cv2.sepFilter2D(signed, -1, smooth, second)
        xy = cv2.sepFilter2D(signed, -1, first, first)
        trace = (xx + yy) * 0.5
        radius = cv2.magnitude((xx - yy) * 0.5, xy)
        normal = trace - radius
        tangent = trace + radius
        response = -normal * sigma**2
        y, x = np.nonzero(
            roi
            & (response >= min_response)
            & (response > strength)
            & (np.abs(tangent) <= -0.6 * normal)
        )
        angle = 0.5 * np.arctan2(2 * xy[y, x], xx[y, x] - yy[y, x])
        nx, ny = -np.sin(angle), np.cos(angle)
        gx = cv2.sepFilter2D(signed, -1, first, smooth)[y, x]
        gy = cv2.sepFilter2D(signed, -1, smooth, first)[y, x]
        displacement = -(nx * gx + ny * gy) / normal[y, x]
        dx, dy = displacement * nx, displacement * ny
        # Retain only a maximum within the pixel; the margin covers discrete ties.
        keep = (np.abs(dx) <= 0.55) & (np.abs(dy) <= 0.55)
        y, x = y[keep], x[keep]
        strength[y, x] = response[y, x]
        offset_x[y, x] = dx[keep]
        offset_y[y, x] = dy[keep]
    return strength, offset_x, offset_y


def _ridge_segments(strength, dx, dy, min_length):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (strength > 0).astype(np.uint8), connectivity=8
    )
    for index in range(1, count):
        left, top, width, height, area = stats[index]
        # Subpixel offsets can extend the integer bounding box by at most 1.1 px.
        if area < 8 or np.hypot(width + 0.1, height + 0.1) < min_length:
            continue
        y, x = np.nonzero(labels[top : top + height, left : left + width] == index)
        x, y = x + left, y + top
        points = np.column_stack((x + dx[y, x], y + dy[y, x])).astype(np.float32)
        vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        direction = np.array((vx, vy))
        origin = np.array((cx, cy))
        relative = points - origin
        along = relative @ direction
        across = relative @ np.array((-vy, vx))
        extent = float(np.ptp(along))
        if extent < 2 * max(1, float(np.ptp(across))):
            continue
        endpoints = origin + np.array([along.min(), along.max()])[:, None] * direction
        yield endpoints.ravel(), float(np.median(strength[y, x])), int(area)


class RidgeDetector(DataclassDetector[RidgeConfig]):
    """Fit line segments to connected subpixel ridge points."""

    plugin_name = "ridge"
    name = "Experimental Hessian ridge detector"
    version = "0.1.0"
    ConfigType = RidgeConfig

    def compute_line_score(self, mask, hough_params):
        return 0.0, []

    def detect(self, context: DetectionContext) -> DetectionResult:
        started = perf_counter()
        current, previous, roi = pair_images(context)
        signed = cv2.subtract(current, previous, dtype=cv2.CV_32F)
        strength, dx, dy = _center_points(
            signed, roi, self.config.scales_px, self.config.min_response
        )
        segments = []
        for line, response, area in _ridge_segments(
            strength, dx, dy, self.config.min_length_px
        ):
            item = measure_segment(
                line,
                signed,
                roi,
                min_length_px=self.config.min_length_px,
                min_contrast=self.config.min_contrast,
                background_offset_px=6,
            )
            segments.append(dict(item, response=response, ridge_pixels=area))
        segments.sort(key=lambda item: item["response"], reverse=True)
        return signed_segment_result(
            self.plugin_name,
            segments,
            started,
            self.config.max_report_segments,
            ridge_pixels=int(np.count_nonzero(strength)),
        )
