"""Pixel measurements shared by the experimental detectors."""

from math import ceil, hypot

import cv2
import numpy as np

from meteor_core.schema import DetectionContext
from meteor_core.utils import ensure_numpy


def pair_residual(context: DetectionContext):
    current = ensure_numpy(context.current_image)
    previous = ensure_numpy(context.previous_image)
    roi = ensure_numpy(context.roi_mask) != 0
    if (
        current.ndim != 2
        or not current.size
        or current.shape != previous.shape
        or current.shape != roi.shape
    ):
        raise ValueError("Expected equally shaped, nonempty grayscale frames and ROI")
    if current.dtype != previous.dtype or current.dtype not in (np.uint8, np.uint16):
        raise ValueError("Expected matching uint8 or uint16 frames (normalize: false)")
    return cv2.absdiff(current, previous), roi


def difference_of_gaussians(residual):
    image = residual.astype(np.float32)
    small = cv2.GaussianBlur(image, (0, 0), 1.0)
    large = cv2.GaussianBlur(image, (0, 0), 3.0)
    return cv2.subtract(small, large, dst=small)


def profile_contrast(residual, roi, origin, direction, normal, positions, offset):
    """Measure median center brightness above two equal-width background bands."""
    height, width = residual.shape

    def sample(offsets):
        points = (
            origin
            + positions[:, None, None] * direction
            + offsets[None, :, None] * normal
        )
        x, y = np.rint(points).astype(np.intp).reshape(-1, 2).T
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y = x[valid], y[valid]
        return residual[y, x][roi[y, x]]

    center = sample(np.arange(-1, 2))
    left = sample(-np.arange(offset, offset + 3))
    right = sample(np.arange(offset, offset + 3))
    if not (center.size and left.size and right.size):
        return None
    return float(np.median(center) - np.median(np.concatenate((left, right))))


def measure_segment(
    line, residual, roi, *, min_length_px, min_contrast, background_offset_px
):
    """Qualify a line using its ROI coverage and the original residual."""
    x0, y0, x1, y1 = map(float, line)
    dx, dy = x1 - x0, y1 - y0
    length = hypot(dx, dy)
    contrast = None
    inside_roi = None
    long_enough = length >= min_length_px
    if long_enough:
        origin = np.array((x0, y0))
        direction = np.array((dx, dy)) / length
        positions = np.linspace(0.0, length, ceil(length) + 1)
        x, y = np.rint(origin + positions[:, None] * direction).astype(np.intp).T
        height, width = roi.shape
        in_bounds = ((x >= 0) & (x < width) & (y >= 0) & (y < height)).all()
        inside_roi = bool(in_bounds and roi[y, x].all())
        if inside_roi:
            normal = np.array((-direction[1], direction[0]))
            contrast = profile_contrast(
                residual,
                roi,
                origin,
                direction,
                normal,
                positions,
                background_offset_px,
            )
    gates = {
        "length": long_enough,
        "roi": inside_roi,
        "contrast": contrast >= min_contrast if contrast is not None else None,
    }
    return {
        "line_xyxy": [x0, y0, x1, y1],
        "length_px": length,
        "contrast": contrast,
        "gates": gates,
        "accepted": all(gates.values()),
    }
