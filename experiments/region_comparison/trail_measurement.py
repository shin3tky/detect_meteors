"""Pixel measurements shared by the experimental detectors."""

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
