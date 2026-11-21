"""Common geometric proposal verification for the v2 evaluation."""

from math import ceil

import cv2
import numpy as np

from meteor_core.detectors.frt_support import TILE as TILE, noise_tiles as noise_tiles

POLARITY = 0.75
CONTIGUOUS = 45.0
THRESHOLDS = {"shared_dn": 9.0, "shared_sigma": 2.0}
DN_PER_SIGMA = 4.5


def noise_scaled_pair(residual):
    """Represent a standardized signed residual as two unsigned analysis images."""
    scale = noise_tiles(residual)
    height, width = residual.shape

    def grid(size):
        starts = np.arange(0, size, TILE)
        centers = (starts + np.minimum(starts + TILE, size) - 1) / 2
        return np.interp(np.arange(size), centers, np.arange(len(starts))).astype(
            np.float32
        )

    x, y = np.meshgrid(grid(width), grid(height))
    sigma = cv2.remap(scale, x, y, cv2.INTER_LINEAR)
    pedestal = 32768
    previous = np.full_like(residual, pedestal, dtype=np.uint16)
    current = np.rint(
        pedestal + np.clip((residual / sigma) * DN_PER_SIGMA, -pedestal, pedestal - 1)
    )
    return current.astype(np.uint16), previous


def _support(supported, axis, spacing):
    edges = np.flatnonzero(np.diff(np.r_[False, supported, False]))
    if not edges.size:
        return []
    longest = (edges[1::2] - edges[::2] - 1) * spacing
    return [
        (*axis[start].tolist(), *axis[stop - 1].tolist())
        for start, stop, length in zip(edges[::2], edges[1::2], longest, strict=True)
        if length >= CONTIGUOUS
    ]


def verify_segments(lines, residual, tiles):
    """Keep continuous center-minus-side support on signed, longitudinal samples."""
    output = {policy: [] for policy in THRESHOLDS}
    height, width = residual.shape
    offsets = np.array([-8, -7, -6, -1, 0, 1, 6, 7, 8])
    for line in lines:
        start, stop = np.asarray(line, dtype=float).reshape(2, 2)
        delta = stop - start
        length = np.hypot(*delta)
        if length < CONTIGUOUS:
            continue
        count = ceil(length)
        axis = np.linspace(start, stop, count + 1)
        normal = np.array((-delta[1], delta[0])) / length
        # Each column is a signed RAW sample at a transverse offset.
        x, y = (
            np.rint(axis[:, None, :] + offsets[:, None] * normal)
            .astype(int)
            .transpose(2, 0, 1)
        )
        if x.min() < 0 or x.max() >= width or y.min() < 0 or y.max() >= height:
            continue
        values = residual[y, x]
        center = np.median(values[:, 3:6], axis=1)
        if (center > 0).mean() < POLARITY:
            continue
        sides = np.median(values[:, [0, 1, 2, 6, 7, 8]], axis=1)
        contrast = center - sides
        sigma = tiles[y[:, 4] // TILE, x[:, 4] // TILE]
        for policy, threshold in THRESHOLDS.items():
            units = contrast / sigma if policy == "shared_sigma" else contrast
            # The box smooths along the axis. Unsmoothed positive signs still
            # control which coordinates can belong to a supported interval.
            smooth = np.convolve(
                np.pad(units, 2, mode="edge"), np.full(5, 0.2), "valid"
            )
            output[policy].extend(
                _support((smooth >= threshold) & (center > 0), axis, length / count)
            )
    return output
