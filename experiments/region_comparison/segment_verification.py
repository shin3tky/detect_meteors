"""Common geometric proposal verification for the v2 evaluation."""

from math import ceil

import numpy as np

TILE = 64
SIGMA_FLOOR = 1.0
POLARITY = 0.75
CONTIGUOUS = 45.0
MIN_LENGTH = 15.0
THRESHOLDS = {"shared_dn": 9.0, "shared_sigma": 2.0}


def noise_tiles(residual):
    """Estimate robust RAW residual noise on 64-pixel tiles, with a 1 DN floor."""
    height, width = residual.shape
    output = np.empty((ceil(height / TILE), ceil(width / TILE)), dtype=np.float32)
    for i, j in np.ndindex(output.shape):
        values = residual[i * TILE : (i + 1) * TILE : 2, j * TILE : (j + 1) * TILE : 2]
        center = np.median(values)
        output[i, j] = max(SIGMA_FLOOR, 1.4826 * np.median(np.abs(values - center)))
    return output


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
        if length < MIN_LENGTH:
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
