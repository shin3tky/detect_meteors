"""Hold out spatially interleaved stellar matches for registration evaluation."""

import numpy as np

from meteor_core.star_registration import TILE_PX, fit_registration


def estimate_registration(previous, current):
    """Fit on spatially interleaved training points and hold out matches for evaluation."""
    previous, current = previous.astype(np.float64), current.astype(np.float64)
    cells = np.floor(previous / TILE_PX).astype(int)
    validation = (cells[:, 0] + 2 * cells[:, 1]) % 4 == 0
    fitted = None
    if validation.sum() >= 6:
        fitted = fit_registration(previous[~validation], current[~validation])
    return fitted, (previous[validation], current[validation])
