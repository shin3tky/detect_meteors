"""Build a causal three-frame background in the current sky coordinates."""

from benchmark_v2 import temporal_median
from star_registration import estimate_registration, match_centroids, warp_reference

HISTORY_DEPTH = 3


def registered_median(history, catalogues, current_stars, *, max_shift):
    """Align each original reference directly, or fall back for the entire window."""
    if len(history) != HISTORY_DEPTH or len(catalogues) != HISTORY_DEPTH:
        raise ValueError("The median background requires three preceding exposures")

    shifts = []
    diagnostics = []
    for catalogue in catalogues:
        source, target = match_centroids(catalogue, current_stars)
        fit, held_out = estimate_registration(source, target)
        matrix = fit[0] if fit else None
        # The registration benchmark reserves a common margin around its sky ROI.
        if matrix is not None and (matrix[:, 2] ** 2).sum() > max_shift**2:
            matrix = None
        shifts.append(matrix)
        diagnostics.append(
            {
                "matches": len(source),
                "validation_matches": len(held_out[0]),
                "inliers": fit[2] if fit else 0,
                "shift_xy": matrix[:, 2].tolist() if matrix is not None else None,
            }
        )

    usable = all(matrix is not None for matrix in shifts)
    references = (
        [
            warp_reference(image, matrix)
            for image, matrix in zip(history, shifts, strict=True)
        ]
        if usable
        else history
    )
    return temporal_median(references), {
        "used_alignment": usable,
        "references": diagnostics,
    }
