"""Align the previous exposure to the current one using stars in the ROI."""

from dataclasses import replace
import logging

import numpy as np

from .base import BaseHook
from ..schema import DetectionContext
from ..star_registration import (
    fit_registration,
    match_centroids,
    star_centroids,
    warp_reference,
)
from ..utils import ensure_numpy

logger = logging.getLogger(__name__)


class StarRegistrationHook(BaseHook[None]):
    """Prepare adjacent RAW exposures for difference-based detection."""

    plugin_name = "star_registration"
    name = "Stellar registration"

    def on_before_detection(self, context: DetectionContext) -> DetectionContext:
        current = ensure_numpy(context.current_image)
        previous = ensure_numpy(context.previous_image)
        roi = ensure_numpy(context.roi_mask)
        if (
            current.ndim != 2
            or current.shape != previous.shape
            or roi.shape != current.shape
        ):
            raise ValueError(
                "Stellar registration requires a matching 2D pair and ROI."
            )
        if current.dtype != previous.dtype or current.dtype not in (
            np.uint16,
            np.float32,
        ):
            raise ValueError(
                "Stellar registration requires matching uint16 or normalized float32 images."
            )

        roi_pixels = roi != 0
        contrast_floor = 6.0
        if current.dtype == np.float32:
            contrast_floor /= np.iinfo(np.uint16).max
        source, target = match_centroids(
            star_centroids(previous, roi_pixels, contrast_floor=contrast_floor),
            star_centroids(current, roi_pixels, contrast_floor=contrast_floor),
        )
        fit = fit_registration(source, target)
        if fit is None:
            return self._with_diagnostics(
                context, status="fit_rejected", matched_stars=len(source)
            )

        _, matrix, inliers = fit
        # A pixel is usable only when all contributing reference samples lie
        # inside the original ROI and image. Use the same interpolation as for
        # the reference so a fraction of a border pixel cannot create a trail.
        coverage = warp_reference(roi_pixels.astype(np.float32), matrix)
        overlap = roi_pixels & (coverage == 1.0)
        if not overlap.any():
            return self._with_diagnostics(
                context, status="no_overlap", matched_stars=len(source)
            )

        aligned = replace(
            context,
            previous_image=warp_reference(previous, matrix),
            roi_mask=np.where(overlap, np.uint8(255), np.uint8(0)),
        )
        return self._with_diagnostics(
            aligned,
            status="applied",
            matched_stars=len(source),
            inliers=inliers,
            matrix=matrix.tolist(),
        )

    @staticmethod
    def _with_diagnostics(context: DetectionContext, **diagnostics) -> DetectionContext:
        logger.debug(
            "Stellar registration for frame %s: %s",
            context.metadata.get("frame_index"),
            diagnostics,
        )
        return replace(
            context, metadata={**context.metadata, "star_registration": diagnostics}
        )
