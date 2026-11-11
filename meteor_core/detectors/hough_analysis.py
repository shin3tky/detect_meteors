"""Measurements and candidate decisions for the built-in Hough detector.

Measurements depend on preprocessing and Hough settings. Candidate thresholds
can be evaluated repeatedly against those measurements without loading images
or running Hough again. This module does not infer whether an object is a meteor.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import cv2
import numpy as np


RotatedRect = Tuple[Tuple[float, float], Tuple[float, float], float]


@dataclass(frozen=True)
class CandidateThresholds:
    """The three thresholds that do not change image measurements."""

    min_area: float
    min_aspect_ratio: float
    min_line_score: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "CandidateThresholds":
        """Read the required candidate thresholds from merged runtime parameters."""
        return cls(
            min_area=params["min_area"],
            min_aspect_ratio=params["min_aspect_ratio"],
            min_line_score=params["min_line_score"],
        )


@dataclass(frozen=True)
class ContourMeasurement:
    """Geometry of one external contour, before candidate filtering."""

    area: float
    rect: RotatedRect

    @classmethod
    def from_contour(cls, contour: np.ndarray) -> "ContourMeasurement":
        """Measure the contour area and its minimum-area enclosing rectangle."""
        center, size, angle = cv2.minAreaRect(contour)
        return cls(
            area=cv2.contourArea(contour),
            rect=((center[0], center[1]), (size[0], size[1]), angle),
        )

    @property
    def aspect_ratio(self) -> float | None:
        """Divide the long side by the short side floored at one pixel.

        A zero-length side yields None because the contour is degenerate.
        """
        width, height = self.rect[1]
        if width == 0 or height == 0:
            return None
        return max(width, height) / max(min(width, height), 1)


@dataclass(frozen=True)
class CandidateDecision:
    """A decision with the contours that passed and the gates that failed.

    Rejection reasons describe this detector's gates, not an astrophysical
    classification. ``max_aspect_ratio`` only includes area-eligible contours,
    preserving the existing DetectionResult semantics.
    """

    accepted_contours: Tuple[int, ...]
    max_aspect_ratio: float
    rejection_reasons: Tuple[str, ...]

    @property
    def is_candidate(self) -> bool:
        return bool(self.accepted_contours)


@dataclass
class HoughAnalysis:
    """Per-pair evidence retained for diagnosis and threshold evaluation.

    Changing ``diff_threshold``, ROI, input preprocessing, or any Hough parameter
    requires fresh measurements. ``score`` is the sum of all Hough line lengths
    in the ROI; it is not associated with an individual contour.
    """

    mask: np.ndarray
    score: float
    lines: list[Tuple[int, int, int, int]]
    contours: Tuple[ContourMeasurement, ...]
    thresholds: CandidateThresholds

    def evaluate(
        self, thresholds: CandidateThresholds | None = None
    ) -> CandidateDecision:
        """Apply the v1.x candidate rule without changing the measurements.

        Args:
            thresholds: Optional replacement for the measured pair's thresholds.

        Returns:
            Accepted contour indices, the maximum valid ratio among contours
            above the area threshold, and the rejection reasons. Area and ratio
            must strictly exceed their thresholds on the same contour. The
            frame-wide score must meet or exceed its threshold.
        """
        thresholds = thresholds if thresholds is not None else self.thresholds
        has_area_eligible_contour = False
        max_ratio = None
        shape_eligible = []
        for index, contour in enumerate(self.contours):
            if not (contour.area > thresholds.min_area):
                continue
            has_area_eligible_contour = True
            ratio = contour.aspect_ratio
            if ratio is None:
                continue
            if max_ratio is None or ratio > max_ratio:
                max_ratio = ratio
            if ratio > thresholds.min_aspect_ratio:
                shape_eligible.append(index)

        score_passed = self.score >= thresholds.min_line_score
        accepted = tuple(shape_eligible) if score_passed else ()

        reasons = []
        if not accepted:
            if not self.contours:
                reasons.append("no_contours")
            elif not has_area_eligible_contour:
                reasons.append("min_area")
            elif max_ratio is None:
                reasons.append("invalid_geometry")
            elif not shape_eligible:
                reasons.append("min_aspect_ratio")
            if not score_passed:
                reasons.append("min_line_score")

        return CandidateDecision(
            accepted, max_ratio if max_ratio is not None else 0.0, tuple(reasons)
        )

    def render_debug(self, decision: CandidateDecision) -> np.ndarray:
        """Render the mask and lines even when the frame was rejected.

        Accepted contours use red boxes, matching normal detection output.
        Callers diagnosing rejections can inspect the mask and yellow lines.
        Normal detection only calls this for candidates.
        """
        debug_image = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in self.lines:
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 1)
        for index in decision.accepted_contours:
            box = np.int64(cv2.boxPoints(self.contours[index].rect))
            cv2.drawContours(debug_image, [box], 0, (0, 0, 255), 2)
        return debug_image
