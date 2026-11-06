"""Bounded, read-only tuning proposals and frame-level sequence comparisons.

Proposals preserve the v1.x candidate rule. They are exploratory settings, not
meteor classifications or an estimate of a false-positive rate.
"""

from dataclasses import dataclass, replace
from itertools import islice, product
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .analysis import HoughReproduction
from .detectors.hough_analysis import (
    CandidateDecision,
    CandidateThresholds,
    HoughAnalysis,
)
from .exceptions import MeteorConfigError, MeteorValidationError
from .roi_selector import create_full_roi_mask, create_roi_mask_from_polygon
from .schema import DetectionParams


MEASUREMENT_PARAMETERS = (
    "diff_threshold",
    "hough_threshold",
    "hough_min_line_length",
    "hough_max_line_gap",
)
GATE_PARAMETERS = ("min_area", "min_aspect_ratio", "min_line_score")


@dataclass(frozen=True)
class TuningOptions:
    """Search limits, not camera-specific recommended thresholds."""

    max_gate_relaxation: float = 0.5
    max_measurements: int = 8
    max_evaluations: int = 256
    max_proposals: int = 5
    max_comparison_pairs: int = 100
    max_comparison_measurements: int = 600

    def __post_init__(self):
        fraction = self.max_gate_relaxation
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (float, int))
            or not 0 <= fraction <= 1
        ):
            raise MeteorConfigError("max_gate_relaxation must be between 0 and 1")
        for key in self.__dataclass_fields__:
            if key == "max_gate_relaxation":
                continue
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MeteorConfigError(
                    "Tuning limits must be positive integers", config_key=key
                )


@dataclass(frozen=True)
class SequenceComparison:
    total_pairs: int
    evaluated_pairs: int
    stop_reason: str
    added_candidates: tuple[str, ...]
    lost_candidates: tuple[str, ...]
    known_candidate_frames: tuple[str, ...]
    known_missed_frames: tuple[str, ...]
    known_unchecked_frames: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.total_pairs > 0 and self.evaluated_pairs == self.total_pairs


@dataclass(frozen=True)
class TuningProposal:
    params: DetectionParams
    changes: dict[str, dict[str, Any]]
    change_cost: float
    decision: CandidateDecision
    matched_contours: tuple[int, ...]
    comparison: SequenceComparison | None = None


@dataclass(frozen=True)
class TuningRun:
    replay: HoughReproduction
    options: TuningOptions
    scope: str
    region: tuple[int, int, int, int] | None
    status: str
    stop_reason: str
    measurements_run: int
    evaluations_run: int
    planned_measurements: int
    proposals: tuple[TuningProposal, ...]
    proposals_truncated: bool = False
    comparison_measurements: int = 0


def _measurement_key(params: DetectionParams) -> tuple[int, ...]:
    return tuple(getattr(params, key) for key in MEASUREMENT_PARAMETERS)


def _param_key(params: DetectionParams) -> tuple[Any, ...]:
    return tuple(params.to_dict().values())


def _changes(
    base: DetectionParams, trial: DetectionParams
) -> dict[str, dict[str, Any]]:
    before = base.to_dict()
    return {
        key: {"before": before[key], "after": value}
        for key, value in trial.to_dict().items()
        if before[key] != value
    }


def _cost(base: DetectionParams, trial: DetectionParams) -> float:
    return sum(
        abs(change["after"] - change["before"]) / max(abs(change["before"]), 1.0)
        for change in _changes(base, trial).values()
    )


def _measurement_grid(base: DetectionParams, grid: Mapping[str, Sequence[int]]):
    if not isinstance(grid, Mapping) or set(grid) - set(MEASUREMENT_PARAMETERS):
        raise MeteorConfigError(
            "The measurement grid supports only difference and Hough parameters"
        )
    columns = []
    for key in MEASUREMENT_PARAMETERS:
        values = grid.get(key, ())
        if key in grid and (not isinstance(values, (list, tuple)) or not values):
            raise MeteorConfigError(
                "Grid values must be non-empty lists of integers", config_key=key
            )
        minimum = 1 if key == "hough_threshold" else 0
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= 2**31 - 1
            for value in values
        ):
            raise MeteorConfigError("Invalid measurement grid value", config_key=key)
        original = getattr(base, key)
        ordered = sorted(
            set(values) - {original}, key=lambda value: (abs(value - original), value)
        )
        columns.append([original, *ordered])
    return columns, math.prod(len(column) for column in columns)


def _region(value: Sequence[int] | None, shape: tuple[int, int]):
    if value is None:
        return None
    if (
        len(value) != 4
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
        or not 0 <= value[0] < value[2] <= shape[1]
        or not 0 <= value[1] < value[3] <= shape[0]
    ):
        raise MeteorConfigError(
            "Analysis region must be x0,y0,x1,y1 within the loaded image (end exclusive)"
        )
    return (value[0], value[1], value[2], value[3])


def _matching_contours(analysis: HoughAnalysis, region) -> tuple[int, ...]:
    if region is None:
        return tuple(range(len(analysis.contours)))
    left, top, right, bottom = region
    contours, _ = cv2.findContours(
        analysis.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) != len(analysis.contours):
        raise MeteorValidationError(
            "Contour evidence no longer matches the measured mask"
        )
    matches = []
    for index, contour in enumerate(contours):
        x, y, width, height = cv2.boundingRect(contour)
        x0, y0 = max(left, x), max(top, y)
        x1, y1 = min(right, x + width), min(bottom, y + height)
        if x0 >= x1 or y0 >= y1:
            continue
        local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(local, [contour], 0, 255, cv2.FILLED, offset=(-x0, -y0))
        if np.any(local & analysis.mask[y0:y1, x0:x1]):
            matches.append(index)
    return tuple(matches)


def _passes(decision: CandidateDecision, matches: tuple[int, ...]) -> bool:
    return bool(set(decision.accepted_contours).intersection(matches))


def _gate_trials(
    base: DetectionParams,
    variant: DetectionParams,
    analysis: HoughAnalysis,
    matches: tuple[int, ...],
    options: TuningOptions,
):
    if not analysis.lines:
        return []
    trials = {}
    for index in matches:
        contour = analysis.contours[index]
        ratio = contour.aspect_ratio
        if ratio is None or contour.area <= 0:
            continue
        trial = replace(
            variant,
            min_area=min(base.min_area, math.ceil(contour.area) - 1),
            min_aspect_ratio=min(
                base.min_aspect_ratio, math.nextafter(ratio, -math.inf)
            ),
            min_line_score=min(base.min_line_score, analysis.score),
        )
        if any(
            getattr(trial, key) < getattr(base, key) * (1 - options.max_gate_relaxation)
            for key in GATE_PARAMETERS
        ):
            continue
        trials[_param_key(trial)] = trial
    return sorted(
        trials.values(), key=lambda trial: (_cost(base, trial), _param_key(trial))
    )


def tune_reproduction(
    replay: HoughReproduction,
    *,
    options: TuningOptions | None = None,
    measurement_grid: Mapping[str, Sequence[int]] | None = None,
    target_region: Sequence[int] | None = None,
) -> TuningRun:
    """Propose minimal gate relaxations per contour within a bounded grid.

    The baseline measurements count toward both budgets. New difference/Hough
    settings always remeasure the original image snapshot. Gate-only changes
    reuse evidence. Region matching establishes contour overlap, not meteor ID.
    """
    options = options if options is not None else TuningOptions()
    region = _region(target_region, replay.image_shape)
    base = replay.settings.params
    columns, planned = _measurement_grid(
        base, measurement_grid if measurement_grid is not None else {}
    )
    run = TuningRun(
        replay,
        options,
        "frame" if region is None else "annotated_region",
        region,
        "reference_missing",
        "reference_missing",
        0,
        0,
        planned,
        (),
    )
    if replay.analysis is None or replay.decision is None:
        return run
    run = replace(run, measurements_run=1, evaluations_run=1)
    if region is not None:
        polygon = replay.settings.roi_polygon
        roi = (
            create_full_roi_mask(replay.image_shape)
            if polygon is None
            else create_roi_mask_from_polygon(
                [list(point) for point in polygon], replay.image_shape
            )
        )
        left, top, right, bottom = region
        if not np.any(roi[top:bottom, left:right]):
            return replace(
                run, status="target_outside_roi", stop_reason="target_outside_roi"
            )
    matches = _matching_contours(replay.analysis, region)
    if _passes(replay.decision, matches):
        return replace(run, status="already_candidate", stop_reason="already_candidate")

    proposals = {}
    evaluated = set()
    measurements, evaluations = 1, 1
    stop = "completed"
    visited = 0
    for values in islice(product(*columns), options.max_measurements):
        if evaluations >= options.max_evaluations:
            stop = "evaluation_limit"
            break
        variant = replace(base, **dict(zip(MEASUREMENT_PARAMETERS, values)))
        if visited == 0:
            analysis = replay.analysis
        else:
            analysis = replay.remeasure(variant)
            measurements += 1
        visited += 1
        matches = _matching_contours(analysis, region)
        for params in _gate_trials(base, variant, analysis, matches, options):
            key = _param_key(params)
            if key in evaluated:
                continue
            if evaluations >= options.max_evaluations:
                stop = "evaluation_limit"
                break
            evaluated.add(key)
            decision = analysis.evaluate(
                CandidateThresholds.from_params(params.to_dict())
            )
            evaluations += 1
            if _passes(decision, matches):
                proposals[key] = TuningProposal(
                    params,
                    _changes(base, params),
                    _cost(base, params),
                    decision,
                    tuple(i for i in decision.accepted_contours if i in matches),
                )
        if stop == "evaluation_limit":
            break
    if stop == "completed" and visited < planned:
        stop = "measurement_limit"
    ordered = sorted(
        proposals.values(),
        key=lambda proposal: (proposal.change_cost, _param_key(proposal.params)),
    )
    return replace(
        run,
        status="proposals_available" if ordered else "no_proposal",
        stop_reason=stop,
        measurements_run=measurements,
        evaluations_run=evaluations,
        proposals=tuple(ordered[: options.max_proposals]),
        proposals_truncated=len(ordered) > options.max_proposals,
    )


def compare_tuning_proposals(
    run: TuningRun,
    pairs: Sequence[tuple[str | Path, str | Path]],
    *,
    known_meteors: Sequence[str | Path] = (),
) -> TuningRun:
    """Compare all returned proposals over the same bounded set of full pairs.

    Unprocessed pairs and known-positive frames remain explicitly unchecked.
    Failures propagate instead of becoming negative detections. Measurements
    are shared only for identical preprocessing/Hough settings within a pair.
    """
    if not run.proposals:
        return run
    normalized = [
        (
            str(Path(current).expanduser().resolve()),
            str(Path(previous).expanduser().resolve()),
        )
        for current, previous in pairs
    ]
    targets = [current for current, _ in normalized]
    if len(set(targets)) != len(targets):
        raise MeteorConfigError("Comparison targets must not be duplicated")
    known = {str(Path(path).expanduser().resolve()) for path in known_meteors}
    groups = {}
    for index, proposal in enumerate(run.proposals):
        groups.setdefault(_measurement_key(proposal.params), []).append(index)
    base_key = _measurement_key(run.replay.settings.params)
    per_pair = len(set(groups) | {base_key})
    baseline_hits, proposed_hits, checked = set(), [set() for _ in run.proposals], set()
    measurements = count = 0
    stop = "no_pairs" if not normalized else "completed"
    for current, previous in normalized:
        if count >= run.options.max_comparison_pairs:
            stop = "pair_limit"
            break
        if measurements + per_pair > run.options.max_comparison_measurements:
            stop = "measurement_limit"
            break
        replay = run.replay.reproduce_pair(current, previous)
        if (
            replay.image_shape != run.replay.image_shape
            or replay.image_dtype != run.replay.image_dtype
        ):
            raise MeteorValidationError(
                "Comparison images must match the target image dimensions and dtype"
            )
        if replay.analysis is None or replay.decision is None:
            raise MeteorValidationError("Comparison pair has no measurements")
        if replay.decision.is_candidate:
            baseline_hits.add(current)
        for key, indices in groups.items():
            analysis = (
                replay.analysis
                if key == base_key
                else replay.remeasure(run.proposals[indices[0]].params)
            )
            for index in indices:
                proposal = run.proposals[index]
                decision = analysis.evaluate(
                    CandidateThresholds.from_params(proposal.params.to_dict())
                )
                if decision.is_candidate:
                    proposed_hits[index].add(current)
        checked.add(current)
        count += 1
        measurements += per_pair
    proposals = []
    for proposal, hits in zip(run.proposals, proposed_hits):
        comparison = SequenceComparison(
            total_pairs=len(normalized),
            evaluated_pairs=count,
            stop_reason=stop,
            added_candidates=tuple(sorted(hits - baseline_hits)),
            lost_candidates=tuple(sorted(baseline_hits - hits)),
            known_candidate_frames=tuple(sorted(known & hits)),
            known_missed_frames=tuple(sorted((known & checked) - hits)),
            known_unchecked_frames=tuple(sorted(known - checked)),
        )
        proposals.append(replace(proposal, comparison=comparison))
    proposals.sort(
        key=lambda proposal: (
            len(proposal.comparison.known_missed_frames) if proposal.comparison else 0,
            proposal.change_cost,
            len(proposal.comparison.added_candidates) if proposal.comparison else 0,
            _param_key(proposal.params),
        )
    )
    return replace(
        run, proposals=tuple(proposals), comparison_measurements=measurements
    )
