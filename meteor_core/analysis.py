"""Read-only reproduction of a built-in Hough decision for an explicit pair.

This API does not run hooks, infer a reference frame, estimate parameters, or
write progress/candidate files. A measured frame is not a confirmed meteor.
"""

from dataclasses import dataclass, field, replace
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .detectors.hough_analysis import CandidateDecision, HoughAnalysis
from .detectors.hough_default import HoughDetector
from .exceptions import MeteorConfigError, MeteorLoadError, MeteorValidationError
from .inputs.base import BaseInputLoader
from .pipeline import _normalize_input_context, _resolve_input_loader
from .roi_selector import create_full_roi_mask, create_roi_mask_from_polygon
from .schema import DEFAULT_LOADER_NAME, DetectionContext, DetectionParams
from .utils import ensure_numpy


@dataclass(frozen=True)
class ReproductionSettings:
    """Effective settings, their sources, and explicit changes to the baseline."""

    params: DetectionParams
    baseline_params: DetectionParams
    parameter_sources: dict[str, str]
    roi_polygon: tuple[tuple[int, int], ...] | None
    roi_source: str
    changes: dict[str, dict[str, Any]]
    limitations: tuple[str, ...]
    progress_path: str | None


@dataclass(frozen=True)
class HoughReproduction:
    """In-memory evidence; missing references never produce a fake decision.

    This is a detector-only replay under the supplied settings, not a claim
    that an entire historical pipeline run (including hooks) was reproduced.
    """

    target_image: str
    reference_image: str | None
    settings: ReproductionSettings
    loader_info: dict[str, Any]
    target_metadata: dict[str, Any]
    reference_metadata: dict[str, Any] | None
    image_shape: tuple[int, int]
    image_dtype: str
    analysis: HoughAnalysis | None
    decision: CandidateDecision | None
    _context: DetectionContext | None = field(default=None, repr=False, compare=False)
    _loader: BaseInputLoader | None = field(default=None, repr=False, compare=False)

    @property
    def status(self) -> str:
        return "measured" if self.analysis is not None else "reference_missing"

    def remeasure(self, params: DetectionParams) -> HoughAnalysis:
        """Measure fresh evidence from the same loaded images and ROI snapshot."""
        if self._context is None:
            raise MeteorValidationError("Cannot remeasure without a reference image")
        detector = HoughDetector()
        context = replace(
            self._context,
            runtime_params=detector.build_runtime_params(
                _validated_params(params.to_dict()).to_dict()
            ),
        )
        return detector.analyze(context)

    def reproduce_pair(
        self, target: str | Path, reference: str | Path
    ) -> "HoughReproduction":
        """Replay another pair with the original loader instance and settings."""
        if self._loader is None:
            raise MeteorConfigError("The original input loader is unavailable")
        return reproduce_hough_pair(
            target, reference, settings=self.settings, input_loader=self._loader
        )


def _validated_params(values: Mapping[str, Any]) -> DetectionParams:
    defaults = DetectionParams().to_dict()
    unknown = set(values) - set(defaults)
    if unknown:
        raise MeteorConfigError(
            "Unsupported built-in Hough parameters",
            context={"unknown_parameters": sorted(unknown)},
        )
    for key, value in values.items():
        minimum = 1 if key == "hough_threshold" else 0
        integer = key not in {"min_aspect_ratio", "min_line_score"}
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < minimum
            or (integer and not isinstance(value, int))
        ):
            raise MeteorConfigError(
                "Invalid built-in Hough parameter",
                config_key=key,
                context={"value": value, "minimum": minimum, "integer": integer},
            )
    return DetectionParams(**{**defaults, **values})


def _validated_roi(value: Any) -> tuple[tuple[int, int], ...] | None:
    if value is None:
        return None
    valid = isinstance(value, (list, tuple)) and len(value) >= 3
    if valid:
        valid = all(
            isinstance(point, (list, tuple))
            and len(point) == 2
            and all(
                isinstance(coord, int)
                and not isinstance(coord, bool)
                and -(2**31) <= coord < 2**31
                for coord in point
            )
            for point in value
        )
    if not valid:
        raise MeteorConfigError(
            "ROI must be null or a polygon with at least three integer points",
            config_key="roi",
        )
    return tuple((point[0], point[1]) for point in value)


def load_reproduction_settings(
    *,
    params: DetectionParams | None = None,
    progress_path: str | Path | None = None,
    parameter_overrides: Mapping[str, Any] | None = None,
    roi_polygon: list[list[int]] | None = None,
    use_full_image: bool = False,
) -> ReproductionSettings:
    """Resolve a baseline without updating the input progress file.

    Recorded ``processing_params`` take precedence over recorded ``params``;
    missing values fall back to supplied params, then defaults. Explicit
    ``parameter_overrides`` are applied last and recorded separately. Supplying
    params does not implicitly override values recorded in progress.

    A polygon or ``use_full_image=True`` explicitly overrides the recorded ROI.
    Missing legacy fields are accepted but identified in ``limitations``.
    """
    if roi_polygon is not None and use_full_image:
        raise MeteorConfigError("Specify either a polygon or full-image ROI")

    values = _validated_params(params.to_dict() if params is not None else {}).to_dict()
    source = "supplied" if params is not None else "default"
    sources = dict.fromkeys(values, source)
    limitations = []
    recorded_roi = None
    roi_source = "default_full_image"
    resolved_progress = None

    if progress_path is not None:
        path = Path(progress_path).expanduser().resolve()
        resolved_progress = str(path)
        try:
            progress = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MeteorConfigError(
                "Cannot read analysis progress file",
                filepath=str(path),
                original_error=exc,
            ) from exc
        if not isinstance(progress, dict) or not any(
            key in progress
            for key in ("processed_files", "detected_files", "params_hash")
        ):
            raise MeteorConfigError(
                "Expected a progress file, not a pipeline configuration",
                filepath=str(path),
            )
        recorded_keys = set()
        for field in ("params", "processing_params"):
            recorded = progress.get(field, {})
            if not isinstance(recorded, dict):
                raise MeteorConfigError(
                    "Recorded parameters must be a mapping",
                    filepath=str(path),
                    config_key=field,
                )
            _validated_params(recorded)
            values.update(recorded)
            recorded_keys.update(recorded)
            sources.update(dict.fromkeys(recorded, f"progress.{field}"))
        if recorded_keys != set(values):
            limitations.append("incomplete_recorded_parameters")
        if "roi" in progress:
            recorded_roi = _validated_roi(progress["roi"])
            roi_source = "progress.roi"
        else:
            limitations.append("missing_recorded_roi")
        limitations.append("original_loader_and_hooks_not_reproduced_from_progress")
    else:
        limitations.append("no_recorded_baseline")

    baseline = _validated_params(values)
    overrides = dict(parameter_overrides) if parameter_overrides is not None else {}
    _validated_params(overrides)
    changes = {
        key: {"before": values[key], "after": value}
        for key, value in overrides.items()
        if values[key] != value
    }
    values.update(overrides)
    sources.update(dict.fromkeys(overrides, "override"))

    roi = recorded_roi
    if roi_polygon is not None or use_full_image:
        roi = _validated_roi(roi_polygon)
        if roi != recorded_roi:
            changes["roi"] = {"before": recorded_roi, "after": roi}
        roi_source = "override_full_image" if use_full_image else "override_polygon"
    elif roi_source == "default_full_image":
        limitations.append("assumed_full_image_roi")

    return ReproductionSettings(
        params=_validated_params(values),
        baseline_params=baseline,
        parameter_sources=sources,
        roi_polygon=roi,
        roi_source=roi_source,
        changes=changes,
        limitations=tuple(limitations),
        progress_path=resolved_progress,
    )


def _image_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise MeteorLoadError("Analysis image file not found", filepath=str(path))
    return path


def reproduce_hough_pair(
    target_image: str | Path,
    reference_image: str | Path | None = None,
    *,
    settings: ReproductionSettings,
    input_loader: BaseInputLoader | None = None,
    loader_name: str | None = None,
    loader_config: dict[str, Any] | None = None,
) -> HoughReproduction:
    """Load an explicit pair and reuse normal Hough measurements and decisions.

    The loader and its normalization follow normal pipeline conventions. Loader
    failures propagate; unsupported plugins are never replaced silently. Hooks,
    auto-params, reference inference, and file output are deliberately absent.
    """
    target = _image_path(target_image)
    reference = _image_path(reference_image) if reference_image is not None else None
    if reference is not None and target.samefile(reference):
        raise MeteorValidationError(
            "Target and reference images must be different files",
            parameter_name="reference_image",
            provided_value=str(reference),
        )
    if input_loader is not None and (
        loader_name is not None or loader_config is not None
    ):
        raise MeteorConfigError(
            "Specify a loader instance or loader name/config, not both"
        )
    loader = _resolve_input_loader(
        input_loader=input_loader,
        loader_name=loader_name or DEFAULT_LOADER_NAME,
        loader_config=loader_config,
    )
    current_context = _normalize_input_context(loader.load(str(target)), str(target))
    current = ensure_numpy(current_context.image_data)
    if current.ndim != 2:
        raise MeteorValidationError(
            "Analysis requires a two-dimensional grayscale image",
            parameter_name="target_image",
            provided_value=current.shape,
        )
    shape = (current.shape[0], current.shape[1])
    polygon = settings.roi_polygon
    mask = (
        create_full_roi_mask(shape)
        if polygon is None
        else create_roi_mask_from_polygon([list(point) for point in polygon], shape)
    )
    if not np.any(mask):
        raise MeteorValidationError(
            "ROI does not overlap the image",
            parameter_name="roi",
            provided_value=polygon,
        )

    previous_context = None
    analysis = None
    decision = None
    context = None
    if reference is not None:
        previous_context = _normalize_input_context(
            loader.load(str(reference)), str(reference)
        )
        previous = ensure_numpy(previous_context.image_data)
        if previous.shape != current.shape or previous.dtype != current.dtype:
            raise MeteorValidationError(
                "Target and reference images must have matching dimensions and dtype",
                parameter_name="reference_image",
                provided_value={"shape": previous.shape, "dtype": str(previous.dtype)},
            )
        detector = HoughDetector()
        context = DetectionContext(
            current_image=current,
            previous_image=previous,
            roi_mask=mask,
            runtime_params=detector.build_runtime_params(settings.params.to_dict()),
            metadata={
                "current": current_context.metadata,
                "previous": previous_context.metadata,
            },
        )
        analysis = detector.analyze(context)
        decision = analysis.evaluate()

    return HoughReproduction(
        target_image=str(target),
        reference_image=str(reference) if reference is not None else None,
        settings=settings,
        loader_info=loader.get_info(),
        target_metadata=current_context.metadata,
        reference_metadata=previous_context.metadata
        if previous_context is not None
        else None,
        image_shape=shape,
        image_dtype=str(current.dtype),
        analysis=analysis,
        decision=decision,
        _context=context,
        _loader=loader,
    )
