"""Explicit, non-overwriting analysis bundles and replayable session settings."""

from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from .analysis import HoughReproduction, source_stamp
from .detectors.hough_analysis import (
    CandidateDecision,
    CandidateThresholds,
    HoughAnalysis,
)
from .detectors.hough_default import HoughDetector
from .exceptions import MeteorConfigError, MeteorValidationError, MeteorWriteError
from .inputs.registry import LoaderRegistry
from .schema import PipelineConfig, VERSION
from .tuning import MEASUREMENT_PARAMETERS, TuningRun


ANALYSIS_REPORT_VERSION = 1
ANALYSIS_REPORT_FORMAT = "detect_meteors.analysis"


def _same_configuration(before: Any, after: Any) -> bool:
    if type(before) is not type(after):
        return False
    if is_dataclass(before) and not isinstance(before, type):
        return all(
            _same_configuration(getattr(before, item.name), getattr(after, item.name))
            for item in fields(before)
        )
    if isinstance(before, Mapping):
        return before.keys() == after.keys() and all(
            _same_configuration(value, after[key]) for key, value in before.items()
        )
    if isinstance(before, (list, tuple)):
        return len(before) == len(after) and all(
            _same_configuration(left, right) for left, right in zip(before, after)
        )
    return bool(before == after)


def _json_value(value: Any, notes: list[str], path: str, *, strict=False) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), notes, path, strict=strict)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(
                getattr(value, item.name), notes, f"{path}.{item.name}", strict=strict
            )
            for item in fields(value)
        }
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"), notes, path, strict=strict)
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {
            key: _json_value(item, notes, f"{path}.{key}", strict=strict)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, notes, f"{path}[{index}]", strict=strict)
            for index, item in enumerate(value)
        ]
    if strict:
        raise MeteorConfigError(
            "Configuration contains a value that cannot be serialized faithfully",
            config_key=path,
        )
    notes.append(f"{path}: represented as text ({type(value).__name__})")
    return str(value)


def _fingerprint(path: str, expected: tuple[int, ...], *, hash_content: bool) -> dict:
    try:
        if source_stamp(path) != expected:
            raise MeteorValidationError(
                "Analysis source changed after it was read", filepath=path
            )
        digest = None
        if hash_content:
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        if source_stamp(path) != expected:
            raise MeteorValidationError(
                "Analysis source changed during hashing", filepath=path
            )
    except OSError as exc:
        raise MeteorValidationError(
            "Analysis source is no longer available", filepath=path, original_error=exc
        ) from exc
    return {
        "path": path,
        "device": expected[0],
        "inode": expected[1],
        "size": expected[2],
        "mtime_ns": expected[3],
        "ctime_ns": expected[4],
        "sha256": digest.hexdigest() if digest is not None else None,
    }


def _decision(decision: CandidateDecision | None) -> dict | None:
    if decision is None:
        return None
    return {
        "is_candidate": decision.is_candidate,
        "max_aspect_ratio": decision.max_aspect_ratio,
        "accepted_contours": list(decision.accepted_contours),
        "rejection_reasons": list(decision.rejection_reasons),
    }


def _session_config(
    replay: HoughReproduction,
    params,
    output: Path,
    base: PipelineConfig | None,
    target_folder: str | Path | None,
) -> PipelineConfig:
    if replay.analysis is None:
        raise MeteorConfigError(
            "A reusable session configuration requires a measured image pair"
        )
    base = base if base is not None else PipelineConfig.with_defaults()
    if base.hooks:
        raise MeteorConfigError(
            "A detector-only session configuration cannot preserve active hooks"
        )
    if base.detector_name not in (None, "hough") or base.detector_config:
        raise MeteorConfigError(
            "Session configuration export requires the built-in Hough detector configuration"
        )
    if base.output_handler_name not in (None, "file"):
        raise MeteorConfigError(
            "Session configuration export supports the built-in file output handler only"
        )
    if base.output_handler_config and set(base.output_handler_config) - {
        "output_folder",
        "debug_folder",
        "output_overwrite",
    }:
        raise MeteorConfigError("Unsupported file output handler configuration")
    if replay._loader is None or not replay.loader_config_available:
        raise MeteorConfigError(
            "The original loader configuration is unavailable for export"
        )
    name = replay.loader_info["plugin_name"]
    try:
        loader_type = LoaderRegistry.get(name)
        if loader_type is not type(replay._loader):
            raise ValueError(
                "The registered loader differs from the loader used for analysis"
            )
        loader_config = _json_value(
            replay.loader_config, [], "loader_config", strict=True
        )
        restored = LoaderRegistry._coerce_config(loader_type, loader_config)
        if not _same_configuration(replay.loader_config, restored):
            raise ValueError("The loader configuration does not round-trip")
    except (KeyError, TypeError, ValueError) as exc:
        raise MeteorConfigError(
            "The original loader settings cannot be recreated from configuration",
            original_error=exc,
        ) from exc
    target = (
        Path(
            target_folder
            if target_folder is not None
            else Path(replay.target_image).parent
        )
        .expanduser()
        .resolve()
    )
    if not target.is_dir():
        raise MeteorConfigError(
            "Session target folder does not exist", filepath=str(target)
        )
    run = output / "run"
    return replace(
        base,
        target_folder=str(target),
        params=params,
        output_folder=str(run / "candidates"),
        debug_folder=str(run / "debug"),
        progress_file=str(run / "progress.json"),
        output_overwrite=False,
        output_handler_name="file",
        output_handler_config={
            "output_folder": str(run / "candidates"),
            "debug_folder": str(run / "debug"),
            "output_overwrite": False,
        },
        input_loader_name=name,
        input_loader_config=loader_config,
        detector_name="hough",
        detector_config={},
        hooks=[],
        roi_polygon=[list(point) for point in replay.settings.roi_polygon]
        if replay.settings.roi_polygon is not None
        else None,
        enable_roi_selection=False,
        num_workers=1,
        enable_parallel=False,
    )


def _validate_replay_snapshot(
    replay: HoughReproduction, tuning: TuningRun | None
) -> None:
    if tuning is not None and tuning.replay is not replay:
        raise MeteorConfigError(
            "Tuning results belong to a different image-pair analysis"
        )
    if replay._context is not None:
        global_params, detector_params = HoughDetector().split_runtime_params(
            replay._context.runtime_params
        )
        if {**global_params, **detector_params} != replay.settings.params.to_dict():
            raise MeteorValidationError(
                "Effective settings changed after measurement; analyze the pair again"
            )
    if tuning is not None:
        baseline = replay.settings.params.to_dict()
        for proposal in tuning.proposals:
            changes = {
                key: {"before": baseline[key], "after": value}
                for key, value in proposal.params.to_dict().items()
                if baseline[key] != value
            }
            if changes != proposal.changes:
                raise MeteorValidationError(
                    "Proposal parameters changed after evaluation; tune the pair again"
                )


def save_analysis_bundle(
    replay: HoughReproduction,
    output_directory: str | Path,
    *,
    tuning: TuningRun | None = None,
    selection: str | int | None = None,
    pipeline_config: PipelineConfig | None = None,
    session_target: str | Path | None = None,
    context: Mapping[str, Any] | None = None,
    source_files: Mapping[str, tuple[int, ...]] | None = None,
) -> Path:
    """Create a new report directory; an explicit selection also saves session.yaml.

    Selection is ``baseline`` or a one-based returned proposal index. Existing
    paths, including symlinks, are never replaced. The manifest is written last;
    failures remove only files created by this call and empty new directories.
    """
    requested = Path(output_directory).expanduser().absolute()
    if os.path.lexists(requested):
        raise MeteorWriteError(
            "Analysis output must be a new directory", destination_path=str(requested)
        )
    output = requested.resolve()
    _validate_replay_snapshot(replay, tuning)
    selected = None
    params = replay.settings.params
    if selection is not None:
        if selection == "baseline":
            selected = {"kind": "baseline"}
        else:
            if not isinstance(selection, (str, int)):
                raise MeteorConfigError("Select baseline or a one-based proposal index")
            try:
                index = int(selection)
            except (ValueError, TypeError) as exc:
                raise MeteorConfigError(
                    "Select baseline or a one-based proposal index"
                ) from exc
            if (
                isinstance(selection, bool)
                or tuning is None
                or not 1 <= index <= len(tuning.proposals)
            ):
                raise MeteorConfigError("Selected proposal is not available")
            selected = {"kind": "proposal", "index": index}
            params = tuning.proposals[index - 1].params
    session = (
        _session_config(replay, params, output, pipeline_config, session_target)
        if selected is not None
        else None
    )
    stamps = dict(replay.source_stamps)
    if (
        replay.settings.progress_path is not None
        and replay.settings.progress_stamp is not None
    ):
        stamps[replay.settings.progress_path] = replay.settings.progress_stamp
    for path, stamp in dict(source_files or {}).items():
        path = str(Path(path).expanduser().resolve())
        if path in stamps and stamps[path] != stamp:
            raise MeteorValidationError("Conflicting source snapshots", filepath=path)
        stamps[path] = stamp
    comparison_stamps = tuning.comparison_inputs if tuning is not None else {}
    for path, stamp in comparison_stamps.items():
        if path in stamps and stamps[path] != stamp:
            raise MeteorValidationError(
                "Source changed between target and sequence analysis", filepath=path
            )
    inputs = [
        _fingerprint(path, stamp, hash_content=True)
        for path, stamp in sorted(stamps.items())
    ]
    comparison_inputs = [
        _fingerprint(path, stamp, hash_content=False)
        for path, stamp in sorted(comparison_stamps.items())
    ]
    notes = []
    effective_config = pipeline_config.to_dict() if pipeline_config else None
    if effective_config is not None:
        effective_config.update(
            params=replay.settings.params.to_dict(),
            roi_polygon=replay.settings.roi_polygon,
            enable_roi_selection=False,
        )
    report: dict[str, Any] = {
        "format": ANALYSIS_REPORT_FORMAT,
        "schema_version": ANALYSIS_REPORT_VERSION,
        "application_version": VERSION,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "detector": HoughDetector().get_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(Path.cwd()),
        "status": replay.status,
        "target_image": replay.target_image,
        "reference_image": replay.reference_image,
        "image_shape": list(replay.image_shape),
        "image_dtype": replay.image_dtype,
        "inputs": inputs,
        "comparison_inputs": comparison_inputs,
        "context": _json_value(dict(context or {}), notes, "context"),
        "loader": {
            "info": replay.loader_info,
            "config_available": replay.loader_config_available,
            "config": _json_value(replay.loader_config, notes, "loader.config"),
        },
        "metadata": {
            "target": _json_value(replay.target_metadata, notes, "metadata.target"),
            "reference": _json_value(
                replay.reference_metadata, notes, "metadata.reference"
            ),
        },
        "settings": {
            "progress_path": replay.settings.progress_path,
            "effective_params": replay.settings.params.to_dict(),
            "baseline_params": replay.settings.baseline_params.to_dict(),
            "parameter_sources": replay.settings.parameter_sources,
            "roi_polygon": replay.settings.roi_polygon,
            "roi_source": replay.settings.roi_source,
            "changes": replay.settings.changes,
            "limitations": replay.settings.limitations,
        },
        "interpretation": {
            "detector_only": True,
            "frame_candidates_are_not_meteor_identifications": True,
            "candidate_additions_are_not_false_positive_rates": True,
        },
        "pipeline_config": _json_value(
            effective_config,
            notes,
            "pipeline_config",
        ),
        "measurements": {},
        "baseline": {
            "decision": _decision(replay.decision),
            "measurement": None,
            "debug_image": None,
        },
        "tuning": None,
        "proposals": [],
        "selected": selected,
        "session_configuration": "session.yaml" if session is not None else None,
        "serialization_notes": notes,
        "assets": {},
    }
    if tuning is not None:
        report["tuning"] = {
            "status": tuning.status,
            "scope": tuning.scope,
            "target_region": tuning.region,
            "stop_reason": tuning.stop_reason,
            "options": asdict(tuning.options),
            "measurement_grid": tuning.measurement_grid,
            "measurements_run": tuning.measurements_run,
            "evaluations_run": tuning.evaluations_run,
            "planned_measurements": tuning.planned_measurements,
            "comparison_measurements": tuning.comparison_measurements,
            "proposals_truncated": tuning.proposals_truncated,
        }
    created_dirs, owned = [], {}

    def write(name: str, data: bytes):
        path = output / name
        with path.open("xb") as stream:
            stat = os.fstat(stream.fileno())
            owned[path] = (stat.st_dev, stat.st_ino)
            if stream.write(data) != len(data):
                raise OSError("Incomplete analysis asset write")
        report["assets"][name] = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def png(name: str, pixels):
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise MeteorWriteError(
                "Failed to encode analysis image", destination_path=str(output / name)
            )
        write(name, encoded.tobytes())

    measurements = {}

    def evidence(params, analysis: HoughAnalysis):
        key = tuple(getattr(params, name) for name in MEASUREMENT_PARAMETERS)
        if key not in measurements:
            identifier = f"measurement-{len(measurements):03d}"
            measurements[key] = (identifier, analysis)
            mask = identifier + "-mask.png"
            png(mask, analysis.mask)
            report["measurements"][identifier] = {
                "params": {
                    name: getattr(params, name) for name in MEASUREMENT_PARAMETERS
                },
                "score": analysis.score,
                "lines": analysis.lines,
                "mask": mask,
                "mask_pixels": int(np.count_nonzero(analysis.mask)),
                "contours": [
                    {
                        "area": contour.area,
                        "rect": contour.rect,
                        "aspect_ratio": contour.aspect_ratio,
                    }
                    for contour in analysis.contours
                ],
            }
        return measurements[key]

    try:
        missing = []
        parent = output
        while not os.path.lexists(parent):
            missing.append(parent)
            parent = parent.parent
        for path in reversed(missing):
            try:
                path.mkdir()
                created_dirs.append(path)
            except FileExistsError:
                if path == output or not path.is_dir():
                    raise
        if output not in created_dirs:
            raise FileExistsError(str(output))
        if replay.analysis is not None and replay.decision is not None:
            identifier, analysis = evidence(replay.settings.params, replay.analysis)
            png("baseline-debug.png", analysis.render_debug(replay.decision))
            report["baseline"] = {
                "decision": _decision(replay.decision),
                "measurement": identifier,
                "debug_image": "baseline-debug.png",
            }
        if tuning is not None:
            for index, proposal in enumerate(tuning.proposals, 1):
                key = tuple(
                    getattr(proposal.params, name) for name in MEASUREMENT_PARAMETERS
                )
                analysis = (
                    measurements[key][1]
                    if key in measurements
                    else replay.remeasure(proposal.params)
                )
                if (
                    analysis.evaluate(
                        CandidateThresholds.from_params(proposal.params.to_dict())
                    )
                    != proposal.decision
                ):
                    raise MeteorValidationError(
                        "Proposal evidence cannot be reproduced"
                    )
                identifier, analysis = evidence(proposal.params, analysis)
                debug = f"proposal-{index:03d}-debug.png"
                png(debug, analysis.render_debug(proposal.decision))
                report["proposals"].append(
                    {
                        "index": index,
                        "params": proposal.params.to_dict(),
                        "changes": proposal.changes,
                        "change_cost": proposal.change_cost,
                        "decision": _decision(proposal.decision),
                        "matched_contours": proposal.matched_contours,
                        "measurement": identifier,
                        "debug_image": debug,
                        "comparison": {
                            **asdict(proposal.comparison),
                            "complete": proposal.comparison.complete,
                        }
                        if proposal.comparison
                        else None,
                    }
                )
        if session is not None:
            write(
                "session.yaml",
                yaml.safe_dump(
                    session.to_dict(), allow_unicode=True, sort_keys=False
                ).encode("utf-8"),
            )
        for path, stamp in {**stamps, **comparison_stamps}.items():
            _fingerprint(path, stamp, hash_content=False)
        # Publish the manifest last. It never lists or hashes itself.
        payload = (
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        with (output / "report.json").open("xb") as stream:
            stat = os.fstat(stream.fileno())
            owned[output / "report.json"] = (stat.st_dev, stat.st_ino)
            if stream.write(payload) != len(payload):
                raise OSError("Incomplete analysis manifest write")
    except BaseException as exc:
        for path, identity in reversed(list(owned.items())):
            try:
                stat = path.stat()
                if (stat.st_dev, stat.st_ino) == identity:
                    path.unlink()
            except OSError:
                pass
        for path in reversed(created_dirs):
            try:
                path.rmdir()
            except OSError:
                pass
        if isinstance(exc, (OSError, cv2.error)):
            raise MeteorWriteError(
                "Failed to save analysis bundle",
                destination_path=str(output),
                original_error=exc,
            ) from exc
        raise
    return output
