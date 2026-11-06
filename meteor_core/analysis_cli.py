"""Reference selection and localized console output for read-only analysis."""

import json
from pathlib import Path

from .analysis import HoughReproduction
from .exceptions import MeteorConfigError, MeteorValidationError
from .i18n import get_message
from .pipeline import collect_files
from .tuning import TuningRun


def select_reference_image(
    target_image: str | Path,
    *,
    reference_image: str | Path | None = None,
    sequence_folder: str | Path | None = None,
) -> tuple[str | None, str]:
    """Use an explicit reference or the normal ordering of an explicit sequence.

    Do not infer a sequence from the target's parent or normal CLI defaults.
    The first frame has no preceding frame. File existence and identity checks
    for explicit pairs remain the reproduction API's responsibility.
    """
    if reference_image is not None and sequence_folder is not None:
        raise MeteorConfigError(
            "Specify a reference image or a sequence folder, not both"
        )
    if reference_image is not None:
        return str(reference_image), "explicit"
    if sequence_folder is None:
        return None, "not_supplied"
    files = collect_files(str(Path(sequence_folder).expanduser().resolve()))
    target = Path(target_image).expanduser().resolve()
    matches = [
        index for index, path in enumerate(files) if Path(path).resolve() == target
    ]
    if len(matches) != 1:
        raise MeteorValidationError(
            "Analysis target must occur exactly once in the selected sequence",
            parameter_name="analysis_sequence",
            provided_value=str(sequence_folder),
            context={"target": str(target), "matches": len(matches)},
        )
    index = matches[0]
    if index == 0:
        return None, "sequence_start"
    return files[index - 1], "filename_order"


def display_reproduction(
    replay: HoughReproduction,
    *,
    reference_source: str,
    locale: str | None = None,
    verbose: bool = False,
) -> None:
    """Print evidence without saving reports or invoking output handlers."""

    def message(key: str, **values) -> str:
        return get_message(f"ui.analysis.{key}", locale=locale, **values)

    print(message("header"))
    print(message("scope"))
    print(message("target", path=replay.target_image))
    if replay.settings.progress_path is not None:
        print(message("baseline", path=replay.settings.progress_path))
    print(
        message(
            "reference",
            path=replay.reference_image or "—",
            source=message(f"reference_source.{reference_source}"),
        )
    )
    print(message("image", width=replay.image_shape[1], height=replay.image_shape[0]))
    print(message("loader", info=json.dumps(replay.loader_info, ensure_ascii=False)))
    print(
        message(
            "roi",
            polygon=json.dumps(replay.settings.roi_polygon),
            source=replay.settings.roi_source,
        )
    )
    print(message("params_header"))
    for name, value in replay.settings.params.to_dict().items():
        print(f"  {name}: {value} [{replay.settings.parameter_sources[name]}]")
    if replay.settings.changes:
        print(message("changes_header"))
        for name, change in replay.settings.changes.items():
            print(f"  {name}: {change['before']} -> {change['after']}")
    print(
        message(
            "metadata",
            metadata=json.dumps(
                replay.target_metadata, ensure_ascii=False, default=str
            ),
        )
    )
    if replay.settings.limitations:
        print(message("limitations_header"))
        for limitation in replay.settings.limitations:
            print("  - " + message(f"limitation.{limitation}"))
    if replay.analysis is None or replay.decision is None:
        print(message("reference_missing"))
        return

    analysis, decision = replay.analysis, replay.decision
    print(
        message(
            "decision",
            candidate=decision.is_candidate,
            score=analysis.score,
            ratio=decision.max_aspect_ratio,
        )
    )
    print(
        message(
            "measurements",
            pixels=int((analysis.mask != 0).sum()),
            lines=len(analysis.lines),
            contours=len(analysis.contours),
        )
    )
    if decision.rejection_reasons:
        reasons = "; ".join(
            f"{reason}: {message('reason.' + reason)}"
            for reason in decision.rejection_reasons
        )
        print(message("rejections", reasons=reasons))
    if verbose:
        print(message("contours_header"))
        for index, contour in enumerate(analysis.contours):
            print(
                message(
                    "contour",
                    index=index,
                    area=contour.area,
                    ratio=contour.aspect_ratio,
                    accepted=index in decision.accepted_contours,
                )
            )
        print(message("lines_header"))
        for line in analysis.lines:
            print(f"  {line}")


def display_tuning(run: TuningRun, *, locale: str | None = None) -> None:
    """Show exploratory settings and comparison coverage without applying them."""

    def message(key: str, **values) -> str:
        return get_message(f"ui.tuning.{key}", locale=locale, **values)

    print(message("header"))
    print(message("scope", scope=run.scope, region=run.region))
    print(message("notice"))
    print(message("status", status=run.status, stop=run.stop_reason))
    print(
        message(
            "work",
            measurements=run.measurements_run,
            planned=run.planned_measurements,
            evaluations=run.evaluations_run,
            comparison=run.comparison_measurements,
        )
    )
    if run.proposals_truncated:
        print(message("truncated"))
    if run.proposals and run.proposals[0].comparison is None:
        print(message("no_comparison"))
    elif run.proposals:
        comparison = run.proposals[0].comparison
        if comparison is not None and not any(
            (
                comparison.known_candidate_frames,
                comparison.known_missed_frames,
                comparison.known_unchecked_frames,
            )
        ):
            print(message("no_known_meteors"))
    for index, proposal in enumerate(run.proposals, 1):
        print(
            message(
                "proposal",
                index=index,
                cost=proposal.change_cost,
                contours=proposal.matched_contours,
            )
        )
        for name, change in proposal.changes.items():
            print(f"  {name}: {change['before']} -> {change['after']}")
        comparison = proposal.comparison
        if comparison is not None:
            print(
                message(
                    "comparison",
                    evaluated=comparison.evaluated_pairs,
                    total=comparison.total_pairs,
                    complete=comparison.complete,
                    stop=comparison.stop_reason,
                )
            )
            for field in (
                "added_candidates",
                "lost_candidates",
                "known_candidate_frames",
                "known_missed_frames",
                "known_unchecked_frames",
            ):
                print(
                    message(
                        field,
                        files=json.dumps(
                            getattr(comparison, field), ensure_ascii=False
                        ),
                    )
                )
