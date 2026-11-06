"""Evidence-derived proposals, exact boundaries, bounded work, and comparison."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from meteor_core.analysis import load_reproduction_settings, reproduce_hough_pair
from meteor_core.exceptions import (
    MeteorConfigError,
    MeteorLoadError,
    MeteorValidationError,
)
from meteor_core.inputs.base import BaseInputLoader
from meteor_core.schema import DetectionParams, InputContext
from meteor_core.tuning import (
    TuningOptions,
    compare_tuning_proposals,
    tune_reproduction,
)


class TuningLoader(BaseInputLoader):
    plugin_name = "tuning_test"

    def __init__(self):
        self.images = {}
        self.loaded = []

    def load(self, filepath):
        self.loaded.append(filepath)
        return InputContext(self.images[Path(filepath).name].copy(), filepath)


class TestTuning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name).resolve()
        self.loader = TuningLoader()
        self.blank = self.image("blank.ORF", np.zeros((96, 96), np.uint16))
        current = np.zeros((96, 96), np.uint16)
        current[20:45, 10:70] = 1000
        self.target = self.image("target.ORF", current)
        self.params = DetectionParams(
            diff_threshold=10, min_area=3, min_aspect_ratio=3.0, min_line_score=30
        )

    def image(self, name, image):
        path = self.directory / name
        path.write_bytes(name.encode())
        self.loader.images[name] = image
        return path

    def replay(self, params=None, reference=True, roi=None):
        return reproduce_hough_pair(
            self.target,
            self.blank if reference else None,
            settings=load_reproduction_settings(
                params=params or self.params, roi_polygon=roi
            ),
            input_loader=self.loader,
        )

    def glow_replay(self):
        image = np.zeros((96, 96), np.uint16)
        image[10:80, 10:80] = 20
        image[10:85, 40:45] = 1000
        self.loader.images[self.target.name] = image
        weak = np.zeros_like(image)
        weak[40:45, 10:80] = 20
        self.weak = self.image("weak.ORF", weak)
        return self.replay()

    def test_gate_proposal_is_measured_not_a_hard_coded_ratio(self):
        replay = self.replay()
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        run = tune_reproduction(replay)
        self.assertEqual(run.status, "proposals_available")
        self.assertEqual(run.scope, "frame")
        proposal = run.proposals[0]
        ratio = replay.analysis.contours[0].aspect_ratio
        self.assertEqual(proposal.params.min_aspect_ratio, np.nextafter(ratio, -np.inf))
        self.assertNotEqual(proposal.params.min_aspect_ratio, 2.0)
        self.assertEqual(set(proposal.changes), {"min_aspect_ratio"})
        self.assertTrue(proposal.decision.is_candidate)
        self.assertEqual(run.measurements_run, 1)
        self.assertEqual(len(self.loader.loaded), 2)
        self.assertEqual(replay.settings.params.min_aspect_ratio, 3.0)
        self.assertEqual(
            before, {p.name: p.read_bytes() for p in self.directory.iterdir()}
        )

    def test_exact_area_ratio_and_score_boundaries_survive_remeasurement(self):
        replay = self.replay(replace(self.params, min_area=10000, min_line_score=1e6))
        run = tune_reproduction(replay, options=TuningOptions(max_gate_relaxation=1))
        proposal = run.proposals[0]
        self.assertEqual(
            proposal.params.min_area, int(replay.analysis.contours[0].area) - 1
        )
        self.assertEqual(proposal.params.min_line_score, replay.analysis.score)
        analysis = replay.remeasure(proposal.params)
        self.assertTrue(analysis.evaluate().is_candidate)
        self.assertFalse(replay.decision.is_candidate)

    def test_relaxation_bounds_can_prevent_any_proposal(self):
        run = tune_reproduction(
            self.replay(), options=TuningOptions(max_gate_relaxation=0)
        )
        self.assertEqual(run.status, "no_proposal")
        self.assertEqual(run.stop_reason, "completed")
        self.assertEqual(run.proposals, ())

    def test_preprocessing_trials_remeasure_the_same_loaded_snapshot(self):
        self.loader.images[self.target.name][20:45, 10:70] = 5
        replay = self.replay()
        self.loader.images[self.target.name][:] = 0
        run = tune_reproduction(replay, measurement_grid={"diff_threshold": [4]})
        self.assertEqual(run.measurements_run, 2)
        self.assertEqual(run.status, "proposals_available")
        self.assertEqual(run.proposals[0].params.diff_threshold, 4)
        self.assertEqual(len(self.loader.loaded), 2)
        self.assertEqual(replay.analysis.score, 0)

    def test_no_hough_evidence_is_not_rescued_by_a_zero_score_threshold(self):
        self.loader.images[self.target.name][:] = 0
        run = tune_reproduction(
            self.replay(), options=TuningOptions(max_gate_relaxation=1)
        )
        self.assertEqual(run.status, "no_proposal")
        self.assertFalse(run.proposals)

    def test_region_prevents_another_object_from_satisfying_the_target(self):
        self.loader.images[self.target.name][70:75, 10:80] = 1000
        replay = self.replay()
        self.assertTrue(replay.decision.is_candidate)
        self.assertEqual(tune_reproduction(replay).status, "already_candidate")
        run = tune_reproduction(replay, target_region=(10, 20, 70, 45))
        self.assertEqual(run.scope, "annotated_region")
        self.assertEqual(run.status, "proposals_available")
        self.assertTrue(run.proposals[0].matched_contours)
        self.assertFalse(
            set(run.proposals[0].matched_contours)
            & set(replay.decision.accepted_contours)
        )

    def test_outside_roi_and_invalid_region_do_not_relax_the_roi(self):
        replay = self.replay(roi=[[0, 0], [8, 0], [8, 95], [0, 95]])
        run = tune_reproduction(replay, target_region=(10, 20, 70, 45))
        self.assertEqual(run.status, "target_outside_roi")
        self.assertFalse(run.proposals)
        for region in ((-1, 0, 2, 2), (0, 0, 97, 96), (1, 1, 1, 5), (True, 0, 2, 2)):
            with self.subTest(region=region), self.assertRaises(MeteorConfigError):
                tune_reproduction(replay, target_region=region)

    def test_missing_reference_has_no_search_or_fabricated_proposals(self):
        replay = self.replay(reference=False)
        run = tune_reproduction(replay)
        self.assertEqual(run.status, "reference_missing")
        self.assertEqual(run.measurements_run, 0)
        self.assertEqual(run.evaluations_run, 0)
        with self.assertRaises(MeteorValidationError):
            replay.remeasure(self.params)

    def test_measurement_and_evaluation_budgets_are_inclusive(self):
        self.loader.images[self.target.name][:] = 5
        replay = self.replay()
        run = tune_reproduction(
            replay,
            measurement_grid={"diff_threshold": [4]},
            options=TuningOptions(max_measurements=1),
        )
        self.assertEqual(run.stop_reason, "measurement_limit")
        self.assertEqual(run.measurements_run, 1)
        run = tune_reproduction(self.replay(), options=TuningOptions(max_evaluations=1))
        self.assertEqual(run.stop_reason, "evaluation_limit")
        self.assertEqual(run.evaluations_run, 1)
        self.assertFalse(run.proposals)

    def test_proposal_limit_is_explicit_and_smallest_change_is_first(self):
        self.loader.images[self.target.name][60:80, 10:48] = 1000
        run = tune_reproduction(self.replay(), options=TuningOptions(max_proposals=1))
        self.assertTrue(run.proposals_truncated)
        self.assertEqual(len(run.proposals), 1)
        self.assertGreater(run.proposals[0].params.min_aspect_ratio, 2.0)

    def test_grid_values_and_limits_are_validated(self):
        replay = self.replay()
        for grid in (
            {"min_area": [1]},
            {"diff_threshold": []},
            {"diff_threshold": [True]},
            {"hough_threshold": [0]},
            {"diff_threshold": [2**40]},
            {"diff_threshold": "5"},
        ):
            with self.subTest(grid=grid), self.assertRaises(MeteorConfigError):
                tune_reproduction(replay, measurement_grid=grid)
        for options in (
            {"max_measurements": 0},
            {"max_evaluations": True},
            {"max_gate_relaxation": float("nan")},
            {"max_gate_relaxation": 1.1},
        ):
            with self.subTest(options=options), self.assertRaises(MeteorConfigError):
                TuningOptions(**options)

    def test_comparison_reports_added_lost_and_known_missed_frames(self):
        replay = self.glow_replay()
        run = tune_reproduction(replay, measurement_grid={"diff_threshold": [30]})
        self.assertEqual(len(run.proposals), 1)
        self.assertEqual(set(run.proposals[0].changes), {"diff_threshold"})
        compared = compare_tuning_proposals(
            run,
            [(self.target, self.blank), (self.weak, self.blank)],
            known_meteors=[self.weak],
        )
        comparison = compared.proposals[0].comparison
        self.assertTrue(comparison.complete)
        self.assertEqual(comparison.added_candidates, (str(self.target),))
        self.assertEqual(comparison.lost_candidates, (str(self.weak),))
        self.assertEqual(comparison.known_missed_frames, (str(self.weak),))
        self.assertEqual(compared.comparison_measurements, 4)
        self.assertEqual(len(self.loader.loaded), 6)

    def test_comparison_limits_leave_known_positives_unchecked(self):
        replay = self.glow_replay()
        pairs = [(self.target, self.blank), (self.weak, self.blank)]
        run = tune_reproduction(
            replay,
            measurement_grid={"diff_threshold": [30]},
            options=TuningOptions(max_comparison_pairs=1),
        )
        compared = compare_tuning_proposals(run, pairs, known_meteors=[self.weak])
        comparison = compared.proposals[0].comparison
        self.assertFalse(comparison.complete)
        self.assertEqual(comparison.stop_reason, "pair_limit")
        self.assertEqual(comparison.known_unchecked_frames, (str(self.weak),))
        self.assertEqual(comparison.known_missed_frames, ())
        run = tune_reproduction(
            replay,
            measurement_grid={"diff_threshold": [30]},
            options=TuningOptions(max_comparison_measurements=1),
        )
        compared = compare_tuning_proposals(run, pairs, known_meteors=[self.weak])
        comparison = compared.proposals[0].comparison
        self.assertEqual(comparison.evaluated_pairs, 0)
        self.assertEqual(comparison.stop_reason, "measurement_limit")
        self.assertEqual(compared.comparison_measurements, 0)

    def test_comparison_does_not_turn_load_errors_into_negative_detections(self):
        run = tune_reproduction(self.replay())
        with self.assertRaises(MeteorLoadError):
            compare_tuning_proposals(
                run, [(self.directory / "missing.ORF", self.blank)]
            )
        with self.assertRaises(MeteorConfigError):
            compare_tuning_proposals(
                run, [(self.target, self.blank), (self.target, self.blank)]
            )

    def test_comparison_keeps_known_candidate_frames_and_preserves_all_files(self):
        run = tune_reproduction(self.replay())
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        compared = compare_tuning_proposals(
            run, [(self.target, self.blank)], known_meteors=[self.target, self.blank]
        )
        comparison = compared.proposals[0].comparison
        self.assertEqual(comparison.known_candidate_frames, (str(self.target),))
        self.assertEqual(comparison.known_unchecked_frames, (str(self.blank),))
        self.assertEqual(
            before, {p.name: p.read_bytes() for p in self.directory.iterdir()}
        )

    def test_comparison_cannot_silently_change_resolution_or_numeric_scale(self):
        run = tune_reproduction(self.replay())
        for image in (np.zeros((48, 48), np.uint16), np.zeros((96, 96), np.float32)):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                current = self.image("different.ORF", image)
                previous = self.image("different_blank.ORF", image)
                with self.assertRaises(MeteorValidationError):
                    compare_tuning_proposals(run, [(current, previous)])


if __name__ == "__main__":
    unittest.main()
