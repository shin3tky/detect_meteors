"""Verify the recovery sweep against the detector and its zero-FP selection."""

from dataclasses import replace
import unittest

import cv2
import numpy as np

from benchmark_hough_recovery import (
    GATES,
    HOUGH,
    HOUGH_PARAMS,
    SHAPE,
    evaluate,
    measure,
    select_thresholds,
    totals,
)
from meteor_core.detectors.hough_analysis import CandidateThresholds
from meteor_core.detectors.hough_default import HoughDetector, HoughDetectorConfig
from meteor_core.schema import DetectionContext, DetectionParams
from summarize import trail_coverage


class HoughRecoveryTests(unittest.TestCase):
    def test_grid_matches_direct_hough_measurements(self):
        previous = np.zeros((96, 96), np.uint16)
        current = previous.copy()
        cv2.line(current, (15, 75), (73, 20), 12, 5)
        context = DetectionContext(
            current, previous, np.full(current.shape, 255, np.uint8), {}, {}
        )
        axis = {"axis": [[15, 75], [73, 20]], "half_width_px": 5}
        settings = DetectionParams(
            hough_threshold=10, hough_min_line_length=15, hough_max_line_gap=5
        ).to_dict()
        detector = HoughDetector(HoughDetectorConfig("positive"))
        candidates, coverage = measure(detector, context, settings, axis)
        self.assertEqual(candidates.shape, SHAPE)
        for profile, hough in enumerate(HOUGH):
            runtime = {
                **settings,
                **dict(
                    zip((f"hough_{name}" for name in HOUGH_PARAMS), hough, strict=True)
                ),
            }
            for cutoff in (9, 13):
                evidence = detector.analyze(
                    replace(
                        context, runtime_params={**runtime, "diff_threshold": cutoff}
                    )
                )
                cutoff_index = GATES["diff_threshold"].index(cutoff)
                self.assertEqual(
                    coverage[profile, cutoff_index],
                    trail_coverage(
                        axis, [(line[:2], line[2:]) for line in evidence.lines]
                    ),
                )
                for index in np.ndindex(*SHAPE[2:]):
                    thresholds = CandidateThresholds(
                        *(
                            values[i]
                            for values, i in zip(
                                tuple(GATES.values())[1:], index, strict=True
                            )
                        )
                    )
                    self.assertEqual(
                        candidates[profile, cutoff_index][index],
                        evidence.evaluate(thresholds).is_candidate,
                    )

    def test_recovery_selection_requires_zero_false_positives_and_preservation(self):
        baseline = (0, 1, 0, 0, 0)
        better = (0, 2, 0, 0, 0)
        has_false_positive = (0, 3, 0, 0, 0)
        loses_baseline = (0, 4, 0, 0, 0)
        eligible = np.zeros(SHAPE, dtype=bool)
        eligible[baseline] = eligible[better] = eligible[has_false_positive] = True
        fp = np.zeros(SHAPE, dtype=np.uint8)
        fp[has_false_positive] = 1
        matched = np.zeros_like(fp)
        (
            matched[baseline],
            matched[better],
            matched[has_false_positive],
            matched[loses_baseline],
        ) = (1, 2, 3, 4)
        self.assertEqual(
            select_thresholds({"square": (eligible, fp, matched)}, baseline),
            ("square", better),
        )

    def test_retry_keeps_baseline_meteor_and_its_axis_evidence(self):
        baseline = (0, 1, 0, 0, 0)
        retry = (0, 2, 0, 0, 0)
        control = np.zeros(SHAPE, dtype=bool)
        control[baseline] = True
        supplemental = np.zeros_like(control)
        supplemental[retry] = True
        original_coverage = np.zeros(SHAPE[:2])
        original_coverage[baseline[:2]] = 0.75
        new_coverage = np.zeros_like(original_coverage)
        new_coverage[retry[:2]] = 0.85
        blank = np.zeros_like(new_coverage)
        data = {
            "truth": {"control": True, "missed": True},
            "pairs": {
                "square": {
                    "control": (control, original_coverage),
                    "missed": (np.zeros_like(control), blank),
                },
                "cross": {
                    "control": (np.zeros_like(control), blank),
                    "missed": (supplemental, new_coverage),
                },
            },
        }
        eligible, fp, matched = totals([data], "cross", baseline, retry_rejected=True)
        self.assertTrue(eligible[retry])
        self.assertEqual(fp[retry], 0)
        self.assertEqual(matched[retry], 2)
        result = evaluate({"test": data}, "cross", retry, keep_baseline=baseline)
        rows = result["sequences"]["test"]
        self.assertTrue(rows["control"]["appearance_match"])
        self.assertEqual(rows["control"]["appearance_coverage"], 0.75)
        self.assertTrue(rows["missed"]["appearance_match"])
