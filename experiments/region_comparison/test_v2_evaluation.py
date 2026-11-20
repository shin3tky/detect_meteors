"""Unit tests for the evaluation protocol and its input transformations."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark_v2 import temporal_median
from inject_v2 import inject
from review_metrics import (
    NEGATIVE_DECLARATION,
    event_matches,
    exact_recall_interval,
    frame_metrics,
    read_truth,
)
from segment_verification import noise_tiles, verify_segments

FIRST, METEOR, NEXT = ("_C140520.ORF", "_C140521.ORF", "_C140522.ORF")


class TruthTests(unittest.TestCase):
    def test_unlisted_frames_require_exhaustive_review(self):
        frames = [FIRST, METEOR, NEXT]
        table = f"| File | Aircraft | Meteor |\n| --- | --- | --- |\n| `{METEOR}` | No | Yes |\n"
        with tempfile.TemporaryDirectory() as folder:
            readme = Path(folder) / "README.md"
            readme.write_text(table)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_truth(readme, frames, set(frames))
            readme.write_text(table + NEGATIVE_DECLARATION)
            self.assertEqual(
                read_truth(readme, frames, set(frames)),
                {FIRST: False, METEOR: True, NEXT: False},
            )
            readme.write_text(table + table + NEGATIVE_DECLARATION)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_truth(readme, frames, set(frames))

    def test_disappearance_can_match_an_event_but_is_a_false_positive_frame(self):
        frames = [FIRST, METEOR, NEXT]
        truth = {FIRST: False, METEOR: True, NEXT: False}
        label = {"current": METEOR, "previous": FIRST}
        pairs = {
            METEOR: {"candidate": False, "coverage": {METEOR: 0.75}},
            NEXT: {"candidate": True, "coverage": {METEOR: 0.50}},
        }
        metrics = frame_metrics(pairs, truth)
        self.assertEqual((metrics["fp"], metrics["fn"]), ([NEXT], [METEOR]))
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertIsNone(metrics["candidates_per_meteor_frame"])
        event = event_matches(pairs, frames, [label])[METEOR]
        self.assertFalse(event["appearance"]["match"])
        self.assertTrue(event["event_match"])
        self.assertIsNone(frame_metrics({}, truth)["recall"])

    def test_exact_interval_with_one_annotated_event(self):
        self.assertAlmostEqual(exact_recall_interval(0, 1)[1], 0.975)
        self.assertAlmostEqual(exact_recall_interval(1, 1)[0], 0.025)
        self.assertIsNone(exact_recall_interval(0, 0))

    def test_match_in_both_pairs_is_one_event(self):
        frames = [FIRST, METEOR, NEXT]
        pairs = {
            frame: {"candidate": True, "coverage": {METEOR: 1.0}}
            for frame in frames[1:]
        }
        events = event_matches(pairs, frames, [{"current": METEOR, "previous": FIRST}])
        self.assertEqual(sum(row["event_match"] for row in events.values()), 1)


class AnalysisInputTests(unittest.TestCase):
    def test_temporal_median_removes_one_previous_transient(self):
        baseline = np.array([[512, 512, 900]], dtype=np.uint16)
        older = baseline + np.array([[0, 3, 0]], dtype=np.uint16)
        prior = baseline + np.array([[30, 1, 0]], dtype=np.uint16)
        np.testing.assert_array_equal(
            temporal_median([baseline, older, prior]), [[512, 513, 900]]
        )
        self.assertEqual(prior[0, 0], 542)

    def test_common_verifier_requires_polarity_and_contiguous_support(self):
        residual = np.zeros((100, 100), dtype=np.float32)
        residual[49:52, 15:82] = 5
        proposals = [(15, 50, 80, 50)]
        tiles = noise_tiles(residual)
        self.assertEqual(
            verify_segments(proposals, residual, tiles),
            {"shared_dn": [], "shared_sigma": proposals},
        )
        self.assertFalse(verify_segments(proposals, -residual, tiles)["shared_sigma"])
        residual[49:52, 45:53] = 0
        self.assertFalse(verify_segments(proposals, residual, tiles)["shared_sigma"])
        self.assertFalse(
            verify_segments([(15, 50, 42, 50)], residual, tiles)["shared_sigma"]
        )

    def test_noise_units_are_invariant_under_scalar_gain_away_from_floor(self):
        rng = np.random.default_rng(1)
        residual = rng.normal(0, 4, size=(100, 100)).astype(np.float32)
        residual[45:56, 10:90] = 7
        residual[49:52, 15:82] += 10
        proposals = [(15, 50, 80, 50)]
        low = verify_segments(proposals, residual, noise_tiles(residual))
        high = verify_segments(proposals, 4 * residual, noise_tiles(4 * residual))
        self.assertTrue(low["shared_sigma"])
        self.assertEqual(low["shared_sigma"], high["shared_sigma"])

    def test_injection_reports_quantized_signal_without_altering_original(self):
        current = np.full((96, 96), 512, dtype=np.uint16)
        axis = np.array([[20, 48], [68, 48]])
        injected, peak, snr = inject(current, np.ones((2, 2)), axis, peak_dn=5)
        self.assertEqual(peak, 5)
        self.assertEqual(injected.max() - current.max(), 5)
        self.assertEqual(current.max(), 512)
        self.assertEqual(snr, np.linalg.norm(injected.astype(np.float32) - current))
        self.assertEqual(int(injected[0, 0]), 512)


if __name__ == "__main__":
    unittest.main()
