"""Exercise the FRT cutoff and distinguish candidates from axis recovery."""

from math import inf, nextafter
import unittest

from benchmark_hough_frt_retry import evaluate, evaluate_folds, train_zero_fp


class HoughFrtRetryTests(unittest.TestCase):
    def test_cutoff_excludes_tied_negatives_and_ignores_hough_accepts(self):
        data = {
            "train": {
                "truth": {"a": False, "b": False, "hough": True, "meteor": True},
                "pairs": {
                    "a": {"frt": {"score": 12.0}},
                    "b": {"frt": {"score": 12.0}},
                    "hough": {"hough": {"candidate": True}},
                    "meteor": {"frt": {"score": 80.0}},
                },
            },
        }
        cut = train_zero_fp(data)
        self.assertEqual(cut["min_snr"], nextafter(12.0, inf))
        self.assertEqual(cut["negative_count"], 2)
        self.assertEqual(
            cut["max_negative_frames"],
            [{"sequence": "train", "frame": "a"}, {"sequence": "train", "frame": "b"}],
        )

    def test_held_out_negatives_do_not_set_the_cutoff(self):
        sequences = {
            name: {
                "frames": ["baseline", "negative"],
                "labels": [],
                "truth": {"negative": False},
                "pairs": {
                    "negative": {
                        "hough": {"candidate": False, "coverage": {}},
                        "frt": {"score": score, "relevant_lines": {}},
                    }
                },
            }
            for name, score in (("low", 16.0), ("high", 24.0))
        }
        folds = evaluate_folds(sequences)
        held_high = folds["high"]
        self.assertEqual(held_high["calibration"]["min_snr"], nextafter(16.0, inf))
        self.assertEqual(held_high["training"]["low"]["frames"]["fp"], [])
        self.assertEqual(held_high["held_out"]["frames"]["fp"], ["negative"])
        self.assertEqual(folds["low"]["held_out"]["frames"]["fp"], [])

    def test_frame_score_can_pass_after_the_matching_axis_is_rejected(self):
        data = {
            "frames": ["previous", "hough", "frt"],
            "truth": {"hough": True, "frt": True},
            "labels": [
                {
                    "current": name,
                    "previous": previous,
                    "axis": [[0, 5], [20, 5]],
                    "half_width_px": 2,
                }
                for name, previous in (("hough", "previous"), ("frt", "hough"))
            ],
            "pairs": {
                "hough": {"hough": {"candidate": True, "coverage": {"hough": 0.9}}},
                "frt": {
                    "hough": {
                        "candidate": False,
                        "coverage": {"hough": 0.0, "frt": 0.0},
                    },
                    "frt": {
                        "score": 40.0,
                        "relevant_lines": {"hough": [], "frt": [(20.0, [0, 5, 20, 5])]},
                    },
                },
            },
        }
        at_score = evaluate(data, 20.0)
        self.assertEqual(at_score["added_by_frt"], ["frt"])
        self.assertTrue(at_score["events"]["frt"]["appearance"]["match"])
        strict = evaluate(data, nextafter(20.0, inf))
        self.assertEqual(strict["frames"]["tp"], ["hough", "frt"])
        self.assertFalse(strict["events"]["frt"]["appearance"]["match"])
        self.assertEqual(strict["events"]["hough"]["appearance"]["coverage"], 0.9)
