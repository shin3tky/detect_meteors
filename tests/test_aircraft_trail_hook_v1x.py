"""
Aircraft trail hook tests (v1.x).
"""

import os
import sys
import importlib.machinery
import importlib.util
import tempfile
import types
import unittest


if importlib.util.find_spec("cv2") is None:
    cv2_stub = types.SimpleNamespace()
    cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", None)
    sys.modules.setdefault("cv2", cv2_stub)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core.hooks.aircraft_trail import (  # noqa: E402
    AircraftTrailConfig,
    AircraftTrailHook,
)
from meteor_core.outputs import ProgressManager  # noqa: E402
from meteor_core.schema import DetectionResult, SortedDetection  # noqa: E402


class TestAircraftTrailHook(unittest.TestCase):
    """Tests for the aircraft trail hook metadata output."""

    def _make_sorted_detection(
        self,
        frame_index: int,
        lines: list[tuple[int, int, int, int]],
        *,
        is_candidate: bool = True,
        score: float = 1.0,
        aspect_ratio: float = 1.0,
    ) -> SortedDetection:
        return SortedDetection(
            frame_index=frame_index,
            prev_frame_index=frame_index - 1,
            filename=f"frame_{frame_index}.CR2",
            filepath=f"/tmp/frame_{frame_index}.CR2",
            is_candidate=is_candidate,
            score=score,
            aspect_ratio=aspect_ratio,
            lines=list(lines),
        )

    def test_hook_attaches_and_updates_metadata(self):
        hook = AircraftTrailHook(AircraftTrailConfig())
        detections = [
            self._make_sorted_detection(1, [(0, 0, 10, 0)]),
            self._make_sorted_detection(2, [(1, 0, 11, 0)]),
        ]

        updated = hook.on_all_detections_sorted(detections)
        first, second = sorted(updated, key=lambda detection: detection.frame_index)

        self.assertIn("aircraft", first.extras)
        self.assertIn("aircraft", second.extras)
        self.assertEqual(
            first.extras["aircraft"]["track_id"], second.extras["aircraft"]["track_id"]
        )
        self.assertEqual(first.extras["aircraft"]["evidence"]["track_frames"], 1)
        self.assertEqual(second.extras["aircraft"]["evidence"]["track_frames"], 2)
        self.assertGreater(
            second.extras["aircraft"]["likelihood"],
            first.extras["aircraft"]["likelihood"],
        )

    def test_hook_handles_missing_lines(self):
        hook = AircraftTrailHook(AircraftTrailConfig())
        detections = [self._make_sorted_detection(1, [], is_candidate=False, score=0.0)]

        updated = hook.on_all_detections_sorted(detections)[0]

        self.assertIn("aircraft", updated.extras)
        self.assertEqual(updated.extras["aircraft"]["likelihood"], 0.0)
        self.assertEqual(updated.extras["aircraft"]["evidence"]["track_frames"], 0)


class TestAircraftTrailProgressPersistence(unittest.TestCase):
    """Tests for persistence of aircraft metadata in progress files."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.progress_path = os.path.join(self.test_dir, "progress.json")

    def tearDown(self):
        for root, _, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            os.rmdir(root)

    def test_progress_manager_records_aircraft_metadata(self):
        manager = ProgressManager(self.progress_path)
        extras = {
            "aircraft": {
                "likelihood": 0.9,
                "track_id": "air-0001",
                "evidence": {
                    "track_frames": 3,
                    "angle_diff_deg": 1.0,
                    "start_distance_px": 2.0,
                    "end_distance_px": 1.5,
                    "speed_consistency": 0.8,
                },
            }
        }
        detection_result = DetectionResult(
            is_candidate=True,
            score=5.0,
            lines=[(0, 0, 10, 0)],
            aspect_ratio=1.2,
            debug_image=None,
            extras=extras,
        )

        manager.record_result(
            filename="test.CR2",
            is_candidate=True,
            score=5.0,
            lines=1,
            ratio=1.2,
            frame_index=1,
            detection_result=detection_result,
        )

        detected_details = manager.progress_data["detected_details"]
        self.assertEqual(len(detected_details), 1)
        self.assertEqual(detected_details[0]["aircraft"], extras["aircraft"])
