"""Read-only pair reproduction, provenance, and missing-input behavior."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from meteor_core.analysis import load_reproduction_settings, reproduce_hough_pair
from meteor_core.exceptions import (
    MeteorConfigError,
    MeteorLoadError,
    MeteorValidationError,
)
from meteor_core.inputs.base import BaseInputLoader
from meteor_core.pipeline import process_image_batch
from meteor_core.roi_selector import create_full_roi_mask
from meteor_core.schema import DetectionParams, InputContext


class ArrayLoader(BaseInputLoader):
    plugin_name = "analysis_test"

    def __init__(self, images):
        self.images = images
        self.loaded = []

    def load(self, filepath):
        self.loaded.append(filepath)
        return InputContext(
            image_data=self.images[Path(filepath).name].copy(),
            filepath=filepath,
            metadata={"filename": Path(filepath).name},
        )


class TestAnalysisReproduction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name).resolve()
        self.target = self.directory / "frame002.ORF"
        self.reference = self.directory / "frame001.ORF"
        self.target.write_bytes(b"target")
        self.reference.write_bytes(b"reference")
        current = np.zeros((96, 96), np.uint16)
        current[20:45, 10:70] = 1000
        self.loader = ArrayLoader(
            {self.target.name: current, self.reference.name: np.zeros_like(current)}
        )
        self.params = DetectionParams(
            diff_threshold=10, min_area=3, min_aspect_ratio=3.0, min_line_score=30.0
        )
        self.progress = self.directory / "progress.json"
        self.write_progress(
            {
                "processed_files": [self.target.name],
                "params": self.params.to_dict(),
                "processing_params": self.params.to_dict(),
                "roi": None,
            }
        )

    def write_progress(self, data):
        self.progress.write_text(json.dumps(data), encoding="utf-8")

    def replay(self, settings=None, reference=True):
        return reproduce_hough_pair(
            self.target,
            self.reference if reference else None,
            settings=settings
            or load_reproduction_settings(progress_path=self.progress),
            input_loader=self.loader,
        )

    def test_replay_matches_normal_pipeline_and_preserves_inputs(self):
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        result = self.replay()
        normal = process_image_batch(
            [(1, str(self.target), str(self.reference))],
            create_full_roi_mask((96, 96)),
            self.params.to_dict(),
            input_loader=self.loader,
        )[0][7]

        self.assertEqual(result.status, "measured")
        self.assertFalse(result.decision.is_candidate)
        self.assertEqual(result.decision.rejection_reasons, ("min_aspect_ratio",))
        self.assertEqual(result.decision.is_candidate, normal.is_candidate)
        self.assertEqual(result.analysis.score, normal.score)
        self.assertEqual(result.analysis.lines, normal.lines)
        self.assertEqual(result.decision.max_aspect_ratio, normal.aspect_ratio)
        self.assertEqual(result.target_metadata, {"filename": self.target.name})
        self.assertEqual(result.reference_metadata, {"filename": self.reference.name})
        self.assertEqual(result.loader_info["plugin_name"], "analysis_test")
        self.assertEqual(
            before, {p.name: p.read_bytes() for p in self.directory.iterdir()}
        )

    def test_override_recovers_frame_without_overwriting_recorded_baseline(self):
        settings = load_reproduction_settings(
            progress_path=self.progress, parameter_overrides={"min_aspect_ratio": 2.0}
        )
        result = self.replay(settings)
        self.assertTrue(result.decision.is_candidate)
        self.assertEqual(settings.baseline_params.min_aspect_ratio, 3.0)
        self.assertEqual(settings.params.min_aspect_ratio, 2.0)
        self.assertEqual(settings.parameter_sources["min_aspect_ratio"], "override")
        self.assertEqual(
            settings.changes, {"min_aspect_ratio": {"before": 3.0, "after": 2.0}}
        )
        self.assertEqual(
            json.loads(self.progress.read_text())["params"]["min_aspect_ratio"], 3.0
        )

    def test_effective_processing_params_take_precedence_over_hashed_params(self):
        self.write_progress(
            {
                "processed_files": [],
                "params": self.params.to_dict(),
                "processing_params": {"min_aspect_ratio": 2.0},
                "roi": None,
            }
        )
        settings = load_reproduction_settings(
            progress_path=self.progress, params=DetectionParams(min_aspect_ratio=9.0)
        )
        self.assertEqual(settings.params.min_aspect_ratio, 2.0)
        self.assertEqual(
            settings.parameter_sources["min_aspect_ratio"], "progress.processing_params"
        )
        self.assertEqual(settings.parameter_sources["min_area"], "progress.params")
        self.assertEqual(settings.changes, {})

    def test_legacy_missing_fields_are_identified_instead_of_claimed_as_recorded(self):
        self.write_progress({"processed_files": [], "params": {"diff_threshold": 7}})
        settings = load_reproduction_settings(progress_path=self.progress)
        self.assertEqual(settings.params.diff_threshold, 7)
        self.assertEqual(settings.parameter_sources["min_area"], "default")
        self.assertIn("incomplete_recorded_parameters", settings.limitations)
        self.assertIn("missing_recorded_roi", settings.limitations)
        self.assertIn("assumed_full_image_roi", settings.limitations)
        self.assertIn(
            "original_loader_and_hooks_not_reproduced_from_progress",
            settings.limitations,
        )

    def test_recorded_roi_is_used_until_explicitly_overridden(self):
        polygon = [[0, 0], [8, 0], [8, 95], [0, 95]]
        data = json.loads(self.progress.read_text())
        data["roi"] = polygon
        self.write_progress(data)
        settings = load_reproduction_settings(progress_path=self.progress)
        excluded = self.replay(settings)
        self.assertEqual(settings.roi_source, "progress.roi")
        self.assertEqual(excluded.analysis.score, 0.0)
        self.assertIn("no_contours", excluded.decision.rejection_reasons)

        full = load_reproduction_settings(
            progress_path=self.progress, use_full_image=True
        )
        self.assertIsNone(full.roi_polygon)
        self.assertEqual(full.roi_source, "override_full_image")
        self.assertEqual(full.changes["roi"]["before"], tuple(map(tuple, polygon)))
        self.assertGreater(self.replay(full).analysis.score, 0.0)

    def test_missing_reference_does_not_guess_or_substitute_a_background(self):
        result = self.replay(reference=False)
        self.assertEqual(result.status, "reference_missing")
        self.assertIsNone(result.analysis)
        self.assertIsNone(result.decision)
        self.assertIsNone(result.reference_image)
        self.assertIsNone(result.reference_metadata)
        self.assertEqual(result.image_shape, (96, 96))
        self.assertEqual(self.loader.loaded, [str(self.target)])

    def test_identical_file_and_hardlink_are_rejected_before_loading(self):
        alias = self.directory / "alias.ORF"
        os.link(self.target, alias)
        for reference in (self.target, alias):
            with (
                self.subTest(reference=reference),
                self.assertRaises(MeteorValidationError),
            ):
                reproduce_hough_pair(
                    self.target,
                    reference,
                    settings=load_reproduction_settings(),
                    input_loader=self.loader,
                )
        self.assertEqual(self.loader.loaded, [])

    def test_missing_image_is_not_treated_as_a_negative_detection(self):
        with self.assertRaises(MeteorLoadError):
            reproduce_hough_pair(
                self.target,
                self.directory / "missing.ORF",
                settings=load_reproduction_settings(),
                input_loader=self.loader,
            )
        self.assertEqual(self.loader.loaded, [])

    def test_dimension_dtype_and_color_mismatches_are_explicit_errors(self):
        for image in (
            np.zeros((48, 48), np.uint16),
            np.zeros((96, 96), np.float32),
        ):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                self.loader.images[self.reference.name] = image
                with self.assertRaises(MeteorValidationError):
                    self.replay()
        self.loader.images[self.target.name] = np.zeros((96, 96, 3), np.uint16)
        with self.assertRaises(MeteorValidationError):
            self.replay(reference=False)

    def test_invalid_progress_cannot_silently_become_default_settings(self):
        for payload in (
            [],
            {"target_folder": "raw", "params": self.params.to_dict()},
            {"processed_files": [], "params": None},
            {"processed_files": [], "params": {"unsupported": 3}},
        ):
            with self.subTest(payload=payload):
                self.write_progress(payload)
                with self.assertRaises(MeteorConfigError):
                    load_reproduction_settings(progress_path=self.progress)
        self.progress.write_text("not JSON", encoding="utf-8")
        with self.assertRaises(MeteorConfigError):
            load_reproduction_settings(progress_path=self.progress)

    def test_invalid_thresholds_and_polygons_are_rejected(self):
        for override in (
            {"min_line_score": float("nan")},
            {"min_aspect_ratio": float("inf")},
            {"min_area": True},
            {"min_area": 1.5},
            {"hough_threshold": 0},
            {"diff_threshold": -1},
        ):
            with self.subTest(override=override), self.assertRaises(MeteorConfigError):
                load_reproduction_settings(parameter_overrides=override)
        for polygon in (
            [[0, 0]],
            [[0, 0], [1, 2], [True, 3]],
            [[0, 0], [1, 2], [2**32, 3]],
        ):
            with self.subTest(polygon=polygon), self.assertRaises(MeteorConfigError):
                load_reproduction_settings(roi_polygon=polygon)
        with self.assertRaises(MeteorConfigError):
            load_reproduction_settings(
                roi_polygon=[[0, 0], [1, 0], [1, 1]], use_full_image=True
            )

    def test_roi_with_no_overlap_is_an_input_error(self):
        settings = load_reproduction_settings(
            roi_polygon=[[100, 100], [120, 100], [120, 120]]
        )
        with self.assertRaises(MeteorValidationError):
            self.replay(settings)

    def test_loader_config_is_used_even_when_no_name_is_supplied(self):
        with patch(
            "meteor_core.inputs.registry.LoaderRegistry.create",
            return_value=self.loader,
        ) as create:
            result = reproduce_hough_pair(
                self.target,
                self.reference,
                settings=load_reproduction_settings(params=self.params),
                loader_config={"normalize": True},
            )
        create.assert_called_once_with("raw", {"normalize": True})
        self.assertEqual(result.status, "measured")

    def test_unknown_loader_and_load_failures_do_not_fall_back(self):
        with self.assertRaises(MeteorConfigError):
            reproduce_hough_pair(
                self.target,
                settings=load_reproduction_settings(),
                loader_name="missing_analysis_loader",
            )
        error = MeteorLoadError("cannot decode reference", filepath=str(self.reference))
        with patch.object(self.loader, "load", side_effect=error):
            with self.assertRaises(MeteorLoadError) as caught:
                self.replay()
        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()
