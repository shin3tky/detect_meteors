"""
Tests for pipeline helpers in meteor_core.pipeline (v1.x).
"""

import os
import sys
import tempfile
import unittest

import numpy as np

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core.exceptions import MeteorConfigError, MeteorLoadError  # noqa: E402
from meteor_core.pipeline import (  # noqa: E402
    _resolve_output_handler,
    collect_files,
    estimate_diff_threshold_from_samples,
    estimate_min_area_from_samples,
    estimate_min_line_score_from_image,
    validate_raw_file,
)
from meteor_core.schema import DEFAULT_DIFF_THRESHOLD, DEFAULT_MIN_AREA  # noqa: E402


class DummyLoader:
    """Minimal loader stub for testing."""

    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload
        self.error = error

    def load(self, filepath):
        if self.error:
            raise self.error
        return self.payload


class TestCollectFiles(unittest.TestCase):
    def test_collect_files_missing_directory(self):
        missing_path = os.path.join(tempfile.gettempdir(), "missing_raw_dir")
        with self.assertRaises(MeteorLoadError) as ctx:
            collect_files(missing_path)

        self.assertEqual(
            ctx.exception.context.get("error_category"), "directory_not_found"
        )

    def test_collect_files_not_directory(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            with self.assertRaises(MeteorLoadError) as ctx:
                collect_files(tmp_file.name)

        self.assertEqual(ctx.exception.context.get("error_category"), "not_a_directory")

    def test_collect_files_no_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(MeteorLoadError) as ctx:
                collect_files(tmp_dir)

        self.assertEqual(ctx.exception.context.get("error_category"), "no_files_found")

    def test_collect_files_finds_supported_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_a = os.path.join(tmp_dir, "a.CR2")
            file_b = os.path.join(tmp_dir, "b.NEF")
            open(file_a, "a", encoding="utf-8").close()
            open(file_b, "a", encoding="utf-8").close()

            files = collect_files(tmp_dir)

        self.assertEqual(files, sorted([file_a, file_b]))


class TestValidateRawFile(unittest.TestCase):
    def test_validate_raw_file_success(self):
        loader = DummyLoader(payload=np.zeros((4, 4), dtype=np.uint8))
        index, path, error = validate_raw_file(0, "sample.CR2", input_loader=loader)

        self.assertEqual(index, 0)
        self.assertEqual(path, "sample.CR2")
        self.assertIsNone(error)

    def test_validate_raw_file_passes_through_meteor_error(self):
        loader = DummyLoader(error=MeteorLoadError("boom", filepath="sample.CR2"))
        _, _, error = validate_raw_file(1, "sample.CR2", input_loader=loader)

        self.assertIsInstance(error, MeteorLoadError)

    def test_validate_raw_file_wraps_unexpected_error(self):
        loader = DummyLoader(error=ValueError("bad"))
        _, _, error = validate_raw_file(2, "sample.CR2", input_loader=loader)

        self.assertIsInstance(error, MeteorLoadError)
        self.assertEqual(error.context.get("error_category"), "validation_failed")


class TestResolveOutputHandler(unittest.TestCase):
    def test_resolve_output_handler_requires_fallback(self):
        with self.assertRaises(MeteorConfigError):
            _resolve_output_handler()

    def test_resolve_output_handler_unknown_name(self):
        with self.assertRaises(MeteorConfigError) as ctx:
            _resolve_output_handler(handler_name="missing-handler")

        self.assertIn("available_handlers", ctx.exception.context)


class TestEstimateHelpers(unittest.TestCase):
    def test_estimate_diff_threshold_with_too_few_samples(self):
        roi_mask = np.full((4, 4), 255, dtype=np.uint8)
        estimated = estimate_diff_threshold_from_samples(
            files=["only_one.cr2"],
            roi_mask=roi_mask,
            sample_size=1,
            input_loader=DummyLoader(payload=np.zeros((4, 4), dtype=np.uint8)),
            locale="en",
        )

        self.assertEqual(estimated, DEFAULT_DIFF_THRESHOLD)

    def test_estimate_min_area_with_no_valid_samples(self):
        roi_mask = np.full((4, 4), 255, dtype=np.uint8)
        estimated = estimate_min_area_from_samples(
            files=["a.cr2"],
            roi_mask=roi_mask,
            diff_threshold=5,
            sample_size=1,
            input_loader=DummyLoader(error=ValueError("nope")),
            locale="en",
        )

        self.assertEqual(estimated, DEFAULT_MIN_AREA)

    def test_estimate_min_line_score_clamps_bounds(self):
        low = estimate_min_line_score_from_image((10, 10), locale="en")
        high = estimate_min_line_score_from_image(
            (4000, 3000), focal_length_mm=100.0, locale="en"
        )

        self.assertEqual(low, 40.0)
        self.assertEqual(high, 150.0)


if __name__ == "__main__":
    unittest.main()
