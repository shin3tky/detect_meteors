"""Unit tests for image I/O helper functions."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest


missing_deps = [
    name for name in ("numpy", "rawpy") if importlib.util.find_spec(name) is None
]
if missing_deps:
    raise unittest.SkipTest(f"Missing dependencies: {', '.join(missing_deps)}")

if importlib.util.find_spec("yaml") is None:

    class _YamlError(Exception):
        """Fallback YAML error type for tests."""

    sys.modules["yaml"] = types.SimpleNamespace(  # type: ignore[assignment]
        safe_load=lambda _: {},
        YAMLError=_YamlError,
    )

image_io = importlib.import_module("meteor_core.image_io")


class TestImageIoHelpers(unittest.TestCase):
    """Coverage for EXIF helper utilities in image_io."""

    def test_exif_ratio_to_float_handles_tuples_and_scalars(self) -> None:
        self.assertEqual(image_io._exif_ratio_to_float((10, 2)), 5.0)
        self.assertIsNone(image_io._exif_ratio_to_float((1, 0)))
        self.assertEqual(image_io._exif_ratio_to_float(3), 3.0)
        self.assertIsNone(image_io._exif_ratio_to_float(None))

    def test_populate_exif_from_dict_prefers_existing_values(self) -> None:
        result = {
            "focal_length": 24.0,
            "focal_length_35mm": None,
            "iso": None,
            "exposure_time": None,
            "f_number": None,
            "camera_make": None,
            "camera_model": None,
            "lens_model": None,
            "image_width": None,
            "image_height": None,
        }
        exif_dict = {
            "FocalLength": (50, 1),
            "FocalLengthIn35mmFilm": "75",
            "ISOSpeedRatings": [1600],
            "ExposureTime": (1, 30),
            "FNumber": (4, 1),
            "Make": "Canon",
            "Model": "EOS",
            "LensModel": "EF 50mm",
            "ExifImageWidth": "4000",
            "ExifImageHeight": "3000",
        }

        updated = image_io._populate_exif_from_dict(result, exif_dict)

        self.assertEqual(updated, 9)
        self.assertEqual(result["focal_length"], 24.0)
        self.assertEqual(result["focal_length_35mm"], 75.0)
        self.assertEqual(result["iso"], 1600)
        self.assertAlmostEqual(result["exposure_time"], 1 / 30)
        self.assertEqual(result["f_number"], 4.0)
        self.assertEqual(result["camera_make"], "Canon")
        self.assertEqual(result["camera_model"], "EOS")
        self.assertEqual(result["lens_model"], "EF 50mm")
        self.assertEqual(result["image_width"], 4000)
        self.assertEqual(result["image_height"], 3000)

    def test_extract_exif_metadata_skips_when_pillow_unavailable(self) -> None:
        original_value = image_io.PILLOW_AVAILABLE
        image_io.PILLOW_AVAILABLE = False
        try:
            result = image_io.extract_exif_metadata("dummy.raw")
        finally:
            image_io.PILLOW_AVAILABLE = original_value

        self.assertTrue(all(value is None for value in result.values()))
