#!/usr/bin/env python
#
# Detect Meteors CLI - RAW Loader Tests
# © 2025 Shinichi Morita (shin3tky)
#

"""
Unit tests for RawImageLoader and RawLoaderConfig with exception handling.
"""

import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch

from meteor_core.exceptions import (
    MeteorLoadError,
    MeteorUnsupportedFormatError,
    MeteorValidationError,
)
from meteor_core.image_io import (
    PILLOW_AVAILABLE,
    _exif_ratio_to_float,
    _exif_strategy_pil_direct,
    _exif_strategy_rawpy_dimensions,
    _populate_exif_from_dict,
)
from meteor_core.inputs.raw import RawImageLoader, RawLoaderConfig, create_raw_loader


class TestRawLoaderConfig(unittest.TestCase):
    """Test cases for RawLoaderConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RawLoaderConfig()
        self.assertEqual(config.binning, 2)
        self.assertFalse(config.normalize)

    def test_valid_binning(self):
        """Test that binning=2 is valid."""
        config = RawLoaderConfig(binning=2)
        self.assertEqual(config.binning, 2)

    def test_invalid_binning_raises_validation_error(self):
        """Test that invalid binning raises MeteorValidationError."""
        with self.assertRaises(MeteorValidationError) as ctx:
            RawLoaderConfig(binning=4)

        err = ctx.exception
        self.assertEqual(err.parameter_name, "binning")
        self.assertEqual(err.provided_value, 4)
        self.assertIn("2", err.expected)

    def test_invalid_binning_1_raises_validation_error(self):
        """Test that binning=1 raises MeteorValidationError."""
        with self.assertRaises(MeteorValidationError) as ctx:
            RawLoaderConfig(binning=1)

        err = ctx.exception
        self.assertEqual(err.parameter_name, "binning")
        self.assertEqual(err.provided_value, 1)

    def test_normalize_option(self):
        """Test normalize configuration option."""
        config = RawLoaderConfig(normalize=True)
        self.assertTrue(config.normalize)


class TestRawImageLoader(unittest.TestCase):
    """Test cases for RawImageLoader."""

    def test_loader_attributes(self):
        """Test loader class attributes."""
        loader = RawImageLoader(RawLoaderConfig())
        self.assertEqual(loader.plugin_name, "raw")
        self.assertEqual(loader.name, "RAW Image Loader")
        self.assertIsNotNone(loader.version)

    def test_load_file_not_found_raises_meteor_load_error(self):
        """Test that loading non-existent file raises MeteorLoadError."""
        loader = RawImageLoader(RawLoaderConfig())

        with self.assertRaises(MeteorLoadError) as ctx:
            loader.load("/nonexistent/path/image.CR2")

        err = ctx.exception
        self.assertIn("not found", err.message.lower())
        self.assertEqual(err.context.get("error_category"), "file_not_found")

    @patch("meteor_core.image_io.rawpy.imread")
    def test_load_unsupported_format_raises_unsupported_format_error(self, mock_imread):
        """Test that unsupported format raises MeteorUnsupportedFormatError."""
        import rawpy

        loader = RawImageLoader(RawLoaderConfig())

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            mock_imread.side_effect = rawpy.LibRawFileUnsupportedError("Not RAW")

            with self.assertRaises(MeteorUnsupportedFormatError) as ctx:
                loader.load(temp_path)

            err = ctx.exception
            self.assertIsNotNone(err.detected_format)
        finally:
            os.unlink(temp_path)

    @patch("meteor_core.image_io.rawpy.imread")
    def test_load_corrupted_file_raises_meteor_load_error(self, mock_imread):
        """Test that corrupted file raises MeteorLoadError."""
        import rawpy

        loader = RawImageLoader(RawLoaderConfig())

        with tempfile.NamedTemporaryFile(delete=False, suffix=".CR2") as f:
            temp_path = f.name

        try:
            mock_imread.side_effect = rawpy.LibRawDataError("Data corrupted")

            with self.assertRaises(MeteorLoadError) as ctx:
                loader.load(temp_path)

            err = ctx.exception
            self.assertEqual(err.context.get("error_category"), "data_corruption")
        finally:
            os.unlink(temp_path)

    @patch("meteor_core.inputs.raw.load_and_bin_raw_fast")
    def test_load_returns_uint16_by_default(self, mock_load):
        """Test that load returns uint16 array by default."""
        import numpy as np

        mock_load.return_value = np.array([[100, 200], [300, 400]], dtype=np.uint16)

        loader = RawImageLoader(RawLoaderConfig(normalize=False))
        result = loader.load("/dummy/path.CR2")

        self.assertEqual(result.image_data.dtype, np.uint16)

    @patch("meteor_core.inputs.raw.load_and_bin_raw_fast")
    def test_load_normalizes_when_configured(self, mock_load):
        """Test that load returns float32 [0,1] when normalize=True."""
        import numpy as np

        # Max uint16 value
        max_val = np.iinfo(np.uint16).max
        mock_load.return_value = np.array(
            [[max_val, 0], [max_val // 2, max_val]], dtype=np.uint16
        )

        loader = RawImageLoader(RawLoaderConfig(normalize=True))
        result = loader.load("/dummy/path.CR2")

        self.assertEqual(result.image_data.dtype, np.float32)
        self.assertAlmostEqual(result.image_data[0, 0], 1.0)
        self.assertAlmostEqual(result.image_data[0, 1], 0.0)

    def test_error_has_diagnostic_info(self):
        """Test that errors contain diagnostic information."""
        loader = RawImageLoader(RawLoaderConfig())

        try:
            loader.load("/nonexistent/file.CR2")
        except MeteorLoadError as e:
            info = e.get_diagnostic_info()
            self.assertEqual(info.error_type, "MeteorLoadError")
            self.assertEqual(info.filepath, "/nonexistent/file.CR2")
            self.assertIn("numpy", info.dependencies)


class TestRawImageLoaderMetadata(unittest.TestCase):
    """Test cases for RawImageLoader.extract_metadata."""

    @patch("meteor_core.inputs.raw.extract_exif_metadata")
    def test_extract_metadata_returns_dict(self, mock_extract):
        """Test that extract_metadata returns a dictionary."""
        mock_extract.return_value = {
            "focal_length": 50.0,
            "iso": 800,
            "exposure_time": 30.0,
        }

        loader = RawImageLoader(RawLoaderConfig())
        result = loader.extract_metadata("/dummy/path.CR2")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["focal_length"], 50.0)
        mock_extract.assert_called_once_with("/dummy/path.CR2")

    @patch("meteor_core.inputs.raw.extract_exif_metadata")
    def test_extract_metadata_handles_empty_result(self, mock_extract):
        """Test that extract_metadata handles empty metadata gracefully."""
        mock_extract.return_value = {
            "focal_length": None,
            "iso": None,
            "exposure_time": None,
        }

        loader = RawImageLoader(RawLoaderConfig())
        result = loader.extract_metadata("/dummy/path.CR2")

        self.assertIsInstance(result, dict)
        self.assertIsNone(result["focal_length"])


class TestCreateRawLoader(unittest.TestCase):
    """Test cases for create_raw_loader factory function."""

    def test_create_with_defaults(self):
        """Test creating loader with default config."""
        loader = create_raw_loader()
        self.assertIsInstance(loader, RawImageLoader)
        self.assertEqual(loader.config.binning, 2)
        self.assertFalse(loader.config.normalize)

    def test_create_with_custom_config(self):
        """Test creating loader with custom config."""
        config = RawLoaderConfig(normalize=True)
        loader = create_raw_loader(config)
        self.assertTrue(loader.config.normalize)

    def test_create_with_invalid_config_raises(self):
        """Test that factory with invalid config raises MeteorValidationError."""
        with self.assertRaises(MeteorValidationError):
            create_raw_loader(RawLoaderConfig(binning=8))


class TestRawLoaderConfigDiagnostics(unittest.TestCase):
    """Test diagnostic information from RawLoaderConfig errors."""

    def test_validation_error_has_diagnostics(self):
        """Test that validation error can generate diagnostic info."""
        try:
            RawLoaderConfig(binning=3)
        except MeteorValidationError as e:
            info = e.get_diagnostic_info()
            self.assertEqual(info.error_type, "MeteorValidationError")
            self.assertIn("binning", info.context.get("parameter_name", ""))

    def test_validation_error_format_for_issue(self):
        """Test that validation error can be formatted for GitHub issue."""
        try:
            RawLoaderConfig(binning=3)
        except MeteorValidationError as e:
            output = e.format_for_issue()
            self.assertIn("## Diagnostic Information", output)
            self.assertIn("MeteorValidationError", output)


class TestImageIoExif(unittest.TestCase):
    """Tests for image_io EXIF helper utilities."""

    def test_exif_ratio_to_float(self):
        self.assertEqual(_exif_ratio_to_float((1, 2)), 0.5)
        self.assertEqual(_exif_ratio_to_float(3), 3.0)
        self.assertIsNone(_exif_ratio_to_float((1, 0)))
        self.assertIsNone(_exif_ratio_to_float("bad"))

    def test_populate_exif_from_dict(self):
        result = {
            "focal_length": None,
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
            "ISOSpeedRatings": [800],
            "ExposureTime": (1, 10),
            "FNumber": (4, 1),
            "Make": "Test",
            "Model": "Camera",
            "LensModel": "Lens",
            "ImageWidth": "1024",
            "ImageLength": "768",
        }
        added = _populate_exif_from_dict(result, exif_dict)
        self.assertGreaterEqual(added, 1)
        self.assertEqual(result["focal_length"], 50.0)
        self.assertEqual(result["focal_length_35mm"], 75.0)
        self.assertEqual(result["iso"], 800)
        self.assertEqual(result["image_width"], 1024)
        self.assertEqual(result["image_height"], 768)

    @unittest.skipUnless(PILLOW_AVAILABLE, "Pillow not available")
    def test_exif_strategy_pil_direct(self):
        result = {
            "focal_length": None,
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
        fake_image = mock.Mock()
        fake_image._getexif.return_value = {37386: (35, 1)}
        with mock.patch("meteor_core.image_io.Image.open", return_value=fake_image):
            extracted = _exif_strategy_pil_direct("dummy.raw", result)
        self.assertTrue(extracted)
        self.assertEqual(result["focal_length"], 35.0)

    def test_exif_strategy_rawpy_dimensions(self):
        result = {
            "focal_length": None,
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

        class FakeRaw:
            def __init__(self):
                self.sizes = mock.Mock(raw_width=1920, raw_height=1080)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with mock.patch("meteor_core.image_io.rawpy.imread", return_value=FakeRaw()):
            extracted = _exif_strategy_rawpy_dimensions("dummy.raw", result)

        self.assertTrue(extracted)
        self.assertEqual(result["image_width"], 1920)
        self.assertEqual(result["image_height"], 1080)


if __name__ == "__main__":
    unittest.main()
