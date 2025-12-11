#!/usr/bin/env python
"""
Test suite for detector plugin architecture (v1.x).

Tests the BaseDetector protocol and detector resolution:
- Custom detector injection
- Detector name-based resolution
- PipelineConfig detector settings
- Error handling for unknown detectors
"""

import unittest
import numpy as np
import sys
import os

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core.detectors import (  # noqa: E402
    BaseDetector,
    HoughDetector,
    discover_detectors,
)
from meteor_core.schema import PipelineConfig, DetectionParams  # noqa: E402
from meteor_core.pipeline import (  # noqa: E402
    _resolve_detector,
    _get_available_detectors,
    _DEFAULT_DETECTOR,
    process_image_batch,
)


class MockDetector(BaseDetector):
    """Mock detector for testing custom detector injection."""

    plugin_name = "mock"
    name = "MockDetector"
    version = "1.0.0-test"

    def __init__(self):
        self.detect_call_count = 0
        self.last_params = None

    def detect(self, current_image, previous_image, roi_mask, params):
        """Mock detection that tracks calls and returns fixed result."""
        self.detect_call_count += 1
        self.last_params = params
        # Return: is_candidate, line_score, line_segments, max_aspect_ratio, debug_img
        return False, 0.0, [], 0.0, None

    def compute_line_score(self, mask, hough_params):
        """Mock line score computation."""
        return 0.0, []


class TestResolveDetector(unittest.TestCase):
    """Tests for _resolve_detector function."""

    def test_resolve_with_explicit_detector(self):
        """Explicit detector instance takes priority."""
        mock = MockDetector()
        result = _resolve_detector(detector=mock)
        self.assertIs(result, mock)

    def test_resolve_with_explicit_detector_ignores_name(self):
        """Explicit detector takes priority over detector_name."""
        mock = MockDetector()
        result = _resolve_detector(detector=mock, detector_name="hough")
        self.assertIs(result, mock)

    def test_resolve_with_detector_name_hough(self):
        """detector_name='hough' creates HoughDetector instance."""
        result = _resolve_detector(detector_name="hough")
        self.assertIsInstance(result, HoughDetector)

    def test_resolve_with_detector_name_case_insensitive(self):
        """detector_name is case-insensitive."""
        result_lower = _resolve_detector(detector_name="hough")
        result_upper = _resolve_detector(detector_name="HOUGH")
        result_mixed = _resolve_detector(detector_name="Hough")

        self.assertIsInstance(result_lower, HoughDetector)
        self.assertIsInstance(result_upper, HoughDetector)
        self.assertIsInstance(result_mixed, HoughDetector)

    def test_resolve_with_unknown_detector_name_raises(self):
        """Unknown detector_name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _resolve_detector(detector_name="unknown_detector")

        self.assertIn("Unknown detector", str(ctx.exception))
        self.assertIn("unknown_detector", str(ctx.exception))

    def test_resolve_default_detector(self):
        """No arguments returns default detector."""
        result = _resolve_detector()
        self.assertIs(result, _DEFAULT_DETECTOR)

    def test_resolve_default_is_hough_detector(self):
        """Default detector is HoughDetector."""
        self.assertIsInstance(_DEFAULT_DETECTOR, HoughDetector)


class TestAvailableDetectors(unittest.TestCase):
    """Tests for discover_detectors() and _get_available_detectors()."""

    def test_hough_detector_registered(self):
        """HoughDetector is registered as 'hough'."""
        available = _get_available_detectors()
        self.assertIn("hough", available)
        self.assertIs(available["hough"], HoughDetector)

    def test_registry_keys_are_lowercase(self):
        """All registry keys should be lowercase."""
        available = _get_available_detectors()
        for key in available:
            self.assertEqual(key, key.lower())

    def test_discover_detectors_returns_dict(self):
        """discover_detectors() returns a dictionary."""
        detectors = discover_detectors()
        self.assertIsInstance(detectors, dict)
        self.assertIn("hough", detectors)

    def test_hough_detector_has_plugin_name(self):
        """HoughDetector has correct plugin_name."""
        self.assertEqual(HoughDetector.plugin_name, "hough")


class TestPipelineConfigDetector(unittest.TestCase):
    """Tests for PipelineConfig detector settings."""

    def test_config_detector_name_default_is_none(self):
        """detector_name defaults to None."""
        config = PipelineConfig(
            target_folder="./raw",
            output_folder="./out",
            debug_folder="./debug",
            params=DetectionParams(),
        )
        self.assertIsNone(config.detector_name)
        self.assertIsNone(config.detector_config)

    def test_config_detector_name_can_be_set(self):
        """detector_name can be set explicitly."""
        config = PipelineConfig(
            target_folder="./raw",
            output_folder="./out",
            debug_folder="./debug",
            params=DetectionParams(),
            detector_name="hough",
        )
        self.assertEqual(config.detector_name, "hough")

    def test_config_detector_config_can_be_set(self):
        """detector_config can be set with dict."""
        detector_cfg = {"threshold": 15, "min_length": 20}
        config = PipelineConfig(
            target_folder="./raw",
            output_folder="./out",
            debug_folder="./debug",
            params=DetectionParams(),
            detector_name="hough",
            detector_config=detector_cfg,
        )
        self.assertEqual(config.detector_config, detector_cfg)

    def test_config_to_dict_includes_detector(self):
        """to_dict() includes detector settings."""
        config = PipelineConfig(
            target_folder="./raw",
            output_folder="./out",
            debug_folder="./debug",
            params=DetectionParams(),
            detector_name="hough",
            detector_config={"key": "value"},
        )
        d = config.to_dict()
        self.assertEqual(d["detector_name"], "hough")
        self.assertEqual(d["detector_config"], {"key": "value"})

    def test_config_from_dict_reads_detector(self):
        """from_dict() reads detector settings."""
        data = {
            "target_folder": "./raw",
            "output_folder": "./out",
            "debug_folder": "./debug",
            "params": {},
            "detector_name": "hough",
            "detector_config": {"key": "value"},
        }
        config = PipelineConfig.from_dict(data)
        self.assertEqual(config.detector_name, "hough")
        self.assertEqual(config.detector_config, {"key": "value"})

    def test_config_from_dict_defaults_detector_to_none(self):
        """from_dict() defaults detector settings to None if not present."""
        data = {
            "target_folder": "./raw",
            "output_folder": "./out",
            "debug_folder": "./debug",
            "params": {},
        }
        config = PipelineConfig.from_dict(data)
        self.assertIsNone(config.detector_name)
        self.assertIsNone(config.detector_config)


class TestProcessImageBatchWithDetector(unittest.TestCase):
    """Tests for process_image_batch with custom detector."""

    def setUp(self):
        """Set up test images and parameters."""
        self.shape = (100, 100)
        self.img_black = np.zeros(self.shape, dtype=np.uint16)
        self.roi_mask = np.ones(self.shape, dtype=np.uint8) * 255
        self.params = {
            "diff_threshold": 10,
            "min_area": 5,
            "min_aspect_ratio": 2.0,
            "min_line_score": 50.0,
            "hough_threshold": 10,
            "hough_min_line_length": 15,
            "hough_max_line_gap": 5,
        }

    def test_custom_detector_is_called(self):
        """Custom detector's detect() method is called."""
        mock = MockDetector()

        # Create temporary test files
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy image files (we'll mock the loader)
            file1 = os.path.join(tmpdir, "img1.raw")
            file2 = os.path.join(tmpdir, "img2.raw")

            # Create a mock loader that returns our test images
            class MockLoader:
                def load(self, filepath):
                    return np.zeros((100, 100), dtype=np.uint16)

            # Process with mock detector and mock loader
            results = process_image_batch(
                batch_data=[(file1, file2)],
                roi_mask=self.roi_mask,
                params=self.params,
                input_loader=MockLoader(),
                detector=mock,
            )
            self.assertEqual(len(results), 1)

            # Verify detector was called
            self.assertEqual(mock.detect_call_count, 1)
            self.assertEqual(mock.last_params, self.params)


class TestBaseDetectorInterface(unittest.TestCase):
    """Tests for BaseDetector interface compliance."""

    def test_hough_detector_has_required_attributes(self):
        """HoughDetector has required name and version attributes."""
        detector = HoughDetector()
        self.assertTrue(hasattr(detector, "name"))
        self.assertTrue(hasattr(detector, "version"))
        self.assertIsInstance(detector.name, str)
        self.assertIsInstance(detector.version, str)

    def test_hough_detector_has_detect_method(self):
        """HoughDetector has detect() method."""
        detector = HoughDetector()
        self.assertTrue(hasattr(detector, "detect"))
        self.assertTrue(callable(detector.detect))

    def test_hough_detector_has_compute_line_score_method(self):
        """HoughDetector has compute_line_score() method."""
        detector = HoughDetector()
        self.assertTrue(hasattr(detector, "compute_line_score"))
        self.assertTrue(callable(detector.compute_line_score))

    def test_hough_detector_get_info(self):
        """HoughDetector.get_info() returns expected structure."""
        detector = HoughDetector()
        info = detector.get_info()

        self.assertIn("plugin_name", info)
        self.assertIn("name", info)
        self.assertIn("version", info)
        self.assertIn("class", info)
        self.assertEqual(info["plugin_name"], "hough")
        self.assertEqual(info["class"], "HoughDetector")

    def test_hough_detector_validate_params(self):
        """HoughDetector.validate_params() works correctly."""
        detector = HoughDetector()

        valid_params = {
            "diff_threshold": 10,
            "min_area": 5,
            "min_aspect_ratio": 2.0,
            "min_line_score": 50.0,
            "hough_threshold": 10,
            "hough_min_line_length": 15,
            "hough_max_line_gap": 5,
        }
        self.assertTrue(detector.validate_params(valid_params))

        # Missing required key
        invalid_params = {"diff_threshold": 10}
        self.assertFalse(detector.validate_params(invalid_params))


if __name__ == "__main__":
    unittest.main()
