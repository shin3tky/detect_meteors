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

from meteor_core.exceptions import MeteorConfigError  # noqa: E402
from meteor_core.detectors import (  # noqa: E402
    BaseDetector,
    HoughDetector,
    DetectorRegistry,
)
from meteor_core.detectors.simple_threshold import (  # noqa: E402
    SimpleThresholdConfig,
    SimpleThresholdDetector,
)
from meteor_core.schema import (  # noqa: E402
    PipelineConfig,
    DetectionParams,
    DetectionContext,
    DetectionResult,
)
from meteor_core.pipeline import (  # noqa: E402
    _resolve_detector,
    _get_default_detector,
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

    def detect(self, context: DetectionContext) -> DetectionResult:
        """Mock detection that tracks calls and returns fixed result."""
        self.detect_call_count += 1
        self.last_params = context.runtime_params
        # Return: is_candidate, line_score, line_segments, max_aspect_ratio, debug_img
        return DetectionResult(
            is_candidate=False,
            score=0.0,
            lines=[],
            aspect_ratio=0.0,
            debug_image=None,
            extras={},
        )

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
        """Unknown detector_name raises MeteorConfigError."""
        with self.assertRaises(MeteorConfigError) as ctx:
            _resolve_detector(detector_name="unknown_detector")

        err = ctx.exception
        self.assertIn("unknown_detector", err.message.lower())
        self.assertEqual(err.plugin_name, "unknown_detector")

    def test_resolve_default_detector(self):
        """No arguments returns default detector."""
        result = _resolve_detector()
        default = _get_default_detector()
        self.assertIs(result, default)

    def test_resolve_default_is_hough_detector(self):
        """Default detector is HoughDetector."""
        default = _get_default_detector()
        self.assertIsInstance(default, HoughDetector)


class TestAvailableDetectors(unittest.TestCase):
    """Tests for DetectorRegistry.discover()."""

    def setUp(self):
        """Reset registry before each test."""
        DetectorRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        DetectorRegistry._reset()

    def test_hough_detector_registered(self):
        """HoughDetector is registered as 'hough'."""
        available = DetectorRegistry.discover()
        self.assertIn("hough", available)
        self.assertIs(available["hough"], HoughDetector)

    def test_registry_keys_are_lowercase(self):
        """All registry keys should be lowercase."""
        available = DetectorRegistry.discover()
        for key in available:
            self.assertEqual(key, key.lower())

    def test_discover_returns_dict(self):
        """DetectorRegistry.discover() returns a dictionary."""
        detectors = DetectorRegistry.discover()
        self.assertIsInstance(detectors, dict)
        self.assertIn("hough", detectors)

    def test_hough_detector_has_plugin_name(self):
        """HoughDetector has correct plugin_name."""
        self.assertEqual(HoughDetector.plugin_name, "hough")

    def test_deprecated_discover_detectors_warns(self):
        """discover_detectors() emits DeprecationWarning."""
        from meteor_core.detectors import discover_detectors

        with self.assertWarns(DeprecationWarning):
            discover_detectors()


class TestSimpleThresholdDetector(unittest.TestCase):
    """Tests for SimpleThresholdDetector behavior."""

    def test_detect_returns_candidate(self):
        config = SimpleThresholdConfig(diff_threshold=5, min_area=0)
        detector = SimpleThresholdDetector(config)
        current = np.zeros((6, 6), dtype=np.uint8)
        previous = np.zeros((6, 6), dtype=np.uint8)
        current[2:4, 2:4] = 20
        roi_mask = np.full_like(current, 255)

        context = DetectionContext(
            current_image=current,
            previous_image=previous,
            roi_mask=roi_mask,
            runtime_params={"global": {}, "detector": {}},
            metadata={},
        )
        result = detector.detect(context)

        self.assertTrue(result.is_candidate)
        self.assertGreater(result.score, 0)
        self.assertEqual(len(result.lines), 1)
        self.assertGreaterEqual(result.aspect_ratio, 1.0)
        self.assertIsNone(result.debug_image)

        score_from_compute, segments_from_compute = detector.compute_line_score(
            roi_mask, {}
        )
        self.assertEqual(score_from_compute, 0.0)
        self.assertEqual(segments_from_compute, [])

    def test_detect_rejects_mismatched_shapes(self):
        detector = SimpleThresholdDetector(SimpleThresholdConfig())
        current = np.zeros((4, 4), dtype=np.uint8)
        previous = np.zeros((5, 5), dtype=np.uint8)
        roi_mask = np.full_like(current, 255)

        with self.assertRaises(ValueError):
            detector.detect(
                DetectionContext(
                    current_image=current,
                    previous_image=previous,
                    roi_mask=roi_mask,
                    runtime_params={"global": {}, "detector": {}},
                    metadata={},
                )
            )


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
        from meteor_core.schema import InputContext

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy image files (we'll mock the loader)
            file1 = os.path.join(tmpdir, "img1.raw")
            file2 = os.path.join(tmpdir, "img2.raw")

            # Create a mock loader that returns our test images
            class MockLoader:
                def load(self, filepath):
                    return InputContext(
                        image_data=np.zeros((100, 100), dtype=np.uint16),
                        filepath=filepath,
                    )

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
            self.assertEqual(
                mock.last_params,
                {"global": self.params, "detector": {"mock": self.params}},
            )


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

        # Missing required key no longer fails at BaseDetector level
        invalid_params = {"diff_threshold": 10}
        self.assertTrue(detector.validate_params(invalid_params))


if __name__ == "__main__":
    unittest.main()
