#!/usr/bin/env python
#
# Detect Meteors CLI - Detector Registry Tests
# © 2025 Shinichi Morita (shin3tky)
#

"""
Unit tests for DetectorRegistry.
"""

import unittest
import warnings
from typing import Dict, List, Tuple

import numpy as np

from meteor_core.detectors.registry import DetectorRegistry
from meteor_core.detectors.base import BaseDetector
from meteor_core.schema import DEFAULT_DETECTOR_NAME, DetectionContext, DetectionResult

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - optional dependency
    BaseModel = None


class TestDetectorRegistryDiscovery(unittest.TestCase):
    """Tests for DetectorRegistry discovery functionality."""

    def setUp(self):
        """Reset registry before each test."""
        DetectorRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        DetectorRegistry._reset()

    def test_discover_returns_dict(self):
        """discover() returns a dictionary."""
        detectors = DetectorRegistry.discover()
        self.assertIsInstance(detectors, dict)

    def test_discover_includes_builtin_hough(self):
        """Built-in hough detector is always discovered."""
        detectors = DetectorRegistry.discover()
        self.assertIn("hough", detectors)

    def test_discover_includes_simple_threshold(self):
        """Built-in simple_threshold detector is discovered."""
        detectors = DetectorRegistry.discover()
        self.assertIn("simple_threshold", detectors)

    def test_discover_caches_result(self):
        """discover() caches results for subsequent calls."""
        detectors1 = DetectorRegistry.discover()
        detectors2 = DetectorRegistry.discover()
        self.assertIs(detectors1, detectors2)

    def test_discover_force_refreshes_cache(self):
        """discover(force=True) refreshes the cache."""
        detectors1 = DetectorRegistry.discover()
        detectors2 = DetectorRegistry.discover(force=True)
        # Should be equal content but different object
        self.assertEqual(detectors1.keys(), detectors2.keys())


class TestDetectorRegistryGet(unittest.TestCase):
    """Tests for DetectorRegistry.get() method."""

    def setUp(self):
        """Reset registry before each test."""
        DetectorRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        DetectorRegistry._reset()

    def test_get_existing_detector(self):
        """get() returns detector class for existing name."""
        detector_cls = DetectorRegistry.get("hough")
        self.assertEqual(detector_cls.plugin_name, "hough")

    def test_get_case_insensitive(self):
        """get() is case-insensitive."""
        detector_cls1 = DetectorRegistry.get("hough")
        detector_cls2 = DetectorRegistry.get("HOUGH")
        detector_cls3 = DetectorRegistry.get("Hough")
        self.assertIs(detector_cls1, detector_cls2)
        self.assertIs(detector_cls2, detector_cls3)

    def test_get_nonexistent_raises_keyerror(self):
        """get() raises KeyError for unknown detector name."""
        with self.assertRaises(KeyError) as ctx:
            DetectorRegistry.get("nonexistent_detector")

        # Error message should include available detectors
        error_msg = str(ctx.exception)
        self.assertIn("nonexistent_detector", error_msg)
        self.assertIn("Available", error_msg)

    def test_get_custom_takes_priority(self):
        """Runtime-registered detector takes priority over discovered."""

        class CustomHoughDetector(BaseDetector):
            plugin_name = "hough"

            def detect(self, context: DetectionContext) -> DetectionResult:
                return DetectionResult(
                    is_candidate=False,
                    score=0.0,
                    lines=[],
                    aspect_ratio=0.0,
                    debug_image=None,
                    extras={},
                )

            def compute_line_score(
                self, mask: np.ndarray, hough_params: Dict[str, int]
            ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
                return 0.0, []

        DetectorRegistry.register(CustomHoughDetector)
        detector_cls = DetectorRegistry.get("hough")
        self.assertIs(detector_cls, CustomHoughDetector)


class TestDetectorRegistryRegister(unittest.TestCase):
    """Tests for DetectorRegistry.register() method."""

    def setUp(self):
        """Reset registry before each test."""
        DetectorRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        DetectorRegistry._reset()

    def test_register_valid_detector(self):
        """register() adds valid detector to registry."""

        class MockDetector(BaseDetector):
            plugin_name = "mock"

            def detect(self, context: DetectionContext) -> DetectionResult:
                return DetectionResult(
                    is_candidate=False,
                    score=0.0,
                    lines=[],
                    aspect_ratio=0.0,
                    debug_image=None,
                    extras={},
                )

            def compute_line_score(
                self, mask: np.ndarray, hough_params: Dict[str, int]
            ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
                return 0.0, []

        DetectorRegistry.register(MockDetector)
        detector_cls = DetectorRegistry.get("mock")
        self.assertIs(detector_cls, MockDetector)

    def test_register_invalid_class_raises(self):
        """register() raises ValueError for invalid class."""

        class NotADetector:
            plugin_name = "invalid"

        with self.assertRaises(ValueError) as ctx:
            DetectorRegistry.register(NotADetector)  # type: ignore

        self.assertIn("Invalid detector class", str(ctx.exception))

    def test_register_empty_plugin_name_raises(self):
        """register() raises ValueError for empty plugin_name."""

        class EmptyNameDetector(BaseDetector):
            plugin_name = ""

            def detect(self, context: DetectionContext) -> DetectionResult:
                return DetectionResult(
                    is_candidate=False,
                    score=0.0,
                    lines=[],
                    aspect_ratio=0.0,
                    debug_image=None,
                    extras={},
                )

            def compute_line_score(
                self, mask: np.ndarray, hough_params: Dict[str, int]
            ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
                return 0.0, []

        with self.assertRaises(ValueError) as ctx:
            DetectorRegistry.register(EmptyNameDetector)

        self.assertIn("non-empty plugin_name", str(ctx.exception))

    def test_register_overwrite_warns(self):
        """register() warns when overwriting existing detector."""

        class MockDetector1(BaseDetector):
            plugin_name = "mock"

            def detect(self, context: DetectionContext) -> DetectionResult:
                return DetectionResult(
                    is_candidate=False,
                    score=0.0,
                    lines=[],
                    aspect_ratio=0.0,
                    debug_image=None,
                    extras={},
                )

            def compute_line_score(
                self, mask: np.ndarray, hough_params: Dict[str, int]
            ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
                return 0.0, []

        class MockDetector2(BaseDetector):
            plugin_name = "mock"

            def detect(self, context: DetectionContext) -> DetectionResult:
                return DetectionResult(
                    is_candidate=True,
                    score=1.0,
                    lines=[],
                    aspect_ratio=1.0,
                    debug_image=None,
                    extras={},
                )

            def compute_line_score(
                self, mask: np.ndarray, hough_params: Dict[str, int]
            ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
                return 1.0, []

        DetectorRegistry.register(MockDetector1)

        with self.assertWarns(UserWarning) as ctx:
            DetectorRegistry.register(MockDetector2)

        self.assertIn("Overwriting", str(ctx.warning))


class TestDetectorRegistryUnregister(unittest.TestCase):
    """Tests for DetectorRegistry.unregister() method."""

    def setUp(self):
        """Reset registry before each test."""
        DetectorRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        DetectorRegistry._reset()

    def test_unregister_custom_detector(self):
        """unregister() removes runtime-registered detector."""

        class MockDetector(BaseDetector):
            plugin_name = "mock"

            def detect(self, context: DetectionContext) -> DetectionResult:
                return DetectionResult(
                    is_candidate=False,
                    score=0.0,
                    lines=[],
                    aspect_ratio=0.0,
                    debug_image=None,
                    extras={},
                )

            def compute_line_score(
                self, mask: np.ndarray, hough_params: Dict[str, int]
            ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
                return 0.0, []

        DetectorRegistry.register(MockDetector)
        self.assertTrue(DetectorRegistry.unregister("mock"))

        with self.assertRaises(KeyError):
            DetectorRegistry.get("mock")

    def test_unregister_nonexistent_returns_false(self):
        """unregister() returns False for non-registered detector."""
        result = DetectorRegistry.unregister("never_registered")
        self.assertFalse(result)

    def test_unregister_discovered_returns_false(self):
        """unregister() cannot remove discovered detectors."""
        # "hough" is discovered, not runtime-registered
        result = DetectorRegistry.unregister("hough")
        self.assertFalse(result)

        # "hough" should still be available
        detector_cls = DetectorRegistry.get("hough")
        self.assertEqual(detector_cls.plugin_name, "hough")


class TestDetectorRegistryListAvailable(unittest.TestCase):
    """Tests for DetectorRegistry.list_available() method."""

    def setUp(self):
        """Reset registry before each test."""
        DetectorRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        DetectorRegistry._reset()

    def test_list_available_includes_discovered(self):
        """list_available() includes discovered detectors."""
        available = DetectorRegistry.list_available()
        self.assertIn("hough", available)

    def test_list_available_includes_custom(self):
        """list_available() includes runtime-registered detectors."""

        class MockDetector(BaseDetector):
            plugin_name = "mock"

            def detect(self, context: DetectionContext) -> DetectionResult:
                return DetectionResult(
                    is_candidate=False,
                    score=0.0,
                    lines=[],
                    aspect_ratio=0.0,
                    debug_image=None,
                    extras={},
                )

            def compute_line_score(
                self, mask: np.ndarray, hough_params: Dict[str, int]
            ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
                return 0.0, []

        DetectorRegistry.register(MockDetector)
        available = DetectorRegistry.list_available()

        self.assertIn("hough", available)
        self.assertIn("mock", available)

    def test_list_available_is_sorted(self):
        """list_available() returns sorted list."""
        available = DetectorRegistry.list_available()
        self.assertEqual(available, sorted(available))


class TestDetectorRegistryCreate(unittest.TestCase):
    """Tests for DetectorRegistry.create() method."""

    def setUp(self):
        """Reset registry before each test."""
        DetectorRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        DetectorRegistry._reset()

    def test_create_hough_detector(self):
        """create() creates hough detector instance without config."""
        detector = DetectorRegistry.create("hough")
        self.assertEqual(detector.plugin_name, "hough")

    def test_create_nonexistent_raises(self):
        """create() raises KeyError for unknown detector."""
        with self.assertRaises(KeyError):
            DetectorRegistry.create("nonexistent_detector")

    def test_create_dataclass_config_detector(self):
        """create() coerces dict config for dataclass-based detectors."""
        detector = DetectorRegistry.create(
            "simple_threshold", {"diff_threshold": 7, "min_area": 2}
        )

        self.assertEqual(detector.config.diff_threshold, 7)
        self.assertEqual(detector.config.min_area, 2)

    def test_create_dataclass_config_default(self):
        """create() uses default config when none is provided."""
        detector = DetectorRegistry.create("simple_threshold")
        self.assertEqual(detector.config.diff_threshold, 5)
        self.assertEqual(detector.config.min_area, 1)

    def test_create_invalid_dataclass_config_raises(self):
        """create() raises TypeError when config dict mismatches dataclass."""
        with self.assertRaises(TypeError):
            DetectorRegistry.create("simple_threshold", {"unknown": 1})

    def test_create_pydantic_config_detector(self):
        """create() handles Pydantic-configured detectors when available."""

        if BaseModel is None:
            self.skipTest("pydantic not installed")

        class PydanticConfig(BaseModel):  # type: ignore[misc]
            flag: bool = False

            class Config:
                extra = "forbid"

        class PydanticConfiguredDetector(BaseDetector):
            plugin_name = "pydantic_configured"
            ConfigType = PydanticConfig

            def __init__(self, config: PydanticConfig) -> None:
                self.config = config

            def detect(self, context: DetectionContext) -> DetectionResult:
                return DetectionResult(
                    is_candidate=False,
                    score=0.0,
                    lines=[],
                    aspect_ratio=0.0,
                    debug_image=None,
                    extras={},
                )

            def compute_line_score(
                self, mask: np.ndarray, hough_params: Dict[str, int]
            ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
                return 0.0, []

        DetectorRegistry.register(PydanticConfiguredDetector)

        detector = DetectorRegistry.create("pydantic_configured", {"flag": True})
        self.assertTrue(detector.config.flag)


class TestDetectorRegistryCreateDefault(unittest.TestCase):
    """Tests for DetectorRegistry.create_default() method."""

    def setUp(self):
        """Reset registry before each test."""
        DetectorRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        DetectorRegistry._reset()

    def test_create_default_returns_hough_detector(self):
        """create_default() returns hough detector."""
        detector = DetectorRegistry.create_default()
        self.assertEqual(detector.plugin_name, DEFAULT_DETECTOR_NAME)


class TestDetectorRegistryReset(unittest.TestCase):
    """Tests for DetectorRegistry._reset() method."""

    def test_reset_clears_discovered_cache(self):
        """_reset() clears discovered detector cache."""
        # Trigger discovery
        DetectorRegistry.discover()
        self.assertIsNotNone(DetectorRegistry._discovered)

        DetectorRegistry._reset()
        self.assertIsNone(DetectorRegistry._discovered)

    def test_reset_clears_custom_detectors(self):
        """_reset() clears runtime-registered detectors."""

        class MockDetector(BaseDetector):
            plugin_name = "mock"

            def detect(self, context: DetectionContext) -> DetectionResult:
                return DetectionResult(
                    is_candidate=False,
                    score=0.0,
                    lines=[],
                    aspect_ratio=0.0,
                    debug_image=None,
                    extras={},
                )

            def compute_line_score(
                self, mask: np.ndarray, hough_params: Dict[str, int]
            ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
                return 0.0, []

        DetectorRegistry.register(MockDetector)
        self.assertEqual(len(DetectorRegistry._custom), 1)

        DetectorRegistry._reset()
        self.assertEqual(len(DetectorRegistry._custom), 0)


class TestDiscoverDetectorsDeprecation(unittest.TestCase):
    """Tests for deprecated discover_detectors function."""

    def setUp(self):
        """Reset registry before each test."""
        DetectorRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        DetectorRegistry._reset()

    def test_discover_detectors_warns(self):
        """discover_detectors() emits DeprecationWarning."""
        from meteor_core.detectors.discovery import discover_detectors

        with self.assertWarns(DeprecationWarning) as ctx:
            discover_detectors()

        self.assertIn("deprecated", str(ctx.warning))
        self.assertIn("DetectorRegistry", str(ctx.warning))

    def test_discover_detectors_returns_same_result(self):
        """discover_detectors() returns same detectors as registry."""
        from meteor_core.detectors.discovery import discover_detectors

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            deprecated_detectors = discover_detectors()

        registry_detectors = DetectorRegistry.discover()

        self.assertEqual(
            set(deprecated_detectors.keys()), set(registry_detectors.keys())
        )


if __name__ == "__main__":
    unittest.main()
