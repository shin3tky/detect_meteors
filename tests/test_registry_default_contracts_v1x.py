#!/usr/bin/env python
#
# Detect Meteors CLI - Registry Default Contract Tests
# © 2025 Shinichi Morita (shin3tky)
#
"""Regression tests that align default creation behavior across registries."""

from dataclasses import dataclass
import sys
import types
import unittest

# Avoid OpenCV shared library dependency during unit tests that only exercise
# registry behavior.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace()

from meteor_core.detectors.base import BaseDetector
from meteor_core.detectors.registry import DetectorRegistry
from meteor_core.inputs.base import BaseInputLoader
from meteor_core.inputs.registry import LoaderRegistry
from meteor_core.outputs.base import BaseOutputHandler
from meteor_core.outputs.registry import OutputHandlerRegistry
from meteor_core.schema import (
    DEFAULT_DETECTOR_NAME,
    DEFAULT_LOADER_NAME,
    DEFAULT_OUTPUT_HANDLER_NAME,
    DetectionContext,
    DetectionResult,
)


@dataclass
class RequiredOnlyConfig:
    """Simple config that requires a positional argument."""

    required: int


class TestRegistryDefaultContracts(unittest.TestCase):
    """Ensure registry defaults and coercion failures behave consistently."""

    def tearDown(self):
        """Always reset all registries to avoid leaking custom plugins."""

        DetectorRegistry._reset()
        LoaderRegistry._reset()
        OutputHandlerRegistry._reset()

    def test_create_default_requires_configtype_for_all_registries(self):
        """Default creation raises TypeError when ConfigType is missing."""

        class DetectorWithoutConfig(BaseDetector):
            plugin_name = DEFAULT_DETECTOR_NAME

            def __init__(self, config):
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

            def compute_line_score(self, mask, hough_params):
                return 0.0, []

        class LoaderWithoutConfig(BaseInputLoader):
            plugin_name = DEFAULT_LOADER_NAME

            def __init__(self, config):
                self.config = config

            def load(self, filepath: str):
                return filepath

        class OutputWithoutConfig(BaseOutputHandler):
            plugin_name = DEFAULT_OUTPUT_HANDLER_NAME

            def __init__(self, config):
                self.config = config

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return filename

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return filename

        DetectorRegistry.register(DetectorWithoutConfig)
        LoaderRegistry.register(LoaderWithoutConfig)
        OutputHandlerRegistry.register(OutputWithoutConfig)

        with self.assertRaises(TypeError):
            DetectorRegistry.create_default()
        with self.assertRaises(TypeError):
            LoaderRegistry.create_default()
        with self.assertRaises(TypeError):
            OutputHandlerRegistry.create_default()

    def test_dict_coercion_errors_are_consistent(self):
        """Dict coercion failures surface as TypeError across registries."""

        class DetectorWithRequiredConfig(BaseDetector):
            plugin_name = "required_config_detector"
            ConfigType = RequiredOnlyConfig

            def __init__(self, config: RequiredOnlyConfig):
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

            def compute_line_score(self, mask, hough_params):
                return 0.0, []

        class LoaderWithRequiredConfig(BaseInputLoader):
            plugin_name = "required_config_loader"
            ConfigType = RequiredOnlyConfig

            def __init__(self, config: RequiredOnlyConfig):
                self.config = config

            def load(self, filepath: str):
                return filepath

        class OutputWithRequiredConfig(BaseOutputHandler):
            plugin_name = "required_config_output"
            ConfigType = RequiredOnlyConfig

            def __init__(self, config: RequiredOnlyConfig):
                self.config = config

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return filename

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return filename

        DetectorRegistry.register(DetectorWithRequiredConfig)
        LoaderRegistry.register(LoaderWithRequiredConfig)
        OutputHandlerRegistry.register(OutputWithRequiredConfig)

        with self.assertRaises(TypeError):
            DetectorRegistry.create("required_config_detector", {})
        with self.assertRaises(TypeError):
            LoaderRegistry.create("required_config_loader", {})
        with self.assertRaises(TypeError):
            OutputHandlerRegistry.create("required_config_output", {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
