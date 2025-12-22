import unittest
from dataclasses import dataclass

import numpy as np

from meteor_core import schema
from meteor_core.detectors import base as base_module


class TrackingDetector(base_module.BaseDetector[None]):
    plugin_name = "tracking"

    def __init__(self):
        self.received_context = None

    def detect(self, context: schema.DetectionContext) -> schema.DetectionResult:
        self.received_context = context
        return schema.DetectionResult(
            is_candidate=False,
            score=0.0,
            lines=[],
            aspect_ratio=0.0,
            debug_image=None,
        )

    def compute_line_score(self, mask: np.ndarray, hough_params):
        return 0.0, []


class PassThroughDetector(base_module.BaseDetector[None]):
    plugin_name = "passthrough"

    def detect(self, context: schema.DetectionContext) -> schema.DetectionResult:
        return base_module.BaseDetector.detect(self, context)

    def compute_line_score(self, mask: np.ndarray, hough_params):
        return base_module.BaseDetector.compute_line_score(self, mask, hough_params)


@dataclass
class ValidConfig:
    value: int = 1


class BaseDetectorTests(unittest.TestCase):
    def test_detect_legacy_builds_context_and_logs(self):
        detector = TrackingDetector()
        current = np.zeros((2, 2), dtype=np.uint8)
        previous = np.ones((2, 2), dtype=np.uint8)
        roi_mask = np.ones((2, 2), dtype=np.uint8)
        params = {"threshold": 1}

        with self.assertLogs("meteor_core.detectors.base", level="WARNING") as logs:
            result = detector.detect_legacy(current, previous, roi_mask, params)

        self.assertIsInstance(result, schema.DetectionResult)
        self.assertIn("Deprecated detect", logs.output[0])
        self.assertIsNotNone(detector.received_context)
        self.assertTrue((detector.received_context.current_image == current).all())
        self.assertTrue((detector.received_context.previous_image == previous).all())
        self.assertTrue((detector.received_context.roi_mask == roi_mask).all())
        self.assertEqual(
            detector.received_context.runtime_params,
            {"global": params, "detector": {"tracking": params}},
        )

    def test_build_runtime_params_uses_default_name(self):
        class EmptyNameDetector(base_module.BaseDetector[None]):
            plugin_name = ""

            def detect(self, context):
                return schema.DetectionResult(False, 0.0, [], 0.0, None)

            def compute_line_score(self, mask, hough_params):
                return 0.0, []

        detector = EmptyNameDetector()
        params = {"threshold": 2}
        runtime_params = detector.build_runtime_params(params)
        self.assertEqual(
            runtime_params,
            {"global": params, "detector": {schema.DEFAULT_DETECTOR_NAME: params}},
        )

    def test_split_runtime_params_valid(self):
        detector = TrackingDetector()
        runtime_params = {
            "global": {"foo": 1},
            "detector": {"tracking": {"bar": 2}},
        }
        global_params, detector_params = detector.split_runtime_params(runtime_params)
        self.assertEqual(global_params, {"foo": 1})
        self.assertEqual(detector_params, {"bar": 2})

    def test_split_runtime_params_requires_dict(self):
        detector = TrackingDetector()
        with self.assertRaises(TypeError):
            detector.split_runtime_params(["not", "a", "dict"])

    def test_split_runtime_params_namespace_types(self):
        detector = TrackingDetector()
        with self.assertRaises(TypeError):
            detector.split_runtime_params(
                {"global": ["bad"], "detector": {"tracking": {}}}
            )
        with self.assertRaises(TypeError):
            detector.split_runtime_params(
                {"global": {}, "detector": {"tracking": ["bad"]}}
            )

    def test_pass_through_abstract_methods(self):
        detector = PassThroughDetector()
        context = schema.DetectionContext(
            current_image=np.zeros((1, 1), dtype=np.uint8),
            previous_image=np.zeros((1, 1), dtype=np.uint8),
            roi_mask=np.ones((1, 1), dtype=np.uint8),
            runtime_params={"global": {}, "detector": {"passthrough": {}}},
            metadata={},
        )
        self.assertIsNone(detector.detect(context))
        self.assertIsNone(detector.compute_line_score(np.zeros((1, 1)), {}))

    def test_validate_params_requires_dict(self):
        detector = TrackingDetector()
        with self.assertRaises(TypeError):
            detector.validate_params(["bad"])


class DataclassDetectorTests(unittest.TestCase):
    def test_dataclass_detector_requires_dataclass_type(self):
        class NotADataclass:
            pass

        class BadDataclassDetector(base_module.DataclassDetector[NotADataclass]):
            plugin_name = "bad_dataclass"
            ConfigType = NotADataclass

            def detect(self, context):
                return schema.DetectionResult(False, 0.0, [], 0.0, None)

            def compute_line_score(self, mask, hough_params):
                return 0.0, []

        with self.assertRaises(TypeError):
            BadDataclassDetector(NotADataclass())

    def test_dataclass_detector_requires_matching_instance(self):
        class MismatchDetector(base_module.DataclassDetector[ValidConfig]):
            plugin_name = "mismatch"
            ConfigType = ValidConfig

            def detect(self, context):
                return schema.DetectionResult(False, 0.0, [], 0.0, None)

            def compute_line_score(self, mask, hough_params):
                return 0.0, []

        with self.assertRaises(TypeError):
            MismatchDetector("wrong")


class PydanticDetectorTests(unittest.TestCase):
    def test_pydantic_detector_import_error(self):
        class ImportErrorDetector(base_module.PydanticDetector[None]):
            plugin_name = "import_error"

            def detect(self, context):
                return schema.DetectionResult(False, 0.0, [], 0.0, None)

            def compute_line_score(self, mask, hough_params):
                return 0.0, []

        original_base_model = base_module.BaseModel
        try:
            base_module.BaseModel = None
            with self.assertRaises(ImportError):
                ImportErrorDetector(None)
        finally:
            base_module.BaseModel = original_base_model

    def test_pydantic_detector_requires_base_model(self):
        if base_module.BaseModel is None:
            self.skipTest("pydantic is not installed")

        class BadPydanticDetector(base_module.PydanticDetector[int]):
            plugin_name = "bad_pydantic"
            ConfigType = int

            def detect(self, context):
                return schema.DetectionResult(False, 0.0, [], 0.0, None)

            def compute_line_score(self, mask, hough_params):
                return 0.0, []

        with self.assertRaises(TypeError):
            BadPydanticDetector(1)

    def test_pydantic_detector_requires_matching_instance(self):
        if base_module.BaseModel is None:
            self.skipTest("pydantic is not installed")

        class Model(base_module.BaseModel):
            value: int

        class MismatchDetector(base_module.PydanticDetector[Model]):
            plugin_name = "pydantic_mismatch"
            ConfigType = Model

            def detect(self, context):
                return schema.DetectionResult(False, 0.0, [], 0.0, None)

            def compute_line_score(self, mask, hough_params):
                return 0.0, []

        with self.assertRaises(TypeError):
            MismatchDetector("wrong")


if __name__ == "__main__":
    unittest.main()
