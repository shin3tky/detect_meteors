"""
Pipeline execution coverage tests for meteor_core.pipeline (v1.x).
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core.exceptions import MeteorConfigError, MeteorLoadError  # noqa: E402
from meteor_core.pipeline import (  # noqa: E402
    MeteorDetectionPipeline,
    _normalize_detection_context,
    process_image_batch,
)
from meteor_core.schema import (  # noqa: E402
    DetectionContext,
    DetectionParams,
    DetectionResult,
    InputContext,
    OutputResult,
    PipelineConfig,
)
from meteor_core.detectors.base import BaseDetector  # noqa: E402
from dataclasses import dataclass

from meteor_core.outputs.base import BaseOutputHandler  # noqa: E402
from meteor_core.hooks import DataclassHook, HookRegistry  # noqa: E402


class DummyInputLoader:
    """Minimal input loader stub."""

    def __init__(self, image_map, error_map=None):
        self.image_map = image_map
        self.error_map = error_map or {}

    def load(self, filepath):
        error = self.error_map.get(filepath)
        if error:
            raise error
        return InputContext(
            image_data=self.image_map[filepath],
            filepath=filepath,
            metadata={"source": os.path.basename(filepath)},
        )


class DummyDetector(BaseDetector[None]):
    plugin_name = "dummy"

    def detect(self, context: DetectionContext) -> DetectionResult:
        return DetectionResult(
            is_candidate=True,
            score=42.0,
            lines=[(0, 0, 1, 1)],
            aspect_ratio=1.5,
            debug_image=None,
        )

    def compute_line_score(self, mask, hough_params):
        return 0.0, []


class DummyOutputHandler(BaseOutputHandler):
    """Minimal output handler stub."""

    def __init__(self):
        self.saved_candidates = []

    def save_candidate(self, source_path, filename, debug_image=None, roi_polygon=None):
        self.saved_candidates.append((source_path, filename, roi_polygon))
        return OutputResult(saved=True, output_path=source_path, debug_path=None)

    def save_debug_image(self, debug_image, filename, roi_polygon=None):
        return "debug.png"


@dataclass
class DummyImageLoadedConfig:
    """Configuration for DummyImageLoadedHook."""


class DummyImageLoadedHook(DataclassHook[DummyImageLoadedConfig]):
    plugin_name = "dummy_image_loaded"
    name = "Dummy Image Loaded Hook"
    ConfigType = DummyImageLoadedConfig

    def on_file_found(self, filepath: str) -> bool:
        return True

    def on_image_loaded(self, context: InputContext) -> InputContext:
        metadata = dict(context.metadata)
        metadata["hook_called"] = True
        metadata["hook_role"] = metadata.get("frame_role")
        return InputContext(
            image_data=context.image_data,
            filepath=context.filepath,
            metadata=metadata,
            loader_info=context.loader_info,
        )


@dataclass
class DummyDetectionCompleteConfig:
    """Configuration for DummyDetectionCompleteHook."""


class DummyDetectionCompleteHook(DataclassHook[DummyDetectionCompleteConfig]):
    plugin_name = "dummy_detection_complete"
    name = "Dummy Detection Complete Hook"
    ConfigType = DummyDetectionCompleteConfig

    def on_file_found(self, filepath: str) -> bool:
        return True

    def on_detection_complete(
        self,
        result: DetectionResult,
        context: DetectionContext,
    ) -> DetectionResult:
        return DetectionResult(
            is_candidate=False,
            score=0.0,
            lines=[],
            aspect_ratio=result.aspect_ratio,
            debug_image="modified",
            extras={"hooked": True},
            metrics=result.metrics,
        )


class TestNormalizeDetectionContext(unittest.TestCase):
    def test_normalize_detection_context_unsupported_version(self):
        context = DetectionContext(
            current_image=np.zeros((2, 2), dtype=np.uint8),
            previous_image=np.zeros((2, 2), dtype=np.uint8),
            roi_mask=np.ones((2, 2), dtype=np.uint8),
            runtime_params={"global": {}},
            metadata={},
            schema_version=999,
        )

        with self.assertRaises(MeteorConfigError):
            _normalize_detection_context(context)


class TestProcessImageBatch(unittest.TestCase):
    def test_process_image_batch_success_and_errors(self):
        img = np.zeros((2, 2), dtype=np.uint8)
        img_map = {
            "a.CR2": img,
            "b.CR2": img,
            "c.CR2": img,
        }
        error_map = {
            "b.CR2": MeteorLoadError("boom", filepath="b.CR2"),
            "c.CR2": ValueError("bad"),
        }
        loader = DummyInputLoader(img_map, error_map=error_map)
        detector = DummyDetector()

        # batch_data: List[Tuple[int, str, str]] = (frame_index, curr_file, prev_file)
        results = process_image_batch(
            [(1, "a.CR2", "a.CR2"), (2, "c.CR2", "a.CR2")],
            roi_mask=np.ones((2, 2), dtype=np.uint8),
            params={"diff_threshold": 8},
            input_loader=loader,
            detector=detector,
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0][0])
        self.assertEqual(results[0][1], "a.CR2")
        self.assertFalse(results[1][0])
        self.assertEqual(results[1][1], "c.CR2")

    def test_process_image_batch_calls_image_loaded_hook(self):
        img = np.zeros((2, 2), dtype=np.uint8)
        img_map = {
            "a.CR2": img,
            "b.CR2": img,
        }
        loader = DummyInputLoader(img_map)
        detector = DummyDetector()
        HookRegistry.register(DummyImageLoadedHook)
        try:
            results = process_image_batch(
                [(1, "a.CR2", "b.CR2")],
                roi_mask=np.ones((2, 2), dtype=np.uint8),
                params={"diff_threshold": 8},
                input_loader=loader,
                detector=detector,
            )
        finally:
            HookRegistry.unregister(DummyImageLoadedHook.plugin_name)

        context_payload = results[0][8]
        self.assertTrue(context_payload["metadata"]["current"]["hook_called"])
        self.assertTrue(context_payload["metadata"]["previous"]["hook_called"])
        self.assertEqual(context_payload["metadata"]["current"]["hook_role"], "current")
        self.assertEqual(
            context_payload["metadata"]["previous"]["hook_role"], "previous"
        )

    def test_process_image_batch_calls_detection_complete_hook(self):
        img = np.zeros((2, 2), dtype=np.uint8)
        img_map = {
            "a.CR2": img,
            "b.CR2": img,
        }
        loader = DummyInputLoader(img_map)
        detector = DummyDetector()
        HookRegistry.register(DummyDetectionCompleteHook)
        try:
            results = process_image_batch(
                [(1, "a.CR2", "b.CR2")],
                roi_mask=np.ones((2, 2), dtype=np.uint8),
                params={"diff_threshold": 8},
                input_loader=loader,
                detector=detector,
            )
        finally:
            HookRegistry.unregister(DummyDetectionCompleteHook.plugin_name)

        self.assertFalse(results[0][0])
        self.assertEqual(results[0][3], 0.0)
        self.assertIsNone(results[0][4])
        self.assertEqual(results[0][7].extras.get("hooked"), True)


class TestPipelineRun(unittest.TestCase):
    def test_pipeline_run_same_target_and_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PipelineConfig(
                target_folder=temp_dir,
                output_folder=temp_dir,
                debug_folder=temp_dir,
                params=DetectionParams(),
                enable_parallel=False,
            )
            pipeline = MeteorDetectionPipeline(
                config, output_handler=DummyOutputHandler()
            )
            self.assertEqual(pipeline.run(enable_roi_selection=False), 0)

    def test_pipeline_run_sequential_saves_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = os.path.join(temp_dir, "raw")
            output_dir = os.path.join(temp_dir, "out")
            debug_dir = os.path.join(temp_dir, "debug")
            os.makedirs(target_dir)
            os.makedirs(output_dir)
            os.makedirs(debug_dir)

            file_a = os.path.join(target_dir, "a.CR2")
            file_b = os.path.join(target_dir, "b.CR2")
            open(file_a, "a", encoding="utf-8").close()
            open(file_b, "a", encoding="utf-8").close()

            img = np.zeros((2, 2), dtype=np.uint8)
            loader = DummyInputLoader({file_a: img, file_b: img})
            handler = DummyOutputHandler()
            detector = DummyDetector()

            config = PipelineConfig(
                target_folder=target_dir,
                output_folder=output_dir,
                debug_folder=debug_dir,
                params=DetectionParams(),
                enable_parallel=False,
                num_workers=1,
                progress_file=os.path.join(temp_dir, "progress.json"),
            )

            pipeline = MeteorDetectionPipeline(
                config,
                input_loader=loader,
                output_handler=handler,
                detector=detector,
            )

            detected = pipeline.run(enable_roi_selection=False, resume=False)
            self.assertEqual(detected, 1)
            self.assertEqual(len(handler.saved_candidates), 1)


if __name__ == "__main__":
    unittest.main()
