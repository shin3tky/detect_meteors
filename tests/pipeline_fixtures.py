"""Small recording plugins shared by pipeline regression tests."""

import multiprocessing
import os
import tempfile
import unittest
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from meteor_core.detectors.base import BaseDetector
from meteor_core.hooks import DataclassHook, HookRegistry
from meteor_core.inputs.base import BaseInputLoader
from meteor_core.outputs.base import BaseOutputHandler
from meteor_core.pipeline import MeteorDetectionPipeline, _init_worker_ignore_interrupt
from meteor_core.schema import DetectionResult, HookConfig, InputContext, OutputResult
from meteor_core.schema import PipelineConfig


@dataclass
class RecordingHookConfig:
    label: str = "first"
    fail_on: str | None = None
    reject_filename: str | None = None


class RecordingHook(DataclassHook[RecordingHookConfig]):
    plugin_name = "test_pipeline_recording_hook"
    ConfigType = RecordingHookConfig
    events = []
    instances = []

    def __init__(self, config):
        super().__init__(config)
        self.calls = []
        self.image_calls = 0
        self.failure = ValueError(f"{config.label} hook failure")
        self.instances.append(self)

    def _record(self, name, detail):
        self.calls.append((name, detail))
        self.events.append((self.config.label, name, detail, os.getpid()))
        if self.config.fail_on == name:
            raise self.failure

    def on_file_found(self, filepath):
        self._record("on_file_found", filepath)
        return Path(filepath).name != self.config.reject_filename

    def on_image_loaded(self, context):
        self.image_calls += 1
        role = context.metadata["frame_role"]
        self._record("on_image_loaded", (Path(context.filepath).name, role))
        changes = [*context.metadata.get("image_hooks", []), self.config.label]
        return replace(
            context,
            metadata={
                **context.metadata,
                "image_hooks": changes,
                "hook_pid": os.getpid(),
                "image_calls": self.image_calls,
            },
        )

    def on_detection_complete(self, result, context):
        self._record("on_detection_complete", context.metadata["frame_index"])
        return replace(
            result,
            score=result.score + 1,
            extras={
                **result.extras,
                "detection_hooks": [
                    *result.extras.get("detection_hooks", []),
                    self.config.label,
                ],
                "detection_hook_pid": os.getpid(),
            },
        )

    def on_output_saved(self, result):
        self._record("on_output_saved", result.saved)

    def on_batch_results_sorted(self, detections):
        self._record(
            "on_batch_results_sorted", tuple(item.frame_index for item in detections)
        )
        return [
            replace(
                item,
                extras={
                    **item.extras,
                    "batch_hooks": [
                        *item.extras.get("batch_hooks", []),
                        self.config.label,
                    ],
                },
            )
            for item in detections
        ]

    def on_all_detections_sorted(self, detections):
        self._record(
            "on_all_detections_sorted", tuple(item.frame_index for item in detections)
        )
        self.final_detections = detections
        return [
            replace(
                item,
                extras={
                    **item.extras,
                    "final_hooks": [
                        *item.extras.get("final_hooks", []),
                        self.config.label,
                    ],
                },
            )
            for item in detections
        ]


class MemoryLoader(BaseInputLoader[None]):
    plugin_name = "test_memory_loader"

    def __init__(self):
        self.errors = {}

    def load(self, filepath):
        error = self.errors.get(Path(filepath).name)
        if error is not None:
            raise error
        return InputContext(
            image_data=np.zeros((4, 4), dtype=np.uint16),
            filepath=filepath,
            metadata={"filename": Path(filepath).name},
        )


class RecordingDetector(BaseDetector[None]):
    plugin_name = "test_recording_detector"

    def __init__(self):
        self.errors = {}
        self.non_candidates = set()

    def detect(self, context):
        filename = context.metadata["current"]["filename"]
        error = self.errors.get(filename)
        if error is not None:
            raise error
        return DetectionResult(
            is_candidate=filename not in self.non_candidates,
            score=42.0,
            lines=[(0, 0, 3, 3)],
            aspect_ratio=3.0,
            debug_image=np.zeros((4, 4, 3), dtype=np.uint8),
        )

    def compute_line_score(self, mask, hough_params):
        return 0.0, []


class RecordingOutput(BaseOutputHandler):
    plugin_name = "test_recording_output"

    def __init__(self, events):
        self.events = events
        self.contexts = []
        self.results = []
        self.fail_on = None
        self.failure = ValueError("output callback failure")
        self.save_error = None
        self.saved = True
        self.legacy_bool = False

    def _record(self, name, detail):
        self.events.append(("output", name, detail, os.getpid()))
        if self.fail_on == name:
            raise self.failure

    def on_detection_result(self, context, result, filepath):
        self.contexts.append(context)
        self.results.append(result)
        self._record("on_detection_result", Path(filepath).name)

    def save_candidate(self, source_path, filename, debug_image=None, roi_polygon=None):
        self._record("save_candidate", filename)
        if self.save_error is not None:
            raise self.save_error
        if self.legacy_bool:
            return self.saved
        return OutputResult(saved=self.saved, output_path=source_path, debug_path=None)

    def save_debug_image(self, debug_image, filename, roi_polygon=None):
        return filename

    def on_candidate_detected(self, filename, saved, score=0.0, aspect_ratio=0.0):
        self._record("on_candidate_detected", (filename, saved))

    def on_batch_complete(self, processed_count, detected_count, batch_size):
        self._record("on_batch_complete", (processed_count, detected_count, batch_size))

    def on_pipeline_complete(self, total_processed, total_detected, elapsed_seconds):
        self._record("on_pipeline_complete", (total_processed, total_detected))


class CompletedBatchExecutor:
    """Complete tasks immediately; tests choose their delivery order explicitly.

    This double exercises orchestration, not process isolation. A separate test
    uses a real spawn pool to check the worker/main-process boundary.
    """

    def __init__(self):
        self.futures = []
        self.shutdown = Mock()

    def submit(self, function, *args):
        future = Future()
        future.cancel = Mock(wraps=future.cancel)
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            # Deliver worker failures (including Ctrl-C) through future.result().
            future.set_exception(exc)
        self.futures.append(future)
        return future


def initialize_worker_hooks():
    """Bootstrap the test hook explicitly; runtime registration is not inherited."""
    _init_worker_ignore_interrupt()
    HookRegistry.register(RecordingHook)


def make_spawn_executor(*, max_workers, initializer):
    """Use fresh workers with the test plugin explicitly bootstrapped."""
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker_hooks,
    )


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = []
        self.enterContext(patch.object(RecordingHook, "events", self.events))
        self.enterContext(patch.object(RecordingHook, "instances", []))
        self.enterContext(patch.dict(HookRegistry._custom))
        HookRegistry.register(RecordingHook)
        self.run_number = 0

    def make_pipeline(self, *, parallel=False, mode="raise", hooks=None, pairs=3):
        self.run_number += 1
        root = self.root / str(self.run_number)
        target = root / "raw"
        target.mkdir(parents=True)
        for index in range(pairs + 1):
            (target / f"{chr(ord('a') + index)}.CR2").touch()
        if hooks is None:
            hooks = [RecordingHookConfig()]
        config = PipelineConfig.with_defaults(
            target_folder=str(target),
            output_folder=str(root / "output"),
            debug_folder=str(root / "debug"),
        )
        config.enable_parallel = parallel
        config.num_workers = 2 if parallel else 1
        config.batch_size = 2
        config.progress_file = str(root / "progress.json")
        config.enable_roi_selection = False
        config.hook_error_mode = mode
        config.hooks = [HookConfig(RecordingHook.plugin_name, item) for item in hooks]
        return MeteorDetectionPipeline(
            config,
            input_loader=MemoryLoader(),
            detector=RecordingDetector(),
            output_handler=RecordingOutput(self.events),
        )

    def use_completed_executor(self, *, reverse=False):
        executor = CompletedBatchExecutor()
        self.enterContext(
            patch("meteor_core.pipeline.ProcessPoolExecutor", return_value=executor)
        )
        self.enterContext(
            patch(
                "meteor_core.pipeline.as_completed",
                side_effect=lambda futures: iter(futures[::-1] if reverse else futures),
            )
        )
        return executor
