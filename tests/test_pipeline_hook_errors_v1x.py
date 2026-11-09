"""Hook failure policy, worker failures, and primary output error regressions."""

import contextlib
import io
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from meteor_core.exceptions import (
    MeteorConfigError,
    MeteorHookError,
    MeteorLoadError,
    MeteorWriteError,
)
from meteor_core.pipeline import _process_worker_batch, process_image_batch
from meteor_core.schema import HookConfig
from tests.pipeline_fixtures import (
    MemoryLoader,
    PipelineTestCase,
    RecordingDetector,
    RecordingHook,
    RecordingHookConfig,
    make_spawn_executor,
)


PIPELINE_CALLBACKS = (
    "on_file_found",
    "on_image_loaded",
    "on_detection_complete",
    "on_output_saved",
    "on_batch_results_sorted",
    "on_all_detections_sorted",
)
OUTPUT_CALLBACKS = (
    "on_detection_result",
    "on_candidate_detected",
    "on_batch_complete",
    "on_pipeline_complete",
)


class TestHookErrorPolicy(PipelineTestCase):
    def test_spawned_worker_hook_errors_keep_warn_and_raise_policy(self):
        for mode in ("warn", "raise"):
            with self.subTest(mode=mode):
                self.events.clear()
                pipeline = self.make_pipeline(
                    parallel=True,
                    mode=mode,
                    pairs=1,
                    hooks=[
                        RecordingHookConfig("failing", fail_on="on_detection_complete"),
                        RecordingHookConfig("following"),
                    ],
                )
                with patch(
                    "meteor_core.pipeline.ProcessPoolExecutor",
                    side_effect=make_spawn_executor,
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        if mode == "raise":
                            with self.assertRaises(MeteorHookError) as raised:
                                pipeline.run(resume=False)
                        else:
                            self.assertEqual(pipeline.run(resume=False), 1)

                if mode == "raise":
                    error = raised.exception
                    self.assertEqual(
                        error.context["hook_name"], "on_detection_complete"
                    )
                    self.assertIsInstance(error.original_error, ValueError)
                    self.assertEqual(str(error.original_error), "failing hook failure")
                    self.assertEqual(pipeline.progress_manager.processed_set, set())
                    self.assertFalse(
                        any(
                            event[1] == "on_all_detections_sorted"
                            for event in self.events
                        )
                    )
                else:
                    result = pipeline.output_handler.results[0]
                    self.assertEqual(result.extras["detection_hooks"], ["following"])
                    self.assertEqual(result.score, 43.0)
                    self.assertEqual(pipeline.progress_manager.processed_set, {"b.CR2"})

    def test_warn_logs_each_callback_failure_and_continues(self):
        for parallel in (False, True):
            for callback in (*PIPELINE_CALLBACKS, *OUTPUT_CALLBACKS):
                with self.subTest(parallel=parallel, callback=callback):
                    self.events.clear()
                    RecordingHook.instances.clear()
                    pipeline = self.make_pipeline(
                        parallel=parallel,
                        mode="warn",
                        pairs=1,
                        hooks=[
                            RecordingHookConfig("failing", fail_on=callback),
                            RecordingHookConfig("following"),
                        ],
                    )
                    pipeline.output_handler.fail_on = callback
                    if parallel:
                        executor = self.use_completed_executor()
                    with self.assertLogs("meteor_core.pipeline", "WARNING") as logs:
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(pipeline.run(resume=False), 1)

                    self.assertTrue(any(callback in line for line in logs.output))
                    self.assertTrue(
                        all(record.levelname == "WARNING" for record in logs.records)
                    )
                    self.assertEqual(pipeline.progress_manager.processed_set, {"b.CR2"})
                    self.assertIn(
                        ("following", "on_all_detections_sorted"),
                        [event[:2] for event in self.events],
                    )
                    if callback in PIPELINE_CALLBACKS:
                        self.assertIn(
                            ("following", callback),
                            [event[:2] for event in self.events],
                        )
                    if parallel:
                        executor.shutdown.assert_called_once_with(
                            wait=True, cancel_futures=False
                        )

    def test_raise_preserves_hook_name_and_original_cause_and_stops_chain(self):
        for parallel in (False, True):
            for callback in (*PIPELINE_CALLBACKS, *OUTPUT_CALLBACKS):
                with self.subTest(parallel=parallel, callback=callback):
                    self.events.clear()
                    RecordingHook.instances.clear()
                    pipeline = self.make_pipeline(
                        parallel=parallel,
                        mode="raise",
                        pairs=1,
                        hooks=[
                            RecordingHookConfig("failing", fail_on=callback),
                            RecordingHookConfig("following"),
                        ],
                    )
                    pipeline.output_handler.fail_on = callback
                    if parallel:
                        self.use_completed_executor()
                    with self.assertLogs("meteor_core.pipeline", "ERROR") as logs:
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(MeteorHookError) as raised:
                                pipeline.run(resume=False)

                    error = raised.exception
                    self.assertEqual(error.context["hook_name"], callback)
                    self.assertIs(error.__cause__, error.original_error)
                    if callback in PIPELINE_CALLBACKS:
                        failed_hooks = [
                            hook
                            for hook in RecordingHook.instances
                            if hook.config.fail_on == callback
                            and any(name == callback for name, _ in hook.calls)
                        ]
                        self.assertEqual(len(failed_hooks), 1)
                        self.assertIs(error.original_error, failed_hooks[0].failure)
                        self.assertNotIn(
                            ("following", callback),
                            [event[:2] for event in self.events],
                        )
                    else:
                        self.assertIs(
                            error.original_error, pipeline.output_handler.failure
                        )
                    self.assertTrue(any(callback in line for line in logs.output))
                    if callback == "on_image_loaded":
                        self.assertEqual(error.context["frame_role"], "current")
                        self.assertTrue(error.filepath.endswith("b.CR2"))
                    if callback != "on_all_detections_sorted":
                        self.assertFalse(
                            any(
                                event[1] == "on_all_detections_sorted"
                                for event in self.events
                            )
                        )

    def test_hook_configuration_errors_are_fatal_even_in_warn_mode(self):
        failure = ValueError("invalid test hook configuration")
        pipeline = self.make_pipeline(mode="warn", pairs=1)
        with patch("meteor_core.pipeline.HookRegistry.create", side_effect=failure):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(MeteorConfigError) as raised:
                    pipeline.run(resume=False)

        self.assertEqual(raised.exception.context["config_key"], "hooks")
        self.assertIs(raised.exception.original_error, failure)
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(self.events, [])

    def test_invalid_hook_schema_warns_and_passes_last_valid_value_to_next_hook(self):
        for callback in ("on_image_loaded", "on_detection_complete"):
            with self.subTest(callback=callback):
                original_callback = getattr(RecordingHook, callback)

                def invalid_version(hook, value, *args):
                    if hook.config.label == "invalid":
                        return replace(value, schema_version=999)
                    return original_callback(hook, value, *args)

                with patch.object(RecordingHook, callback, invalid_version):
                    with self.assertLogs("meteor_core.pipeline", "WARNING"):
                        results = process_image_batch(
                            [(1, "b.CR2", "a.CR2")],
                            np.ones((4, 4), dtype=np.uint8),
                            {},
                            input_loader=MemoryLoader(),
                            detector=RecordingDetector(),
                            hook_error_mode="warn",
                            hook_configs=[
                                HookConfig(
                                    RecordingHook.plugin_name,
                                    RecordingHookConfig(label),
                                )
                                for label in ("invalid", "following")
                            ],
                        )

                self.assertTrue(results[0][0])
                if callback == "on_image_loaded":
                    self.assertEqual(
                        results[0][8]["metadata"]["current"]["image_hooks"],
                        ["following"],
                    )
                else:
                    self.assertEqual(
                        results[0][7].extras["detection_hooks"], ["following"]
                    )


class TestWorkerAndOutputFailures(PipelineTestCase):
    def test_load_and_detector_errors_fail_one_pair_and_allow_later_pairs(self):
        for component in ("loader", "detector"):
            for failure in (
                MeteorLoadError("unreadable frame"),
                ValueError("bad pixels"),
            ):
                with self.subTest(component=component, failure=type(failure).__name__):
                    loader, detector = MemoryLoader(), RecordingDetector()
                    failing_component = loader if component == "loader" else detector
                    failing_component.errors["b.CR2"] = failure
                    with self.assertLogs("meteor_core.pipeline", "ERROR"):
                        results = process_image_batch(
                            [(1, "b.CR2", "a.CR2"), (3, "d.CR2", "c.CR2")],
                            np.ones((4, 4), dtype=np.uint8),
                            {},
                            input_loader=loader,
                            detector=detector,
                            hook_configs=[HookConfig(RecordingHook.plugin_name)],
                        )

                    self.assertEqual(
                        results[0],
                        (False, "b.CR2", "b.CR2", 0.0, None, 0.0, 0, None, None),
                    )
                    self.assertTrue(results[1][0])
                    self.assertEqual(results[1][1], "d.CR2")

    def test_sequential_primary_write_errors_propagate_in_both_hook_modes(self):
        for mode in ("warn", "raise"):
            with self.subTest(mode=mode):
                self.events.clear()
                pipeline = self.make_pipeline(mode=mode, pairs=1)
                failure = MeteorWriteError("cannot save candidate")
                pipeline.output_handler.save_error = failure
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(MeteorWriteError) as raised:
                        pipeline.run(resume=False)

                self.assertIs(raised.exception, failure)
                self.assertEqual(pipeline.progress_manager.processed_set, set())
                self.assertFalse(
                    any(
                        event[1]
                        in {
                            "on_output_saved",
                            "on_candidate_detected",
                            "on_batch_results_sorted",
                            "on_pipeline_complete",
                            "on_all_detections_sorted",
                        }
                        for event in self.events
                    )
                )

    def test_parallel_primary_write_errors_propagate_in_both_hook_modes(self):
        for mode in ("warn", "raise"):
            for error_type in (MeteorWriteError, OSError, ValueError):
                with self.subTest(mode=mode, error_type=error_type.__name__):
                    self.events.clear()
                    executor = self.use_completed_executor()
                    pipeline = self.make_pipeline(parallel=True, mode=mode, pairs=2)
                    pipeline.config.batch_size = 1
                    failure = error_type("cannot save candidate")
                    pipeline.output_handler.save_error = failure
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(error_type) as raised:
                            pipeline.run(resume=False)

                    self.assertIs(raised.exception, failure)
                    self.assertEqual(pipeline.progress_manager.processed_set, set())
                    self.assertEqual(
                        [
                            event[2]
                            for event in self.events
                            if event[1] == "save_candidate"
                        ],
                        ["b.CR2"],
                    )
                    self.assertFalse(
                        any(
                            event[1]
                            in {
                                "on_output_saved",
                                "on_candidate_detected",
                                "on_batch_results_sorted",
                                "on_pipeline_complete",
                                "on_all_detections_sorted",
                            }
                            for event in self.events
                        )
                    )
                    executor.shutdown.assert_called_once_with(
                        wait=True, cancel_futures=False
                    )

    def test_parallel_worker_batch_failure_does_not_stop_later_batches(self):
        self.use_completed_executor()
        pipeline = self.make_pipeline(parallel=True, pairs=2)
        pipeline.config.batch_size = 1

        def fail_first_batch(batch, *args):
            if batch[0][0] == 1:
                raise RuntimeError("worker batch failed")
            return _process_worker_batch(batch, *args)

        output = io.StringIO()
        with patch(
            "meteor_core.pipeline._process_worker_batch", side_effect=fail_first_batch
        ):
            with contextlib.redirect_stdout(output):
                self.assertEqual(pipeline.run(resume=False), 1)

        self.assertIn("worker batch failed", output.getvalue())
        self.assertEqual(pipeline.progress_manager.processed_set, {"c.CR2"})
        self.assertEqual(
            [event[2] for event in self.events if event[1] == "save_candidate"],
            ["c.CR2"],
        )

    def test_skipped_saves_still_notify_hooks_and_record_candidates(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                self.events.clear()
                pipeline = self.make_pipeline(parallel=parallel, pairs=1)
                pipeline.output_handler.saved = False
                if parallel:
                    self.use_completed_executor()
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(pipeline.run(resume=False), 1)

                self.assertIn(
                    ("first", "on_output_saved", False),
                    [event[:3] for event in self.events],
                )
                self.assertIn(
                    ("output", "on_candidate_detected", ("b.CR2", False)),
                    [event[:3] for event in self.events],
                )
                self.assertEqual(pipeline.progress_manager.detected_set, {"b.CR2"})

    def test_legacy_boolean_save_results_are_normalized_before_hook_notification(self):
        for saved in (False, True):
            with self.subTest(saved=saved):
                self.events.clear()
                pipeline = self.make_pipeline(pairs=1)
                pipeline.output_handler.legacy_bool = True
                pipeline.output_handler.saved = saved
                with self.assertWarnsRegex(DeprecationWarning, "legacy boolean"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(pipeline.run(resume=False), 1)
                self.assertIn(
                    ("first", "on_output_saved", saved),
                    [event[:3] for event in self.events],
                )
