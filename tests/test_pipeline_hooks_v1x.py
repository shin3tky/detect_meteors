"""Regression coverage for observable pipeline hook order and lifetime."""

import contextlib
import io
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from meteor_core.exceptions import MeteorLoadError
from meteor_core.pipeline import _apply_output_saved_hooks, process_image_batch
from meteor_core.schema import HookConfig, OutputResult
from meteor_core.utils import compute_params_hash
from tests.pipeline_fixtures import (
    MemoryLoader,
    PipelineTestCase,
    RecordingDetector,
    RecordingHook,
    RecordingHookConfig,
    make_spawn_executor,
)


class TestImagePairHooks(PipelineTestCase):
    def test_replacement_contexts_and_results_follow_configured_hook_order(self):
        results = process_image_batch(
            [(1, "b.CR2", "a.CR2")],
            np.ones((4, 4), dtype=np.uint8),
            {},
            input_loader=MemoryLoader(),
            detector=RecordingDetector(),
            hook_configs=[
                HookConfig(RecordingHook.plugin_name, RecordingHookConfig(label))
                for label in ("first", "second")
            ],
        )

        self.assertEqual(
            [event[:3] for event in self.events],
            [
                ("first", "on_image_loaded", ("b.CR2", "current")),
                ("second", "on_image_loaded", ("b.CR2", "current")),
                ("first", "on_image_loaded", ("a.CR2", "previous")),
                ("second", "on_image_loaded", ("a.CR2", "previous")),
                ("first", "on_detection_complete", 1),
                ("second", "on_detection_complete", 1),
            ],
        )
        result, payload = results[0][7:]
        self.assertEqual(result.score, 44.0)
        self.assertEqual(result.extras["detection_hooks"], ["first", "second"])
        for role in ("current", "previous"):
            metadata = payload["metadata"][role]
            self.assertEqual(metadata["frame_role"], role)
            self.assertEqual(metadata["image_hooks"], ["first", "second"])

    def test_hook_instance_is_shared_within_batch_and_recreated_for_next_batch(self):
        options = dict(
            roi_mask=np.ones((4, 4), dtype=np.uint8),
            params={},
            input_loader=MemoryLoader(),
            detector=RecordingDetector(),
            hook_configs=[HookConfig(RecordingHook.plugin_name)],
        )
        first = process_image_batch(
            [(1, "b.CR2", "a.CR2"), (2, "c.CR2", "b.CR2")], **options
        )
        second = process_image_batch([(3, "d.CR2", "c.CR2")], **options)

        self.assertEqual(
            [item[8]["metadata"]["previous"]["image_calls"] for item in first],
            [2, 4],
        )
        self.assertEqual(second[0][8]["metadata"]["previous"]["image_calls"], 2)
        self.assertEqual(len(RecordingHook.instances), 2)

    def test_output_hook_receives_snapshot_without_changing_saved_result(self):
        original = OutputResult(
            saved=True,
            output_path="output/b.CR2",
            debug_path=None,
            handler_info={"name": "original"},
            metrics={"writes": 1},
        )

        def change_snapshot(snapshot):
            snapshot.saved = False
            snapshot.handler_info["name"] = "changed"
            snapshot.metrics["writes"] = 99

        hook = Mock()
        hook.on_output_saved.side_effect = change_snapshot
        _apply_output_saved_hooks(original, [hook], "b.CR2", "raise")

        hook.on_output_saved.assert_called_once()
        self.assertIsNot(hook.on_output_saved.call_args.args[0], original)
        self.assertTrue(original.saved)
        self.assertEqual(original.handler_info, {"name": "original"})
        self.assertEqual(original.metrics, {"writes": 1})


class TestPipelineHookLifecycle(PipelineTestCase):
    def test_no_configured_hooks_creates_no_hook_instances(self):
        pipeline = self.make_pipeline(hooks=[], pairs=1)
        pipeline.config.hooks = None
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.run(resume=False), 1)
        self.assertEqual(RecordingHook.instances, [])

    def test_file_filter_short_circuits_and_pairs_only_kept_absolute_paths(self):
        pipeline = self.make_pipeline(
            hooks=[
                RecordingHookConfig("first", reject_filename="b.CR2"),
                RecordingHookConfig("second"),
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.run(resume=False), 2)

        file_events = [event for event in self.events if event[1] == "on_file_found"]
        self.assertEqual(
            [(event[0], Path(event[2]).name) for event in file_events],
            [
                ("first", "a.CR2"),
                ("second", "a.CR2"),
                ("first", "b.CR2"),
                ("first", "c.CR2"),
                ("second", "c.CR2"),
                ("first", "d.CR2"),
                ("second", "d.CR2"),
            ],
        )
        for event in file_events:
            self.assertTrue(os.path.isabs(event[2]))
            self.assertEqual(event[2], os.path.normpath(event[2]))
        self.assertEqual(
            [
                (
                    item["metadata"]["current"]["filename"],
                    item["metadata"]["previous"]["filename"],
                )
                for item in pipeline.output_handler.contexts
            ],
            [("c.CR2", "a.CR2"), ("d.CR2", "c.CR2")],
        )

    def test_save_notifications_progress_and_sorted_hooks_keep_their_order(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                self.events.clear()
                pipeline = self.make_pipeline(
                    parallel=parallel,
                    pairs=1,
                    hooks=[RecordingHookConfig(label) for label in ("first", "second")],
                )
                if parallel:
                    self.use_completed_executor()
                record_result = pipeline.progress_manager.record_result

                def record(*args, **kwargs):
                    count = record_result(*args, **kwargs)
                    self.events.append(
                        ("progress", "record_result", args[0], os.getpid())
                    )
                    return count

                with patch.object(pipeline.progress_manager, "record_result", record):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(pipeline.run(resume=False), 1)

                start = next(
                    i
                    for i, event in enumerate(self.events)
                    if event[1] == "on_detection_result"
                )
                self.assertEqual(
                    [event[:2] for event in self.events[start:]],
                    [
                        ("output", "on_detection_result"),
                        ("output", "save_candidate"),
                        ("first", "on_output_saved"),
                        ("second", "on_output_saved"),
                        ("output", "on_candidate_detected"),
                        ("progress", "record_result"),
                        ("first", "on_batch_results_sorted"),
                        ("second", "on_batch_results_sorted"),
                        ("output", "on_batch_complete"),
                        ("output", "on_pipeline_complete"),
                        ("first", "on_all_detections_sorted"),
                        ("second", "on_all_detections_sorted"),
                    ],
                )
                final = [
                    hook
                    for hook in RecordingHook.instances
                    if hasattr(hook, "final_detections")
                ][-2:]
                self.assertEqual(
                    final[-1].final_detections[0].extras["batch_hooks"],
                    ["first", "second"],
                )
                self.assertEqual(
                    final[-1].final_detections[0].extras["final_hooks"], ["first"]
                )

    def test_parallel_batch_order_and_global_order_are_distinct(self):
        self.use_completed_executor(reverse=True)
        pipeline = self.make_pipeline(parallel=True)
        real_process_batch = process_image_batch

        def reversed_results(*args):
            return list(reversed(real_process_batch(*args)))

        with patch("meteor_core.pipeline.process_image_batch", reversed_results):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.run(resume=False), 3)

        self.assertEqual(
            [
                event[2]
                for event in self.events
                if event[1] == "on_batch_results_sorted"
            ],
            [(3,), (1, 2)],
        )
        self.assertEqual(
            [
                event[2]
                for event in self.events
                if event[1] == "on_all_detections_sorted"
            ],
            [(1, 2, 3)],
        )

        self.assertEqual(
            [event[2] for event in self.events if event[1] == "on_batch_complete"],
            [(1, 1, 1), (3, 3, 2)],
        )

    def test_file_and_final_hooks_share_state_separately_from_output_hooks(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                RecordingHook.instances.clear()
                pipeline = self.make_pipeline(parallel=parallel)
                if parallel:
                    self.use_completed_executor()
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.run(resume=False)

                file_hook = next(
                    hook
                    for hook in RecordingHook.instances
                    if any(name == "on_file_found" for name, _ in hook.calls)
                )
                output_hook = next(
                    hook
                    for hook in RecordingHook.instances
                    if any(name == "on_output_saved" for name, _ in hook.calls)
                )
                self.assertTrue(hasattr(file_hook, "final_detections"))
                self.assertIsNot(file_hook, output_hook)
                self.assertIn(
                    "on_batch_results_sorted", [name for name, _ in output_hook.calls]
                )
                self.assertNotIn(
                    "on_image_loaded", [name for name, _ in file_hook.calls]
                )
                self.assertNotIn(
                    "on_image_loaded", [name for name, _ in output_hook.calls]
                )

    def test_sorted_hooks_include_non_candidates_and_exclude_failed_frames(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                self.events.clear()
                pipeline = self.make_pipeline(parallel=parallel)
                pipeline.detector.non_candidates.add("c.CR2")
                pipeline.input_loader.errors["d.CR2"] = MeteorLoadError("bad frame")
                if parallel:
                    self.use_completed_executor()
                with self.assertLogs("meteor_core.pipeline", level="WARNING"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(pipeline.run(resume=False), 1)

                final = next(
                    hook
                    for hook in reversed(RecordingHook.instances)
                    if hasattr(hook, "final_detections")
                ).final_detections
                self.assertEqual(
                    [(item.frame_index, item.is_candidate) for item in final],
                    [(1, True), (2, False)],
                )
                self.assertEqual(
                    pipeline.progress_manager.processed_set, {"b.CR2", "c.CR2", "d.CR2"}
                )
                self.assertEqual(
                    [event[2] for event in self.events if event[1] == "save_candidate"],
                    ["b.CR2"],
                )

    def test_resume_final_hook_receives_only_new_frames(self):
        pipeline = self.make_pipeline()
        manager = pipeline.progress_manager
        manager.set_params_hash(compute_params_hash(pipeline.params.to_dict()))
        manager.record_result("b.CR2", True, frame_index=1, prev_frame_index=0)

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.run(resume=True), 3)

        self.assertEqual(
            [
                event[2]
                for event in self.events
                if event[1] == "on_all_detections_sorted"
            ],
            [(2, 3)],
        )
        self.assertEqual(
            [event[2] for event in self.events if event[1] == "save_candidate"],
            ["c.CR2", "d.CR2"],
        )

    def test_interrupt_saves_completed_progress_and_skips_completion_hooks(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                self.events.clear()
                pipeline = self.make_pipeline(parallel=parallel)
                pipeline.config.batch_size = 1
                pipeline.detector.errors["c.CR2"] = KeyboardInterrupt()
                executor = self.use_completed_executor() if parallel else None
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(pipeline.run(resume=False), 1)

                progress = json.loads(Path(pipeline.progress_file).read_text())
                self.assertEqual(progress["processed_files"], ["b.CR2"])
                self.assertEqual(progress["detected_files"], ["b.CR2"])
                self.assertFalse(
                    any(
                        event[1] in {"on_pipeline_complete", "on_all_detections_sorted"}
                        for event in self.events
                    )
                )
                if executor is not None:
                    for future in executor.futures:
                        future.cancel.assert_called_once()
                    executor.shutdown.assert_called_once_with(
                        wait=False, cancel_futures=True
                    )

    def test_spawned_workers_run_image_hooks_and_main_process_runs_sorted_hooks(self):
        pipeline = self.make_pipeline(parallel=True, pairs=2)

        with patch(
            "meteor_core.pipeline.ProcessPoolExecutor", side_effect=make_spawn_executor
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.run(resume=False), 2)

        contexts = pipeline.output_handler.contexts
        self.assertEqual(len(contexts), 2)
        for context in contexts:
            for role in ("current", "previous"):
                self.assertNotEqual(context["metadata"][role]["hook_pid"], os.getpid())
        for result, context in zip(pipeline.output_handler.results, contexts):
            self.assertEqual(result.extras["detection_hooks"], ["first"])
            self.assertEqual(
                result.extras["detection_hook_pid"],
                context["metadata"]["current"]["hook_pid"],
            )
        self.assertEqual(
            [context["metadata"]["previous"]["image_calls"] for context in contexts],
            [2, 4],
        )
        parent_events = {event[1] for event in self.events}
        self.assertTrue(
            {
                "on_file_found",
                "on_output_saved",
                "on_batch_results_sorted",
                "on_all_detections_sorted",
            }
            <= parent_events
        )
        self.assertNotIn("on_image_loaded", parent_events)
        self.assertNotIn("on_detection_complete", parent_events)
        self.assertTrue(all(event[3] == os.getpid() for event in self.events))
