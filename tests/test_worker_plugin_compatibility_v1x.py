"""Spawn compatibility for plugins selected by the normal discovery paths."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4

from meteor_core import schema
from meteor_core.config_io import dump_pipeline_config, load_pipeline_config
from meteor_core.detectors import DetectorRegistry
from meteor_core.exceptions import MeteorConfigError
from meteor_core.hooks import HookRegistry
from meteor_core.inputs import LoaderRegistry
from meteor_core.pipeline import MeteorDetectionPipeline
from meteor_core.schema import HookConfig
from tests.pipeline_fixtures import PipelineTestCase, make_spawn_executor


LOCAL_LOADER = """
from dataclasses import replace
from pathlib import Path

import numpy as np
from meteor_core.inputs import BaseInputLoader
from meteor_core.schema import InputContext


class LocalLoader(BaseInputLoader):
    plugin_name = "compat_local_loader"

    def __init__(self, config=None):
        self.loads = 0

    def load(self, filepath):
        self.loads += 1
        return InputContext(
            np.zeros((4, 4), np.uint16), filepath,
            metadata={"filename": Path(filepath).name, "loads": self.loads},
            schema_version=0,
        )


def upgrade_input(context):
    return replace(
        context, metadata={**context.metadata, "input_converted": True}, schema_version=1
    )
"""

LOCAL_HOOK = """
from meteor_core.hooks import BaseHook


class LocalHook(BaseHook):
    plugin_name = "compat_shadowed_hook"

    def __init__(self):
        self.calls = 0

    def on_file_found(self, filepath):
        return not filepath.endswith("excluded.CR2")

    def on_image_loaded(self, context):
        self.calls += 1
        context.metadata["local_hook_calls"] = self.calls
        return context
"""

ENTRY_POINT_PLUGIN = """
from dataclasses import replace
from meteor_core.hooks import BaseHook
from tests.pipeline_fixtures import RecordingDetector


class EntryDetector(RecordingDetector):
    plugin_name = "compat_entry_detector"

    def __init__(self, config=None):
        super().__init__()

    def detect(self, context):
        return replace(super().detect(context), schema_version=0)


def upgrade_result(result):
    return replace(result, extras={"result_converted": True}, schema_version=1)


class EntryHook(BaseHook):
    plugin_name = "compat_entry_hook"

    def on_detection_complete(self, result, context):
        result.extras["entry_hook"] = True
        return result


class ShadowedHook(BaseHook):
    plugin_name = "compat_shadowed_hook"

    def on_image_loaded(self, context):
        raise AssertionError("The local hook should take precedence")
"""


class TestWorkerPluginCompatibility(PipelineTestCase):
    def make_discovered_pipeline(
        self, *, pairs: int, mode: str = "raise", versioned_config: bool = False
    ) -> tuple[MeteorDetectionPipeline, Path]:
        loader_dir = self.root / "loaders"
        hook_dir = self.root / "hooks"
        entry_dir = self.root / "installed"
        for directory in (loader_dir, hook_dir, entry_dir):
            directory.mkdir()
        # These files have the same basename and must load as distinct modules.
        loader_file = loader_dir / "plugin.py"
        loader_file.write_text(LOCAL_LOADER, encoding="utf-8")
        (hook_dir / "plugin.py").write_text(LOCAL_HOOK, encoding="utf-8")

        module_name = f"compat_spawn_{uuid4().hex}"
        (entry_dir / f"{module_name}.py").write_text(
            ENTRY_POINT_PLUGIN, encoding="utf-8"
        )
        metadata = entry_dir / "compat_fixture-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: compat-fixture\nVersion: 1.0\n",
            encoding="utf-8",
        )
        (metadata / "entry_points.txt").write_text(
            f"[detect_meteors.detector]\nentry = {module_name}:EntryDetector\n"
            f"[detect_meteors.hook]\nentry = {module_name}:EntryHook\n"
            f"shadowed = {module_name}:ShadowedHook\n",
            encoding="utf-8",
        )
        self.enterContext(patch.object(sys, "path", [str(entry_dir), *sys.path]))
        self.enterContext(patch("meteor_core.inputs.discovery.PLUGIN_DIR", loader_dir))
        self.enterContext(patch("meteor_core.hooks.discovery.PLUGIN_DIR", hook_dir))
        for registry in (LoaderRegistry, DetectorRegistry, HookRegistry):
            self.enterContext(patch.object(registry, "_discovered", None))
        self.enterContext(patch.dict(schema._INPUT_CONTEXT_CONVERTERS, clear=True))
        self.enterContext(patch.dict(schema._DETECTION_RESULT_CONVERTERS, clear=True))

        loader_class = LoaderRegistry.get("compat_local_loader")
        loader_module = sys.modules[loader_class.__module__]
        schema.register_input_context_converter(0, loader_module.upgrade_input)

        base = self.make_pipeline(parallel=True, pairs=pairs, hooks=[], mode=mode)
        config = base.config
        config.input_loader_name = "compat_local_loader"
        config.detector_name = "compat_entry_detector"
        config.hooks = [
            HookConfig("compat_shadowed_hook"),
            HookConfig("compat_entry_hook"),
        ]
        if versioned_config:
            path = self.root / "pipeline.yaml"
            path.write_text(dump_pipeline_config(config), encoding="utf-8")
            config = load_pipeline_config(path)
        (Path(config.target_folder) / "excluded.CR2").touch()
        pipeline = MeteorDetectionPipeline(config, output_handler=base.output_handler)
        schema.register_detection_result_converter(
            0, sys.modules[module_name].upgrade_result
        )
        return pipeline, loader_file

    def test_spawn_uses_selected_sources_converters_and_batch_state(self):
        pipeline, _ = self.make_discovered_pipeline(pairs=3, versioned_config=True)
        with (
            patch(
                "meteor_core.pipeline.ProcessPoolExecutor",
                side_effect=make_spawn_executor,
            ),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(pipeline.run(resume=False), 3)

        contexts = sorted(
            (context["metadata"] for context in pipeline.output_handler.contexts),
            key=lambda metadata: metadata["frame_index"],
        )
        self.assertEqual(
            [metadata["current"]["filename"] for metadata in contexts],
            ["b.CR2", "c.CR2", "d.CR2"],
        )
        # The first parent load prepared the ROI. Each worker batch starts from
        # that loader snapshot, and its hook starts with a fresh call count.
        self.assertEqual([item["current"]["loads"] for item in contexts], [2, 4, 2])
        self.assertEqual(
            [item["current"]["local_hook_calls"] for item in contexts], [1, 3, 1]
        )
        for item in contexts:
            self.assertTrue(item["current"]["input_converted"])
            self.assertTrue(item["previous"]["input_converted"])
        for result in pipeline.output_handler.results:
            self.assertEqual(result.schema_version, 1)
            self.assertEqual(
                result.extras, {"result_converted": True, "entry_hook": True}
            )

    def test_worker_source_mismatch_is_fatal_and_pair_can_be_resumed(self):
        pipeline, loader_file = self.make_discovered_pipeline(pairs=1, mode="warn")

        def change_source_then_spawn(**kwargs):
            loader_file.write_text(LOCAL_LOADER + "\n# changed\n", encoding="utf-8")
            return make_spawn_executor(**kwargs)

        with (
            patch(
                "meteor_core.pipeline.ProcessPoolExecutor",
                side_effect=change_source_then_spawn,
            ),
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(
                MeteorConfigError, "Selected plugin source changed"
            ):
                pipeline.run(resume=False)
        self.assertEqual(pipeline.progress_manager.processed_set, set())
        self.assertEqual(pipeline.output_handler.results, [])

        loader_file.write_text(LOCAL_LOADER, encoding="utf-8")
        with (
            patch(
                "meteor_core.pipeline.ProcessPoolExecutor",
                side_effect=make_spawn_executor,
            ),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(pipeline.run(resume=True), 1)
        self.assertEqual(pipeline.progress_manager.processed_set, {"b.CR2"})

    def test_unimportable_converter_fails_before_creating_workers(self):
        pipeline = self.make_pipeline(parallel=True, pairs=1, hooks=[])
        self.enterContext(patch.dict(schema._INPUT_CONTEXT_CONVERTERS, clear=True))
        schema.register_input_context_converter(0, lambda context: context)

        with (
            patch("meteor_core.pipeline.ProcessPoolExecutor") as executor,
            redirect_stdout(StringIO()),
            self.assertRaisesRegex(
                MeteorConfigError, "Converter for InputContext schema version 0"
            ),
        ):
            pipeline.run(resume=False)

        executor.assert_not_called()
        self.assertEqual(pipeline.progress_manager.processed_set, set())

    def test_unserializable_explicit_instance_reports_configuration_error(self):
        pipeline = self.make_pipeline(parallel=True, pairs=1, hooks=[])
        pipeline.input_loader.worker_probe = lambda: None

        with (
            patch(
                "meteor_core.pipeline.ProcessPoolExecutor",
                side_effect=make_spawn_executor,
            ),
            redirect_stdout(StringIO()),
            self.assertRaisesRegex(
                MeteorConfigError, "Could not serialize plugin state"
            ),
        ):
            pipeline.run(resume=False)

        self.assertEqual(pipeline.progress_manager.processed_set, set())
