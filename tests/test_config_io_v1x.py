"""
Tests for configuration loading helpers in meteor_core.config_io (v1.x).
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detect_meteors_cli as cli  # noqa: E402
from meteor_core import dump_pipeline_config, load_pipeline_config  # noqa: E402
from meteor_core.exceptions import MeteorConfigError  # noqa: E402
from meteor_core.schema import (  # noqa: E402
    DEFAULT_DEBUG_FOLDER,
    DEFAULT_MIN_AREA,
    DEFAULT_OUTPUT_FOLDER,
)


class TestLoadPipelineConfigPartial(unittest.TestCase):
    def test_load_pipeline_config_allows_partial_config(self):
        payload = {
            "target_folder": "custom_raw",
            "params": {"diff_threshold": 12},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "pipeline.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            config = load_pipeline_config(config_path)

        self.assertEqual(config.target_folder, "custom_raw")
        self.assertEqual(config.output_folder, DEFAULT_OUTPUT_FOLDER)
        self.assertEqual(config.debug_folder, DEFAULT_DEBUG_FOLDER)
        self.assertEqual(config.params.diff_threshold, 12)
        self.assertEqual(config.params.min_area, DEFAULT_MIN_AREA)

    def test_versioned_and_legacy_files_resolve_same_cli_settings(self):
        payload = {
            "target_folder": "画像",
            "params": {"diff_threshold": 12},
            "detector_name": "custom_detector",
            "detector_config": {"schema_version": "plugin-specific"},
            "hooks": [
                {"name": "second", "config": {"schema_version": 2}},
                {"name": "first"},
            ],
            "roi_polygon": [[0, 0], [8, 0], [8, 9]],
            "enable_roi_selection": False,
        }
        parser = cli.build_arg_parser()
        with tempfile.TemporaryDirectory() as tmp_dir:
            for suffix, encode in ((".yaml", yaml.safe_dump), (".json", json.dumps)):
                with self.subTest(suffix=suffix):
                    path = Path(tmp_dir) / f"pipeline{suffix}"
                    argv = ["--config", str(path), "--diff-threshold", "24"]
                    args = parser.parse_args(argv)
                    with patch.object(sys, "argv", ["detect-meteors", *argv]):
                        path.write_text(encode(payload), encoding="utf-8")
                        legacy = cli._build_pipeline_config(args)
                        path.write_text(
                            encode({"schema_version": 1, **payload}), encoding="utf-8"
                        )
                        versioned = cli._build_pipeline_config(args)

                    self.assertEqual(versioned.to_dict(), legacy.to_dict())
                    self.assertEqual(versioned.params.diff_threshold, 24)
                    self.assertEqual(versioned.params.min_area, DEFAULT_MIN_AREA)
                    self.assertEqual(versioned.detector_name, "custom_detector")
                    self.assertEqual(
                        versioned.detector_config, payload["detector_config"]
                    )
                    self.assertEqual(
                        [hook.name for hook in versioned.hooks], ["second", "first"]
                    )
                    self.assertEqual(versioned.hooks[0].config, {"schema_version": 2})
                    self.assertEqual(versioned.roi_polygon, payload["roi_polygon"])
                    self.assertFalse(versioned.enable_roi_selection)

    def test_dump_writes_versioned_yaml_and_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "legacy.yaml"
            path.write_text("target_folder: 画像\n", encoding="utf-8")
            config = load_pipeline_config(path)
            for fmt, decode in (("yaml", yaml.safe_load), ("json", json.loads)):
                with self.subTest(format=fmt):
                    text = dump_pipeline_config(config, format=fmt)
                    saved = Path(tmp_dir) / f"saved.{fmt}"
                    saved.write_text(text, encoding="utf-8")
                    self.assertEqual(decode(text)["schema_version"], 1)
                    self.assertNotIn("schema_version", config.to_dict())
                    self.assertEqual(
                        load_pipeline_config(saved).to_dict(), config.to_dict()
                    )

    def test_rejects_unsupported_and_malformed_versions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for suffix, encode in ((".yml", yaml.safe_dump), (".json", json.dumps)):
                path = Path(tmp_dir) / f"pipeline{suffix}"
                for version in (2, 0, -1, "1", 1.0, True, False, None):
                    with self.subTest(suffix=suffix, version=version):
                        path.write_text(
                            encode({"schema_version": version}), encoding="utf-8"
                        )
                        with self.assertRaisesRegex(
                            MeteorConfigError, r"schema_version.*Expected integer 1"
                        ) as raised:
                            load_pipeline_config(path)
                        self.assertEqual(raised.exception.config_key, "schema_version")
                        self.assertEqual(raised.exception.filepath, str(path))


if __name__ == "__main__":
    unittest.main()
