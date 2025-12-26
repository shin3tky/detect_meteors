"""
Tests for configuration loading helpers in meteor_core.config_io (v1.x).
"""

import json
import os
import sys
import tempfile
import unittest

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core.config_io import load_pipeline_config  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
