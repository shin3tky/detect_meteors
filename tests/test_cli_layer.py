import io
import sys
import types
import unittest
from contextlib import redirect_stdout

sys.modules.setdefault("cv2", types.SimpleNamespace())

from detect_meteors import cli, services


class TestCliLayer(unittest.TestCase):
    def test_parse_defaults_and_overrides(self):
        args = cli.parse_args([
            "--target",
            "custom",
            "--workers",
            "2",
            "--batch-size",
            "8",
        ])

        self.assertEqual(args.target, "custom")
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.output, services.DEFAULT_OUTPUT_FOLDER)
        self.assertIsNone(args.detector_plugin)
        self.assertIsNone(args.preprocessor_plugin)
        self.assertIsNone(args.output_writer_plugin)
        self.assertFalse(args.list_plugins)

    def test_plugin_selection_arguments(self):
        args = cli.parse_args([
            "--detector-plugin",
            "alt_detector",
            "--preprocessor-plugin",
            "alt_pre",
            "--output-writer-plugin",
            "alt_writer",
        ])

        self.assertEqual(args.detector_plugin, "alt_detector")
        self.assertEqual(args.preprocessor_plugin, "alt_pre")
        self.assertEqual(args.output_writer_plugin, "alt_writer")

    def test_list_plugins_flag_sets_action(self):
        args = cli.parse_args(["--list-plugins"])

        self.assertTrue(args.list_plugins)
        self.assertIsNone(args.detector_plugin)
        self.assertIsNone(args.preprocessor_plugin)
        self.assertIsNone(args.output_writer_plugin)

    def test_help_output_available(self):
        parser = cli.build_arg_parser()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            parser.print_help()
        output = buffer.getvalue()
        self.assertIn("Meteor detection tool", output)
        self.assertIn("--target", output)
        self.assertIn("--show-exif", output)


if __name__ == "__main__":
    unittest.main()
