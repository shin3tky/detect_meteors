"""CLI option coverage for v1.5.x features."""

import sys
import types
import unittest

if "cv2" not in sys.modules:  # pragma: no cover - optional dependency shim
    sys.modules["cv2"] = types.SimpleNamespace()

from detect_meteors_cli import build_arg_parser


class TestDiagnosticFlag(unittest.TestCase):
    """Ensure --save-diagnostic matches documented behavior."""

    def test_save_diagnostic_without_value_generates_default(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--save-diagnostic"])
        self.assertEqual(
            args.save_diagnostic,
            "",
            "Expected empty string to trigger auto-generated diagnostic filename",
        )

    def test_save_diagnostic_with_filename(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--save-diagnostic", "report.md"])
        self.assertEqual(args.save_diagnostic, "report.md")


class TestPluginOptionFlags(unittest.TestCase):
    """Ensure plugin-related CLI options are accepted and parsed."""

    def test_plugin_options_default_to_none(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.config)
        self.assertIsNone(args.input_loader)
        self.assertIsNone(args.input_loader_config)
        self.assertIsNone(args.detector)
        self.assertIsNone(args.detector_config)
        self.assertIsNone(args.output_handler)
        self.assertIsNone(args.output_handler_config)

    def test_plugin_options_accept_values(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--config",
                "config.yml",
                "--input-loader",
                "raw",
                "--input-loader-config",
                '{"key": "value"}',
                "--detector",
                "hough",
                "--detector-config",
                "detector.json",
                "--output-handler",
                "file",
                "--output-handler-config",
                "output.yml",
            ]
        )
        self.assertEqual(args.config, "config.yml")
        self.assertEqual(args.input_loader, "raw")
        self.assertEqual(args.input_loader_config, '{"key": "value"}')
        self.assertEqual(args.detector, "hough")
        self.assertEqual(args.detector_config, "detector.json")
        self.assertEqual(args.output_handler, "file")
        self.assertEqual(args.output_handler_config, "output.yml")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
