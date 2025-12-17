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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
