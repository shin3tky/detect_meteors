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
