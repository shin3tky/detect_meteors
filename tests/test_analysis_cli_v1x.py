"""Analysis CLI routing, provenance, sequence selection, and read-only behavior."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import detect_meteors_cli as cli
from meteor_core.inputs.base import BaseInputLoader
from meteor_core.schema import DetectionParams, InputContext


class CliArrayLoader(BaseInputLoader):
    plugin_name = "cli_test"

    def __init__(self, images):
        self.images = images
        self.loaded = []

    def load(self, filepath):
        self.loaded.append(filepath)
        return InputContext(
            self.images[Path(filepath).name].copy(),
            filepath,
            metadata={"camera_model": "Test camera"},
        )


class TestAnalysisCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name).resolve()
        self.target = self.directory / "frame002.ORF"
        self.previous = self.directory / "frame001.ORF"
        self.target.write_bytes(b"current")
        self.previous.write_bytes(b"previous")
        current = np.zeros((96, 96), np.uint16)
        current[20:45, 10:70] = 1000
        self.loader = CliArrayLoader(
            {self.target.name: current, self.previous.name: np.zeros_like(current)}
        )
        self.params = DetectionParams(
            diff_threshold=10, min_area=3, min_aspect_ratio=3.0, min_line_score=30.0
        )
        self.progress = self.directory / "progress.json"
        self.progress.write_text(
            json.dumps(
                {
                    "params": self.params.to_dict(),
                    "processing_params": self.params.to_dict(),
                    "roi": None,
                    "processed_files": [self.target.name],
                }
            )
        )
        self.config = self.directory / "config.json"
        self.config.write_text(json.dumps({"params": self.params.to_dict()}))

    def run_cli(self, options):
        stdout, stderr = io.StringIO(), io.StringIO()
        before = {
            p.name: p.read_bytes() for p in self.directory.iterdir() if p.is_file()
        }
        with (
            patch.object(
                sys, "argv", ["detect_meteors_cli.py", "--locale", "en", *options]
            ),
            patch.object(cli, "_configure_logging"),
            patch.object(cli, "_run_main") as normal_run,
            patch(
                "meteor_core.inputs.registry.LoaderRegistry.create",
                return_value=self.loader,
            ) as create,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            try:
                cli.main()
                code = 0
            except SystemExit as exc:
                code = exc.code
        normal_run.assert_not_called()
        self.assertEqual(
            before,
            {p.name: p.read_bytes() for p in self.directory.iterdir() if p.is_file()},
        )
        self.assertFalse((self.directory / "candidates").exists())
        return code, stdout.getvalue(), stderr.getvalue(), create

    def pair_options(self):
        return [
            "--analyze-image",
            str(self.target),
            "--reference-image",
            str(self.previous),
            "--analysis-progress",
            str(self.progress),
        ]

    def test_explicit_pair_prints_evidence_without_starting_the_output_pipeline(self):
        with patch.object(cli, "MeteorDetectionPipeline") as pipeline:
            code, out, err, _ = self.run_cli(self.pair_options())
        self.assertEqual(code, 0, err)
        self.assertIn("Candidate: False", out)
        self.assertIn("min_aspect_ratio:", out)
        self.assertIn("progress.processing_params", out)
        self.assertIn("explicit reference", out)
        self.assertIn(str(self.previous), out)
        self.assertIn("Mask pixels:", out)
        self.assertIn("Test camera", out)
        pipeline.assert_not_called()

    def test_explicit_parameter_spellings_preserve_original_values_and_sources(self):
        for option in (
            ["--min-aspect-ratio", "2.0"],
            ["--min-aspect-ratio=2.0"],
            ["--min-aspect-r=2.0"],
        ):
            with self.subTest(option=option):
                code, out, err, _ = self.run_cli(self.pair_options() + option)
                self.assertEqual(code, 0, err)
                self.assertIn("Candidate: True", out)
                self.assertIn("min_aspect_ratio: 2.0 [override]", out)
                self.assertIn("min_aspect_ratio: 3.0 -> 2.0", out)

    def test_config_baseline_is_not_replaced_before_override_tracking(self):
        options = [
            "--analyze-image",
            str(self.target),
            "--reference-image",
            str(self.previous),
            "--config",
            str(self.config),
            "--min-aspect-ratio=2.0",
        ]
        code, out, err, _ = self.run_cli(options)
        self.assertEqual(code, 0, err)
        self.assertIn("min_aspect_ratio: 3.0 -> 2.0", out)
        self.assertIn("min_line_score: 30.0 [supplied]", out)

    def test_recorded_settings_take_precedence_over_config_defaults(self):
        self.config.write_text(json.dumps({"params": {"min_aspect_ratio": 9.0}}))
        code, out, err, _ = self.run_cli(
            self.pair_options() + ["--config", str(self.config)]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("min_aspect_ratio: 3.0 [progress.processing_params]", out)
        self.assertNotIn("Explicit changes from the baseline", out)

    def test_missing_reference_returns_two_without_guessing_from_parent(self):
        code, out, _, _ = self.run_cli(["--analyze-image", str(self.target)])
        self.assertEqual(code, 2)
        self.assertIn("reference_missing", out)
        self.assertNotIn("Candidate:", out)
        self.assertEqual(self.loader.loaded, [str(self.target)])

    def test_empty_analysis_paths_cannot_start_normal_processing(self):
        code, _, _, create = self.run_cli(["--analyze-image", ""])
        self.assertEqual(code, 1)
        create.assert_not_called()
        for flag in ("--reference-image", "--analysis-sequence", "--analysis-progress"):
            with self.subTest(flag=flag):
                code, _, _, create = self.run_cli([flag, ""])
                self.assertEqual(code, 2)
                create.assert_not_called()

    def test_explicit_sequence_uses_normal_filename_order(self):
        code, out, err, _ = self.run_cli(
            [
                "--analyze-image",
                str(self.target),
                "--analysis-sequence",
                str(self.directory),
                "--analysis-progress",
                str(self.progress),
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertIn(str(self.previous), out)
        self.assertIn("preceding frame in explicitly selected sequence", out)
        self.assertEqual(self.loader.loaded, [str(self.target), str(self.previous)])

    def test_first_sequence_frame_has_no_reference(self):
        code, out, _, _ = self.run_cli(
            [
                "--analyze-image",
                str(self.previous),
                "--analysis-sequence",
                str(self.directory),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("first frame in explicitly selected sequence", out)
        self.assertNotIn("Candidate:", out)

    def test_target_outside_selected_sequence_is_an_error(self):
        sequence = self.directory / "other"
        sequence.mkdir()
        (sequence / "other.ORF").write_bytes(b"other")
        code, _, err, _ = self.run_cli(
            [
                "--analyze-image",
                str(self.target),
                "--analysis-sequence",
                str(sequence),
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("exactly once", err)
        self.assertEqual(self.loader.loaded, [])

    def test_reference_selection_methods_are_mutually_exclusive(self):
        code, _, _, create = self.run_cli(
            self.pair_options() + ["--analysis-sequence", str(self.directory)]
        )
        self.assertEqual(code, 2)
        create.assert_not_called()

    def test_analysis_options_require_an_analysis_target(self):
        for flag, value in (
            ("--reference-image", self.previous),
            ("--analysis-progress", self.progress),
            ("--analysis-sequence", self.directory),
        ):
            with self.subTest(flag=flag):
                code, _, err, create = self.run_cli([flag, str(value)])
                self.assertEqual(code, 2)
                self.assertIn("require --analyze-image", err)
                create.assert_not_called()

    def test_incompatible_modes_are_rejected_before_any_deletion_or_estimation(self):
        for option in (
            ["--remove-progress"],
            ["--remove-prog"],
            ["--auto-params"],
            ["--show-exif"],
            ["--debug-image"],
            ["--progress-file=" + str(self.progress)],
            ["--target=" + str(self.directory)],
            ["-t" + str(self.directory)],
        ):
            with self.subTest(option=option):
                with (
                    patch.object(cli.os, "remove") as remove,
                    patch.object(cli, "_run_auto_params") as estimate,
                ):
                    code, _, err, create = self.run_cli(
                        ["--analyze-image", str(self.target), *option]
                    )
                self.assertEqual(code, 2)
                self.assertIn("cannot be combined", err)
                remove.assert_not_called()
                estimate.assert_not_called()
                create.assert_not_called()

    def test_progress_and_config_inputs_cannot_be_swapped(self):
        for option in (
            ["--config", str(self.progress)],
            ["--analysis-progress", str(self.config)],
        ):
            with self.subTest(option=option):
                code, _, err, create = self.run_cli(
                    ["--analyze-image", str(self.target), *option]
                )
                self.assertEqual(code, 1)
                self.assertIn("progress", err.lower())
                create.assert_not_called()

    def test_hooks_require_explicit_disabling_for_detector_only_replay(self):
        self.config.write_text(json.dumps({"hooks": [{"name": "aircraft_trail"}]}))
        options = self.pair_options() + ["--config", str(self.config)]
        code, _, err, create = self.run_cli(options)
        self.assertEqual(code, 1)
        self.assertIn("does not run hooks", err)
        create.assert_not_called()
        code, out, err, _ = self.run_cli(options + ["--hooks", ""])
        self.assertEqual(code, 0, err)
        self.assertIn("Candidate: False", out)

    def test_other_detectors_and_invalid_hough_config_are_not_silently_replaced(self):
        for option in (
            ["--detector", "simple_threshold"],
            ["--detector-config", '{"unsupported": 3}'],
        ):
            with self.subTest(option=option):
                code, _, err, create = self.run_cli(self.pair_options() + option)
                self.assertEqual(code, 1)
                self.assertIn("detector", err.lower())
                create.assert_not_called()

    def test_loader_config_is_forwarded(self):
        code, _, err, create = self.run_cli(
            self.pair_options() + ["--input-loader-config", '{"normalize": true}']
        )
        self.assertEqual(code, 0, err)
        create.assert_called_once_with("raw", {"normalize": True})

    def test_roi_overrides_are_reported_and_conflicts_rejected(self):
        options = self.pair_options() + ["--roi", "0,0;8,0;8,95;0,95"]
        code, out, err, _ = self.run_cli(options)
        self.assertEqual(code, 0, err)
        self.assertIn("override_polygon", out)
        self.assertIn("no_contours:", out)
        code, _, _, _ = self.run_cli(options + ["--no-roi"])
        self.assertEqual(code, 2)

    def test_japanese_summary_and_verbose_evidence_are_available(self):
        code, out, err, _ = self.run_cli(
            self.pair_options() + ["--locale", "ja", "--verbose"]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("Hough画像ペア分析", out)
        self.assertIn("不採用の条件:", out)
        self.assertIn("輪郭の計測結果:", out)
        self.assertIn("Hough線分（", out)
        self.assertNotIn("ui.analysis.", out)

    def test_normal_detection_dispatch_remains_available(self):
        with (
            patch.object(sys, "argv", ["detect_meteors_cli.py"]),
            patch.object(cli, "_configure_logging"),
            patch.object(cli, "_run_main") as normal,
            patch.object(cli, "_run_image_analysis") as analysis,
        ):
            cli.main()
        normal.assert_called_once()
        analysis.assert_not_called()


if __name__ == "__main__":
    unittest.main()
