"""Persisted evidence, exact replay, source protection, and failed-write cleanup."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

import detect_meteors_cli as cli
from meteor_core.analysis import (
    load_reproduction_settings,
    reproduce_hough_pair,
    source_stamp,
)
from meteor_core.analysis_reports import save_analysis_bundle
from meteor_core.config_io import load_pipeline_config
from meteor_core.exceptions import (
    MeteorConfigError,
    MeteorValidationError,
    MeteorWriteError,
)
from meteor_core.inputs.raw import RawImageLoader, RawLoaderConfig
from meteor_core.pipeline import MeteorDetectionPipeline
from meteor_core.schema import DetectionParams, InputContext, PipelineConfig
from meteor_core.tuning import compare_tuning_proposals, tune_reproduction


class TestAnalysisReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.target = self.root / "frame002.ORF"
        self.reference = self.root / "frame001.ORF"
        self.target.write_bytes(b"original target RAW")
        self.reference.write_bytes(b"original reference RAW")
        self.output = self.root / "analysis"
        self.current = np.zeros((96, 96), np.uint16)
        self.current[20:45, 10:70] = 1000
        self.params = DetectionParams(
            diff_threshold=10, min_area=3, min_aspect_ratio=3, min_line_score=30
        )
        current = self.current
        reference_name = self.reference.name

        def load(loader, filepath):
            image = (
                np.zeros_like(current)
                if Path(filepath).name == reference_name
                else current.copy()
            )
            return InputContext(
                image,
                filepath,
                metadata={"camera_model": "Test camera", "iso": np.int64(4000)},
            )

        loader_patch = patch.object(RawImageLoader, "load", load)
        loader_patch.start()
        self.addCleanup(loader_patch.stop)
        self.loader = RawImageLoader(RawLoaderConfig())

    def replay(self, *, roi=None, reference=True, progress=None):
        settings = load_reproduction_settings(
            params=self.params,
            progress_path=progress,
            roi_polygon=roi,
            use_full_image=roi is None and progress is None,
        )
        return reproduce_hough_pair(
            self.target,
            self.reference if reference else None,
            settings=settings,
            input_loader=self.loader,
        )

    def manifest(self):
        return json.loads((self.output / "report.json").read_text())

    def run_cli(self, arguments):
        out, err = io.StringIO(), io.StringIO()
        with (
            patch.object(
                sys, "argv", ["detect_meteors_cli.py", "--locale", "en", *arguments]
            ),
            patch.object(cli, "_configure_logging"),
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            try:
                cli.main()
                code = 0
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_manifest_and_pngs_retain_rejected_frame_evidence_and_hashes(self):
        replay = self.replay()
        save_analysis_bundle(replay, self.output)
        report = self.manifest()
        self.assertEqual(report["format"], "detect_meteors.analysis")
        self.assertEqual(report["schema_version"], 1)
        self.assertFalse(report["baseline"]["decision"]["is_candidate"])
        measurement = report["measurements"][report["baseline"]["measurement"]]
        self.assertEqual(measurement["score"], replay.analysis.score)
        self.assertEqual(measurement["lines"], list(map(list, replay.analysis.lines)))
        np.testing.assert_array_equal(
            cv2.imread(str(self.output / measurement["mask"]), cv2.IMREAD_GRAYSCALE),
            replay.analysis.mask,
        )
        np.testing.assert_array_equal(
            cv2.imread(str(self.output / report["baseline"]["debug_image"])),
            replay.analysis.render_debug(replay.decision),
        )
        for name, asset in report["assets"].items():
            data = (self.output / name).read_bytes()
            self.assertEqual(
                asset, {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
        self.assertNotIn("report.json", report["assets"])
        for source in report["inputs"]:
            self.assertEqual(
                source["sha256"],
                hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest(),
            )
        self.assertEqual(report["metadata"]["target"]["iso"], 4000)
        self.assertIsNone(report["session_configuration"])
        self.assertFalse((self.output / "session.yaml").exists())

    def test_tuning_report_contains_grid_comparison_and_exact_proposal_values(self):
        replay = self.replay()
        run = compare_tuning_proposals(
            tune_reproduction(replay),
            [(self.target, self.reference)],
            known_meteors=[self.target],
        )
        save_analysis_bundle(replay, self.output, tuning=run)
        report = self.manifest()
        proposal = report["proposals"][0]
        self.assertEqual(proposal["params"], run.proposals[0].params.to_dict())
        self.assertTrue(proposal["comparison"]["complete"])
        self.assertEqual(
            proposal["comparison"]["known_candidate_frames"], [str(self.target)]
        )
        self.assertEqual(report["tuning"]["measurement_grid"]["diff_threshold"], [10])
        self.assertEqual(proposal["measurement"], report["baseline"]["measurement"])
        self.assertIsNone(report["selected"])

    def test_fresh_measurement_proposal_has_distinct_mask_and_replayable_settings(self):
        self.current[20:45, 10:70] = 5
        replay = self.replay()
        run = tune_reproduction(replay, measurement_grid={"diff_threshold": [4]})
        save_analysis_bundle(replay, self.output, tuning=run, selection=1)
        report = self.manifest()
        baseline = report["measurements"][report["baseline"]["measurement"]]
        proposal = report["proposals"][0]
        measured = report["measurements"][proposal["measurement"]]
        self.assertEqual(baseline["mask_pixels"], 0)
        self.assertGreater(measured["mask_pixels"], 0)
        self.assertNotEqual(baseline["mask"], measured["mask"])
        self.assertEqual(measured["params"]["diff_threshold"], 4)
        session = self.output / "session.yaml"
        session_data = yaml.safe_load(session.read_text(encoding="utf-8"))
        self.assertEqual(session_data.pop("schema_version"), 1)
        legacy_session = self.root / "legacy_session.yaml"
        legacy_session.write_text(yaml.safe_dump(session_data), encoding="utf-8")
        config = load_pipeline_config(session)
        self.assertEqual(
            config.to_dict(), load_pipeline_config(legacy_session).to_dict()
        )
        self.assertEqual(config.params.to_dict(), proposal["params"])
        code, _, err = self.run_cli(["--config", str(session)])
        self.assertEqual(code, 0, err)
        progress = json.loads((self.output / "run/progress.json").read_text())
        self.assertEqual(progress["total_detected"], 1)

    def test_non_json_metadata_is_not_mistaken_for_replayable_configuration(self):
        replay = replace(
            self.replay(), target_metadata={"value": float("nan"), "object": object()}
        )
        save_analysis_bundle(replay, self.output)
        report = self.manifest()
        self.assertEqual(report["metadata"]["target"]["value"], "nan")
        self.assertEqual(len(report["serialization_notes"]), 2)
        self.assertIsNone(report["session_configuration"])

    def test_comparison_only_source_changes_invalidate_export(self):
        replay = self.replay()
        other = self.root / "frame003.ORF"
        other.write_bytes(b"comparison target")
        run = compare_tuning_proposals(
            tune_reproduction(replay), [(other, self.reference)]
        )
        other.write_bytes(b"changed comparison target")
        with self.assertRaises(MeteorValidationError):
            save_analysis_bundle(replay, self.output, tuning=run)
        self.assertFalse(self.output.exists())

    def test_selected_session_round_trips_loader_roi_and_strict_float_boundary(self):
        polygon = [[0, 0], [90, 0], [90, 90], [0, 90]]
        replay = self.replay(roi=polygon)
        run = tune_reproduction(replay)
        save_analysis_bundle(replay, self.output, tuning=run, selection=1)
        config = load_pipeline_config(self.output / "session.yaml")
        self.assertEqual(config.params.to_dict(), run.proposals[0].params.to_dict())
        self.assertEqual(config.input_loader_config, {"binning": 2, "normalize": False})
        self.assertEqual(config.roi_polygon, polygon)
        self.assertFalse(config.enable_roi_selection)
        self.assertFalse(config.output_overwrite)
        self.assertEqual(config.progress_file, str(self.output / "run/progress.json"))
        self.assertFalse((self.output / "run").exists())
        settings = load_reproduction_settings(
            params=config.params, fallback_roi_polygon=config.roi_polygon
        )
        again = reproduce_hough_pair(
            self.target,
            self.reference,
            settings=settings,
            loader_name=config.input_loader_name,
            loader_config=config.input_loader_config,
        )
        self.assertEqual(again.decision, run.proposals[0].decision)
        self.assertEqual(again.analysis.lines, replay.analysis.lines)
        self.assertEqual(self.manifest()["selected"], {"kind": "proposal", "index": 1})

    def test_saved_config_drives_normal_pipeline_without_interactive_roi(self):
        replay = self.replay()
        run = tune_reproduction(replay)
        save_analysis_bundle(replay, self.output, tuning=run, selection="1")
        config = load_pipeline_config(self.output / "session.yaml")
        before = self.target.read_bytes(), self.reference.read_bytes()
        with (
            patch(
                "meteor_core.pipeline.select_roi",
                side_effect=AssertionError("unexpected GUI"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            pipeline = MeteorDetectionPipeline(config)
            count = pipeline.run()
        self.assertEqual(count, 1)
        actual = pipeline._all_sorted_detections[0]
        self.assertEqual(actual.is_candidate, run.proposals[0].decision.is_candidate)
        self.assertEqual(
            actual.aspect_ratio, run.proposals[0].decision.max_aspect_ratio
        )
        self.assertEqual(actual.score, replay.analysis.score)
        self.assertEqual(actual.lines, replay.analysis.lines)
        self.assertEqual(
            before, (self.target.read_bytes(), self.reference.read_bytes())
        )
        self.assertEqual(
            (self.output / "run/candidates" / self.target.name).read_bytes(),
            self.target.read_bytes(),
        )
        # Normal progress writes the legacy full-image marker; analysis can read it.
        settings = load_reproduction_settings(progress_path=config.progress_file)
        self.assertIsNone(settings.roi_polygon)
        self.assertEqual(settings.params.to_dict(), config.params.to_dict())

    def test_normal_cli_can_explicitly_override_saved_polygon_with_full_image(self):
        replay = self.replay(roi=[[0, 0], [8, 0], [8, 95], [0, 95]])
        save_analysis_bundle(replay, self.output, selection="baseline")
        with patch(
            "meteor_core.pipeline.select_roi",
            side_effect=AssertionError("unexpected GUI"),
        ):
            code, out, err = self.run_cli(
                [
                    "--config",
                    str(self.output / "session.yaml"),
                    "--no-roi",
                    "--min-aspect-ratio",
                    "2.0",
                ]
            )
        self.assertEqual(code, 0, err)
        progress = json.loads((self.output / "run/progress.json").read_text())
        self.assertEqual(progress["total_detected"], 1)
        self.assertEqual(progress["roi"], "full_image")

    def test_report_only_missing_reference_does_not_fabricate_images_or_decisions(self):
        replay = self.replay(reference=False)
        save_analysis_bundle(replay, self.output)
        report = self.manifest()
        self.assertEqual(report["status"], "reference_missing")
        self.assertIsNone(report["baseline"]["decision"])
        self.assertEqual(report["measurements"], {})
        self.assertEqual(report["assets"], {})
        self.assertEqual(list(self.output.iterdir()), [self.output / "report.json"])

    def test_existing_file_directory_symlink_and_raw_are_never_replaced(self):
        replay = self.replay()
        folder = self.root / "existing"
        folder.mkdir()
        marker = folder / "keep.txt"
        marker.write_bytes(b"keep")
        link = self.root / "link"
        link.symlink_to(self.target)
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "not-created")
        for path in (self.target, folder, link, dangling):
            with self.subTest(path=path), self.assertRaises(MeteorWriteError):
                save_analysis_bundle(replay, path)
        self.assertEqual(self.target.read_bytes(), b"original target RAW")
        self.assertEqual(marker.read_bytes(), b"keep")
        self.assertTrue(link.is_symlink())
        self.assertTrue(dangling.is_symlink())
        self.assertFalse((self.root / "not-created").exists())

    def test_invalid_selection_or_missing_reference_leaves_no_outputs(self):
        replay = self.replay()
        run = tune_reproduction(replay)
        for selection in (0, -1, True, 1.5, "unknown", 99):
            with (
                self.subTest(selection=selection),
                self.assertRaises(MeteorConfigError),
            ):
                save_analysis_bundle(
                    replay, self.output, tuning=run, selection=selection
                )
            self.assertFalse(self.output.exists())
        with self.assertRaises(MeteorConfigError):
            save_analysis_bundle(
                self.replay(reference=False), self.output, selection="baseline"
            )
        with self.assertRaises(MeteorConfigError):
            save_analysis_bundle(self.replay(), self.output, tuning=run)
        self.assertFalse(self.output.exists())

    def test_changed_sources_are_rejected_before_creating_the_bundle(self):
        replay = self.replay()
        self.target.write_bytes(b"changed")
        with self.assertRaises(MeteorValidationError):
            save_analysis_bundle(replay, self.output)
        self.assertFalse(self.output.exists())
        replay = self.replay()
        config = self.root / "source.json"
        config.write_text("{}")
        stamp = source_stamp(config)
        config.write_text('{"changed":true}')
        with self.assertRaises(MeteorValidationError):
            save_analysis_bundle(replay, self.output, source_files={str(config): stamp})
        self.assertFalse(self.output.exists())

    def test_mutated_parameters_cannot_be_saved_as_historical_evidence(self):
        replay = self.replay()
        replay.settings.params.min_aspect_ratio = 2
        with self.assertRaises(MeteorValidationError):
            save_analysis_bundle(replay, self.output)
        replay = self.replay()
        run = tune_reproduction(replay)
        run.proposals[0].params.min_aspect_ratio = 1.8
        with self.assertRaises(MeteorValidationError):
            save_analysis_bundle(replay, self.output, tuning=run, selection=1)
        self.assertFalse(self.output.exists())

    def test_failed_image_write_cleans_up_only_new_artifacts(self):
        replay = self.replay()
        output = self.root / "new-parent" / "analysis"
        original = cv2.imencode
        calls = 0

        def encode(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated disk failure")
            return original(*args)

        with (
            patch("meteor_core.analysis_reports.cv2.imencode", side_effect=encode),
            self.assertRaises(MeteorWriteError),
        ):
            save_analysis_bundle(replay, output)
        self.assertFalse((self.root / "new-parent").exists())
        self.assertEqual(self.target.read_bytes(), b"original target RAW")

    def test_failed_write_does_not_delete_an_unowned_file_in_the_new_directory(self):
        replay = self.replay()

        def encode(*args):
            (self.output / "keep.txt").write_bytes(b"unowned")
            raise OSError("simulated failure")

        with (
            patch("meteor_core.analysis_reports.cv2.imencode", side_effect=encode),
            self.assertRaises(MeteorWriteError),
        ):
            save_analysis_bundle(replay, self.output)
        self.assertEqual((self.output / "keep.txt").read_bytes(), b"unowned")
        self.assertFalse((self.output / "report.json").exists())

    def test_unsupported_session_configuration_is_not_silently_changed(self):
        replay = self.replay()
        for data in (
            {"hooks": [{"name": "aircraft_trail"}]},
            {"detector_name": "simple_threshold"},
            {"output_handler_name": "custom"},
            {"output_handler_config": {"unknown": 3}},
        ):
            with self.subTest(data=data), self.assertRaises(MeteorConfigError):
                save_analysis_bundle(
                    replay,
                    self.output,
                    selection="baseline",
                    pipeline_config=PipelineConfig.from_partial_dict(data),
                )
        bad = replace(replay, loader_config_available=False)
        with self.assertRaises(MeteorConfigError):
            save_analysis_bundle(bad, self.output, selection="baseline")
        self.assertFalse(self.output.exists())

    def test_config_roi_defaults_validation_and_progress_precedence(self):
        path = self.root / "config.json"
        path.write_text(
            '{"roi_polygon":[[0,0],[8,0],[8,95]],"enable_roi_selection":false}'
        )
        config = load_pipeline_config(path)
        self.assertFalse(config.enable_roi_selection)
        polygon = config.roi_polygon
        settings = load_reproduction_settings(fallback_roi_polygon=polygon)
        self.assertEqual(settings.roi_source, "supplied.roi")
        progress = self.root / "progress.json"
        progress.write_text(
            json.dumps(
                {
                    "processed_files": [],
                    "params": self.params.to_dict(),
                    "roi": "full_image",
                }
            )
        )
        settings = load_reproduction_settings(
            progress_path=progress, fallback_roi_polygon=polygon
        )
        self.assertIsNone(settings.roi_polygon)
        self.assertEqual(settings.roi_source, "progress.roi")
        for data in ({"enable_roi_selection": "false"}, {"roi_polygon": [[0, 0]]}):
            path.write_text(json.dumps(data))
            with self.assertRaises(MeteorConfigError):
                load_pipeline_config(path)

    def test_cli_exports_report_and_selected_config_then_reloads_without_roi_gui(self):
        args = [
            "--analyze-image",
            str(self.target),
            "--reference-image",
            str(self.reference),
            "--diff-threshold",
            "10",
            "--min-area",
            "3",
            "--min-line-score",
            "30",
            "--tune-params",
            "--no-roi",
            "--analysis-output",
            str(self.output),
            "--analysis-proposal",
            "1",
        ]
        code, out, err = self.run_cli(args)
        self.assertEqual(code, 0, err)
        self.assertIn("Analysis bundle saved:", out)
        report = self.manifest()
        self.assertTrue(report["proposals"][0]["decision"]["is_candidate"])
        with patch(
            "meteor_core.pipeline.select_roi",
            side_effect=AssertionError("unexpected GUI"),
        ):
            code, out, err = self.run_cli(
                ["--config", str(self.output / "session.yaml")]
            )
        self.assertEqual(code, 0, err)
        progress = json.loads((self.output / "run/progress.json").read_text())
        self.assertEqual(progress["total_detected"], 1)
        self.assertEqual(
            progress["detected_details"][0]["ratio"],
            report["proposals"][0]["decision"]["max_aspect_ratio"],
        )
        code, out, err = self.run_cli(
            [
                "--analyze-image",
                str(self.target),
                "--reference-image",
                str(self.reference),
                "--config",
                str(self.output / "session.yaml"),
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("Candidate: True", out)
        self.assertNotIn("full image was assumed", out)

    def test_cli_rejects_unsupported_pipeline_version_before_processing(self):
        path = self.root / "unsupported.yaml"
        path.write_text("schema_version: 2\n", encoding="utf-8")
        with (
            patch.object(cli, "MeteorDetectionPipeline") as pipeline,
            patch("meteor_core.pipeline._resolve_detector") as detector,
        ):
            for mode in ([], ["--analyze-image", str(self.target)]):
                with self.subTest(mode=mode):
                    code, _, err = self.run_cli(["--config", str(path), *mode])
                    self.assertEqual(code, 1)
                    self.assertIn("schema_version 2", err)
                    self.assertIn("Expected integer 1", err)
                    self.assertIn(path.name, err)
        pipeline.assert_not_called()
        detector.assert_not_called()

    def test_cli_output_modifiers_require_explicit_analysis_and_selection(self):
        for options in (
            ["--analysis-output", str(self.output)],
            ["--analyze-image", str(self.target), "--analysis-proposal", "baseline"],
            [
                "--analyze-image",
                str(self.target),
                "--analysis-output",
                str(self.output),
                "--analysis-proposal",
                "1",
            ],
        ):
            with self.subTest(options=options):
                code, _, _ = self.run_cli(options)
                self.assertEqual(code, 2)
                self.assertFalse(self.output.exists())

    def test_cli_can_save_reference_missing_report_with_exit_two(self):
        code, out, err = self.run_cli(
            [
                "--analyze-image",
                str(self.target),
                "--analysis-output",
                str(self.output),
                "--locale",
                "ja",
            ]
        )
        self.assertEqual(code, 2, err)
        self.assertIn("分析結果を保存しました", out)
        self.assertIsNone(self.manifest()["baseline"]["decision"])


if __name__ == "__main__":
    unittest.main()
