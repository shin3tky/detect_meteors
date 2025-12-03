import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("cv2", types.SimpleNamespace())

from detect_meteors import app, services


class TestAppLayer(unittest.TestCase):
    def make_args(self, **overrides):
        defaults = dict(
            remove_progress=False,
            progress_file=services.DEFAULT_PROGRESS_FILE,
            list_sensor_types=False,
            list_plugins=False,
            roi=None,
            no_roi=False,
            detector_plugin=None,
            preprocessor_plugin=None,
            output_writer_plugin=None,
            sensor_type=None,
            focal_length=None,
            focal_factor=None,
            sensor_width=None,
            pixel_pitch=None,
            show_exif=False,
            show_npf=False,
            target=services.DEFAULT_TARGET_FOLDER,
            output=services.DEFAULT_OUTPUT_FOLDER,
            debug_dir=services.DEFAULT_DEBUG_FOLDER,
            diff_threshold=services.DEFAULT_DIFF_THRESHOLD,
            min_area=services.DEFAULT_MIN_AREA,
            min_aspect_ratio=services.DEFAULT_MIN_ASPECT_RATIO,
            hough_threshold=services.DEFAULT_HOUGH_THRESHOLD,
            hough_min_line_length=services.DEFAULT_HOUGH_MIN_LINE_LENGTH,
            hough_max_line_gap=services.DEFAULT_HOUGH_MAX_LINE_GAP,
            min_line_score=services.DEFAULT_MIN_LINE_SCORE,
            workers=services.DEFAULT_NUM_WORKERS,
            batch_size=services.DEFAULT_BATCH_SIZE,
            auto_batch_size=False,
            no_parallel=False,
            profile=False,
            validate_raw=False,
            no_resume=False,
            auto_params=False,
            output_overwrite=False,
            fisheye=False,
            plugin_dir=None,
            plugin_entrypoint_group=None,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_invalid_sensor_type_rejected(self):
        args = self.make_args(sensor_type="INVALID")
        with self.assertRaises(ValueError):
            app.run(args)

    def test_roi_and_threshold_flags_propagated_to_detection(self):
        args = self.make_args(
            roi="0,0;1,1",
            diff_threshold=5,
            min_area=10,
            min_line_score=2.5,
        )

        argv_backup = sys.argv
        sys.argv = [
            "prog",
            "--diff-threshold",
            "5",
            "--min-area",
            "10",
            "--min-line-score",
            "2.5",
        ]

        with patch(
            "detect_meteors.services.parse_roi_polygon_string", return_value=[[0, 0], [1, 1]]
        ) as parse_roi, patch(
            "detect_meteors.services.apply_sensor_preset",
            return_value=(None, None, None, None, {}),
        ), patch("detect_meteors.services.validate_sensor_overrides", return_value=[]), patch(
            "detect_meteors.services.detect_meteors_advanced", return_value=7
        ) as detect_advanced:
            result = app.run(args)

        sys.argv = argv_backup

        self.assertEqual(result["action"], "detect")
        self.assertEqual(result["detected_count"], 7)
        parse_roi.assert_called_once_with("0,0;1,1")
        kwargs = detect_advanced.call_args.kwargs
        self.assertFalse(kwargs["enable_roi_selection"])
        self.assertEqual(kwargs["roi_polygon_cli"], [[0, 0], [1, 1]])
        self.assertTrue(kwargs["user_specified_diff_threshold"])
        self.assertTrue(kwargs["user_specified_min_area"])
        self.assertTrue(kwargs["user_specified_min_line_score"])

    def test_show_exif_path_formats_response(self):
        args = self.make_args(show_exif=True, target="folder")
        sample_exif = {
            "focal_length_35mm": 50.0,
            "f_number": 2.0,
            "exposure_time": 1.5,
            "iso": 800,
        }
        sample_npf = {
            "npf_max_exposure_sec": 10.0,
            "trail_score": 1.0,
            "motion_blur_score": 1.0,
            "rating": "A",
            "has_complete_data": True,
        }

        with patch("detect_meteors.services.collect_files", return_value=["file1.ORF"]), patch(
            "detect_meteors.services.extract_exif_metadata", return_value=sample_exif
        ), patch("detect_meteors.services.calculate_npf_metrics", return_value=sample_npf), patch(
            "detect_meteors.services.validate_sensor_overrides", return_value=[]
        ), patch("detect_meteors.npf.build_warnings", return_value=["warn"]), patch(
            "detect_meteors.exif.format_exif_info", return_value="EXIF"
        ), patch("detect_meteors.exif.format_fisheye_info", return_value="FISHEYE"):
            result = app.run(args)

        self.assertEqual(result["action"], "show_exif")
        self.assertEqual(result["files_found"], 1)
        self.assertEqual(result["warnings"], ["warn"])
        self.assertEqual(result["exif_text"], "EXIF")

    def test_list_plugins_returns_registered_metadata(self):
        args = self.make_args(list_plugins=True)

        with patch("detect_meteors.plugin_loader.load_plugins") as load_plugins:
            result = app.run(args)

        load_plugins.assert_called_once_with(plugin_folder=None, entrypoint_group=None)
        self.assertEqual(result["action"], "list_plugins")
        self.assertTrue(any(entry["name"] == "default" for entry in result["detectors"]))
        self.assertTrue(
            any(entry["name"] == "default" for entry in result["preprocessors"])
        )
        self.assertTrue(
            any(entry["name"] == "default" for entry in result["output_writers"])
        )
        self.assertIn("warnings", result)

    def test_specified_plugins_are_used_for_detection_pipeline(self):
        class CustomPreprocessor:
            plugin_info = app.PluginInfo(
                name="custom_pre", version="0.1.0", capabilities=["preprocessor"]
            )

            def __init__(self):
                self.seen_targets = []

            def preprocess(self, target_folder: str) -> str:
                self.seen_targets.append(target_folder)
                return f"custom:{target_folder}"

        class CustomDetector(app.Detector):
            plugin_info = app.PluginInfo(
                name="custom_detector", version="0.1.0", capabilities=["detector"]
            )

            def detect(self, **kwargs):  # type: ignore[override]
                self.kwargs = kwargs
                return 99

        class CustomWriter:
            plugin_info = app.PluginInfo(
                name="custom_writer", version="0.1.0", capabilities=["writer"]
            )

            def write(self, detected_count: int, warnings):
                return {"count": detected_count, "warnings": warnings, "writer": "custom"}

        preprocessor = CustomPreprocessor()
        detector = CustomDetector()
        writer = CustomWriter()

        app.register_preprocessor("custom_pre", preprocessor)
        app.register_detector("custom_detector", detector)
        app.register_output_writer("custom_writer", writer)

        args = self.make_args(
            preprocessor_plugin="custom_pre",
            detector_plugin="custom_detector",
            output_writer_plugin="custom_writer",
        )

        try:
            with patch("detect_meteors.services.apply_sensor_preset", return_value=(None, None, None, None, {})), patch(
                "detect_meteors.services.validate_sensor_overrides", return_value=[]
            ):
                result = app.run(args)
        finally:
            app.unregister_preprocessor("custom_pre")
            app.unregister_detector("custom_detector")
            app.unregister_output_writer("custom_writer")

        self.assertEqual(result["count"], 99)
        self.assertEqual(preprocessor.seen_targets, [services.DEFAULT_TARGET_FOLDER])
        self.assertEqual(detector.kwargs["target_folder"], "custom:rawfiles")


if __name__ == "__main__":
    unittest.main()
