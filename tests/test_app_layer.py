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
            roi=None,
            no_roi=False,
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


if __name__ == "__main__":
    unittest.main()
