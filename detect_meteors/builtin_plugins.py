"""Built-in plugin implementations for detect_meteors."""

from typing import Any, Dict, List

from detect_meteors import app, services


class DefaultPreprocessor:
    """Pass-through preprocessor used by default."""

    plugin_info = app.PluginInfo(
        name="default",
        version="1.0.0",
        capabilities=["pass_through_preprocessor"],
    )

    def initialize(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def preprocess(self, target_folder: str) -> str:
        return target_folder


class AdvancedDetector(app.Detector):
    """Wrap the existing advanced detector implementation."""

    plugin_info = app.PluginInfo(
        name="default",
        version="1.0.0",
        capabilities=["advanced_detection"],
    )

    def detect(
        self,
        *,
        target_folder: str,
        output_folder: str,
        debug_folder: str,
        diff_threshold: float,
        min_area: int,
        min_aspect_ratio: float,
        hough_threshold: int,
        hough_min_line_length: int,
        hough_max_line_gap: int,
        min_line_score: float,
        enable_roi_selection: bool,
        roi_polygon_cli: Any,
        num_workers: int,
        batch_size: int,
        auto_batch_size: bool,
        enable_parallel: bool,
        profile: bool,
        validate_raw: bool,
        progress_file: str,
        resume: bool,
        auto_params: bool,
        user_specified_diff_threshold: bool,
        user_specified_min_area: bool,
        user_specified_min_line_score: bool,
        focal_length_mm: Any,
        focal_factor: Any,
        sensor_width_mm: Any,
        pixel_pitch_um: Any,
        output_overwrite: bool,
        fisheye: bool,
    ) -> int:
        return services.detect_meteors_advanced(
            target_folder=target_folder,
            output_folder=output_folder,
            debug_folder=debug_folder,
            diff_threshold=diff_threshold,
            min_area=min_area,
            min_aspect_ratio=min_aspect_ratio,
            hough_threshold=hough_threshold,
            hough_min_line_length=hough_min_line_length,
            hough_max_line_gap=hough_max_line_gap,
            min_line_score=min_line_score,
            enable_roi_selection=enable_roi_selection,
            roi_polygon_cli=roi_polygon_cli,
            num_workers=num_workers,
            batch_size=batch_size,
            auto_batch_size=auto_batch_size,
            enable_parallel=enable_parallel,
            profile=profile,
            validate_raw=validate_raw,
            progress_file=progress_file,
            resume=resume,
            auto_params=auto_params,
            user_specified_diff_threshold=user_specified_diff_threshold,
            user_specified_min_area=user_specified_min_area,
            user_specified_min_line_score=user_specified_min_line_score,
            focal_length_mm=focal_length_mm,
            focal_factor=focal_factor,
            sensor_width_mm=sensor_width_mm,
            pixel_pitch_um=pixel_pitch_um,
            output_overwrite=output_overwrite,
            fisheye=fisheye,
        )


class DefaultOutputWriter:
    """Return results in the format expected by the CLI runner."""

    plugin_info = app.PluginInfo(
        name="default",
        version="1.0.0",
        capabilities=["cli_output_writer"],
    )

    def initialize(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def write(self, detected_count: int, warnings: List[str]) -> Dict[str, Any]:
        return {"action": "detect", "detected_count": detected_count, "warnings": warnings}


DEFAULT_PLUGIN_NAME = app._DEFAULT_IMPLEMENTATION  # type: ignore[attr-defined]

DETECTORS = {DEFAULT_PLUGIN_NAME: AdvancedDetector()}
PREPROCESSORS = {DEFAULT_PLUGIN_NAME: DefaultPreprocessor()}
OUTPUT_WRITERS = {DEFAULT_PLUGIN_NAME: DefaultOutputWriter()}
