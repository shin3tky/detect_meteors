"""Application runner for detect_meteors CLI."""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from detect_meteors import exif as exif_utils
from detect_meteors import npf
from detect_meteors import plugin_loader
from detect_meteors import services


@dataclass(frozen=True)
class PluginInfo:
    """Metadata describing a plugin implementation."""

    name: str
    version: str
    capabilities: List[str]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Plugin name must be provided")
        if not self.version:
            raise ValueError("Plugin version must be provided")
        if not self.capabilities:
            raise ValueError("Plugin capabilities must be provided")
        if any(not capability for capability in self.capabilities):
            raise ValueError("Plugin capabilities must not contain empty values")


@dataclass
class RegisteredPlugin:
    implementation: Any
    info: PluginInfo


@runtime_checkable
class Preprocessor(Protocol):
    """Prepare input data for meteor detection."""

    plugin_info: PluginInfo

    def preprocess(self, target_folder: str) -> str:
        """Return the folder that should be passed to the detector."""


class Detector(ABC):
    """Perform meteor detection against a prepared target folder."""

    plugin_info: PluginInfo

    @abstractmethod
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
        """Run detection and return the count of detected meteors."""

    # Optional lifecycle hooks
    def initialize(self) -> None:  # pragma: no cover - optional
        """Prepare the detector before use."""
        return None

    def shutdown(self) -> None:  # pragma: no cover - optional
        """Clean up any detector resources."""
        return None


@runtime_checkable
class OutputWriter(Protocol):
    """Persist or format the detection results."""

    plugin_info: PluginInfo

    def write(self, detected_count: int, warnings: List[str]) -> Dict[str, Any]:
        """Return a serialized representation of the run result."""


class DefaultPreprocessor:
    """Pass-through preprocessor used by default."""

    plugin_info = PluginInfo(
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


class AdvancedDetector(Detector):
    """Wrap the existing advanced detector implementation."""

    plugin_info = PluginInfo(
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

    plugin_info = PluginInfo(
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


_DEFAULT_IMPLEMENTATION = "default"
_DETECTOR_REGISTRY: Dict[str, RegisteredPlugin] = {}
_PREPROCESSOR_REGISTRY: Dict[str, RegisteredPlugin] = {}
_OUTPUT_WRITER_REGISTRY: Dict[str, RegisteredPlugin] = {}


def _call_lifecycle_hook(plugin: Any, hook_name: str) -> None:
    hook = getattr(plugin, hook_name, None)
    if callable(hook):
        hook()


def _resolve_plugin_info(
    name: str, plugin: Any, plugin_info: Optional[PluginInfo]
) -> PluginInfo:
    info = plugin_info or getattr(plugin, "plugin_info", None)
    if info is None:
        raise ValueError(f"Plugin info must be provided for '{name}'")
    if info.name != name:
        raise ValueError(
            f"Plugin info name '{info.name}' does not match registry key '{name}'"
        )
    return info


def register_detector(
    name: str, detector: Detector, *, override: bool = False, plugin_info: Optional[PluginInfo] = None
) -> None:
    if name in _DETECTOR_REGISTRY and not override:
        raise ValueError(f"Detector '{name}' already registered")
    if name in _DETECTOR_REGISTRY:
        _call_lifecycle_hook(_DETECTOR_REGISTRY[name].implementation, "shutdown")
    info = _resolve_plugin_info(name, detector, plugin_info)
    _call_lifecycle_hook(detector, "initialize")
    _DETECTOR_REGISTRY[name] = RegisteredPlugin(detector, info)


def register_preprocessor(
    name: str,
    preprocessor: Preprocessor,
    *,
    override: bool = False,
    plugin_info: Optional[PluginInfo] = None,
) -> None:
    if name in _PREPROCESSOR_REGISTRY and not override:
        raise ValueError(f"Preprocessor '{name}' already registered")
    if name in _PREPROCESSOR_REGISTRY:
        _call_lifecycle_hook(_PREPROCESSOR_REGISTRY[name].implementation, "shutdown")
    info = _resolve_plugin_info(name, preprocessor, plugin_info)
    _call_lifecycle_hook(preprocessor, "initialize")
    _PREPROCESSOR_REGISTRY[name] = RegisteredPlugin(preprocessor, info)


def register_output_writer(
    name: str,
    output_writer: OutputWriter,
    *,
    override: bool = False,
    plugin_info: Optional[PluginInfo] = None,
) -> None:
    if name in _OUTPUT_WRITER_REGISTRY and not override:
        raise ValueError(f"Output writer '{name}' already registered")
    if name in _OUTPUT_WRITER_REGISTRY:
        _call_lifecycle_hook(_OUTPUT_WRITER_REGISTRY[name].implementation, "shutdown")
    info = _resolve_plugin_info(name, output_writer, plugin_info)
    _call_lifecycle_hook(output_writer, "initialize")
    _OUTPUT_WRITER_REGISTRY[name] = RegisteredPlugin(output_writer, info)


def unregister_detector(name: str) -> None:
    try:
        registered = _DETECTOR_REGISTRY.pop(name)
    except KeyError as exc:
        raise KeyError(f"Detector '{name}' not registered") from exc
    _call_lifecycle_hook(registered.implementation, "shutdown")


def unregister_preprocessor(name: str) -> None:
    try:
        registered = _PREPROCESSOR_REGISTRY.pop(name)
    except KeyError as exc:
        raise KeyError(f"Preprocessor '{name}' not registered") from exc
    _call_lifecycle_hook(registered.implementation, "shutdown")


def unregister_output_writer(name: str) -> None:
    try:
        registered = _OUTPUT_WRITER_REGISTRY.pop(name)
    except KeyError as exc:
        raise KeyError(f"Output writer '{name}' not registered") from exc
    _call_lifecycle_hook(registered.implementation, "shutdown")


def get_detector(name: str = _DEFAULT_IMPLEMENTATION) -> Detector:
    try:
        return _DETECTOR_REGISTRY[name].implementation
    except KeyError as exc:
        raise KeyError(f"Detector '{name}' not registered") from exc


def get_preprocessor(name: str = _DEFAULT_IMPLEMENTATION) -> Preprocessor:
    try:
        return _PREPROCESSOR_REGISTRY[name].implementation
    except KeyError as exc:
        raise KeyError(f"Preprocessor '{name}' not registered") from exc


def get_output_writer(name: str = _DEFAULT_IMPLEMENTATION) -> OutputWriter:
    try:
        return _OUTPUT_WRITER_REGISTRY[name].implementation
    except KeyError as exc:
        raise KeyError(f"Output writer '{name}' not registered") from exc


# Register built-in implementations for immediate use.
register_detector(_DEFAULT_IMPLEMENTATION, AdvancedDetector())
register_preprocessor(_DEFAULT_IMPLEMENTATION, DefaultPreprocessor())
register_output_writer(_DEFAULT_IMPLEMENTATION, DefaultOutputWriter())


def _sensor_type_listing() -> Dict[str, Any]:
    primary_types = [
        "1INCH",
        "MFT",
        "APSC",
        "APSC_CANON",
        "APSH",
        "FF",
        "MF44X33",
        "MF54X40",
    ]

    listing: List[Dict[str, Any]] = []
    for sensor_type in primary_types:
        preset = services.SENSOR_PRESETS.get(sensor_type)
        if preset:
            listing.append(
                {
                    "type": sensor_type,
                    "description": preset["description"],
                    "focal_factor": preset["focal_factor"],
                    "sensor_width": preset["sensor_width"],
                    "pixel_pitch": preset["pixel_pitch"],
                }
            )

    aliases = {
        "1INCH": ["1-INCH", "1_INCH"],
        "APSC": ["APS-C", "APS_C"],
        "APSC_CANON": ["APS-C_CANON"],
        "APSH": ["APS-H", "APS_H"],
        "FF": ["FULLFRAME"],
        "MF44X33": ["MF44-33", "MF44_33"],
        "MF54X40": ["MF54-40", "MF54_40"],
    }

    return {"list": listing, "aliases": aliases}


def run(args):
    """Execute the application logic using parsed arguments."""

    if args.remove_progress:
        removed = False
        if os.path.exists(args.progress_file):
            os.remove(args.progress_file)
            removed = True
        return {
            "action": "remove_progress",
            "progress_file": args.progress_file,
            "removed": removed,
        }

    if args.list_sensor_types:
        return {"action": "list_sensor_types", "data": _sensor_type_listing()}

    roi_polygon_cli = None
    enable_roi_selection = services.DEFAULT_ENABLE_ROI_SELECTION

    if args.roi is not None:
        roi_polygon_cli = services.parse_roi_polygon_string(args.roi)
        enable_roi_selection = False
    elif args.no_roi:
        enable_roi_selection = False

    user_specified_diff_threshold = "--diff-threshold" in sys.argv
    user_specified_min_area = "--min-area" in sys.argv
    user_specified_min_line_score = "--min-line-score" in sys.argv

    if args.sensor_type and services.get_sensor_preset(args.sensor_type) is None:
        raise ValueError(
            f"Invalid --sensor-type value: '{args.sensor_type}'. "
            "Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40"
        )

    (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    ) = services.apply_sensor_preset(args, verbose=False)

    warnings = services.validate_sensor_overrides(
        args, preset, sensor_width_value, pixel_pitch_value, collect_only=True
    )

    if args.focal_factor and focal_factor_value is None:
        raise ValueError(
            f"Invalid --focal-factor value: '{args.focal_factor}'. "
            "Valid values: MFT, APS-C, APS-H, FF, or numeric (e.g., 2.0)"
        )

    if args.show_exif or args.show_npf:
        files = services.collect_files(args.target)
        if not files:
            raise FileNotFoundError("No RAW files found in target folder.")

        exif_data = services.extract_exif_metadata(files[0])

        focal_length_source = "Unknown"
        if focal_length_value:
            focal_length_source = "CLI (--focal-length)"
            exif_data["focal_length_35mm"] = focal_length_value
        elif exif_data.get("focal_length_35mm"):
            focal_length_source = "EXIF"
        elif exif_data.get("focal_length") and focal_factor_value:
            if args.focal_factor:
                focal_length_source = f"Calculated (--focal-factor {args.focal_factor})"
            else:
                focal_length_source = f"Calculated (--sensor-type {args.sensor_type})"
            exif_data["focal_length_35mm"] = exif_data["focal_length"] * focal_factor_value
        elif exif_data.get("focal_length"):
            focal_length_source = "EXIF (actual, no 35mm equiv.)"

        npf_metrics = None
        if args.show_npf or (
            exif_data.get("focal_length_35mm") and exif_data.get("f_number")
        ):
            npf_metrics = services.calculate_npf_metrics(
                exif_data,
                sensor_width_mm=sensor_width_value,
                pixel_pitch_um=pixel_pitch_value,
                fisheye=args.fisheye,
                fisheye_model=services.DEFAULT_FISHEYE_MODEL,
            )

        if args.auto_params and (
            exif_data.get("focal_length_35mm") is None
            or exif_data.get("exposure_time") is None
            or exif_data.get("f_number") is None
        ):
            warnings.append(
                "Auto-parameter estimation requires focal_length_35mm, exposure_time, and f_number"
            )
        if (
            exif_data.get("focal_length_35mm") is None
            and exif_data.get("focal_length") is None
            and args.focal_factor is None
            and args.sensor_type is None
        ):
            warnings.append(
                "35mm equivalent not found. Consider using --sensor-type or --focal-factor"
            )
        if not exif_data.get("iso"):
            warnings.append("ISO value not available")
        if not exif_data.get("exposure_time"):
            warnings.append("Exposure time not available")

        warnings.extend(
            npf.build_warnings(
                exif_data=exif_data,
                npf_metrics=npf_metrics,
                auto_params=args.auto_params,
                sensor_type=args.sensor_type,
                focal_factor_arg=args.focal_factor,
                sensor_width_value=sensor_width_value,
                show_npf=args.show_npf,
            )
        )

        fisheye_text = None
        if args.fisheye and exif_data.get("focal_length_35mm"):
            fisheye_text = exif_utils.format_fisheye_info(
                exif_data["focal_length_35mm"], services.DEFAULT_FISHEYE_MODEL
            )

        return {
            "action": "show_exif",
            "target": args.target,
            "files_found": len(files),
            "first_file": os.path.basename(files[0]),
            "exif_text": exif_utils.format_exif_info(
                exif_data,
                focal_length_source,
                focal_factor_value,
                npf_metrics,
            ),
            "fisheye_text": fisheye_text,
            "warnings": warnings,
            "show_usage_examples": args.show_npf,
        }

    plugin_loader.load_plugins()

    preprocessor = get_preprocessor()
    processed_target = preprocessor.preprocess(args.target)

    detector = get_detector()
    detected_count = detector.detect(
        target_folder=processed_target,
        output_folder=args.output,
        debug_folder=args.debug_dir,
        diff_threshold=args.diff_threshold,
        min_area=args.min_area,
        min_aspect_ratio=args.min_aspect_ratio,
        hough_threshold=args.hough_threshold,
        hough_min_line_length=args.hough_min_line_length,
        hough_max_line_gap=args.hough_max_line_gap,
        min_line_score=args.min_line_score,
        enable_roi_selection=enable_roi_selection,
        roi_polygon_cli=roi_polygon_cli,
        num_workers=args.workers,
        batch_size=args.batch_size,
        auto_batch_size=args.auto_batch_size,
        enable_parallel=not args.no_parallel,
        profile=args.profile,
        validate_raw=args.validate_raw,
        progress_file=args.progress_file,
        resume=not args.no_resume,
        auto_params=args.auto_params,
        user_specified_diff_threshold=user_specified_diff_threshold,
        user_specified_min_area=user_specified_min_area,
        user_specified_min_line_score=user_specified_min_line_score,
        focal_length_mm=focal_length_value,
        focal_factor=focal_factor_value,
        sensor_width_mm=sensor_width_value,
        pixel_pitch_um=pixel_pitch_value,
        output_overwrite=args.output_overwrite,
        fisheye=args.fisheye,
    )

    output_writer = get_output_writer()
    return output_writer.write(detected_count, warnings)
