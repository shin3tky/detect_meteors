#!/usr/bin/env python
#
# Detect Meteors CLI - Schema definitions
# © 2025 Shinichi Morita (shin3tky)
#

"""
Data structures, constants, and type definitions for meteor detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing as mp

# ==========================================
# Version
# ==========================================
VERSION = "1.6.1"

# ==========================================
# Default Settings
# ==========================================
DEFAULT_PROGRESS_FILE = "progress.json"
DEFAULT_LOADER_NAME = "raw"
DEFAULT_DETECTOR_NAME = "hough"
DEFAULT_OUTPUT_HANDLER_NAME = "file"

DEFAULT_TARGET_FOLDER = "rawfiles"
DEFAULT_OUTPUT_FOLDER = "candidates"
DEFAULT_DEBUG_FOLDER = "debug_masks"

EXTENSIONS = [
    "*.ORF",  # Olympus
    "*.RW2",  # Panasonic
    "*.X3F",  # Sigma
    "*.RWL",  # Leica
    "*.RAF",  # Fujifilm
    "*.ARW",  # Sony
    "*.SRF",  # Sony
    "*.SR2",  # Sony
    "*.CR2",  # Canon
    "*.CR3",  # Canon
    "*.CRW",  # Canon
    "*.NEF",  # Nikon
    "*.NRW",  # Nikon
    "*.FFF",  # Hasselblad
    "*.3FR",  # Hasselblad
    "*.DCR",  # Kodak
    "*.KDC",  # Kodak
    "*.KC2",  # Kodak
    "*.SRW",  # Samsung
    "*.RAW",  # RAW
    "*.DNG",  # Adobe
]

DEFAULT_DIFF_THRESHOLD = 8
DEFAULT_MIN_AREA = 10
DEFAULT_MIN_ASPECT_RATIO = 3.0

DEFAULT_HOUGH_THRESHOLD = 10
DEFAULT_HOUGH_MIN_LINE_LENGTH = 15
DEFAULT_HOUGH_MAX_LINE_GAP = 5
DEFAULT_MIN_LINE_SCORE = 80.0

DEFAULT_ENABLE_ROI_SELECTION = True
DEFAULT_NUM_WORKERS = max(1, mp.cpu_count() - 1)
DEFAULT_BATCH_SIZE = 10  # Batch processing size
AUTO_BATCH_MEMORY_FRACTION = 0.6  # Portion of free RAM to use when auto-sizing batches

# NPF Rule related default values
# Default pixel pitch for fallback (μm)
DEFAULT_PIXEL_PITCH_UM = 4.0  # Typical value for APS-C/MFT cameras

# ==========================================
# Fisheye Projection Models
# ==========================================
FISHEYE_PROJECTION_MODELS = {
    "EQUISOLID": {
        "name": "Equisolid Angle Projection",
        "description": "Equal-area projection (r = 2f × sin(θ/2))",
    },
    # Future projection models can be added here:
    # "EQUIDISTANT": {
    #     "name": "Equidistant Projection",
    #     "description": "Linear angle-to-distance mapping (r = f × θ)",
    # },
    # "STEREOGRAPHIC": {
    #     "name": "Stereographic Projection",
    #     "description": "Conformal projection (r = 2f × tan(θ/2))",
    # },
}

# Default fisheye projection model
DEFAULT_FISHEYE_MODEL = "EQUISOLID"

# ==========================================
# Sensor Presets
# ==========================================
# Each preset contains:
#   - focal_factor: Crop factor for 35mm equivalent conversion
#   - sensor_width: Sensor width in mm
#   - pixel_pitch: Typical pixel pitch in μm (None = calculate from resolution)
#   - description: Human-readable description
# Ordered by sensor size (smallest to largest)
SENSOR_PRESETS: Dict[str, Dict[str, Any]] = {
    # 1-inch sensor (smallest)
    "1INCH": {
        "focal_factor": 2.7,
        "sensor_width": 13.2,
        "pixel_pitch": 2.4,  # Typical for 20MP 1-inch (e.g., Sony RX100)
        "description": "1-inch sensor (13.2×8.8mm)",
    },
    "1_INCH": {
        "focal_factor": 2.7,
        "sensor_width": 13.2,
        "pixel_pitch": 2.4,
        "description": "1-inch sensor (13.2×8.8mm)",
    },
    # Micro Four Thirds
    "MFT": {
        "focal_factor": 2.0,
        "sensor_width": 17.3,
        "pixel_pitch": 3.7,  # Typical for 20MP MFT (e.g., OM-1, GH6)
        "description": "Micro Four Thirds (17.3×13mm)",
    },
    # APS-C (Sony/Nikon/Fuji)
    "APSC": {
        "focal_factor": 1.5,
        "sensor_width": 23.5,
        "pixel_pitch": 3.9,  # Typical for 26MP APS-C (e.g., Sony a6700, Fuji X-T5)
        "description": "APS-C Sony/Nikon/Fuji (23.5×15.6mm)",
    },
    "APS_C": {
        "focal_factor": 1.5,
        "sensor_width": 23.5,
        "pixel_pitch": 3.9,
        "description": "APS-C Sony/Nikon/Fuji (23.5×15.6mm)",
    },
    # APS-C (Canon)
    "APSC_CANON": {
        "focal_factor": 1.6,
        "sensor_width": 22.3,
        "pixel_pitch": 3.2,  # Typical for 32MP Canon APS-C (e.g., R7)
        "description": "APS-C Canon (22.3×14.9mm)",
    },
    "APS_C_CANON": {
        "focal_factor": 1.6,
        "sensor_width": 22.3,
        "pixel_pitch": 3.2,
        "description": "APS-C Canon (22.3×14.9mm)",
    },
    # APS-H
    "APSH": {
        "focal_factor": 1.3,
        "sensor_width": 27.9,
        "pixel_pitch": 5.7,  # Typical for 16MP APS-H (e.g., Canon 1D Mark IV)
        "description": "APS-H Canon (27.9×18.6mm)",
    },
    "APS_H": {
        "focal_factor": 1.3,
        "sensor_width": 27.9,
        "pixel_pitch": 5.7,
        "description": "APS-H Canon (27.9×18.6mm)",
    },
    # Full Frame 35mm
    "FF": {
        "focal_factor": 1.0,
        "sensor_width": 36.0,
        "pixel_pitch": 4.3,  # Typical for 45-50MP FF (e.g., Sony a7RV, Canon R5)
        "description": "Full Frame 35mm (36×24mm)",
    },
    "FULLFRAME": {
        "focal_factor": 1.0,
        "sensor_width": 36.0,
        "pixel_pitch": 4.3,
        "description": "Full Frame 35mm (36×24mm)",
    },
    # Medium Format 44x33 (Fujifilm GFX, Pentax 645Z, Hasselblad X)
    "MF44X33": {
        "focal_factor": 0.79,
        "sensor_width": 43.8,
        "pixel_pitch": 3.76,  # Typical for 100MP (e.g., GFX100, X2D 100C)
        "description": "Medium Format 44×33 (43.8×32.9mm) - GFX/645Z/X2D",
    },
    "MF44_33": {
        "focal_factor": 0.79,
        "sensor_width": 43.8,
        "pixel_pitch": 3.76,
        "description": "Medium Format 44×33 (43.8×32.9mm) - GFX/645Z/X2D",
    },
    # Medium Format 54x40 (Hasselblad H6D-100c)
    "MF54X40": {
        "focal_factor": 0.64,
        "sensor_width": 53.4,
        "pixel_pitch": 4.6,  # 100MP H6D-100c (11600×8700, 4.6μm)
        "description": "Medium Format 54×40 (53.4×40mm) - Hasselblad H6D-100c",
    },
    "MF54_40": {
        "focal_factor": 0.64,
        "sensor_width": 53.4,
        "pixel_pitch": 4.6,
        "description": "Medium Format 54×40 (53.4×40mm) - Hasselblad H6D-100c",
    },
}

# Legacy compatibility: CROP_FACTORS dictionary for parse_focal_factor()
CROP_FACTORS = {key: preset["focal_factor"] for key, preset in SENSOR_PRESETS.items()}

# Legacy compatibility: DEFAULT_SENSOR_WIDTHS dictionary
DEFAULT_SENSOR_WIDTHS = {
    key: preset["sensor_width"] for key, preset in SENSOR_PRESETS.items()
}


# ==========================================
# Data Classes
# ==========================================
@dataclass
class HoughParams:
    """Hough transform parameters."""

    threshold: int = DEFAULT_HOUGH_THRESHOLD
    min_line_length: int = DEFAULT_HOUGH_MIN_LINE_LENGTH
    max_line_gap: int = DEFAULT_HOUGH_MAX_LINE_GAP

    def to_dict(self) -> Dict[str, int]:
        return {
            "threshold": self.threshold,
            "min_line_length": self.min_line_length,
            "max_line_gap": self.max_line_gap,
        }


@dataclass
class DetectionParams:
    """Detection parameters for meteor detection."""

    diff_threshold: int = DEFAULT_DIFF_THRESHOLD
    min_area: int = DEFAULT_MIN_AREA
    min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO
    hough_threshold: int = DEFAULT_HOUGH_THRESHOLD
    hough_min_line_length: int = DEFAULT_HOUGH_MIN_LINE_LENGTH
    hough_max_line_gap: int = DEFAULT_HOUGH_MAX_LINE_GAP
    min_line_score: float = DEFAULT_MIN_LINE_SCORE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diff_threshold": self.diff_threshold,
            "min_area": self.min_area,
            "min_aspect_ratio": self.min_aspect_ratio,
            "hough_threshold": self.hough_threshold,
            "hough_min_line_length": self.hough_min_line_length,
            "hough_max_line_gap": self.hough_max_line_gap,
            "min_line_score": self.min_line_score,
        }

    def get_hough_params(self) -> HoughParams:
        return HoughParams(
            threshold=self.hough_threshold,
            min_line_length=self.hough_min_line_length,
            max_line_gap=self.hough_max_line_gap,
        )


@dataclass
class DetectionContext:
    """Input bundle for detector execution."""

    current_image: Any
    previous_image: Any
    roi_mask: Any
    runtime_params: Dict[str, Any]
    metadata: Dict[str, Any]
    schema_version: int = 1


@dataclass
class PipelineConfig:
    """Configuration for MeteorDetectionPipeline.

    This dataclass consolidates all pipeline configuration into a single object,
    making it easier to manage, serialize, and pass around.

    Attributes:
        target_folder: Input folder containing RAW files to process.
        output_folder: Output folder for detected meteor candidates.
        debug_folder: Folder for debug mask images.
        params: Detection algorithm parameters.
        num_workers: Number of parallel worker processes.
        batch_size: Number of image pairs to process per batch.
        auto_batch_size: Whether to automatically adjust batch size based on memory.
        enable_parallel: Whether to enable parallel processing.
        progress_file: Path to the progress tracking JSON file.
        output_overwrite: Whether to overwrite existing files in output folder.
        detector_name: Name of detector to use (e.g., "hough"). If None, uses default.
        detector_config: Configuration dict for the detector. Structure depends on detector.
        output_handler_name: Name of output handler to use (e.g., "file"). If None, uses default.
        output_handler_config: Configuration dict for the output handler.

    Example:
        >>> config = PipelineConfig(
        ...     target_folder="./raw",
        ...     output_folder="./candidates",
        ...     debug_folder="./debug",
        ...     params=DetectionParams(),
        ... )
        >>> # Use with MeteorDetectionPipeline
        >>> pipeline = MeteorDetectionPipeline(config)
    """

    # Required fields (no defaults)
    target_folder: str
    output_folder: str
    debug_folder: str
    params: DetectionParams

    # Optional fields with defaults
    num_workers: int = DEFAULT_NUM_WORKERS
    batch_size: int = DEFAULT_BATCH_SIZE
    auto_batch_size: bool = False
    enable_parallel: bool = True
    progress_file: str = DEFAULT_PROGRESS_FILE
    output_overwrite: bool = False

    # Detector configuration
    detector_name: Optional[str] = None
    detector_config: Optional[Dict[str, Any]] = None

    # Output handler configuration
    output_handler_name: Optional[str] = None
    output_handler_config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "target_folder": self.target_folder,
            "output_folder": self.output_folder,
            "debug_folder": self.debug_folder,
            "params": self.params.to_dict(),
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "auto_batch_size": self.auto_batch_size,
            "enable_parallel": self.enable_parallel,
            "progress_file": self.progress_file,
            "output_overwrite": self.output_overwrite,
            "detector_name": self.detector_name,
            "detector_config": self.detector_config,
            "output_handler_name": self.output_handler_name,
            "output_handler_config": self.output_handler_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            PipelineConfig instance.
        """
        params_data = data.get("params", {})
        if isinstance(params_data, dict):
            params = DetectionParams(**params_data)
        else:
            params = params_data

        return cls(
            target_folder=data["target_folder"],
            output_folder=data["output_folder"],
            debug_folder=data["debug_folder"],
            params=params,
            num_workers=data.get("num_workers", DEFAULT_NUM_WORKERS),
            batch_size=data.get("batch_size", DEFAULT_BATCH_SIZE),
            auto_batch_size=data.get("auto_batch_size", False),
            enable_parallel=data.get("enable_parallel", True),
            progress_file=data.get("progress_file", DEFAULT_PROGRESS_FILE),
            output_overwrite=data.get("output_overwrite", False),
            detector_name=data.get("detector_name"),
            detector_config=data.get("detector_config"),
            output_handler_name=data.get("output_handler_name"),
            output_handler_config=data.get("output_handler_config"),
        )

    @classmethod
    def with_defaults(
        cls,
        target_folder: str = DEFAULT_TARGET_FOLDER,
        output_folder: str = DEFAULT_OUTPUT_FOLDER,
        debug_folder: str = DEFAULT_DEBUG_FOLDER,
    ) -> "PipelineConfig":
        """Create configuration with all default values.

        Args:
            target_folder: Input folder (default: "rawfiles").
            output_folder: Output folder (default: "candidates").
            debug_folder: Debug folder (default: "debug_masks").

        Returns:
            PipelineConfig with default settings.
        """
        return cls(
            target_folder=target_folder,
            output_folder=output_folder,
            debug_folder=debug_folder,
            params=DetectionParams(),
        )


@dataclass
class ExifData:
    """EXIF metadata extracted from RAW file."""

    focal_length: Optional[float] = None
    focal_length_35mm: Optional[float] = None
    iso: Optional[int] = None
    exposure_time: Optional[float] = None
    f_number: Optional[float] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focal_length": self.focal_length,
            "focal_length_35mm": self.focal_length_35mm,
            "iso": self.iso,
            "exposure_time": self.exposure_time,
            "f_number": self.f_number,
            "camera_make": self.camera_make,
            "camera_model": self.camera_model,
            "lens_model": self.lens_model,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExifData":
        return cls(
            focal_length=data.get("focal_length"),
            focal_length_35mm=data.get("focal_length_35mm"),
            iso=data.get("iso"),
            exposure_time=data.get("exposure_time"),
            f_number=data.get("f_number"),
            camera_make=data.get("camera_make"),
            camera_model=data.get("camera_model"),
            lens_model=data.get("lens_model"),
            image_width=data.get("image_width"),
            image_height=data.get("image_height"),
        )


@dataclass
class NPFMetrics:
    """NPF Rule calculation metrics."""

    pixel_pitch_um: Optional[float] = None
    npf_recommended_sec: Optional[float] = None
    star_trail_px: Optional[float] = None
    compliance_level: str = "UNKNOWN"
    overshoot_factor: float = 0.0
    sensor_width_mm: Optional[float] = None
    has_complete_data: bool = False
    fisheye: bool = False
    fisheye_model: Optional[str] = None
    effective_focal_length: Optional[float] = None
    trail_length_ratio: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixel_pitch_um": self.pixel_pitch_um,
            "npf_recommended_sec": self.npf_recommended_sec,
            "star_trail_px": self.star_trail_px,
            "compliance_level": self.compliance_level,
            "overshoot_factor": self.overshoot_factor,
            "sensor_width_mm": self.sensor_width_mm,
            "has_complete_data": self.has_complete_data,
            "fisheye": self.fisheye,
            "fisheye_model": self.fisheye_model,
            "effective_focal_length": self.effective_focal_length,
            "trail_length_ratio": self.trail_length_ratio,
        }


@dataclass
class DetectionResult:
    """Result returned by detectors."""

    is_candidate: bool
    score: float
    lines: List[Tuple[int, int, int, int]]
    aspect_ratio: float
    debug_image: Optional[Any]
    extras: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_candidate": self.is_candidate,
            "score": self.score,
            "lines": self.lines,
            "aspect_ratio": self.aspect_ratio,
            "extras": self.extras,
        }


@dataclass
class ROIData:
    """ROI (Region of Interest) selection data."""

    mask: Any  # numpy array
    polygon: List[List[int]]
    bounding_rect: Tuple[int, int, int, int]


@dataclass
class ProgressData:
    """Progress tracking data for resumable processing."""

    version: str = VERSION
    params_hash: str = ""
    processed_files: List[str] = field(default_factory=list)
    detected_files: List[str] = field(default_factory=list)
    total_processed: int = 0
    total_detected: int = 0
    created_at: Optional[str] = None
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "params_hash": self.params_hash,
            "processed_files": self.processed_files,
            "detected_files": self.detected_files,
            "total_processed": self.total_processed,
            "total_detected": self.total_detected,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgressData":
        return cls(
            version=data.get("version", VERSION),
            params_hash=data.get("params_hash", ""),
            processed_files=data.get("processed_files", []),
            detected_files=data.get("detected_files", []),
            total_processed=data.get("total_processed", 0),
            total_detected=data.get("total_detected", 0),
            created_at=data.get("created_at"),
            last_updated=data.get("last_updated"),
        )


@dataclass
class OptimizationInfo:
    """Information about parameter optimization."""

    quality_score: float = 0.0
    quality_level: str = "UNKNOWN"
    adjustments: List[str] = field(default_factory=list)
