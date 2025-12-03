"""Command-line argument parsing for detect_meteors."""

import argparse

from detect_meteors import services


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Meteor detection tool with comprehensive auto-parameter estimation (v1.3.1)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Detect Meteors CLI (https://github.com/shin3tky/detect_meteors) {services.VERSION}",
    )

    parser.add_argument("-t", "--target", default=services.DEFAULT_TARGET_FOLDER, help="Input RAW image folder")
    parser.add_argument(
        "-o",
        "--output",
        default=services.DEFAULT_OUTPUT_FOLDER,
        help="Folder to copy detected candidate RAW files",
    )
    parser.add_argument(
        "--debug-dir",
        default=services.DEFAULT_DEBUG_FOLDER,
        help="Folder to save mask/debug images",
    )

    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=services.DEFAULT_DIFF_THRESHOLD,
        help=f"Threshold for difference binarization (default: {services.DEFAULT_DIFF_THRESHOLD})",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=services.DEFAULT_MIN_AREA,
        help=f"Minimum contour area (default: {services.DEFAULT_MIN_AREA})",
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=services.DEFAULT_MIN_ASPECT_RATIO,
        help=f"Minimum aspect ratio (default: {services.DEFAULT_MIN_ASPECT_RATIO})",
    )

    parser.add_argument(
        "--hough-threshold",
        type=int,
        default=services.DEFAULT_HOUGH_THRESHOLD,
        help=f"Hough line detection threshold (default: {services.DEFAULT_HOUGH_THRESHOLD})",
    )
    parser.add_argument(
        "--hough-min-line-length",
        type=int,
        default=services.DEFAULT_HOUGH_MIN_LINE_LENGTH,
        help=f"Minimum line length (default: {services.DEFAULT_HOUGH_MIN_LINE_LENGTH})",
    )
    parser.add_argument(
        "--hough-max-line-gap",
        type=int,
        default=services.DEFAULT_HOUGH_MAX_LINE_GAP,
        help=f"Maximum line gap (default: {services.DEFAULT_HOUGH_MAX_LINE_GAP})",
    )
    parser.add_argument(
        "--min-line-score",
        type=float,
        default=services.DEFAULT_MIN_LINE_SCORE,
        help=f"Minimum line score (default: {services.DEFAULT_MIN_LINE_SCORE})",
    )
    parser.add_argument("--no-roi", action="store_true", help="Skip ROI selection")
    parser.add_argument("--roi", type=str, default=None, help='Specify ROI polygon as "x1,y1;x2,y2;..."')

    parser.add_argument(
        "--workers",
        type=int,
        default=services.DEFAULT_NUM_WORKERS,
        help=f"Number of workers (default: {services.DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=services.DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {services.DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument("--auto-batch-size", action="store_true", help="Auto-adjust batch size for memory")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--profile", action="store_true", help="Display timing profile")
    parser.add_argument("--validate-raw", action="store_true", help="Validate RAW files first")
    parser.add_argument("--progress-file", default=services.DEFAULT_PROGRESS_FILE, help="Progress JSON file path")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing progress")
    parser.add_argument("--remove-progress", action="store_true", help="Delete progress and exit")

    parser.add_argument(
        "--auto-params",
        action="store_true",
        help="Auto-estimate diff_threshold, min_area, and min_line_score based on EXIF (v1.4.0 with NPF Rule)",
    )
    parser.add_argument(
        "--sensor-type",
        type=str,
        default=None,
        metavar="TYPE",
        help="Sensor type preset for NPF Rule parameters. "
        "Sets focal_factor, sensor_width, and pixel_pitch automatically. "
        "Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40. "
        "Individual options (--focal-factor, --sensor-width, --pixel-pitch) override preset values.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Focal length in mm (35mm equivalent; used for NPF Rule and parameter estimation). "
        "Overrides --sensor-type preset if specified.",
    )
    parser.add_argument(
        "--focal-factor",
        type=str,
        default=None,
        help="Crop factor for 35mm equivalent calculation. "
        "Use sensor type (MFT, APS-C, APS-H, etc.) or numeric value (e.g., 2.0, 1.5). "
        "Common values: MFT=2.0, APS-C=1.5, APS-C_CANON=1.6, APS-H=1.3, FF=1.0. "
        "Overrides --sensor-type preset if specified.",
    )
    parser.add_argument(
        "--sensor-width",
        type=float,
        default=None,
        help="Sensor width in mm (for NPF Rule calculation). "
        "Common values: MFT=17.3, APS-C=23.5, APS-C(Canon)=22.3, FF=36.0. "
        "Overrides --sensor-type preset if specified.",
    )
    parser.add_argument(
        "--pixel-pitch",
        type=float,
        default=None,
        help="Pixel pitch in micrometers (μm) for NPF Rule. "
        "If not specified, calculated from sensor width and image resolution, or uses default. "
        "Overrides --sensor-type preset if specified.",
    )
    parser.add_argument("--list-sensor-types", action="store_true", help="Display available sensor type presets and exit")
    parser.add_argument(
        "--show-exif",
        action="store_true",
        help="Display EXIF metadata and NPF Rule analysis from first RAW file and exit",
    )
    parser.add_argument("--show-npf", action="store_true", help="Display NPF Rule analysis details (implies --show-exif)")
    parser.add_argument(
        "--output-overwrite",
        action="store_true",
        help="Force overwrite existing files in output folder (default: skip existing files)",
    )

    parser.add_argument(
        "--fisheye",
        action="store_true",
        help="Enable fisheye lens correction. Applies equisolid angle projection compensation "
        "to account for varying effective focal length across the image. "
        "NPF calculation uses edge (worst case) effective focal length. "
        "Star trail estimation accounts for longer trails at image edges.",
    )

    return parser


def parse_args(argv=None):
    """Parse CLI arguments."""

    parser = build_arg_parser()
    return parser.parse_args(argv)

