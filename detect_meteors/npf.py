"""NPF rule helpers for warnings and text output."""

from typing import Any, Dict, List, Optional


def build_warnings(
    *,
    exif_data: Dict[str, Any],
    npf_metrics: Optional[Dict[str, Any]],
    auto_params: bool,
    sensor_type: Optional[str],
    focal_factor_arg: Optional[str],
    sensor_width_value: Optional[float],
    show_npf: bool,
) -> List[str]:
    """Generate warning messages for EXIF and NPF analysis."""

    warnings: List[str] = []

    if auto_params and (
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
        and focal_factor_arg is None
        and sensor_type is None
    ):
        warnings.append(
            "35mm equivalent not found. Consider using --sensor-type or --focal-factor"
        )

    if not exif_data.get("iso"):
        warnings.append("ISO value not available")
    if not exif_data.get("exposure_time"):
        warnings.append("Exposure time not available")

    if show_npf or npf_metrics:
        if not sensor_width_value and not exif_data.get("image_width"):
            warnings.append(
                "Sensor width not specified. Use --sensor-type or --sensor-width for accurate NPF calculation"
            )
        if npf_metrics and not npf_metrics.get("has_complete_data"):
            warnings.append("NPF calculation using default/estimated values")

    return warnings


def format_warnings_block(warnings: List[str]) -> str:
    """Return formatted warning text block."""

    if not warnings:
        return ""

    lines = ["=" * 60, "⚠ Warnings:"]
    lines.extend([f"  • {warning}" for warning in warnings])
    lines.append("=" * 60)
    return "\n".join(lines)


def format_usage_examples() -> str:
    """Provide usage examples for NPF analysis presentation."""

    examples = [
        "=" * 60,
        "Usage Examples:",
        "=" * 60,
        "\nUse --sensor-type for easy setup (recommended):",
        "  --sensor-type MFT           # Micro Four Thirds",
        "  --sensor-type APS-C         # APS-C (Sony/Nikon/Fuji)",
        "  --sensor-type APS-C_CANON   # APS-C (Canon)",
        "  --sensor-type FF            # Full Frame",
        "",
        "Override pixel pitch or focal factor explicitly:",
        "  --pixel-pitch 3.76          # Specify pixel pitch in μm",
        "  --focal-factor 1.5          # Crop factor override",
        "  --sensor-width 24           # Sensor width in mm (for NPF)",
    ]

    return "\n".join(examples)
