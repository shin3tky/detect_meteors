"""Helpers for EXIF-related formatting and display."""

from typing import Any, Dict, Optional

from detect_meteors import services


def format_fisheye_info(
    nominal_focal_length_mm: float, projection_model: str = "EQUISOLID"
) -> str:
    """Return a formatted fisheye correction summary."""

    edge_focal = services.calculate_fisheye_edge_focal_length(
        nominal_focal_length_mm, projection_model
    )
    model_info = services.FISHEYE_PROJECTION_MODELS.get(
        projection_model.upper(), {"name": projection_model, "description": ""}
    )

    trail_ratio = services.get_fisheye_max_trail_ratio(projection_model)

    lines = [
        "=" * 60,
        "Fisheye Correction",
        "=" * 60,
        f"  Projection model:   {model_info['name']}",
        f"  Nominal focal:      {nominal_focal_length_mm:.1f}mm (center)",
        f"  Effective focal:    {edge_focal:.1f}mm (edge)",
        f"  Trail length ratio: {trail_ratio:.2f}× (edge vs center)",
        "  NPF calculation:    Based on edge (worst case)",
        "=" * 60,
    ]

    return "\n".join(lines)


def format_exif_info(
    exif_data: Dict[str, Any],
    focal_length_source: str = "EXIF",
    focal_factor: Optional[float] = None,
    npf_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Format EXIF information and optional NPF details for display."""

    lines = ["=" * 60, "Camera Settings (EXIF Metadata)", "=" * 60]

    if exif_data.get("camera_make") or exif_data.get("camera_model"):
        camera_str = f"{exif_data.get('camera_make', '')} {exif_data.get('camera_model', '')}".strip()
        if camera_str:
            lines.append(f"  Camera:           {camera_str}")

    if exif_data.get("lens_model"):
        lines.append(f"  Lens:             {exif_data['lens_model']}")

    if exif_data.get("focal_length_35mm"):
        lines.append(
            f"  Focal length:     {exif_data['focal_length_35mm']:.1f}mm (35mm equiv.) [{focal_length_source}]"
        )
    elif exif_data.get("focal_length"):
        lines.append(
            f"  Focal length:     {exif_data['focal_length']:.1f}mm (actual) [{focal_length_source}]"
        )
        if focal_factor:
            equiv = exif_data["focal_length"] * focal_factor
            lines.append(
                f"                    ~ {equiv:.1f}mm (35mm equiv., calculated with factor {focal_factor})"
            )
        else:
            lines.append("                    ⚠ No 35mm equivalent found")
    else:
        lines.append("  Focal length:     Not available")

    if exif_data.get("iso"):
        lines.append(f"  ISO:              {exif_data['iso']}")
    else:
        lines.append("  ISO:              Not available")

    if exif_data.get("exposure_time"):
        exp = exif_data["exposure_time"]
        exp_str = f"1/{int(round(1/exp))}" if exp < 1 else f"{exp:.1f}s"
        lines.append(f"  Exposure:         {exp_str}")
    else:
        lines.append("  Exposure:         Not available")

    if exif_data.get("f_number"):
        lines.append(f"  Aperture:         f/{exif_data['f_number']:.1f}")
    else:
        lines.append("  Aperture:         Not available")

    if exif_data.get("camera_serial"):
        lines.append(f"  Serial:           {exif_data['camera_serial']}")

    if npf_metrics:
        lines.extend([
            "",
            "=" * 60,
            "NPF Rule Analysis",
            "=" * 60,
            f"  NPF Max Exposure:  {npf_metrics['npf_max_exposure_sec']:.1f}s",
            f"  Star Trail Score:  {npf_metrics['trail_score']:.1f} (lower is better)",
            f"  Motion Blur Score: {npf_metrics['motion_blur_score']:.1f}",
            f"  Rating:            {npf_metrics['rating']}",
            f"  Data completeness: {'Yes' if npf_metrics.get('has_complete_data') else 'No'}",
        ])

    lines.append("=" * 60)
    return "\n".join(lines)
