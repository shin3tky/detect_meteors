"""Application runner for detect_meteors CLI."""

import os
import sys

from detect_meteors_cli import (
    DEFAULT_ENABLE_ROI_SELECTION,
    DEFAULT_FISHEYE_MODEL,
    apply_sensor_preset,
    calculate_npf_metrics,
    collect_files,
    detect_meteors_advanced,
    display_exif_info,
    display_fisheye_info,
    extract_exif_metadata,
    get_sensor_preset,
    list_sensor_types,
    parse_roi_polygon_string,
    validate_sensor_overrides,
)


def run(args):
    """Execute the application logic using parsed arguments."""

    if args.remove_progress:
        if os.path.exists(args.progress_file):
            os.remove(args.progress_file)
            print(f"Removed progress file: {args.progress_file}")
        else:
            print(f"Progress file not found: {args.progress_file}")
        return

    # --list-sensor-types: Display available sensor types and exit
    if args.list_sensor_types:
        list_sensor_types()
        return

    # --show-exif or --show-npf: Display EXIF info and NPF analysis then exit
    if args.show_exif or args.show_npf:
        print(f"\n{'='*60}")
        if args.show_npf:
            print("EXIF Metadata & NPF Rule Analysis")
        else:
            print("EXIF Metadata Viewer")
        print(f"{'='*60}\n")
        print(f"Target folder: {args.target}")

        # Validate --sensor-type if specified
        if args.sensor_type and get_sensor_preset(args.sensor_type) is None:
            print(f"⚠ Error: Invalid --sensor-type value: '{args.sensor_type}'")
            print(
                f"  Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40"
            )
            return

        # Apply sensor preset (with individual args taking priority)
        (
            focal_factor_value,
            sensor_width_value,
            focal_length_value,
            pixel_pitch_value,
            preset,
        ) = apply_sensor_preset(args, verbose=True)

        # Validate sensor overrides
        validate_sensor_overrides(args, preset, sensor_width_value, pixel_pitch_value)

        # Validate focal_factor if specified directly
        if args.focal_factor and focal_factor_value is None:
            print(f"⚠ Error: Invalid --focal-factor value: '{args.focal_factor}'")
            print(f"  Valid values: MFT, APS-C, APS-H, FF, or numeric (e.g., 2.0)")
            return

        try:
            files = collect_files(args.target)
            if not files:
                print("⚠ No RAW files found in target folder.")
                return

            print(f"Found {len(files)} RAW files")
            print(f"Reading EXIF from first file: {os.path.basename(files[0])}\n")

            exif_data = extract_exif_metadata(files[0])

            # Focal length priority processing
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

            # NPF Rule Analysis (when --show-npf or sufficient information exists)
            npf_metrics = None
            if args.show_npf or (
                exif_data.get("focal_length_35mm") and exif_data.get("f_number")
            ):
                npf_metrics = calculate_npf_metrics(
                    exif_data,
                    sensor_width_mm=sensor_width_value,
                    pixel_pitch_um=pixel_pitch_value,
                    fisheye=args.fisheye,
                    fisheye_model=DEFAULT_FISHEYE_MODEL,
                )

            # Display fisheye info if enabled
            if args.fisheye and exif_data.get("focal_length_35mm"):
                display_fisheye_info(exif_data["focal_length_35mm"], DEFAULT_FISHEYE_MODEL)

            display_exif_info(exif_data, focal_length_source, focal_factor_value, npf_metrics)

            # Display warnings
            warnings = []
            if (
                not exif_data.get("focal_length")
                and not exif_data.get("focal_length_35mm")
                and not focal_length_value
            ):
                warnings.append("Focal length not available")
            elif (
                not exif_data.get("focal_length_35mm")
                and not focal_factor_value
                and not focal_length_value
            ):
                warnings.append(
                    f"35mm equivalent not found. Consider using --sensor-type or --focal-factor"
                )
            if not exif_data.get("iso"):
                warnings.append("ISO value not available")
            if not exif_data.get("exposure_time"):
                warnings.append("Exposure time not available")

            # NPF-related warnings
            if args.show_npf or npf_metrics:
                if not sensor_width_value and not exif_data.get("image_width"):
                    warnings.append(
                        "Sensor width not specified. Use --sensor-type or --sensor-width for accurate NPF calculation"
                    )
                if npf_metrics and not npf_metrics.get("has_complete_data"):
                    warnings.append("NPF calculation using default/estimated values")

            if warnings:
                print(f"{'='*60}")
                print("⚠ Warnings:")
                for warning in warnings:
                    print(f"  • {warning}")
                print(f"{'='*60}\n")

            # Display Usage Examples
            if args.show_npf:
                print(f"{'='*60}")
                print("Usage Examples:")
                print(f"{'='*60}")
                print("\nUse --sensor-type for easy setup (recommended):")
                print(f"  --sensor-type MFT           # Micro Four Thirds")
                print(f"  --sensor-type APS-C         # APS-C (Sony/Nikon/Fuji)")
                print(f"  --sensor-type APS-C_CANON   # APS-C (Canon)")
                print(f"  --sensor-type FF            # Full Frame")
                print("\nOr specify individual parameters (overrides --sensor-type):")
                print(f"  --sensor-width 17.3   # Sensor width in mm")
                print(f"  --pixel-pitch 3.7     # Pixel pitch in micrometers")
                print(f"  --focal-factor 2.0    # Crop factor")
                print(f"\n{'='*60}\n")
                if warnings:
                    print("⚠ Warnings:")
                    for warning in warnings:
                        print(f"  • {warning}")
                    print(f"{'='*60}\n")

        except FileNotFoundError as e:
            print(f"⚠ Error: {e}")
        except Exception as e:
            print(f"⚠ Error reading EXIF: {e}")

        return

    roi_polygon_cli = None
    enable_roi_selection = DEFAULT_ENABLE_ROI_SELECTION

    if args.roi is not None:
        roi_polygon_cli = parse_roi_polygon_string(args.roi)
        enable_roi_selection = False
    elif args.no_roi:
        enable_roi_selection = False

    # Determine user specifications
    user_specified_diff_threshold = "--diff-threshold" in sys.argv
    user_specified_min_area = "--min-area" in sys.argv
    user_specified_min_line_score = "--min-line-score" in sys.argv

    # Validate --sensor-type if specified
    if args.sensor_type and get_sensor_preset(args.sensor_type) is None:
        print(f"⚠ Error: Invalid --sensor-type value: '{args.sensor_type}'")
        print(
            f"  Valid types: 1INCH, MFT, APS-C, APS-C_CANON, APS-H, FF, MF44X33, MF54X40"
        )
        return

    # Apply sensor preset (with individual args taking priority)
    (
        focal_factor_value,
        sensor_width_value,
        focal_length_value,
        pixel_pitch_value,
        preset,
    ) = apply_sensor_preset(args, verbose=False)

    # Validate sensor overrides
    validate_sensor_overrides(args, preset, sensor_width_value, pixel_pitch_value)

    # Validate focal_factor if specified directly
    if args.focal_factor and focal_factor_value is None:
        print(f"⚠ Error: Invalid --focal-factor value: '{args.focal_factor}'")
        print(f"  Valid values: MFT, APS-C, APS-H, FF, or numeric (e.g., 2.0)")
        return

    detect_meteors_advanced(
        target_folder=args.target,
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

