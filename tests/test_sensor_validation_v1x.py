"""
Test suite for sensor override validation (v1.5.2+).

Tests the validate_sensor_overrides() function that checks if
--sensor-width and --pixel-pitch overrides deviate significantly
from --sensor-type preset values.
"""

import unittest
import sys
import os
from argparse import Namespace
from io import StringIO
from contextlib import redirect_stdout

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detect_meteors_cli import (
    get_sensor_preset,
    apply_sensor_preset,
    validate_sensor_overrides,
    SENSOR_PRESETS,
)


class TestValidateSensorOverrides(unittest.TestCase):
    """Test validate_sensor_overrides() function."""

    def test_no_warning_without_sensor_type(self):
        """Test that no warning is issued when --sensor-type is not specified."""
        args = Namespace(
            sensor_type=None,
            sensor_width=17.3,
            pixel_pitch=3.7,
        )

        # Capture stdout
        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, None, 17.3, 3.7)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning expected

    def test_no_warning_without_overrides(self):
        """Test that no warning is issued when using preset values without overrides."""
        args = Namespace(
            sensor_type="MFT",
            sensor_width=None,  # Not overridden
            pixel_pitch=None,  # Not overridden
        )

        preset = get_sensor_preset("MFT")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 17.3, 3.7)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning expected

    def test_no_warning_with_small_sensor_width_deviation(self):
        """Test that no warning is issued for small sensor_width deviation (< 30%)."""
        args = Namespace(
            sensor_type="MFT",
            sensor_width=17.5,  # 1.2% deviation from 17.3mm
            pixel_pitch=None,
        )

        preset = get_sensor_preset("MFT")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 17.5, 3.7)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning expected

    def test_warning_with_large_sensor_width_deviation(self):
        """Test that warning is issued for large sensor_width deviation (> 30%)."""
        args = Namespace(
            sensor_type="MFT",
            sensor_width=23.5,  # 35.8% deviation from 17.3mm (APS-C size)
            pixel_pitch=None,
        )

        preset = get_sensor_preset("MFT")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 23.5, 3.7)

        output = f.getvalue()
        self.assertIn("Warning", output)
        self.assertIn("--sensor-width", output)
        self.assertIn("23.5mm", output)
        self.assertIn("17.3mm", output)
        self.assertIn("MFT", output)

    def test_no_warning_with_small_pixel_pitch_deviation(self):
        """Test that no warning is issued for small pixel_pitch deviation (< 50%)."""
        args = Namespace(
            sensor_type="APSC",
            sensor_width=None,
            pixel_pitch=4.0,  # 2.6% deviation from 3.9μm
        )

        preset = get_sensor_preset("APSC")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 23.5, 4.0)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning expected

    def test_warning_with_large_pixel_pitch_deviation(self):
        """Test that warning is issued for large pixel_pitch deviation (> 50%)."""
        args = Namespace(
            sensor_type="APSC",
            sensor_width=None,
            pixel_pitch=6.0,  # 53.8% deviation from 3.9μm
        )

        preset = get_sensor_preset("APSC")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 23.5, 6.0)

        output = f.getvalue()
        self.assertIn("Warning", output)
        self.assertIn("--pixel-pitch", output)
        self.assertIn("6.0", output)
        self.assertIn("3.9", output)
        self.assertIn("APSC", output)

    def test_multiple_warnings_with_both_deviations(self):
        """Test that both warnings are issued when both parameters deviate."""
        args = Namespace(
            sensor_type="FF",
            sensor_width=23.5,  # 34.7% deviation from 36.0mm
            pixel_pitch=7.0,  # 62.8% deviation from 4.3μm
        )

        preset = get_sensor_preset("FF")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 23.5, 7.0)

        output = f.getvalue()
        # Should have 2 warnings
        self.assertEqual(output.count("Warning"), 2)
        self.assertIn("--sensor-width", output)
        self.assertIn("--pixel-pitch", output)

    def test_sensor_width_at_threshold_boundary(self):
        """Test sensor_width at exactly 30% deviation (should not warn)."""
        args = Namespace(
            sensor_type="MFT",
            sensor_width=22.49,  # Exactly 30% deviation from 17.3mm
            pixel_pitch=None,
        )

        preset = get_sensor_preset("MFT")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 22.49, 3.7)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning at exactly 30%

    def test_sensor_width_just_over_threshold(self):
        """Test sensor_width just over 30% deviation (should warn)."""
        args = Namespace(
            sensor_type="MFT",
            sensor_width=22.6,  # Just over 30% deviation from 17.3mm
            pixel_pitch=None,
        )

        preset = get_sensor_preset("MFT")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 22.6, 3.7)

        output = f.getvalue()
        self.assertIn("Warning", output)

    def test_pixel_pitch_at_threshold_boundary(self):
        """Test pixel_pitch at exactly 50% deviation (should not warn)."""
        args = Namespace(
            sensor_type="APSC",
            sensor_width=None,
            pixel_pitch=5.85,  # Exactly 50% deviation from 3.9μm
        )

        preset = get_sensor_preset("APSC")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 23.5, 5.85)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning at exactly 50%

    def test_pixel_pitch_just_over_threshold(self):
        """Test pixel_pitch just over 50% deviation (should warn)."""
        args = Namespace(
            sensor_type="APSC",
            sensor_width=None,
            pixel_pitch=5.9,  # Just over 50% deviation from 3.9μm
        )

        preset = get_sensor_preset("APSC")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 23.5, 5.9)

        output = f.getvalue()
        self.assertIn("Warning", output)


class TestValidationWithDifferentSensorTypes(unittest.TestCase):
    """Test validation across different sensor types."""

    def test_1inch_sensor_validation(self):
        """Test validation with 1-inch sensor preset."""
        args = Namespace(
            sensor_type="1INCH",
            sensor_width=17.3,  # Using MFT width (31% larger)
            pixel_pitch=None,
        )

        preset = get_sensor_preset("1INCH")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 17.3, 2.4)

        output = f.getvalue()
        self.assertIn("Warning", output)
        self.assertIn("1INCH", output)

    def test_canon_apsc_validation(self):
        """Test validation with Canon APS-C preset."""
        args = Namespace(
            sensor_type="APSC_CANON",
            sensor_width=None,
            pixel_pitch=3.3,  # 3.1% deviation, should not warn
        )

        preset = get_sensor_preset("APSC_CANON")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 22.3, 3.3)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning expected

    def test_medium_format_44x33_validation(self):
        """Test validation with MF44X33 preset."""
        args = Namespace(
            sensor_type="MF44X33",
            sensor_width=36.0,  # Using FF width (17.8% smaller)
            pixel_pitch=None,
        )

        preset = get_sensor_preset("MF44X33")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 36.0, 3.76)

        output = f.getvalue()
        self.assertEqual(output, "")  # Less than 30%, no warning

    def test_medium_format_54x40_validation(self):
        """Test validation with MF54X40 preset."""
        args = Namespace(
            sensor_type="MF54X40",
            sensor_width=36.0,  # Using FF width (32.6% smaller)
            pixel_pitch=None,
        )

        preset = get_sensor_preset("MF54X40")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 36.0, 4.6)

        output = f.getvalue()
        self.assertIn("Warning", output)  # Over 30%, should warn


class TestValidationWithApplySensorPreset(unittest.TestCase):
    """Test validation integration with apply_sensor_preset()."""

    def test_apply_sensor_preset_returns_preset(self):
        """Test that apply_sensor_preset() now returns preset dictionary."""
        args = Namespace(
            sensor_type="MFT",
            focal_factor=None,
            sensor_width=None,
            focal_length=None,
            pixel_pitch=None,
        )

        result = apply_sensor_preset(args, verbose=False)

        # Should return 5-tuple now (v1.5.2+)
        self.assertEqual(len(result), 5)
        focal_factor, sensor_width, focal_length, pixel_pitch, preset = result

        # Check values
        self.assertEqual(focal_factor, 2.0)
        self.assertEqual(sensor_width, 17.3)
        self.assertIsNone(focal_length)
        self.assertAlmostEqual(pixel_pitch, 3.7, places=1)

        # Check preset dict
        self.assertIsNotNone(preset)
        self.assertEqual(preset["focal_factor"], 2.0)
        self.assertEqual(preset["sensor_width"], 17.3)

    def test_apply_sensor_preset_with_overrides_returns_preset(self):
        """Test apply_sensor_preset() returns preset even with overrides."""
        args = Namespace(
            sensor_type="APSC",
            focal_factor=None,
            sensor_width=24.0,  # Override
            focal_length=None,
            pixel_pitch=4.0,  # Override
        )

        focal_factor, sensor_width, focal_length, pixel_pitch, preset = (
            apply_sensor_preset(args, verbose=False)
        )

        # Overridden values
        self.assertEqual(sensor_width, 24.0)
        self.assertEqual(pixel_pitch, 4.0)

        # Preset should still be APSC preset
        self.assertIsNotNone(preset)
        self.assertEqual(preset["focal_factor"], 1.5)
        self.assertEqual(preset["sensor_width"], 23.5)  # Original preset value
        self.assertAlmostEqual(preset["pixel_pitch"], 3.9, places=1)

    def test_apply_sensor_preset_no_sensor_type_returns_none_preset(self):
        """Test that apply_sensor_preset() returns None for preset when no sensor_type."""
        args = Namespace(
            sensor_type=None,
            focal_factor=None,
            sensor_width=17.3,
            focal_length=None,
            pixel_pitch=3.7,
        )

        focal_factor, sensor_width, focal_length, pixel_pitch, preset = (
            apply_sensor_preset(args, verbose=False)
        )

        self.assertIsNone(preset)


class TestValidationRealWorldScenarios(unittest.TestCase):
    """Test validation with real-world astrophotography scenarios."""

    def test_correct_mft_setup_no_warning(self):
        """Test typical correct MFT setup doesn't trigger warnings."""
        # OM System OM-1: 20MP MFT sensor with correct settings
        args = Namespace(
            sensor_type="MFT",
            sensor_width=17.4,  # Slightly different but within tolerance
            pixel_pitch=3.7,  # Typical 20MP MFT
        )

        preset = get_sensor_preset("MFT")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 17.4, 3.7)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning expected

    def test_wrong_sensor_type_selected_warns(self):
        """Test that selecting wrong sensor type triggers warning."""
        # User selected MFT but actually has APS-C camera
        # MFT preset: sensor_width=17.3mm, pixel_pitch=3.7μm
        args = Namespace(
            sensor_type="MFT",
            sensor_width=23.5,  # APS-C width (35.8% deviation from 17.3mm)
            pixel_pitch=6.0,  # 62.2% deviation from 3.7μm (exceeds 50% threshold)
        )

        preset = get_sensor_preset("MFT")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 23.5, 6.0)

        output = f.getvalue()
        # Should have 2 warnings (both parameters wrong)
        self.assertEqual(output.count("Warning"), 2)

    def test_gfx_100_correct_setup(self):
        """Test Fujifilm GFX 100 correct setup."""
        # GFX 100 II: 102MP MF44x33 sensor
        args = Namespace(
            sensor_type="MF44X33",
            sensor_width=43.8,
            pixel_pitch=3.76,
        )

        preset = get_sensor_preset("MF44X33")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 43.8, 3.76)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning expected

    def test_user_measured_values_slightly_different(self):
        """Test user-measured values slightly different from preset."""
        # User measured their sensor and got slightly different values
        args = Namespace(
            sensor_type="FF",
            sensor_width=35.9,  # Measured 36.0mm → 35.9mm (0.3% diff)
            pixel_pitch=4.4,  # Measured 4.3μm → 4.4μm (2.3% diff)
        )

        preset = get_sensor_preset("FF")

        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, 35.9, 4.4)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning expected (small deviations)


class TestValidationEdgeCases(unittest.TestCase):
    """Test edge cases for validation."""

    def test_validation_with_none_values(self):
        """Test that validation handles None values gracefully."""
        args = Namespace(
            sensor_type="MFT",
            sensor_width=None,
            pixel_pitch=None,
        )

        preset = get_sensor_preset("MFT")

        # Should not raise error
        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, preset, None, None)

        output = f.getvalue()
        self.assertEqual(output, "")

    def test_validation_with_zero_values(self):
        """Test validation with zero preset values (edge case)."""
        args = Namespace(
            sensor_type="MFT",
            sensor_width=17.3,
            pixel_pitch=3.7,
        )

        # Mock preset with zero values (shouldn't happen in practice)
        mock_preset = {
            "sensor_width": 0,
            "pixel_pitch": 0,
        }

        # Should not crash (division by zero protection)
        f = StringIO()
        with redirect_stdout(f):
            # This would cause division by zero, but function should handle it
            # In real code, this scenario won't occur as presets are validated
            try:
                validate_sensor_overrides(args, mock_preset, 17.3, 3.7)
            except ZeroDivisionError:
                self.fail("validate_sensor_overrides should handle zero values")

    def test_validation_with_preset_missing_keys(self):
        """Test validation when preset is missing expected keys."""
        args = Namespace(
            sensor_type="MFT",
            sensor_width=17.3,
            pixel_pitch=3.7,
        )

        # Incomplete preset
        incomplete_preset = {
            "focal_factor": 2.0,
            # Missing sensor_width and pixel_pitch
        }

        # Should not crash
        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args, incomplete_preset, 17.3, 3.7)

        output = f.getvalue()
        self.assertEqual(output, "")  # No warning if preset keys missing


class TestValidationThresholdCalculations(unittest.TestCase):
    """Test threshold calculation accuracy."""

    def test_30_percent_threshold_calculation_sensor_width(self):
        """Test 30% threshold calculation for sensor_width."""
        # MFT sensor_width: 17.3mm
        # 30% deviation: 17.3 * 0.3 = 5.19mm
        # Upper bound: 17.3 + 5.19 = 22.49mm
        # Lower bound: 17.3 - 5.19 = 12.11mm

        preset = get_sensor_preset("MFT")
        args_upper = Namespace(sensor_type="MFT", sensor_width=22.49, pixel_pitch=None)
        args_over = Namespace(sensor_type="MFT", sensor_width=22.5, pixel_pitch=None)

        # At threshold (22.49mm) - no warning
        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args_upper, preset, 22.49, 3.7)
        self.assertEqual(f.getvalue(), "")

        # Just over threshold (22.5mm) - warning
        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args_over, preset, 22.5, 3.7)
        self.assertIn("Warning", f.getvalue())

    def test_50_percent_threshold_calculation_pixel_pitch(self):
        """Test 50% threshold calculation for pixel_pitch."""
        # APSC pixel_pitch: 3.9μm
        # 50% deviation: 3.9 * 0.5 = 1.95μm
        # Upper bound: 3.9 + 1.95 = 5.85μm
        # Lower bound: 3.9 - 1.95 = 1.95μm

        preset = get_sensor_preset("APSC")
        args_upper = Namespace(sensor_type="APSC", sensor_width=None, pixel_pitch=5.85)
        args_over = Namespace(sensor_type="APSC", sensor_width=None, pixel_pitch=5.86)

        # At threshold (5.85μm) - no warning
        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args_upper, preset, 23.5, 5.85)
        self.assertEqual(f.getvalue(), "")

        # Just over threshold (5.86μm) - warning
        f = StringIO()
        with redirect_stdout(f):
            validate_sensor_overrides(args_over, preset, 23.5, 5.86)
        self.assertIn("Warning", f.getvalue())


if __name__ == "__main__":
    unittest.main()
