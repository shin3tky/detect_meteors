"""
Test suite for sensor preset integration with NPF calculations (v1.5.0+).

Tests that sensor presets work correctly with NPF Rule calculations
for practical astrophotography scenarios.
"""

import unittest
import sys
import os

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detect_meteors_cli import (
    get_sensor_preset,
    calculate_pixel_pitch,
    calculate_npf_rule,
)


class TestSensorPresetNPFIntegration(unittest.TestCase):
    """Integration tests for sensor preset with NPF calculations."""

    def test_mft_npf_calculation(self):
        """Test NPF calculation with MFT preset values."""
        preset = get_sensor_preset("MFT")

        # Typical MFT camera: 20MP (5184x3888) sensor
        image_width_px = 5184
        calculated_pitch = calculate_pixel_pitch(preset["sensor_width"], image_width_px)

        # NPF calculation for 25mm f/1.8 lens (50mm equivalent)
        focal_length_35mm = 50.0
        aperture = 1.8

        npf_time = calculate_npf_rule(focal_length_35mm, aperture, calculated_pitch)

        # NPF should be reasonable (between 1s and 20s for typical settings)
        self.assertGreater(npf_time, 1.0)
        self.assertLess(npf_time, 20.0)

    def test_apsc_npf_calculation(self):
        """Test NPF calculation with APS-C preset values."""
        preset = get_sensor_preset("APSC")

        # Typical APS-C camera: 26MP (6000x4000) sensor
        image_width_px = 6000
        calculated_pitch = calculate_pixel_pitch(preset["sensor_width"], image_width_px)

        # NPF calculation for 16mm f/2.8 lens (24mm equivalent)
        focal_length_35mm = 24.0
        aperture = 2.8

        npf_time = calculate_npf_rule(focal_length_35mm, aperture, calculated_pitch)

        # NPF should be reasonable
        self.assertGreater(npf_time, 3.0)
        self.assertLess(npf_time, 25.0)

    def test_ff_npf_calculation(self):
        """Test NPF calculation with Full Frame preset values."""
        preset = get_sensor_preset("FF")

        # Typical FF camera: 45MP (8192x5464) sensor
        image_width_px = 8192
        calculated_pitch = calculate_pixel_pitch(preset["sensor_width"], image_width_px)

        # NPF calculation for 24mm f/2.8 lens
        focal_length_35mm = 24.0
        aperture = 2.8

        npf_time = calculate_npf_rule(focal_length_35mm, aperture, calculated_pitch)

        # NPF for FF with 24mm should be longer than MFT equivalent
        self.assertGreater(npf_time, 5.0)
        self.assertLess(npf_time, 30.0)

    def test_1inch_npf_calculation(self):
        """Test NPF calculation with 1-inch sensor preset values."""
        preset = get_sensor_preset("1INCH")

        # Typical 1-inch camera: 20MP (5472x3648) sensor
        image_width_px = 5472
        calculated_pitch = calculate_pixel_pitch(preset["sensor_width"], image_width_px)

        # NPF calculation for 9mm f/2.8 lens (24mm equivalent)
        focal_length_35mm = 24.0
        aperture = 2.8

        npf_time = calculate_npf_rule(focal_length_35mm, aperture, calculated_pitch)

        # 1-inch sensor has small pixels, shorter NPF time
        self.assertGreater(npf_time, 2.0)
        self.assertLess(npf_time, 15.0)


class TestPixelPitchVsPreset(unittest.TestCase):
    """Test that preset pixel pitch values are reasonable."""

    def test_mft_pixel_pitch_reasonable(self):
        """Test MFT preset pixel pitch is close to calculated value."""
        preset = get_sensor_preset("MFT")
        # 20MP MFT sensor
        calculated = calculate_pixel_pitch(preset["sensor_width"], 5184)

        # Should be within 1μm of each other
        self.assertAlmostEqual(
            preset["pixel_pitch"],
            calculated,
            delta=1.0,
            msg="MFT preset pixel_pitch should be close to calculated",
        )

    def test_apsc_pixel_pitch_reasonable(self):
        """Test APS-C preset pixel pitch is close to calculated value."""
        preset = get_sensor_preset("APSC")
        # 26MP APS-C sensor
        calculated = calculate_pixel_pitch(preset["sensor_width"], 6000)

        self.assertAlmostEqual(
            preset["pixel_pitch"],
            calculated,
            delta=1.0,
            msg="APS-C preset pixel_pitch should be close to calculated",
        )

    def test_ff_pixel_pitch_reasonable(self):
        """Test FF preset pixel pitch is close to calculated value."""
        preset = get_sensor_preset("FF")
        # 45MP FF sensor
        calculated = calculate_pixel_pitch(preset["sensor_width"], 8256)

        self.assertAlmostEqual(
            preset["pixel_pitch"],
            calculated,
            delta=1.0,
            msg="FF preset pixel_pitch should be close to calculated",
        )


class TestNPFComparisonAcrossSensors(unittest.TestCase):
    """Test NPF behavior across different sensor types."""

    def test_wider_sensor_longer_npf(self):
        """Test that wider sensors generally allow longer NPF times."""
        # Using same equivalent focal length and aperture
        focal_length_35mm = 24.0
        aperture = 2.8

        # Calculate NPF for each sensor type with typical resolution
        sensors = [
            ("1INCH", 5472),  # 20MP 1-inch
            ("MFT", 5184),  # 20MP MFT
            ("APSC", 6000),  # 26MP APS-C
            ("FF", 8256),  # 45MP FF
        ]

        npf_times = {}
        for sensor_type, resolution in sensors:
            preset = get_sensor_preset(sensor_type)
            pixel_pitch = calculate_pixel_pitch(preset["sensor_width"], resolution)
            npf_time = calculate_npf_rule(focal_length_35mm, aperture, pixel_pitch)
            npf_times[sensor_type] = npf_time

        # Smaller sensors (smaller pixels) should generally have shorter NPF
        # This is a simplified test - actual relationship depends on resolution
        self.assertGreater(
            npf_times["FF"],
            npf_times["1INCH"],
            "FF should have longer NPF than 1-inch (given similar relative resolution)",
        )


if __name__ == "__main__":
    unittest.main()
