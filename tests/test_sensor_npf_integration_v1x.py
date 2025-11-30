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

    def test_mf44x33_npf_calculation(self):
        """Test NPF calculation with Medium Format 44x33 preset values."""
        preset = get_sensor_preset("MF44X33")

        # Typical MF44x33 camera: 100MP (11648x8736) sensor (e.g., GFX100, X2D)
        image_width_px = 11648
        calculated_pitch = calculate_pixel_pitch(preset["sensor_width"], image_width_px)

        # NPF calculation for 45mm f/2.8 lens (35mm equivalent on MF)
        # 45mm * 0.79 = ~35.5mm equivalent
        focal_length_35mm = 35.5
        aperture = 2.8

        npf_time = calculate_npf_rule(focal_length_35mm, aperture, calculated_pitch)

        # NPF should be reasonable for medium format
        self.assertGreater(npf_time, 3.0)
        self.assertLess(npf_time, 20.0)

    def test_mf54x40_npf_calculation(self):
        """Test NPF calculation with Medium Format 54x40 preset values."""
        preset = get_sensor_preset("MF54X40")

        # Typical MF54x40 camera: 100MP (11600x8700) sensor (e.g., H6D-100c)
        image_width_px = 11600
        calculated_pitch = calculate_pixel_pitch(preset["sensor_width"], image_width_px)

        # NPF calculation for 50mm f/3.5 lens (32mm equivalent on MF54x40)
        # 50mm * 0.64 = 32mm equivalent
        focal_length_35mm = 32.0
        aperture = 3.5

        npf_time = calculate_npf_rule(focal_length_35mm, aperture, calculated_pitch)

        # NPF should be reasonable for large medium format
        self.assertGreater(npf_time, 5.0)
        self.assertLess(npf_time, 25.0)


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

    def test_mf44x33_pixel_pitch_reasonable(self):
        """Test MF44X33 preset pixel pitch is close to calculated value."""
        preset = get_sensor_preset("MF44X33")
        # 100MP MF44x33 sensor (GFX100, X2D)
        calculated = calculate_pixel_pitch(preset["sensor_width"], 11648)

        self.assertAlmostEqual(
            preset["pixel_pitch"],
            calculated,
            delta=0.5,
            msg="MF44X33 preset pixel_pitch should be close to calculated",
        )

    def test_mf54x40_pixel_pitch_reasonable(self):
        """Test MF54X40 preset pixel pitch is close to calculated value."""
        preset = get_sensor_preset("MF54X40")
        # 100MP MF54x40 sensor (H6D-100c)
        calculated = calculate_pixel_pitch(preset["sensor_width"], 11600)

        self.assertAlmostEqual(
            preset["pixel_pitch"],
            calculated,
            delta=0.5,
            msg="MF54X40 preset pixel_pitch should be close to calculated",
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
            ("MF44X33", 11648),  # 100MP MF44x33
            ("MF54X40", 11600),  # 100MP MF54x40
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

    def test_medium_format_npf_characteristics(self):
        """Test that medium format sensors have reasonable NPF characteristics."""
        focal_length_35mm = 35.0
        aperture = 2.8

        # Get NPF for medium format sensors
        sensors = [
            ("FF", 8256),  # 45MP FF
            ("MF44X33", 11648),  # 100MP MF44x33
            ("MF54X40", 11600),  # 100MP MF54x40
        ]

        npf_times = {}
        for sensor_type, resolution in sensors:
            preset = get_sensor_preset(sensor_type)
            pixel_pitch = calculate_pixel_pitch(preset["sensor_width"], resolution)
            npf_time = calculate_npf_rule(focal_length_35mm, aperture, pixel_pitch)
            npf_times[sensor_type] = npf_time

        # All should be positive and reasonable
        for sensor_type, npf_time in npf_times.items():
            self.assertGreater(npf_time, 0, f"{sensor_type} NPF should be positive")
            self.assertLess(npf_time, 60, f"{sensor_type} NPF should be reasonable")


class TestMediumFormatSpecificScenarios(unittest.TestCase):
    """Test medium format specific scenarios for astrophotography."""

    def test_gfx_typical_setup(self):
        """Test typical Fujifilm GFX setup for astrophotography."""
        preset = get_sensor_preset("MF44X33")

        # GFX with 32mm f/1.8 lens (25mm equivalent)
        # 32mm * 0.79 = 25.3mm equivalent
        focal_length_35mm = 25.3
        aperture = 1.8

        # GFX 100 II: 11648 x 8736 pixels
        pixel_pitch = calculate_pixel_pitch(preset["sensor_width"], 11648)
        npf_time = calculate_npf_rule(focal_length_35mm, aperture, pixel_pitch)

        # Should allow reasonable exposure for wide-field astrophotography
        self.assertGreater(npf_time, 5.0)
        self.assertLess(npf_time, 25.0)

    def test_pentax_645z_typical_setup(self):
        """Test typical Pentax 645Z setup for astrophotography."""
        preset = get_sensor_preset("MF44X33")

        # 645Z with 35mm f/3.5 lens (28mm equivalent)
        # 35mm * 0.79 = 27.7mm equivalent
        focal_length_35mm = 27.7
        aperture = 3.5

        # 645Z: 8256 x 6192 pixels (51MP)
        pixel_pitch = calculate_pixel_pitch(preset["sensor_width"], 8256)
        npf_time = calculate_npf_rule(focal_length_35mm, aperture, pixel_pitch)

        # Should allow reasonable exposure
        self.assertGreater(npf_time, 5.0)
        self.assertLess(npf_time, 30.0)

    def test_hasselblad_h6d_typical_setup(self):
        """Test typical Hasselblad H6D-100c setup for astrophotography."""
        preset = get_sensor_preset("MF54X40")

        # H6D with 50mm f/3.5 lens (32mm equivalent)
        # 50mm * 0.64 = 32mm equivalent
        focal_length_35mm = 32.0
        aperture = 3.5

        # H6D-100c: 11600 x 8700 pixels (100MP)
        pixel_pitch = calculate_pixel_pitch(preset["sensor_width"], 11600)
        npf_time = calculate_npf_rule(focal_length_35mm, aperture, pixel_pitch)

        # Should allow reasonable exposure for medium format
        self.assertGreater(npf_time, 5.0)
        self.assertLess(npf_time, 30.0)


if __name__ == "__main__":
    unittest.main()
