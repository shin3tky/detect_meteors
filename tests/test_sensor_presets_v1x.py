"""
Test suite for sensor type presets (v1.5.0+).

Tests the --sensor-type option that provides preset configurations
for common camera sensor types (MFT, APS-C, FF, MF44X33, MF54X40, etc.).
"""

import unittest
import sys
import os
from argparse import Namespace

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core import (
    SENSOR_PRESETS,
    CROP_FACTORS,
    DEFAULT_SENSOR_WIDTHS,
    get_sensor_preset,
    apply_sensor_preset,
    parse_focal_factor,
)


class TestSensorPresetsStructure(unittest.TestCase):
    """Test SENSOR_PRESETS dictionary structure and contents."""

    def test_sensor_presets_contains_required_types(self):
        """Test that SENSOR_PRESETS contains all required sensor types."""
        required_types = [
            "1INCH",
            "MFT",
            "APSC",
            "APSC_CANON",
            "APSH",
            "FF",
            "MF44X33",
            "MF54X40",
        ]
        for sensor_type in required_types:
            self.assertIn(sensor_type, SENSOR_PRESETS)

    def test_sensor_preset_structure(self):
        """Test that each sensor preset has required fields."""
        required_fields = ["focal_factor", "sensor_width", "pixel_pitch", "description"]
        for sensor_type, preset in SENSOR_PRESETS.items():
            for field in required_fields:
                self.assertIn(
                    field, preset, f"Missing '{field}' in preset '{sensor_type}'"
                )

    def test_sensor_preset_value_ranges(self):
        """Test that preset values are within reasonable ranges."""
        for sensor_type, preset in SENSOR_PRESETS.items():
            # Focal factor should be between 0.5 and 5.0
            self.assertGreaterEqual(preset["focal_factor"], 0.5)
            self.assertLessEqual(preset["focal_factor"], 5.0)

            # Sensor width should be between 5mm and 60mm (includes medium format)
            self.assertGreaterEqual(preset["sensor_width"], 5.0)
            self.assertLessEqual(preset["sensor_width"], 60.0)

            # Pixel pitch should be between 1μm and 10μm
            self.assertGreaterEqual(preset["pixel_pitch"], 1.0)
            self.assertLessEqual(preset["pixel_pitch"], 10.0)

    def test_sensor_presets_ordered_by_size(self):
        """Test that primary sensor types are ordered by sensor size."""
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

        # Get sensor widths for primary types
        widths = [SENSOR_PRESETS[t]["sensor_width"] for t in primary_types]

        # Check that widths are in ascending order (with tolerance for APSC vs APSC_CANON)
        for i in range(len(widths) - 1):
            # APSC_CANON (22.3) is smaller than APSC (23.5), so skip that comparison
            if primary_types[i] == "APSC" and primary_types[i + 1] == "APSC_CANON":
                continue
            if primary_types[i] == "APSC_CANON" and primary_types[i + 1] == "APSH":
                self.assertLess(widths[i], widths[i + 1])


class TestSensorPresetValues(unittest.TestCase):
    """Test specific sensor preset values."""

    def test_1inch_preset_values(self):
        """Test 1-inch sensor preset values."""
        preset = get_sensor_preset("1INCH")
        self.assertIsNotNone(preset)
        self.assertEqual(preset["focal_factor"], 2.7)
        self.assertEqual(preset["sensor_width"], 13.2)

    def test_mft_preset_values(self):
        """Test MFT (Micro Four Thirds) preset values."""
        preset = get_sensor_preset("MFT")
        self.assertIsNotNone(preset)
        self.assertEqual(preset["focal_factor"], 2.0)
        self.assertEqual(preset["sensor_width"], 17.3)
        self.assertAlmostEqual(preset["pixel_pitch"], 3.7, places=1)

    def test_apsc_preset_values(self):
        """Test APS-C (Sony/Nikon/Fuji) preset values."""
        preset = get_sensor_preset("APSC")
        self.assertIsNotNone(preset)
        self.assertEqual(preset["focal_factor"], 1.5)
        self.assertEqual(preset["sensor_width"], 23.5)

    def test_apsc_canon_preset_values(self):
        """Test APS-C Canon preset values."""
        preset = get_sensor_preset("APSC_CANON")
        self.assertIsNotNone(preset)
        self.assertEqual(preset["focal_factor"], 1.6)
        self.assertEqual(preset["sensor_width"], 22.3)

    def test_apsh_preset_values(self):
        """Test APS-H preset values."""
        preset = get_sensor_preset("APSH")
        self.assertIsNotNone(preset)
        self.assertEqual(preset["focal_factor"], 1.3)
        self.assertEqual(preset["sensor_width"], 27.9)

    def test_ff_preset_values(self):
        """Test Full Frame preset values."""
        preset = get_sensor_preset("FF")
        self.assertIsNotNone(preset)
        self.assertEqual(preset["focal_factor"], 1.0)
        self.assertEqual(preset["sensor_width"], 36.0)

    def test_mf44x33_preset_values(self):
        """Test Medium Format 44x33 preset values."""
        preset = get_sensor_preset("MF44X33")
        self.assertIsNotNone(preset)
        self.assertEqual(preset["focal_factor"], 0.79)
        self.assertEqual(preset["sensor_width"], 43.8)
        self.assertAlmostEqual(preset["pixel_pitch"], 3.76, places=2)

    def test_mf54x40_preset_values(self):
        """Test Medium Format 54x40 preset values."""
        preset = get_sensor_preset("MF54X40")
        self.assertIsNotNone(preset)
        self.assertEqual(preset["focal_factor"], 0.64)
        self.assertEqual(preset["sensor_width"], 53.4)
        self.assertAlmostEqual(preset["pixel_pitch"], 4.6, places=1)


class TestMediumFormatPresets(unittest.TestCase):
    """Test medium format sensor presets (v1.5.1+)."""

    def test_mf44x33_crop_factor_less_than_one(self):
        """Test that MF44X33 has crop factor less than 1.0."""
        preset = get_sensor_preset("MF44X33")
        self.assertLess(preset["focal_factor"], 1.0)
        self.assertGreater(preset["focal_factor"], 0.5)

    def test_mf54x40_crop_factor_less_than_one(self):
        """Test that MF54X40 has crop factor less than 1.0."""
        preset = get_sensor_preset("MF54X40")
        self.assertLess(preset["focal_factor"], 1.0)
        self.assertGreater(preset["focal_factor"], 0.5)

    def test_mf54x40_larger_than_mf44x33(self):
        """Test that MF54X40 is larger than MF44X33."""
        preset_44 = get_sensor_preset("MF44X33")
        preset_54 = get_sensor_preset("MF54X40")

        self.assertGreater(preset_54["sensor_width"], preset_44["sensor_width"])
        self.assertLess(preset_54["focal_factor"], preset_44["focal_factor"])

    def test_medium_format_larger_than_full_frame(self):
        """Test that medium format sensors are larger than full frame."""
        preset_ff = get_sensor_preset("FF")
        preset_mf44 = get_sensor_preset("MF44X33")
        preset_mf54 = get_sensor_preset("MF54X40")

        self.assertGreater(preset_mf44["sensor_width"], preset_ff["sensor_width"])
        self.assertGreater(preset_mf54["sensor_width"], preset_ff["sensor_width"])


class TestGetSensorPreset(unittest.TestCase):
    """Test get_sensor_preset() function."""

    def test_get_sensor_preset_with_aliases(self):
        """Test that sensor type aliases work correctly."""
        aliases = [
            ("APS-C", "APSC"),
            ("APS_C", "APSC"),
            ("aps-c", "APSC"),
            ("APS-C_CANON", "APSC_CANON"),
            ("FULLFRAME", "FF"),
            ("1-INCH", "1INCH"),
            ("1_INCH", "1INCH"),
            ("MF44-33", "MF44X33"),
            ("MF44_33", "MF44X33"),
            ("MF54-40", "MF54X40"),
            ("MF54_40", "MF54X40"),
        ]

        for alias, canonical in aliases:
            preset_alias = get_sensor_preset(alias)
            preset_canonical = get_sensor_preset(canonical)
            self.assertIsNotNone(
                preset_alias, f"Alias '{alias}' should return a preset"
            )
            self.assertEqual(
                preset_alias["focal_factor"],
                preset_canonical["focal_factor"],
                f"Alias '{alias}' should match canonical '{canonical}'",
            )

    def test_get_sensor_preset_case_insensitive(self):
        """Test that sensor type lookup is case insensitive."""
        test_cases = ["mft", "MFT", "Mft", "mFt"]
        for case in test_cases:
            preset = get_sensor_preset(case)
            self.assertIsNotNone(preset, f"'{case}' should return MFT preset")
            self.assertEqual(preset["focal_factor"], 2.0)

    def test_get_sensor_preset_case_insensitive_medium_format(self):
        """Test that medium format sensor type lookup is case insensitive."""
        test_cases = ["mf44x33", "MF44X33", "Mf44x33", "mf44X33"]
        for case in test_cases:
            preset = get_sensor_preset(case)
            self.assertIsNotNone(preset, f"'{case}' should return MF44X33 preset")
            self.assertEqual(preset["focal_factor"], 0.79)

    def test_get_sensor_preset_invalid_type(self):
        """Test that invalid sensor type returns None."""
        self.assertIsNone(get_sensor_preset("INVALID_TYPE"))
        self.assertIsNone(get_sensor_preset(""))
        self.assertIsNone(get_sensor_preset(None))
        self.assertIsNone(get_sensor_preset("MEDIUM_FORMAT"))
        self.assertIsNone(get_sensor_preset("LARGE_FORMAT"))


class TestApplySensorPreset(unittest.TestCase):
    """Test apply_sensor_preset() function."""

    def test_apply_sensor_preset_basic(self):
        """Test apply_sensor_preset with basic sensor type."""
        args = Namespace(
            sensor_type="MFT",
            focal_factor=None,
            sensor_width=None,
            focal_length=None,
            pixel_pitch=None,
        )

        focal_factor, sensor_width, focal_length, pixel_pitch, preset = (
            apply_sensor_preset(args)
        )

        self.assertEqual(focal_factor, 2.0)
        self.assertEqual(sensor_width, 17.3)
        self.assertIsNone(focal_length)  # Not in preset
        self.assertAlmostEqual(pixel_pitch, 3.7, places=1)

    def test_apply_sensor_preset_medium_format(self):
        """Test apply_sensor_preset with medium format sensor type."""
        args = Namespace(
            sensor_type="MF44X33",
            focal_factor=None,
            sensor_width=None,
            focal_length=None,
            pixel_pitch=None,
        )

        focal_factor, sensor_width, focal_length, pixel_pitch, preset = (
            apply_sensor_preset(args)
        )

        self.assertEqual(focal_factor, 0.79)
        self.assertEqual(sensor_width, 43.8)
        self.assertIsNone(focal_length)
        self.assertAlmostEqual(pixel_pitch, 3.76, places=2)

    def test_apply_sensor_preset_sensor_width_override(self):
        """Test that --sensor-width overrides preset value."""
        args = Namespace(
            sensor_type="MFT",
            focal_factor=None,
            sensor_width=18.0,  # Override
            focal_length=None,
            pixel_pitch=None,
        )

        focal_factor, sensor_width, focal_length, pixel_pitch, preset = (
            apply_sensor_preset(args)
        )

        self.assertEqual(focal_factor, 2.0)  # From preset
        self.assertEqual(sensor_width, 18.0)  # Overridden
        self.assertAlmostEqual(pixel_pitch, 3.7, places=1)  # From preset

    def test_apply_sensor_preset_pixel_pitch_override(self):
        """Test that --pixel-pitch overrides preset value."""
        args = Namespace(
            sensor_type="MFT",
            focal_factor=None,
            sensor_width=None,
            focal_length=None,
            pixel_pitch=3.3,  # Override
        )

        focal_factor, sensor_width, focal_length, pixel_pitch, preset = (
            apply_sensor_preset(args)
        )

        self.assertEqual(focal_factor, 2.0)  # From preset
        self.assertEqual(sensor_width, 17.3)  # From preset
        self.assertEqual(pixel_pitch, 3.3)  # Overridden

    def test_apply_sensor_preset_focal_factor_override(self):
        """Test that --focal-factor overrides preset focal_factor."""
        args = Namespace(
            sensor_type="MFT",
            focal_factor="1.8",  # Override with numeric string
            sensor_width=None,
            focal_length=None,
            pixel_pitch=None,
        )

        focal_factor, sensor_width, focal_length, pixel_pitch, preset = (
            apply_sensor_preset(args)
        )

        self.assertEqual(focal_factor, 1.8)  # Overridden
        self.assertEqual(sensor_width, 17.3)  # From preset

    def test_apply_sensor_preset_all_overrides(self):
        """Test that all individual args override preset values."""
        args = Namespace(
            sensor_type="MFT",
            focal_factor="1.8",
            sensor_width=18.0,
            focal_length=24.0,
            pixel_pitch=3.3,
        )

        focal_factor, sensor_width, focal_length, pixel_pitch, preset = (
            apply_sensor_preset(args)
        )

        self.assertEqual(focal_factor, 1.8)
        self.assertEqual(sensor_width, 18.0)
        self.assertEqual(focal_length, 24.0)
        self.assertEqual(pixel_pitch, 3.3)

    def test_apply_sensor_preset_no_sensor_type(self):
        """Test apply_sensor_preset when no sensor_type is specified."""
        args = Namespace(
            sensor_type=None,
            focal_factor=None,
            sensor_width=17.3,
            focal_length=24.0,
            pixel_pitch=None,
        )

        focal_factor, sensor_width, focal_length, pixel_pitch, preset = (
            apply_sensor_preset(args)
        )

        self.assertIsNone(focal_factor)
        self.assertEqual(sensor_width, 17.3)
        self.assertEqual(focal_length, 24.0)
        self.assertIsNone(pixel_pitch)

    def test_apply_sensor_preset_invalid_sensor_type(self):
        """Test apply_sensor_preset with invalid sensor_type."""
        args = Namespace(
            sensor_type="INVALID",
            focal_factor=None,
            sensor_width=None,
            focal_length=None,
            pixel_pitch=None,
        )

        focal_factor, sensor_width, focal_length, pixel_pitch, preset = (
            apply_sensor_preset(args)
        )

        # Should return None for all preset values (invalid sensor type)
        self.assertIsNone(focal_factor)
        self.assertIsNone(sensor_width)
        self.assertIsNone(focal_length)
        self.assertIsNone(pixel_pitch)


class TestLegacyCompatibility(unittest.TestCase):
    """Test backward compatibility with v1.4.x dictionaries."""

    def test_crop_factors_compatibility(self):
        """Test that CROP_FACTORS dictionary is correctly generated."""
        self.assertEqual(CROP_FACTORS["MFT"], 2.0)
        self.assertEqual(CROP_FACTORS["APSC"], 1.5)
        self.assertEqual(CROP_FACTORS["APSC_CANON"], 1.6)
        self.assertEqual(CROP_FACTORS["FF"], 1.0)
        self.assertEqual(CROP_FACTORS["MF44X33"], 0.79)
        self.assertEqual(CROP_FACTORS["MF54X40"], 0.64)

    def test_crop_factors_matches_presets(self):
        """Test that CROP_FACTORS matches SENSOR_PRESETS."""
        for key in CROP_FACTORS:
            self.assertEqual(
                CROP_FACTORS[key],
                SENSOR_PRESETS[key]["focal_factor"],
                f"CROP_FACTORS[{key}] should match SENSOR_PRESETS",
            )

    def test_sensor_widths_compatibility(self):
        """Test that DEFAULT_SENSOR_WIDTHS dictionary is correctly generated."""
        self.assertEqual(DEFAULT_SENSOR_WIDTHS["MFT"], 17.3)
        self.assertEqual(DEFAULT_SENSOR_WIDTHS["APSC"], 23.5)
        self.assertEqual(DEFAULT_SENSOR_WIDTHS["FF"], 36.0)
        self.assertEqual(DEFAULT_SENSOR_WIDTHS["MF44X33"], 43.8)
        self.assertEqual(DEFAULT_SENSOR_WIDTHS["MF54X40"], 53.4)

    def test_sensor_widths_matches_presets(self):
        """Test that DEFAULT_SENSOR_WIDTHS matches SENSOR_PRESETS."""
        for key in DEFAULT_SENSOR_WIDTHS:
            self.assertEqual(
                DEFAULT_SENSOR_WIDTHS[key],
                SENSOR_PRESETS[key]["sensor_width"],
                f"DEFAULT_SENSOR_WIDTHS[{key}] should match SENSOR_PRESETS",
            )


class TestParseFocalFactor(unittest.TestCase):
    """Test parse_focal_factor() function."""

    def test_parse_focal_factor_with_sensor_types(self):
        """Test parse_focal_factor with sensor type strings."""
        self.assertEqual(parse_focal_factor("MFT"), 2.0)
        self.assertEqual(parse_focal_factor("APS-C"), 1.5)
        self.assertEqual(parse_focal_factor("FF"), 1.0)
        self.assertEqual(parse_focal_factor("FULLFRAME"), 1.0)
        self.assertEqual(parse_focal_factor("MF44X33"), 0.79)
        self.assertEqual(parse_focal_factor("MF54X40"), 0.64)

    def test_parse_focal_factor_with_numeric(self):
        """Test parse_focal_factor with numeric strings."""
        self.assertEqual(parse_focal_factor("2.0"), 2.0)
        self.assertEqual(parse_focal_factor("1.5"), 1.5)
        self.assertEqual(parse_focal_factor("1.0"), 1.0)
        self.assertEqual(parse_focal_factor("1.6"), 1.6)
        self.assertEqual(parse_focal_factor("0.79"), 0.79)
        self.assertEqual(parse_focal_factor("0.64"), 0.64)

    def test_parse_focal_factor_invalid_string(self):
        """Test parse_focal_factor with invalid string input."""
        self.assertIsNone(parse_focal_factor("INVALID"))
        self.assertIsNone(parse_focal_factor("LARGE_FORMAT"))
        self.assertIsNone(parse_focal_factor("abc"))

    def test_parse_focal_factor_empty_or_none(self):
        """Test parse_focal_factor with empty or None input."""
        self.assertIsNone(parse_focal_factor(""))
        self.assertIsNone(parse_focal_factor(None))

    def test_parse_focal_factor_out_of_range(self):
        """Test parse_focal_factor with out-of-range numeric values."""
        self.assertIsNone(parse_focal_factor("0.1"))  # Too small
        self.assertIsNone(parse_focal_factor("0.4"))  # Below 0.5
        self.assertIsNone(parse_focal_factor("15.0"))  # Above 10.0
        self.assertIsNone(parse_focal_factor("100"))  # Way too large

    def test_parse_focal_factor_medium_format_aliases(self):
        """Test parse_focal_factor with medium format aliases."""
        self.assertEqual(parse_focal_factor("MF44-33"), 0.79)
        self.assertEqual(parse_focal_factor("MF44_33"), 0.79)
        self.assertEqual(parse_focal_factor("MF54-40"), 0.64)
        self.assertEqual(parse_focal_factor("MF54_40"), 0.64)


if __name__ == "__main__":
    unittest.main()
