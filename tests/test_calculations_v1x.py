"""
Test suite for calculation functions in meteor_core (v1.x).

Covers:
- NPF Rule calculations
- Pixel pitch calculations
- Star trail length estimation
- Meteor trail length estimation
- NPF compliance evaluation
- Shooting quality score calculation
- Focal factor parsing
"""

import unittest
import sys
import os

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core import (
    calculate_npf_rule,
    calculate_pixel_pitch,
    estimate_star_trail_length,
    estimate_meteor_trail_length,
    evaluate_npf_compliance,
    calculate_shooting_quality_score,
    parse_focal_factor,
    DEFAULT_SENSOR_WIDTHS,
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_LINE_SCORE,
)


class TestCalculateNPFRule(unittest.TestCase):
    """Test calculate_npf_rule() function."""

    def test_npf_rule_basic(self):
        """Test NPF Rule with basic values from README."""
        # Case: 24mm, f/2.8, 3.3μm
        # (35 * 2.8 + 30 * 3.3) / 24 = (98 + 99) / 24 = 8.2083...
        result = calculate_npf_rule(24.0, 2.8, 3.3)
        self.assertAlmostEqual(result, 8.208, places=3)

    def test_npf_rule_mft_example(self):
        """Test NPF Rule with MFT sensor example."""
        # MFT 25mm f/1.8 with 3.7μm pixel pitch
        # (35 * 1.8 + 30 * 3.7) / 50 = (63 + 111) / 50 = 3.48
        # Note: 25mm on MFT = 50mm equivalent
        result = calculate_npf_rule(50.0, 1.8, 3.7)
        self.assertAlmostEqual(result, 3.48, places=2)

    def test_npf_rule_ff_example(self):
        """Test NPF Rule with Full Frame sensor example."""
        # FF 50mm f/2.8 with 5.9μm pixel pitch
        # (35 * 2.8 + 30 * 5.9) / 50 = (98 + 177) / 50 = 5.5
        result = calculate_npf_rule(50.0, 2.8, 5.9)
        self.assertAlmostEqual(result, 5.5, places=1)

    def test_npf_rule_wide_angle(self):
        """Test NPF Rule with wide angle lens."""
        # 14mm f/2.8 with 4.0μm pixel pitch
        # Wider angle = longer allowed exposure
        result = calculate_npf_rule(14.0, 2.8, 4.0)
        self.assertGreater(result, 10.0)  # Wide angle should allow longer exposure

    def test_npf_rule_telephoto(self):
        """Test NPF Rule with telephoto lens."""
        # 200mm f/2.8 with 4.0μm pixel pitch
        # Telephoto = shorter allowed exposure
        result = calculate_npf_rule(200.0, 2.8, 4.0)
        self.assertLess(result, 2.0)  # Telephoto should require shorter exposure

    def test_npf_rule_medium_format(self):
        """Test NPF Rule with medium format sensor."""
        # MF44x33 with 45mm lens (35.5mm equiv), f/2.8, 3.76μm pixel pitch
        result = calculate_npf_rule(35.5, 2.8, 3.76)
        self.assertGreater(result, 3.0)
        self.assertLess(result, 15.0)

    def test_npf_rule_zero_focal_length(self):
        """Test NPF Rule with zero focal length returns 0."""
        result = calculate_npf_rule(0, 2.8, 3.3)
        self.assertEqual(result, 0.0)

    def test_npf_rule_zero_aperture(self):
        """Test NPF Rule with zero aperture returns 0."""
        result = calculate_npf_rule(24.0, 0, 3.3)
        self.assertEqual(result, 0.0)

    def test_npf_rule_zero_pixel_pitch(self):
        """Test NPF Rule with zero pixel pitch returns 0."""
        result = calculate_npf_rule(24.0, 2.8, 0)
        self.assertEqual(result, 0.0)

    def test_npf_rule_negative_values(self):
        """Test NPF Rule with negative values returns 0."""
        self.assertEqual(calculate_npf_rule(-24.0, 2.8, 3.3), 0.0)
        self.assertEqual(calculate_npf_rule(24.0, -2.8, 3.3), 0.0)
        self.assertEqual(calculate_npf_rule(24.0, 2.8, -3.3), 0.0)


class TestCalculatePixelPitch(unittest.TestCase):
    """Test calculate_pixel_pitch() function."""

    def test_pixel_pitch_mft(self):
        """Test pixel pitch calculation for MFT sensor."""
        # MFT: 17.3mm sensor, 5184px width (20MP)
        # 17.3 * 1000 / 5184 = 3.338μm
        result = calculate_pixel_pitch(17.3, 5184)
        self.assertAlmostEqual(result, 3.338, places=2)

    def test_pixel_pitch_apsc(self):
        """Test pixel pitch calculation for APS-C sensor."""
        # APS-C: 23.5mm sensor, 6000px width (26MP)
        # 23.5 * 1000 / 6000 = 3.917μm
        result = calculate_pixel_pitch(23.5, 6000)
        self.assertAlmostEqual(result, 3.917, places=2)

    def test_pixel_pitch_ff(self):
        """Test pixel pitch calculation for Full Frame sensor."""
        # FF: 36mm sensor, 8256px width (45MP)
        # 36 * 1000 / 8256 = 4.36μm
        result = calculate_pixel_pitch(36.0, 8256)
        self.assertAlmostEqual(result, 4.36, places=2)

    def test_pixel_pitch_1inch(self):
        """Test pixel pitch calculation for 1-inch sensor."""
        # 1-inch: 13.2mm sensor, 5472px width (20MP)
        # 13.2 * 1000 / 5472 = 2.41μm
        result = calculate_pixel_pitch(13.2, 5472)
        self.assertAlmostEqual(result, 2.41, places=2)

    def test_pixel_pitch_mf44x33(self):
        """Test pixel pitch calculation for MF44x33 sensor."""
        # MF44x33: 43.8mm sensor, 11648px width (100MP)
        # 43.8 * 1000 / 11648 = 3.76μm
        result = calculate_pixel_pitch(43.8, 11648)
        self.assertAlmostEqual(result, 3.76, places=2)

    def test_pixel_pitch_mf54x40(self):
        """Test pixel pitch calculation for MF54x40 sensor."""
        # MF54x40: 53.4mm sensor, 11600px width (100MP)
        # 53.4 * 1000 / 11600 = 4.60μm
        result = calculate_pixel_pitch(53.4, 11600)
        self.assertAlmostEqual(result, 4.60, places=2)

    def test_pixel_pitch_high_resolution(self):
        """Test pixel pitch for high resolution sensor."""
        # FF: 36mm sensor, 10000px width (very high res)
        # Smaller pixel pitch
        result = calculate_pixel_pitch(36.0, 10000)
        self.assertAlmostEqual(result, 3.6, places=1)

    def test_pixel_pitch_low_resolution(self):
        """Test pixel pitch for low resolution sensor."""
        # FF: 36mm sensor, 4000px width (low res)
        # Larger pixel pitch
        result = calculate_pixel_pitch(36.0, 4000)
        self.assertAlmostEqual(result, 9.0, places=1)


class TestEstimateStarTrailLength(unittest.TestCase):
    """Test estimate_star_trail_length() function."""

    def test_star_trail_basic(self):
        """Test star trail length estimation with basic values."""
        # 24mm, 10s exposure, 5000px width at equator
        result = estimate_star_trail_length(24.0, 10.0, 5000, 0.0)
        # Should return a reasonable positive value
        self.assertGreater(result, 0.0)
        self.assertLess(result, 50.0)  # Shouldn't be excessively long

    def test_star_trail_longer_exposure(self):
        """Test that longer exposure gives longer trails."""
        short_trail = estimate_star_trail_length(24.0, 5.0, 5000, 0.0)
        long_trail = estimate_star_trail_length(24.0, 20.0, 5000, 0.0)

        self.assertGreater(long_trail, short_trail)
        # Should be roughly proportional (4x exposure = 4x trail)
        self.assertAlmostEqual(long_trail / short_trail, 4.0, places=1)

    def test_star_trail_longer_focal_length(self):
        """Test that longer focal length gives longer trails."""
        wide_trail = estimate_star_trail_length(14.0, 10.0, 5000, 0.0)
        tele_trail = estimate_star_trail_length(200.0, 10.0, 5000, 0.0)

        # Telephoto should have longer trails (narrower FOV)
        self.assertGreater(tele_trail, wide_trail)

    def test_star_trail_higher_resolution(self):
        """Test that higher resolution gives longer trails (more pixels)."""
        low_res_trail = estimate_star_trail_length(24.0, 10.0, 3000, 0.0)
        high_res_trail = estimate_star_trail_length(24.0, 10.0, 8000, 0.0)

        # Higher resolution should have longer trails in pixels
        self.assertGreater(high_res_trail, low_res_trail)

    def test_star_trail_declination_equator(self):
        """Test star trail at equator (maximum motion)."""
        equator_trail = estimate_star_trail_length(24.0, 10.0, 5000, 0.0)
        mid_lat_trail = estimate_star_trail_length(24.0, 10.0, 5000, 45.0)
        pole_trail = estimate_star_trail_length(24.0, 10.0, 5000, 89.0)

        # Equator should have longest trails
        self.assertGreater(equator_trail, mid_lat_trail)
        self.assertGreater(mid_lat_trail, pole_trail)

    def test_star_trail_zero_values(self):
        """Test star trail with zero values returns 0."""
        self.assertEqual(estimate_star_trail_length(0, 10.0, 5000, 0.0), 0.0)
        self.assertEqual(estimate_star_trail_length(24.0, 0, 5000, 0.0), 0.0)
        self.assertEqual(estimate_star_trail_length(24.0, 10.0, 0, 0.0), 0.0)

    def test_star_trail_negative_values(self):
        """Test star trail with negative values returns 0."""
        self.assertEqual(estimate_star_trail_length(-24.0, 10.0, 5000, 0.0), 0.0)
        self.assertEqual(estimate_star_trail_length(24.0, -10.0, 5000, 0.0), 0.0)
        self.assertEqual(estimate_star_trail_length(24.0, 10.0, -5000, 0.0), 0.0)


class TestEstimateMeteorTrailLength(unittest.TestCase):
    """Test estimate_meteor_trail_length() function."""

    def test_meteor_trail_longer_than_star(self):
        """Test that meteor trails are longer than star trails."""
        star_trail = estimate_star_trail_length(24.0, 10.0, 5000, 0.0)
        meteor_trail = estimate_meteor_trail_length(24.0, 10.0, 5000, 3.0)

        self.assertGreater(meteor_trail, star_trail)

    def test_meteor_trail_speed_factor(self):
        """Test meteor trail with different speed factors."""
        slow_meteor = estimate_meteor_trail_length(24.0, 10.0, 5000, 2.0)
        fast_meteor = estimate_meteor_trail_length(24.0, 10.0, 5000, 5.0)

        # Faster meteor should leave longer trail
        self.assertGreater(fast_meteor, slow_meteor)
        self.assertAlmostEqual(fast_meteor / slow_meteor, 2.5, places=1)

    def test_meteor_trail_default_factor(self):
        """Test meteor trail with default speed factor (3.0)."""
        star_trail = estimate_star_trail_length(24.0, 10.0, 5000, 0.0)
        meteor_trail = estimate_meteor_trail_length(24.0, 10.0, 5000)

        # Default factor is 3.0
        self.assertAlmostEqual(meteor_trail / star_trail, 3.0, places=1)


class TestEvaluateNPFCompliance(unittest.TestCase):
    """Test evaluate_npf_compliance() function."""

    def test_compliance_ok(self):
        """Test OK compliance level."""
        level, factor = evaluate_npf_compliance(5.0, 10.0)  # 50% of NPF limit
        self.assertEqual(level, "OK")
        self.assertAlmostEqual(factor, 0.5, places=2)

    def test_compliance_exactly_at_limit(self):
        """Test compliance exactly at NPF limit."""
        level, factor = evaluate_npf_compliance(10.0, 10.0)
        self.assertEqual(level, "OK")
        self.assertAlmostEqual(factor, 1.0, places=2)

    def test_compliance_warning(self):
        """Test WARNING compliance level."""
        level, factor = evaluate_npf_compliance(12.0, 10.0)  # 20% over
        self.assertEqual(level, "WARNING")
        self.assertAlmostEqual(factor, 1.2, places=2)

    def test_compliance_critical(self):
        """Test CRITICAL compliance level."""
        level, factor = evaluate_npf_compliance(20.0, 10.0)  # 100% over
        self.assertEqual(level, "CRITICAL")
        self.assertAlmostEqual(factor, 2.0, places=2)

    def test_compliance_unknown_zero_npf(self):
        """Test UNKNOWN compliance when NPF is zero."""
        level, factor = evaluate_npf_compliance(10.0, 0)
        self.assertEqual(level, "UNKNOWN")
        self.assertEqual(factor, 0.0)


class TestCalculateShootingQualityScore(unittest.TestCase):
    """Test calculate_shooting_quality_score() function."""

    def test_quality_excellent(self):
        """Test excellent quality score."""
        exif_data = {
            "iso": 800,
            "focal_length_35mm": 20,
        }
        npf_metrics = {
            "overshoot_factor": 0.8,  # Under NPF limit
        }

        score, level = calculate_shooting_quality_score(exif_data, npf_metrics)
        self.assertEqual(level, "EXCELLENT")
        self.assertGreaterEqual(score, 0.8)

    def test_quality_good(self):
        """Test good quality score."""
        exif_data = {
            "iso": 3200,
            "focal_length_35mm": 35,
        }
        npf_metrics = {
            "overshoot_factor": 1.2,  # Slight overshoot
        }

        score, level = calculate_shooting_quality_score(exif_data, npf_metrics)
        self.assertIn(level, ["EXCELLENT", "GOOD"])
        self.assertGreaterEqual(score, 0.6)

    def test_quality_fair(self):
        """Test fair quality score."""
        exif_data = {
            "iso": 4000,
            "focal_length_35mm": 35,
        }
        npf_metrics = {
            "overshoot_factor": 1.8,  # Moderate overshoot
        }

        score, level = calculate_shooting_quality_score(exif_data, npf_metrics)
        self.assertIn(level, ["GOOD", "FAIR", "POOR"])
        self.assertLess(score, 0.8)  # Should not be excellent

    def test_quality_poor(self):
        """Test poor quality score (bad conditions)."""
        exif_data = {
            "iso": 12800,
            "focal_length_35mm": 100,
        }
        npf_metrics = {
            "overshoot_factor": 3.0,  # Critical overshoot
        }

        score, level = calculate_shooting_quality_score(exif_data, npf_metrics)
        self.assertEqual(level, "POOR")
        self.assertLess(score, 0.4)

    def test_quality_npf_dominates(self):
        """Test that NPF compliance has largest impact on score."""
        # Good ISO and focal length but critical NPF
        exif_data = {
            "iso": 800,
            "focal_length_35mm": 20,
        }
        npf_metrics_good = {"overshoot_factor": 0.5}
        npf_metrics_bad = {"overshoot_factor": 3.0}

        score_good, _ = calculate_shooting_quality_score(exif_data, npf_metrics_good)
        score_bad, _ = calculate_shooting_quality_score(exif_data, npf_metrics_bad)

        # Bad NPF should significantly reduce score
        self.assertGreater(score_good, score_bad * 2)

    def test_quality_missing_exif(self):
        """Test quality score with missing EXIF data."""
        exif_data = {}
        npf_metrics = {"overshoot_factor": 1.0}

        # Should not crash, return reasonable default
        score, level = calculate_shooting_quality_score(exif_data, npf_metrics)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestParseFocalFactor(unittest.TestCase):
    """Test parse_focal_factor() function."""

    def test_parse_numeric_string(self):
        """Test parsing numeric string values."""
        self.assertEqual(parse_focal_factor("2.0"), 2.0)
        self.assertEqual(parse_focal_factor("1.5"), 1.5)
        self.assertEqual(parse_focal_factor("1.0"), 1.0)
        self.assertEqual(parse_focal_factor("1.6"), 1.6)
        self.assertEqual(parse_focal_factor("0.79"), 0.79)
        self.assertEqual(parse_focal_factor("0.64"), 0.64)

    def test_parse_sensor_type_string(self):
        """Test parsing sensor type strings."""
        self.assertEqual(parse_focal_factor("MFT"), 2.0)
        self.assertEqual(parse_focal_factor("APS-C"), 1.5)
        self.assertEqual(parse_focal_factor("FF"), 1.0)
        self.assertEqual(parse_focal_factor("FULLFRAME"), 1.0)
        self.assertEqual(parse_focal_factor("MF44X33"), 0.79)
        self.assertEqual(parse_focal_factor("MF54X40"), 0.64)

    def test_parse_case_insensitive(self):
        """Test that parsing is case insensitive."""
        self.assertEqual(parse_focal_factor("mft"), 2.0)
        self.assertEqual(parse_focal_factor("Mft"), 2.0)
        self.assertEqual(parse_focal_factor("aps-c"), 1.5)
        self.assertEqual(parse_focal_factor("APS-C"), 1.5)
        self.assertEqual(parse_focal_factor("FullFrame"), 1.0)
        self.assertEqual(parse_focal_factor("mf44x33"), 0.79)

    def test_parse_with_hyphens_and_underscores(self):
        """Test parsing with different separators."""
        self.assertEqual(parse_focal_factor("APS-C"), 1.5)
        self.assertEqual(parse_focal_factor("APS_C"), 1.5)
        self.assertEqual(parse_focal_factor("APS-C_CANON"), 1.6)
        self.assertEqual(parse_focal_factor("MF44-33"), 0.79)
        self.assertEqual(parse_focal_factor("MF44_33"), 0.79)
        self.assertEqual(parse_focal_factor("MF54-40"), 0.64)

    def test_parse_invalid_string(self):
        """Test parsing invalid strings."""
        self.assertIsNone(parse_focal_factor("invalid"))
        self.assertIsNone(parse_focal_factor("LARGE_FORMAT"))
        self.assertIsNone(parse_focal_factor("abc123"))

    def test_parse_empty_or_none(self):
        """Test parsing empty or None values."""
        self.assertIsNone(parse_focal_factor(""))
        self.assertIsNone(parse_focal_factor(None))

    def test_parse_out_of_range_numeric(self):
        """Test parsing out-of-range numeric values."""
        self.assertIsNone(parse_focal_factor("0.1"))  # Too small (< 0.5)
        self.assertIsNone(parse_focal_factor("0.4"))  # Just under limit
        self.assertIsNone(parse_focal_factor("10.1"))  # Just over limit
        self.assertIsNone(parse_focal_factor("15.0"))  # Too large (> 10.0)

    def test_parse_boundary_values(self):
        """Test parsing boundary numeric values."""
        self.assertEqual(parse_focal_factor("0.5"), 0.5)  # Minimum valid
        self.assertEqual(parse_focal_factor("10.0"), 10.0)  # Maximum valid


class TestDefaultSensorWidths(unittest.TestCase):
    """Test DEFAULT_SENSOR_WIDTHS constant."""

    def test_sensor_widths_exist(self):
        """Test that common sensor widths are defined."""
        self.assertIn("MFT", DEFAULT_SENSOR_WIDTHS)
        self.assertIn("APSC", DEFAULT_SENSOR_WIDTHS)
        self.assertIn("FF", DEFAULT_SENSOR_WIDTHS)
        self.assertIn("MF44X33", DEFAULT_SENSOR_WIDTHS)
        self.assertIn("MF54X40", DEFAULT_SENSOR_WIDTHS)

    def test_sensor_widths_values(self):
        """Test that sensor width values are correct."""
        self.assertEqual(DEFAULT_SENSOR_WIDTHS["MFT"], 17.3)
        self.assertEqual(DEFAULT_SENSOR_WIDTHS["APSC"], 23.5)
        self.assertEqual(DEFAULT_SENSOR_WIDTHS["FF"], 36.0)
        self.assertEqual(DEFAULT_SENSOR_WIDTHS["MF44X33"], 43.8)
        self.assertEqual(DEFAULT_SENSOR_WIDTHS["MF54X40"], 53.4)

    def test_sensor_widths_reasonable_range(self):
        """Test that all sensor widths are in reasonable range."""
        for sensor, width in DEFAULT_SENSOR_WIDTHS.items():
            self.assertGreater(width, 5.0, f"{sensor} width too small")
            self.assertLess(width, 60.0, f"{sensor} width too large")

    def test_sensor_widths_ordering(self):
        """Test that medium format sensors are larger than full frame."""
        self.assertGreater(
            DEFAULT_SENSOR_WIDTHS["MF44X33"], DEFAULT_SENSOR_WIDTHS["FF"]
        )
        self.assertGreater(
            DEFAULT_SENSOR_WIDTHS["MF54X40"], DEFAULT_SENSOR_WIDTHS["MF44X33"]
        )


class TestDefaultParameters(unittest.TestCase):
    """Test default parameter constants."""

    def test_default_diff_threshold(self):
        """Test default diff_threshold value."""
        self.assertEqual(DEFAULT_DIFF_THRESHOLD, 8)
        self.assertGreater(DEFAULT_DIFF_THRESHOLD, 0)

    def test_default_min_area(self):
        """Test default min_area value."""
        self.assertEqual(DEFAULT_MIN_AREA, 10)
        self.assertGreater(DEFAULT_MIN_AREA, 0)

    def test_default_min_line_score(self):
        """Test default min_line_score value."""
        self.assertEqual(DEFAULT_MIN_LINE_SCORE, 80.0)
        self.assertGreater(DEFAULT_MIN_LINE_SCORE, 0)


if __name__ == "__main__":
    unittest.main()
