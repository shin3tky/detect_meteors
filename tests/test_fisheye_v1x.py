#!/usr/bin/env python
"""
Test suite for fisheye correction functions (v1.5.3)
"""
import unittest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core import (  # noqa: E402
    calculate_fisheye_effective_focal_length,
    calculate_fisheye_edge_focal_length,
    calculate_fisheye_trail_length_ratio,
    get_fisheye_max_trail_ratio,
    calculate_npf_metrics,
    calculate_npf_rule,
    FISHEYE_PROJECTION_MODELS,
    DEFAULT_FISHEYE_MODEL,
)


class TestFisheyeEffectiveFocalLength(unittest.TestCase):
    """Tests for calculate_fisheye_effective_focal_length()"""

    def test_center_focal_length_unchanged(self):
        """At center (r=0), effective focal length equals nominal"""
        nominal_fl = 8.0  # 8mm fisheye
        effective_fl = calculate_fisheye_effective_focal_length(nominal_fl, 0.0)
        self.assertAlmostEqual(effective_fl, nominal_fl, places=5)

    def test_edge_focal_length_shorter(self):
        """At edge (r=1), effective focal length is shorter"""
        nominal_fl = 8.0
        effective_fl = calculate_fisheye_effective_focal_length(nominal_fl, 1.0)
        # For equisolid with 90° max angle: cos(45°) ≈ 0.707
        expected = nominal_fl * math.cos(math.pi / 4)  # cos(45°)
        self.assertAlmostEqual(effective_fl, expected, places=5)

    def test_mid_radius_intermediate(self):
        """At mid radius (r=0.5), effective focal length is intermediate"""
        nominal_fl = 8.0
        center_fl = calculate_fisheye_effective_focal_length(nominal_fl, 0.0)
        mid_fl = calculate_fisheye_effective_focal_length(nominal_fl, 0.5)
        edge_fl = calculate_fisheye_effective_focal_length(nominal_fl, 1.0)

        self.assertGreater(mid_fl, edge_fl)
        self.assertLess(mid_fl, center_fl)

    def test_negative_radius_clamped(self):
        """Negative radius is clamped to 0"""
        nominal_fl = 8.0
        effective_fl = calculate_fisheye_effective_focal_length(nominal_fl, -0.5)
        self.assertAlmostEqual(effective_fl, nominal_fl, places=5)

    def test_over_one_radius_clamped(self):
        """Radius > 1 is clamped to 1"""
        nominal_fl = 8.0
        effective_fl_1 = calculate_fisheye_effective_focal_length(nominal_fl, 1.0)
        effective_fl_2 = calculate_fisheye_effective_focal_length(nominal_fl, 1.5)
        self.assertAlmostEqual(effective_fl_1, effective_fl_2, places=5)

    def test_unknown_projection_returns_nominal(self):
        """Unknown projection model returns nominal focal length"""
        nominal_fl = 8.0
        effective_fl = calculate_fisheye_effective_focal_length(
            nominal_fl, 0.5, projection_model="UNKNOWN"
        )
        self.assertAlmostEqual(effective_fl, nominal_fl, places=5)


class TestFisheyeEdgeFocalLength(unittest.TestCase):
    """Tests for calculate_fisheye_edge_focal_length()"""

    def test_edge_focal_length_calculation(self):
        """Edge focal length is correctly calculated"""
        nominal_fl = 16.0  # 16mm fisheye (35mm equiv)
        edge_fl = calculate_fisheye_edge_focal_length(nominal_fl)

        # For equisolid: edge_fl = nominal * cos(45°)
        expected = nominal_fl * math.cos(math.pi / 4)
        self.assertAlmostEqual(edge_fl, expected, places=5)

    def test_edge_focal_length_ratio(self):
        """Edge focal length is approximately 0.707x nominal"""
        nominal_fl = 8.0
        edge_fl = calculate_fisheye_edge_focal_length(nominal_fl)

        ratio = edge_fl / nominal_fl
        self.assertAlmostEqual(ratio, math.cos(math.pi / 4), places=5)


class TestFisheyeTrailLengthRatio(unittest.TestCase):
    """Tests for calculate_fisheye_trail_length_ratio()"""

    def test_center_ratio_is_one(self):
        """At center (r=0), trail length ratio is 1.0"""
        ratio = calculate_fisheye_trail_length_ratio(0.0)
        self.assertAlmostEqual(ratio, 1.0, places=5)

    def test_edge_ratio_greater_than_one(self):
        """At edge (r=1), trail length ratio is > 1.0"""
        ratio = calculate_fisheye_trail_length_ratio(1.0)
        self.assertGreater(ratio, 1.0)

    def test_edge_ratio_value(self):
        """At edge, trail length ratio is 1/cos(45°) ≈ 1.414"""
        ratio = calculate_fisheye_trail_length_ratio(1.0)
        expected = 1.0 / math.cos(math.pi / 4)  # ≈ 1.414
        self.assertAlmostEqual(ratio, expected, places=5)

    def test_ratio_increases_with_radius(self):
        """Trail length ratio increases toward edge"""
        ratios = [
            calculate_fisheye_trail_length_ratio(r) for r in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        for i in range(len(ratios) - 1):
            self.assertLess(ratios[i], ratios[i + 1])


class TestFisheyeMaxTrailRatio(unittest.TestCase):
    """Tests for get_fisheye_max_trail_ratio()"""

    def test_max_ratio_equals_edge_ratio(self):
        """Max trail ratio equals ratio at r=1"""
        max_ratio = get_fisheye_max_trail_ratio()
        edge_ratio = calculate_fisheye_trail_length_ratio(1.0)
        self.assertAlmostEqual(max_ratio, edge_ratio, places=5)

    def test_max_ratio_approximately_1_414(self):
        """Max trail ratio is approximately sqrt(2) for equisolid"""
        max_ratio = get_fisheye_max_trail_ratio()
        self.assertAlmostEqual(max_ratio, math.sqrt(2), places=5)


class TestNPFMetricsWithFisheye(unittest.TestCase):
    """Tests for calculate_npf_metrics() with fisheye correction"""

    def setUp(self):
        """Set up common test data"""
        self.exif_data = {
            "focal_length_35mm": 16.0,  # 16mm fisheye
            "f_number": 2.8,
            "exposure_time": 20.0,
            "image_width": 6000,
            "image_height": 4000,
            "iso": 3200,
        }

    def test_fisheye_disabled_by_default(self):
        """Fisheye is disabled by default"""
        metrics = calculate_npf_metrics(self.exif_data)
        self.assertFalse(metrics.get("fisheye", False))
        self.assertIsNone(metrics.get("fisheye_model"))

    def test_fisheye_enabled(self):
        """Fisheye flag is properly stored"""
        metrics = calculate_npf_metrics(self.exif_data, fisheye=True)
        self.assertTrue(metrics.get("fisheye"))
        self.assertEqual(metrics.get("fisheye_model"), "EQUISOLID")

    def test_effective_focal_length_stored(self):
        """Effective focal length is stored in metrics"""
        metrics = calculate_npf_metrics(self.exif_data, fisheye=True)
        self.assertIsNotNone(metrics.get("effective_focal_length"))

    def test_fisheye_shorter_effective_focal_length(self):
        """Fisheye results in shorter effective focal length"""
        metrics_normal = calculate_npf_metrics(self.exif_data, fisheye=False)
        metrics_fisheye = calculate_npf_metrics(self.exif_data, fisheye=True)

        # Effective focal length should be shorter with fisheye
        self.assertLess(
            metrics_fisheye["effective_focal_length"],
            metrics_normal["effective_focal_length"],
        )

    def test_fisheye_longer_npf_exposure(self):
        """Fisheye results in longer NPF recommended exposure"""
        # NPF = (35*F + 30*PP) / FL
        # Shorter FL -> longer NPF time
        metrics_normal = calculate_npf_metrics(self.exif_data, fisheye=False)
        metrics_fisheye = calculate_npf_metrics(self.exif_data, fisheye=True)

        self.assertGreater(
            metrics_fisheye["npf_recommended_sec"],
            metrics_normal["npf_recommended_sec"],
        )

    def test_fisheye_longer_star_trail(self):
        """Fisheye results in longer star trail estimate"""
        metrics_normal = calculate_npf_metrics(self.exif_data, fisheye=False)
        metrics_fisheye = calculate_npf_metrics(self.exif_data, fisheye=True)

        self.assertGreater(
            metrics_fisheye["star_trail_px"],
            metrics_normal["star_trail_px"],
        )

    def test_trail_length_ratio_applied(self):
        """Trail length ratio is properly applied"""
        metrics_normal = calculate_npf_metrics(self.exif_data, fisheye=False)
        metrics_fisheye = calculate_npf_metrics(self.exif_data, fisheye=True)

        expected_ratio = get_fisheye_max_trail_ratio()
        actual_ratio = (
            metrics_fisheye["star_trail_px"] / metrics_normal["star_trail_px"]
        )

        self.assertAlmostEqual(actual_ratio, expected_ratio, places=4)

    def test_trail_length_ratio_stored(self):
        """Trail length ratio is stored in metrics"""
        metrics = calculate_npf_metrics(self.exif_data, fisheye=True)
        expected_ratio = get_fisheye_max_trail_ratio()
        self.assertAlmostEqual(metrics["trail_length_ratio"], expected_ratio, places=5)


class TestFisheyeNPFIntegration(unittest.TestCase):
    """Integration tests for fisheye + NPF calculations"""

    def test_8mm_fisheye_ff(self):
        """Test 8mm f/2.8 fisheye on full frame"""
        exif_data = {
            "focal_length_35mm": 8.0,
            "f_number": 2.8,
            "exposure_time": 30.0,
            "image_width": 6000,
            "iso": 6400,
        }

        metrics = calculate_npf_metrics(exif_data, sensor_width_mm=36.0, fisheye=True)

        # Effective focal length at edge should be ~5.66mm
        expected_eff_fl = 8.0 * math.cos(math.pi / 4)
        self.assertAlmostEqual(
            metrics["effective_focal_length"], expected_eff_fl, places=1
        )

        # NPF time should be based on shorter effective focal length
        # NPF = (35*2.8 + 30*PP) / 5.66
        # This should be longer than without fisheye
        npf_without_fisheye = calculate_npf_rule(8.0, 2.8, metrics["pixel_pitch_um"])
        self.assertGreater(metrics["npf_recommended_sec"], npf_without_fisheye)

    def test_fisheye_compliance_more_lenient(self):
        """Fisheye correction makes compliance more lenient"""
        exif_data = {
            "focal_length_35mm": 16.0,
            "f_number": 2.8,
            "exposure_time": 25.0,  # Might be CRITICAL without fisheye
            "image_width": 6000,
            "iso": 3200,
        }

        metrics_normal = calculate_npf_metrics(exif_data, fisheye=False)
        metrics_fisheye = calculate_npf_metrics(exif_data, fisheye=True)

        # With longer recommended exposure, overshoot should be lower
        self.assertLess(
            metrics_fisheye["overshoot_factor"],
            metrics_normal["overshoot_factor"],
        )


class TestFisheyeProjectionModels(unittest.TestCase):
    """Tests for fisheye projection model configuration"""

    def test_equisolid_model_exists(self):
        """EQUISOLID model is defined"""
        self.assertIn("EQUISOLID", FISHEYE_PROJECTION_MODELS)

    def test_default_model_is_equisolid(self):
        """Default model is EQUISOLID"""
        self.assertEqual(DEFAULT_FISHEYE_MODEL, "EQUISOLID")

    def test_model_has_required_fields(self):
        """Each model has required fields"""
        for model_name, model_info in FISHEYE_PROJECTION_MODELS.items():
            self.assertIn("name", model_info)
            self.assertIn("description", model_info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
