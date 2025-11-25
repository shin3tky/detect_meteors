import unittest
import sys
import os

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detect_meteors_cli import (
    calculate_npf_rule,
    parse_focal_factor,
    DEFAULT_SENSOR_WIDTHS,
)


class TestMeteorUtils(unittest.TestCase):

    def test_calculate_npf_rule(self):
        """Test NPF Rule"""
        # Case 1: 24mm, f/2.8, 3.3um (README)
        # (35 * 2.8 + 30 * 3.3) / 24 = (98 + 99) / 24 = 8.2083...
        result = calculate_npf_rule(24.0, 2.8, 3.3)
        self.assertAlmostEqual(result, 8.208, places=3)

        # Case 2: Zero divide
        self.assertEqual(calculate_npf_rule(0, 2.8, 3.3), 0.0)

    def test_parse_focal_factor(self):
        """Parse focal factor"""
        # numeric string
        self.assertEqual(parse_focal_factor("2.0"), 2.0)

        # string
        self.assertEqual(parse_focal_factor("MFT"), 2.0)
        self.assertEqual(parse_focal_factor("aps-c"), 1.5)
        self.assertEqual(parse_focal_factor("FullFrame"), 1.0)

        # invalid
        self.assertIsNone(parse_focal_factor("invalid"))


if __name__ == "__main__":
    unittest.main()
