"""Checks for the synthetic 2-inch hexagonal PMT layout."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT / "generator"))

from build_geometry import GeometryConfig, pmt_layout


class GeometryLayoutTest(unittest.TestCase):
    def test_public_geometry_and_hexagonal_layout(self) -> None:
        config = GeometryConfig()
        positions, channels = pmt_layout(config)

        self.assertEqual(2.0 * config.active_radius_mm, 2000.0)
        self.assertEqual(2.0 * config.aperture_radius_mm, 50.8)
        self.assertEqual(config.lxe_depth_mm, 2000.0)
        self.assertEqual(len(positions), 721)
        np.testing.assert_array_equal(
            channels, np.arange(len(positions), dtype=np.int32)
        )

        row_pitch = math.sqrt(3.0) * 0.5 * config.pmt_pitch_mm
        for x, y in positions:
            row = round(float(y) / row_pitch)
            self.assertAlmostEqual(float(y), row * row_pitch, places=9)
            offset = 0.5 * config.pmt_pitch_mm if row % 2 else 0.0
            column = round((float(x) - offset) / config.pmt_pitch_mm)
            self.assertAlmostEqual(
                float(x), column * config.pmt_pitch_mm + offset, places=9
            )


if __name__ == "__main__":
    unittest.main()
