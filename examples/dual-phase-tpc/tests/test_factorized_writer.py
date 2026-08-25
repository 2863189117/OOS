#!/usr/bin/env python3
"""Regression tests for the example-owned factorized LXe block writer."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from generator.write_intrinsic_lxe_block import write_factorized_block


class FactorizedWriterTest(unittest.TestCase):
    def test_runtime_state_count_includes_position_azimuth(self) -> None:
        phase_grid = {
            "position_radial_bins": 2,
            "position_phi_bins": 3,
            "direction_mu_bins": 2,
            "direction_phi_bins": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "block.h5"
            write_factorized_block(
                output,
                coefficients=np.full((2, 2, 2, 1, 1), 0.25 + 0.0j),
                expected_return=np.full((2, 2, 2), 0.25),
                audit_values=np.full((2, 2, 2, 1), 0.75),
                surface_radius_mm=np.asarray([0.0]),
                surface_ring_area_mm2=np.asarray([1.0]),
                angular_weight=np.asarray([1.0]),
                surface_element=np.asarray([0]),
                barycentric=np.asarray([[1.0, 0.0, 0.0]]),
                side=np.asarray([1]),
                direction_local=np.asarray([[0.0, 0.0, 1.0]]),
                stokes=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                reference_axis_local=np.asarray([[1.0, 0.0, 0.0]]),
                phase_grid=phase_grid,
                audit_names=["lxe_nonreturn"],
                metadata={"state_count": 8},
            )
            with h5py.File(output, "r") as handle:
                raw = bytes(
                    np.asarray(
                        handle["/metadata/generator_json"], dtype=np.uint8
                    )
                )
            metadata = json.loads(raw.decode())
            self.assertEqual(metadata["coefficient_row_count"], 8)
            self.assertEqual(metadata["state_count"], 24)


if __name__ == "__main__":
    unittest.main()
