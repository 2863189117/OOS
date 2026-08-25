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

    def test_joint_surface_angle_coefficients_are_preserved(self) -> None:
        phase_grid = {
            "position_radial_bins": 1,
            "position_phi_bins": 1,
            "direction_mu_bins": 1,
            "direction_phi_bins": 1,
        }
        coefficients = np.asarray([[[[[[0.10, 0.30]]]]]], dtype=np.complex128)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "joint-block.h5"
            write_factorized_block(
                output,
                coefficients=coefficients,
                expected_return=np.asarray([[[0.40]]]),
                audit_values=np.asarray([[[[0.60]]]]),
                surface_radius_mm=np.asarray([0.0]),
                surface_ring_area_mm2=np.asarray([1.0]),
                angular_weight=np.asarray([0.25, 0.75]),
                surface_element=np.asarray([0, 0]),
                barycentric=np.asarray(
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
                ),
                side=np.asarray([1, 1]),
                direction_local=np.asarray(
                    [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]
                ),
                stokes=np.asarray(
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                ),
                reference_axis_local=np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                ),
                phase_grid=phase_grid,
                audit_names=["lxe_nonreturn"],
            )
            with h5py.File(output, "r") as handle:
                actual = np.asarray(handle["/function/coefficients_real"])
                raw = bytes(
                    np.asarray(
                        handle["/metadata/generator_json"], dtype=np.uint8
                    )
                )
            np.testing.assert_array_equal(actual, coefficients.real)
            self.assertEqual(
                json.loads(raw.decode())["coefficient_layout"],
                "joint_surface_angle_v1",
            )

    def test_joint_surface_angle_axis_must_match_egress_basis(self) -> None:
        phase_grid = {
            "position_radial_bins": 1,
            "position_phi_bins": 1,
            "direction_mu_bins": 1,
            "direction_phi_bins": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "angle axis"):
                write_factorized_block(
                    Path(directory) / "invalid.h5",
                    coefficients=np.zeros((1, 1, 1, 1, 1, 3), complex),
                    expected_return=np.asarray([[[0.0]]]),
                    audit_values=np.asarray([[[[1.0]]]]),
                    surface_radius_mm=np.asarray([0.0]),
                    surface_ring_area_mm2=np.asarray([1.0]),
                    angular_weight=np.asarray([0.5, 0.5]),
                    surface_element=np.asarray([0, 0]),
                    barycentric=np.asarray(
                        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
                    ),
                    side=np.asarray([1, 1]),
                    direction_local=np.asarray(
                        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
                    ),
                    stokes=np.asarray(
                        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                    ),
                    reference_axis_local=np.asarray(
                        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
                    ),
                    phase_grid=phase_grid,
                    audit_names=["lxe_nonreturn"],
                )


if __name__ == "__main__":
    unittest.main()
