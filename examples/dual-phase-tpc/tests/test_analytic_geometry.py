"""Checks for identity-bearing analytic geometry discretization controls."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT / "generator"))

from build_analytic_geometry import (  # noqa: E402
    AnalyticDiscretization,
    _ragged_ring_phi_counts,
    build,
)
from build_geometry import GXE_LXE, GeometryConfig  # noqa: E402


def compact_geometry() -> GeometryConfig:
    return replace(
        GeometryConfig(),
        active_radius_mm=100.0,
        field_cage_thickness_mm=5.0,
        wall_radius_mm=110.0,
        pmt_array_radius_mm=70.0,
        lxe_depth_mm=200.0,
    )


class AnalyticGeometryTest(unittest.TestCase):
    def test_ragged_ring_counts_follow_arc_minimum_and_multiple(self) -> None:
        np.testing.assert_array_equal(
            _ragged_ring_phi_counts(
                np.asarray([0.0, 10.0, 100.0]),
                target_arc_mm=25.0,
                min_phi=5,
                phi_multiple=6,
            ),
            [6, 6, 30],
        )

    def test_lxe_surface_phi_bins_control_elements_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for phi_bins in (80, 160):
                output = directory / f"geometry-{phi_bins}.h5"
                discretization = AnalyticDiscretization.profile("debug")
                if phi_bins != 80:
                    discretization = replace(
                        discretization,
                        lxe_surface_phi_bins=phi_bins,
                    )
                build(
                    output,
                    compact_geometry(),
                    discretization,
                    validation_profile="smoke",
                    force=False,
                )
                with h5py.File(output, "r") as handle:
                    primitive_surface = np.asarray(
                        handle["/analytic/surface_id"], dtype=np.uint32
                    )
                    element_primitive = np.asarray(
                        handle["/analytic/elements/primitive_index"],
                        dtype=np.uint32,
                    )
                    lxe_elements = np.count_nonzero(
                        primitive_surface[element_primitive] == GXE_LXE
                    )
                    identity = json.loads(
                        bytes(
                            handle["/metadata/analytic_generator_json"][:]
                        ).decode()
                    )

                self.assertEqual(lxe_elements, 40 * phi_bins)
                self.assertEqual(
                    identity["analytic_discretization"][
                        "lxe_surface_phi_bins"
                    ],
                    phi_bins,
                )

    def test_ragged_geometry_owns_ring_points_areas_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ragged-geometry.h5"
            config = compact_geometry()
            discretization = replace(
                AnalyticDiscretization.profile("debug"),
                lxe_surface_target_arc_mm=25.0,
                lxe_surface_min_phi=5,
                lxe_surface_phi_multiple=6,
            )
            build(
                output,
                config,
                discretization,
                validation_profile="smoke",
                force=True,
            )
            with h5py.File(output, "r") as handle:
                primitive_surface = np.asarray(
                    handle["/analytic/surface_id"], dtype=np.uint32
                )
                owner = np.asarray(
                    handle["/analytic/elements/primitive_index"],
                    dtype=np.uint32,
                )
                selected = np.flatnonzero(
                    primitive_surface[owner] == GXE_LXE
                )
                element_id = np.asarray(
                    handle["/analytic/elements/surface_element"],
                    dtype=np.uint64,
                )[selected]
                centers = np.asarray(
                    handle["/analytic/elements/center_mm"], dtype=float
                )[selected]
                area = np.asarray(
                    handle["/analytic/elements/area_mm2"], dtype=float
                )[selected]
            order = np.argsort(element_id)
            element_id = element_id[order]
            centers = centers[order]
            area = area[order]
            radial_node, radial_weight = np.polynomial.legendre.leggauss(40)
            radius = config.active_radius_mm * np.sqrt(
                0.5 * (radial_node + 1.0)
            )
            ring_area = (
                math.pi
                * config.active_radius_mm**2
                * 0.5
                * radial_weight
            )
            radial_outer_u = np.cumsum(0.5 * radial_weight)
            radial_outer_u[-1] = 1.0
            radial_outer_radius = config.active_radius_mm * np.sqrt(
                radial_outer_u
            )
            phi_count = _ragged_ring_phi_counts(
                radial_outer_radius,
                target_arc_mm=25.0,
                min_phi=5,
                phi_multiple=6,
            )
            np.testing.assert_array_equal(
                element_id, np.arange(phi_count.sum())
            )
            offset = np.concatenate([[0], np.cumsum(phi_count)])
            for ring in range(40):
                values = slice(offset[ring], offset[ring + 1])
                np.testing.assert_allclose(
                    np.linalg.norm(centers[values, :2], axis=1),
                    radius[ring],
                    rtol=0.0,
                    atol=2.0e-12,
                )
                self.assertAlmostEqual(
                    float(area[values].sum()),
                    float(ring_area[ring]),
                    places=9,
                )


if __name__ == "__main__":
    unittest.main()
