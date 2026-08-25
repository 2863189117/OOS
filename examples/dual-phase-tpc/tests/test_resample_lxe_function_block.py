#!/usr/bin/env python3
"""Tests for geometry-only resampling of a factorized LXe block."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from generator.reference.lxe_diffusion_return import (  # noqa: E402
    LXeDiffusionConfig,
)
from generator.resample_lxe_function_block import (  # noqa: E402
    cesaro_modal_weights,
    diffuse_escape_quadrature,
    normalize_modal_m0,
    positive_ring_contraction,
    resample_lxe_function_block,
)
from generator.write_intrinsic_lxe_block import (  # noqa: E402
    write_factorized_block,
)


def write_json(handle: h5py.File, path: str, value: object) -> None:
    handle.create_dataset(
        path,
        data=np.frombuffer(
            json.dumps(value, sort_keys=True).encode(), dtype=np.uint8
        ),
    )


def analytic_geometry(
    path: Path,
    *,
    radius_mm: float,
    phi_bins: int | tuple[int, int],
) -> None:
    counts = (
        (phi_bins, phi_bins)
        if isinstance(phi_bins, int)
        else tuple(phi_bins)
    )
    surface_radii = np.asarray([0.5, 0.75])
    ring_area = np.asarray([1.0, 2.0])
    centers = np.asarray(
        [
            [radius * math.cos(angle), radius * math.sin(angle), 0.0]
            for radius, count in zip(surface_radii, counts)
            for angle in (
                2.0 * math.pi * (np.arange(count) + 0.5) / count
            )
        ]
    )
    areas = np.concatenate(
        [
            np.full(count, area / count, dtype=float)
            for area, count in zip(ring_area, counts)
        ]
    )
    element_count = len(centers)
    with h5py.File(path, "w") as handle:
        write_json(
            handle,
            "/metadata/generator_json",
            {
                "config": {
                    "active_radius_mm": radius_mm,
                    "lxe_depth_mm": 2.0,
                }
            },
        )
        write_json(
            handle,
            "/metadata/analytic_generator_json",
            {
                "analytic_discretization": {
                    "lxe_surface_target_arc_mm": None,
                    "lxe_surface_min_phi": min(counts),
                    "lxe_surface_phi_multiple": 1,
                }
            },
        )
        analytic = handle.create_group("analytic")
        analytic.create_dataset("kind", data=np.asarray([1], dtype=np.uint32))
        analytic.create_dataset("center_mm", data=[[0.0, 0.0, 0.0]])
        analytic.create_dataset("axis_x", data=[[1.0, 0.0, 0.0]])
        analytic.create_dataset("axis_y", data=[[0.0, 1.0, 0.0]])
        analytic.create_dataset("axis_z", data=[[0.0, 0.0, 1.0]])
        analytic.create_dataset(
            "parameters", data=[[radius_mm, 0.0, 0.0, 0.0]]
        )
        analytic.create_dataset("normal_sign", data=[1.0])
        analytic.create_dataset("surface_id", data=[1])
        analytic.create_dataset("minus_domain_id", data=[1])
        analytic.create_dataset("plus_domain_id", data=[0])
        elements = analytic.create_group("elements")
        elements.create_dataset(
            "primitive_index", data=np.zeros(element_count, dtype=np.uint32)
        )
        elements.create_dataset("coordinates", data=centers)
        elements.create_dataset(
            "bounds", data=np.zeros((element_count, 4), dtype=float)
        )
        elements.create_dataset("center_mm", data=centers)
        elements.create_dataset("area_mm2", data=areas)
        elements.create_dataset(
            "normal", data=np.tile([0.0, 0.0, 1.0], (element_count, 1))
        )
        elements.create_dataset(
            "surface_element", data=np.arange(100, 100 + element_count)
        )


def source_block(
    path: Path, *, positive_density: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = LXeDiffusionConfig(radius_mm=1.0, depth_mm=2.0)
    _, angular_weight, _ = diffuse_escape_quadrature(
        config, liquid_mu_order=2, direction_phi=2
    )
    expected_return = np.asarray([[[0.1]], [[0.2]]])
    if positive_density:
        coefficients = np.zeros((2, 1, 1, 2, 2), dtype=np.complex128)
        coefficients[0, 0, 0, 0] = [0.02, 0.04]
        coefficients[0, 0, 0, 1] = [0.004, 0.008]
        coefficients[1, 0, 0, 0] = [0.04, 0.08]
        coefficients[1, 0, 0, 1] = [0.008, 0.016]
    else:
        coefficients = (
            np.arange(8, dtype=float).reshape(2, 1, 1, 2, 2)
            + 1j * np.arange(8, 16, dtype=float).reshape(2, 1, 1, 2, 2)
        ) / 100.0
    audit_values = np.asarray([[[[0.9]]], [[[0.8]]]])
    source_phi_bins = 2
    egress_count = source_phi_bins * angular_weight.size * 2
    write_factorized_block(
        path,
        coefficients=coefficients,
        expected_return=expected_return,
        audit_values=audit_values,
        surface_radius_mm=np.asarray([0.5, 0.75]),
        surface_ring_area_mm2=np.asarray([1.0, 2.0]),
        angular_weight=angular_weight,
        surface_element=np.arange(egress_count),
        barycentric=np.tile([1.0, 0.0, 0.0], (egress_count, 1)),
        side=np.ones(egress_count, dtype=np.uint64),
        direction_local=np.tile([0.0, 0.0, 1.0], (egress_count, 1)),
        stokes=np.tile([1.0, 0.0, 0.0, 0.0], (egress_count, 1)),
        reference_axis_local=np.tile([1.0, 0.0, 0.0], (egress_count, 1)),
        phase_grid={
            "position_radial_bins": 1,
            "position_phi_bins": 3,
            "direction_mu_bins": 1,
            "direction_phi_bins": 2,
        },
        audit_names=["lxe_nonreturn"],
        metadata={
            "lxe_config": asdict(config),
            "return_mu_order": 2,
            "return_direction_phi": 2,
            "generator_sha256": "modal-generator",
        },
    )
    return coefficients, expected_return, audit_values


def joint_source_block(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = LXeDiffusionConfig(radius_mm=1.0, depth_mm=2.0)
    direction, angular_weight, angular_stokes = diffuse_escape_quadrature(
        config, liquid_mu_order=2, direction_phi=2
    )
    expected_return = np.asarray([[[0.1]], [[0.2]]])
    coefficients = np.zeros(
        (2, 1, 1, 2, 2, angular_weight.size), dtype=np.complex128
    )
    for direction_phi, expected in enumerate(expected_return[:, 0, 0]):
        m0 = expected * angular_weight / 3.0
        coefficients[direction_phi, 0, 0, 0, 0] = m0
        coefficients[direction_phi, 0, 0, 0, 1] = m0
        coefficients[direction_phi, 0, 0, 1, 0] = 0.05 * m0 * (1.0 + 0.2j)
        coefficients[direction_phi, 0, 0, 1, 1] = 0.03 * m0 * (1.0 - 0.1j)
    audit_values = np.asarray([[[[0.9]]], [[[0.8]]]])
    source_phi_bins = 2
    egress_count = source_phi_bins * angular_weight.size * 2
    write_factorized_block(
        path,
        coefficients=coefficients,
        expected_return=expected_return,
        audit_values=audit_values,
        surface_radius_mm=np.asarray([0.5, 0.75]),
        surface_ring_area_mm2=np.asarray([1.0, 2.0]),
        angular_weight=angular_weight,
        surface_element=np.arange(egress_count),
        barycentric=np.tile([1.0, 0.0, 0.0], (egress_count, 1)),
        side=np.ones(egress_count, dtype=np.uint64),
        direction_local=np.tile([0.0, 0.0, 1.0], (egress_count, 1)),
        stokes=np.tile(angular_stokes, (2 * source_phi_bins, 1)),
        reference_axis_local=np.tile([1.0, 0.0, 0.0], (egress_count, 1)),
        phase_grid={
            "position_radial_bins": 1,
            "position_phi_bins": 3,
            "direction_mu_bins": 1,
            "direction_phi_bins": 2,
        },
        audit_names=["lxe_nonreturn"],
        metadata={
            "lxe_config": asdict(config),
            "return_mu_order": 2,
            "return_direction_phi": 2,
            "generator_sha256": "joint-modal-generator",
        },
    )
    return coefficients, expected_return, direction


class ResampleLXeFunctionBlockTest(unittest.TestCase):
    def test_resampling_preserves_modal_payload_and_refines_egress(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "source.h5"
            geometry = directory / "geometry.h5"
            output = directory / "resampled.h5"
            coefficients, expected, audit = source_block(source)
            analytic_geometry(geometry, radius_mm=1.0, phi_bins=4)

            summary = resample_lxe_function_block(
                source, geometry, output, surface_phi_bins=4
            )

            self.assertEqual(summary["source_surface_phi_bins"], 2)
            self.assertEqual(summary["surface_phi_bins"], 4)
            with h5py.File(output, "r") as handle:
                actual_coefficients = np.asarray(
                    handle["/function/coefficients_real"]
                ) + 1j * np.asarray(handle["/function/coefficients_imag"])
                np.testing.assert_array_equal(actual_coefficients, coefficients)
                np.testing.assert_array_equal(
                    handle["/function/expected_return"], expected
                )
                np.testing.assert_array_equal(
                    handle["/function/audit_values"], audit
                )
                metadata = json.loads(
                    bytes(
                        np.asarray(
                            handle["/metadata/generator_json"], dtype=np.uint8
                        )
                    ).decode()
                )
                angular_count = int(
                    handle["/function/angular_weight"].shape[0]
                )
                elements = np.asarray(
                    handle["/nonlocal/egress/surface_element"]
                )
                output_direction = np.asarray(
                    handle["/nonlocal/egress/direction_local"]
                )
            self.assertEqual(metadata["surface_phi_bins"], 4)
            self.assertEqual(metadata["resampled_from_surface_phi_bins"], 2)
            self.assertEqual(metadata["generator_sha256"], "modal-generator")
            self.assertEqual(metadata["geometry_contract"]["radius_mm"], 1.0)
            self.assertEqual(len(elements), 2 * 4 * angular_count)
            np.testing.assert_array_equal(
                elements,
                np.repeat(np.arange(100, 108), angular_count),
            )
            for point in range(1, 8):
                np.testing.assert_allclose(
                    output_direction[point * angular_count],
                    output_direction[0],
                    rtol=0.0,
                    atol=2.0e-15,
                )

    def test_rejects_non_refinement_and_incompatible_lxe_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "source.h5"
            geometry = directory / "geometry.h5"
            source_block(source)
            analytic_geometry(geometry, radius_mm=1.0, phi_bins=4)
            with self.assertRaisesRegex(ValueError, "greater than the input"):
                resample_lxe_function_block(
                    source,
                    geometry,
                    directory / "same.h5",
                    surface_phi_bins=2,
                )

            incompatible = directory / "incompatible.h5"
            analytic_geometry(incompatible, radius_mm=1.1, phi_bins=4)
            with self.assertRaisesRegex(ValueError, "radius_mm"):
                resample_lxe_function_block(
                    source,
                    incompatible,
                    directory / "wrong-radius.h5",
                    surface_phi_bins=4,
                )

    def test_ragged_v2_uses_geometry_points_areas_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "source.h5"
            geometry = directory / "geometry.h5"
            output = directory / "ragged.h5"
            source_coefficients, expected, _ = source_block(
                source, positive_density=True
            )
            analytic_geometry(
                geometry, radius_mm=1.0, phi_bins=(4, 6)
            )

            summary = resample_lxe_function_block(
                source,
                geometry,
                output,
                ragged_rings=True,
                positivity_phi_bins=32,
            )

            normalized, _ = normalize_modal_m0(
                source_coefficients,
                expected,
                np.asarray([1.0, 2.0]),
            )
            with h5py.File(output, "r") as handle:
                metadata = json.loads(
                    bytes(
                        np.asarray(
                            handle["/metadata/generator_json"],
                            dtype=np.uint8,
                        )
                    ).decode()
                )
                offsets = np.asarray(
                    handle["/function/surface_ring_offsets"]
                )
                phi = np.asarray(handle["/function/surface_phi_rad"])
                area = np.asarray(handle["/function/surface_area_mm2"])
                elements = np.asarray(
                    handle["/nonlocal/egress/surface_element"]
                )
                angular_count = len(handle["/function/angular_weight"])
                actual_coefficients = np.asarray(
                    handle["/function/coefficients_real"]
                ) + 1j * np.asarray(
                    handle["/function/coefficients_imag"]
                )

            self.assertEqual(
                summary["surface_layout"]["kind"], "ragged_ring_v1"
            )
            self.assertEqual(metadata["schema"], "oos.nonlocal.function.v2")
            self.assertEqual(metadata["surface_layout"], "ragged_ring_v1")
            self.assertEqual(metadata["surface_point_count"], 10)
            self.assertTrue(
                metadata["ragged_surface_quadrature"][
                    "geometry_is_unique_egress_source"
                ]
            )
            self.assertEqual(
                metadata["modal_filter"]["kind"], "none"
            )
            self.assertEqual(
                metadata["modal_filter"]["weights"], [1.0, 1.0]
            )
            self.assertIsNone(metadata["positivity_correction"])
            self.assertIn("dense_phi_positivity_audit", metadata)
            self.assertTrue(metadata["nonnegative_density_on_audit_grid"])
            np.testing.assert_array_equal(offsets, [0, 4, 10])
            np.testing.assert_allclose(
                phi[:4], 2.0 * math.pi * (np.arange(4) + 0.5) / 4
            )
            np.testing.assert_allclose(
                phi[4:], 2.0 * math.pi * (np.arange(6) + 0.5) / 6
            )
            np.testing.assert_allclose(area[:4], 0.25)
            np.testing.assert_allclose(area[4:], 1.0 / 3.0)
            np.testing.assert_array_equal(
                elements,
                np.repeat(np.arange(100, 110), angular_count),
            )
            np.testing.assert_allclose(actual_coefficients, normalized)
            m0_integral = np.einsum(
                "...s,s->...",
                actual_coefficients[..., 0, :].real,
                np.asarray([1.0, 2.0]),
            )
            np.testing.assert_allclose(
                m0_integral, expected, rtol=0.0, atol=2.0e-15
            )
            with self.assertRaisesRegex(
                ValueError, "re-filtering an existing v2 block"
            ):
                resample_lxe_function_block(
                    output,
                    geometry,
                    directory / "ragged-twice.h5",
                    ragged_rings=True,
                    positivity_phi_bins=32,
                )

    def test_rank_six_ragged_resampling_preserves_angles_and_rotates_delta(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "joint-source.h5"
            geometry = directory / "geometry.h5"
            output = directory / "joint-ragged.h5"
            coefficients, expected, direction = joint_source_block(source)
            analytic_geometry(geometry, radius_mm=1.0, phi_bins=(4, 6))

            resample_lxe_function_block(
                source,
                geometry,
                output,
                ragged_rings=True,
                positivity_phi_bins=32,
            )

            with h5py.File(output, "r") as handle:
                actual = np.asarray(
                    handle["/function/coefficients_real"]
                ) + 1j * np.asarray(handle["/function/coefficients_imag"])
                phi = np.asarray(handle["/function/surface_phi_rad"])
                output_direction = np.asarray(
                    handle["/nonlocal/egress/direction_local"]
                )
                metadata = json.loads(
                    bytes(
                        np.asarray(
                            handle["/metadata/generator_json"], dtype=np.uint8
                        )
                    ).decode()
                )

            np.testing.assert_array_equal(actual, coefficients)
            m0 = np.einsum(
                "...sa,s->...", actual[..., 0, :, :].real, [1.0, 2.0]
            )
            np.testing.assert_allclose(m0, expected, rtol=0.0, atol=2.0e-15)
            self.assertEqual(
                metadata["strict_m0_normalization"]["method"],
                "angle_summed_audit_no_coefficient_change",
            )
            self.assertEqual(
                metadata["coefficient_layout"], "joint_surface_angle_v1"
            )
            angular_count = len(direction)
            for point, point_phi in enumerate(phi):
                cosine = math.cos(float(point_phi))
                sine = math.sin(float(point_phi))
                reference = direction[0]
                expected_direction = np.asarray(
                    [
                        cosine * reference[0] - sine * reference[1],
                        sine * reference[0] + cosine * reference[1],
                        reference[2],
                    ]
                )
                np.testing.assert_allclose(
                    output_direction[point * angular_count],
                    expected_direction,
                    rtol=0.0,
                    atol=2.0e-15,
                )

    def test_rank_six_uniform_resampling_rotates_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "joint-source.h5"
            geometry = directory / "geometry.h5"
            output = directory / "joint-uniform.h5"
            coefficients, _, direction = joint_source_block(source)
            analytic_geometry(geometry, radius_mm=1.0, phi_bins=4)

            resample_lxe_function_block(
                source, geometry, output, surface_phi_bins=4
            )

            with h5py.File(output, "r") as handle:
                actual = np.asarray(
                    handle["/function/coefficients_real"]
                ) + 1j * np.asarray(handle["/function/coefficients_imag"])
                output_direction = np.asarray(
                    handle["/nonlocal/egress/direction_local"]
                )
            np.testing.assert_array_equal(actual, coefficients)
            angular_count = len(direction)
            surface_phi = 2.0 * math.pi * (np.arange(4) + 0.5) / 4
            for point_phi_index, point_phi in enumerate(surface_phi):
                cosine = math.cos(float(point_phi))
                sine = math.sin(float(point_phi))
                reference = direction[0]
                expected_direction = np.asarray(
                    [
                        cosine * reference[0] - sine * reference[1],
                        sine * reference[0] + cosine * reference[1],
                        reference[2],
                    ]
                )
                np.testing.assert_allclose(
                    output_direction[point_phi_index * angular_count],
                    expected_direction,
                    rtol=0.0,
                    atol=2.0e-15,
                )

    def test_positive_ring_contraction_preserves_m0_and_removes_undershoot(
        self,
    ) -> None:
        coefficients = np.zeros((1, 1, 1, 3, 1), dtype=np.complex128)
        coefficients[..., 0, 0] = 1.0
        coefficients[..., 2, 0] = 2.0
        expected = np.ones((1, 1, 1), dtype=float)

        projected, audit = positive_ring_contraction(
            coefficients,
            expected,
            np.ones(1),
            phi_bins=64,
            tolerance=1.0e-13,
        )

        phi = 2.0 * math.pi * np.arange(4096) / 4096
        density = sum(
            projected[0, 0, 0, order, 0] * np.exp(1j * order * phi)
            for order in range(3)
        ).real
        self.assertGreaterEqual(float(density.min()), -2.0e-15)
        self.assertAlmostEqual(projected[0, 0, 0, 0, 0].real, 1.0)
        # Fejer would reduce m=2 from 2 to 2/3.  The ringwise contraction
        # retains more than 0.9 even with the between-grid derivative margin.
        self.assertGreater(projected[0, 0, 0, 2, 0].real, 0.9)
        self.assertEqual(audit["contracted_row_ring_count"], 1)
        self.assertTrue(audit["m0_preserved_bitwise"])
        self.assertAlmostEqual(
            np.einsum(
                "...s,s->...", projected[..., 0, :].real, np.ones(1)
            ).item(),
            1.0,
        )

    def test_ragged_rejects_phi_aliasing_and_cesaro_weights_are_exact(self) -> None:
        caratheodory = cesaro_modal_weights(33, kind="caratheodory")
        self.assertEqual(caratheodory[0], 1.0)
        self.assertAlmostEqual(caratheodory[1], 0.9957341762950345)
        self.assertTrue(np.all(np.diff(caratheodory) < 0.0))
        phi = 2.0 * math.pi * np.arange(65536) / 65536
        kernel = np.ones_like(phi)
        for order, weight in enumerate(caratheodory[1:], start=1):
            kernel += 2.0 * weight * np.cos(order * phi)
        self.assertGreaterEqual(float(kernel.min()), -2.0e-14)
        np.testing.assert_allclose(
            cesaro_modal_weights(4, kind="fejer"),
            [1.0, 0.75, 0.5, 0.25],
        )
        np.testing.assert_allclose(
            cesaro_modal_weights(4, kind="cesaro", cesaro_order=2.0),
            [1.0, 0.6, 0.3, 0.1],
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "source.h5"
            geometry = directory / "geometry.h5"
            source_block(source)
            analytic_geometry(
                geometry, radius_mm=1.0, phi_bins=(2, 4)
            )
            with self.assertRaisesRegex(ValueError, r"2\*M\+1=3"):
                resample_lxe_function_block(
                    source,
                    geometry,
                    directory / "aliased.h5",
                    ragged_rings=True,
                    positivity_phi_bins=32,
                )

    def test_ragged_rejects_noncanonical_analytic_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "source.h5"
            geometry = directory / "geometry.h5"
            source_block(source, positive_density=True)
            analytic_geometry(geometry, radius_mm=1.0, phi_bins=(4, 6))
            with h5py.File(geometry, "r+") as handle:
                handle["/analytic/center_mm"][0] = [1.0, 0.0, 0.0]
            with self.assertRaisesRegex(ValueError, "origin-centered"):
                resample_lxe_function_block(
                    source,
                    geometry,
                    directory / "translated.h5",
                    ragged_rings=True,
                    positivity_phi_bins=32,
                )


if __name__ == "__main__":
    unittest.main()
