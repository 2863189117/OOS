#!/usr/bin/env python3
"""Numerical parity checks for the optional LXe CUDA generator backend."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import qmc


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from generator.reference.lxe_cuda import CudaModalProjector, cuda_available
from generator.reference.lxe_cuda import propagate_scalar_collision_orders_cuda
from generator.build_lxe_function_block import diffuse_escape_quadrature
from generator.reference.lxe_cylinder_green import (
    LXeCylinderGreen,
    LXeEntryRay,
    ModeTruncation,
)
from generator.reference.lxe_diffusion_return import (
    LXeDiffusionConfig,
    configured_top_extrapolation_length_mm,
)
from generator.reference.lxe_p1_vector_tail import (
    p1_escape_angular_channels,
    surface_current_fourier_coefficients_from_sources,
)
from generator.reference.lxe_response_operator import (
    _explicit_exit_angular_fourier_coefficients,
    _explicit_exit_fourier_coefficients,
    build_lxe_response_operator,
    surface_radial_quadrature,
)
from generator.reference.lxe_scalar_collision import (
    ScalarCollisionConfig,
    propagate_scalar_collision_orders,
)
from generator.reference.phase_space_grid import LXePhaseSpaceGrid


class LXeCollisionSnapshotTest(unittest.TestCase):
    def test_joint_explicit_projection_is_conservative_and_covariant(self) -> None:
        surface_radius, surface_area = surface_radial_quadrature(100.0, 5)
        direction_phi_bins = 4
        angular_direction = np.asarray(
            [
                [
                    math.sqrt(1.0 - mu * mu) * math.cos(phi),
                    math.sqrt(1.0 - mu * mu) * math.sin(phi),
                    mu,
                ]
                for mu in (0.35, 0.75)
                for phi in (
                    2.0
                    * math.pi
                    * (np.arange(direction_phi_bins, dtype=float) + 0.5)
                    / direction_phi_bins
                )
            ]
        )
        position = np.asarray([[13.0, -7.0], [44.0, 31.0], [-71.0, 12.0]])
        direction = np.asarray(
            [
                [0.40, 0.30, math.sqrt(0.75)],
                [-0.20, 0.60, math.sqrt(0.60)],
                [0.50, -0.60, math.sqrt(0.39)],
            ]
        )
        weight = np.asarray([0.11, 0.07, 0.03])
        maximum_order = 3
        joint = _explicit_exit_angular_fourier_coefficients(
            position,
            direction,
            weight,
            surface_radius,
            surface_area,
            angular_direction,
            direction_phi_bins=direction_phi_bins,
            maximum_order=maximum_order,
        )
        legacy = _explicit_exit_fourier_coefficients(
            position,
            weight,
            surface_radius,
            surface_area,
            maximum_order,
        )
        np.testing.assert_allclose(
            joint.sum(axis=2), legacy, rtol=2.0e-14, atol=2.0e-16
        )
        self.assertAlmostEqual(
            float(np.sum(joint[0].real * surface_area[:, np.newaxis])),
            float(weight.sum()),
            places=14,
        )

        rotation_angle = 0.37
        rotation = np.asarray(
            [
                [math.cos(rotation_angle), -math.sin(rotation_angle)],
                [math.sin(rotation_angle), math.cos(rotation_angle)],
            ]
        )
        rotated_position = position @ rotation.T
        rotated_direction = direction.copy()
        rotated_direction[:, :2] = direction[:, :2] @ rotation.T
        rotated = _explicit_exit_angular_fourier_coefficients(
            rotated_position,
            rotated_direction,
            weight,
            surface_radius,
            surface_area,
            angular_direction,
            direction_phi_bins=direction_phi_bins,
            maximum_order=maximum_order,
        )
        covariance = np.exp(
            -1j * np.arange(maximum_order + 1) * rotation_angle
        )
        np.testing.assert_allclose(
            rotated,
            joint * covariance[:, np.newaxis, np.newaxis],
            rtol=3.0e-14,
            atol=3.0e-16,
        )

    def test_named_top_boundary_override_is_explicit(self) -> None:
        default = LXeDiffusionConfig()
        overridden = LXeDiffusionConfig(
            top_boundary_model="milne_sn_rayleigh",
            top_boundary_length_mm=1089.87,
        )
        self.assertNotAlmostEqual(
            configured_top_extrapolation_length_mm(default), 1089.87, places=2
        )
        self.assertEqual(
            configured_top_extrapolation_length_mm(overridden), 1089.87
        )
        audit = LXeCylinderGreen(overridden).top_boundary_audit()
        self.assertEqual(audit["boundary_model"], "milne_sn_rayleigh")
        self.assertEqual(audit["extrapolation_length_mm"], 1089.87)
        with self.assertRaisesRegex(ValueError, "requires a non-default"):
            LXeDiffusionConfig(top_boundary_length_mm=1089.87).validate()

    def test_milne_egress_basis_matches_vector_p1_normal_channel(self) -> None:
        liquid = LXeDiffusionConfig(
            top_boundary_model="milne_sn_rayleigh",
            top_boundary_length_mm=1089.8684625,
        )
        direction, weight, _ = diffuse_escape_quadrature(
            liquid, liquid_mu_order=4, direction_phi=8
        )
        channels = p1_escape_angular_channels(
            liquid, liquid_mu_order=4, direction_phi_bins=8
        )
        np.testing.assert_allclose(
            direction, channels.gas_direction, rtol=0.0, atol=2.0e-13
        )
        np.testing.assert_allclose(weight, channels.normal, rtol=2.0e-13, atol=2.0e-15)

    def test_handoff_preserves_post_collision_directions(self) -> None:
        liquid = LXeDiffusionConfig(
            radius_mm=100.0,
            depth_mm=200.0,
            rayleigh_length_mm=34.0,
            absorption_length_mm=7000.0,
            side_reflectivity=0.9,
            bottom_reflectivity=0.0,
        )
        controls = ScalarCollisionConfig(sample_power=6, maximum_events=16)
        cubature = qmc.Sobol(
            d=1 + 3 * controls.maximum_events, scramble=False
        ).random_base2(controls.sample_power)
        direction = np.asarray([0.35, -0.2, 0.0])
        direction[2] = math.sqrt(1.0 - float(np.dot(direction, direction)))
        snapshot = propagate_scalar_collision_orders(
            [
                LXeEntryRay(
                    entry_xy_mm=np.asarray([20.0, -15.0]),
                    liquid_direction=direction,
                    weight=1.0,
                )
            ],
            liquid,
            collision_orders=(2,),
            controls=controls,
            cubature=cubature,
        ).snapshots[2]

        self.assertEqual(snapshot.source_xyz_mm.shape, snapshot.source_direction.shape)
        self.assertEqual(len(snapshot.source_direction), len(snapshot.source_weight))
        np.testing.assert_allclose(
            np.linalg.norm(snapshot.source_direction, axis=1),
            1.0,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        self.assertAlmostEqual(snapshot.audit["pre_switch_accounted"], 1.0, places=12)

    def test_joint_response_operator_preserves_angle_summed_return(self) -> None:
        liquid = LXeDiffusionConfig(
            radius_mm=100.0,
            depth_mm=200.0,
            rayleigh_length_mm=34.0,
            absorption_length_mm=7000.0,
            side_reflectivity=0.9,
            bottom_reflectivity=0.0,
        )
        angular_phi_bins = 2
        angular_direction = np.asarray(
            [
                [
                    math.sqrt(1.0 - mu * mu) * math.cos(phi),
                    math.sqrt(1.0 - mu * mu) * math.sin(phi),
                    mu,
                ]
                for mu in (0.4, 0.8)
                for phi in math.pi * (np.arange(angular_phi_bins) + 0.5)
            ]
        )
        angular_weight = np.full(len(angular_direction), 0.25)
        grid = LXePhaseSpaceGrid(
            radius_mm=liquid.radius_mm,
            position_radial_bins=1,
            position_phi_bins=4,
            direction_mu_bins=1,
            direction_phi_bins=1,
            direction_phi_relative_to_position=True,
            deposition="nearest",
            radial_node_spacing="chebyshev",
            direction_mu_minimum=0.804,
        )
        operator = build_lxe_response_operator(
            LXeCylinderGreen(
                liquid,
                ModeTruncation(
                    azimuthal_maximum=2,
                    radial_modes=6,
                    samples_per_expected_root=24,
                ),
            ),
            grid,
            surface_radial_order=3,
            first_scatter_order=2,
            explicit_collision_order=2,
            collision_sample_power=5,
            collision_maximum_events=8,
            processes=1,
            angular_direction=angular_direction,
            angular_weight=angular_weight,
            angular_direction_phi_bins=angular_phi_bins,
        )

        self.assertEqual(operator.coefficients.shape, (1, 1, 1, 3, 3, 4))
        integrated = np.einsum(
            "...sa,s->...",
            operator.coefficients[..., 0, :, :].real,
            operator.surface_ring_area_mm2,
        )
        np.testing.assert_allclose(
            integrated, operator.expected_return, rtol=3.0e-12, atol=3.0e-12
        )
        phase = np.zeros((1, grid.size), dtype=float)
        phase[0, 0] = 1.0
        surface, audit = operator.apply(phase, surface_phi_bins=8)
        self.assertEqual(surface.shape, (1, 3 * 8 * 4))
        self.assertAlmostEqual(
            float(surface.sum()), float(audit["surface_return_expected"][0]), places=12
        )

    def test_full_vector_p1_builder_is_raw_audited_and_angle_conservative(self) -> None:
        liquid = LXeDiffusionConfig(
            radius_mm=100.0,
            depth_mm=200.0,
            rayleigh_length_mm=34.0,
            absorption_length_mm=7000.0,
            side_reflectivity=0.9,
            bottom_reflectivity=0.0,
        )
        channels = p1_escape_angular_channels(
            liquid, liquid_mu_order=2, direction_phi_bins=4
        )
        grid = LXePhaseSpaceGrid(
            radius_mm=liquid.radius_mm,
            position_radial_bins=1,
            position_phi_bins=4,
            direction_mu_bins=1,
            direction_phi_bins=1,
            direction_phi_relative_to_position=True,
            deposition="nearest",
            radial_node_spacing="chebyshev",
            direction_mu_minimum=0.804,
        )
        green = LXeCylinderGreen(
            liquid,
            ModeTruncation(
                azimuthal_maximum=2,
                radial_modes=6,
                samples_per_expected_root=24,
            ),
        )
        common = dict(
            surface_radial_order=4,
            first_scatter_order=2,
            explicit_collision_order=2,
            collision_sample_power=5,
            collision_maximum_events=8,
            processes=1,
            angular_direction=channels.gas_direction,
            angular_weight=channels.normal,
            angular_direction_phi_bins=4,
        )
        scalar = build_lxe_response_operator(green, grid, **common)
        vector = build_lxe_response_operator(
            green,
            grid,
            diffuse_tail_angular_model="full_vector_p1_raw",
            p1_vector_audit_phi_bins=9,
            **common,
        )
        np.testing.assert_allclose(
            vector.expected_return,
            scalar.expected_return,
            rtol=2.0e-13,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            vector.coefficients.sum(axis=-1),
            scalar.coefficients.sum(axis=-1),
            rtol=3.0e-11,
            atol=3.0e-14,
        )
        self.assertGreater(
            float(np.max(np.abs(vector.coefficients - scalar.coefficients))),
            1.0e-12,
        )
        self.assertEqual(
            vector.metadata["diffuse_tail_angular_model"],
            "full_vector_p1_raw",
        )
        self.assertTrue(vector.metadata["p1_vector_candidate_uses_raw"])
        self.assertEqual(
            vector.metadata["p1_vector_limiter_policy"],
            "audit_only_raw_coefficients_unchanged",
        )
        for name in (
            "p1_vector_raw_negative_tail_fraction",
            "p1_vector_limiter_l1_fraction",
            "p1_vector_limited_surface_node_count",
        ):
            self.assertIn(name, vector.audit_names)
            values = vector.audit_values[..., vector.audit_names.index(name)]
            self.assertTrue(np.all(np.isfinite(values)))


@unittest.skipUnless(cuda_available(), "a CuPy CUDA runtime is not available")
class LXeCudaParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.liquid = LXeDiffusionConfig(
            radius_mm=100.0,
            depth_mm=200.0,
            rayleigh_length_mm=34.0,
            absorption_length_mm=7000.0,
            side_reflectivity=0.9,
            bottom_reflectivity=0.0,
        )
        cls.controls = ScalarCollisionConfig(sample_power=6, maximum_events=16)
        cls.cubature = qmc.Sobol(
            d=1 + 3 * cls.controls.maximum_events, scramble=False
        ).random_base2(cls.controls.sample_power)
        direction = np.asarray([0.35, -0.2, 0.0])
        direction[2] = math.sqrt(1.0 - float(np.dot(direction, direction)))
        cls.entry = LXeEntryRay(
            entry_xy_mm=np.asarray([20.0, -15.0]),
            liquid_direction=direction,
            weight=1.0,
        )

    def test_collision_audits_match_cpu(self) -> None:
        cpu = propagate_scalar_collision_orders(
            [self.entry],
            self.liquid,
            collision_orders=(2,),
            controls=self.controls,
            cubature=self.cubature,
        ).snapshots[2]
        cuda = propagate_scalar_collision_orders_cuda(
            [self.entry],
            self.liquid,
            collision_orders=(2,),
            controls=self.controls,
            cubature=self.cubature,
        ).snapshots[2]
        self.assertEqual(cpu.source_xyz_mm.shape, cuda.source_xyz_mm.shape)
        self.assertEqual(cpu.source_direction.shape, cuda.source_direction.shape)
        self.assertEqual(
            cpu.explicit_exits.position_xy_mm.shape,
            cuda.explicit_exits.position_xy_mm.shape,
        )
        np.testing.assert_allclose(
            cpu.source_xyz_mm, cuda.source_xyz_mm, rtol=2.0e-13, atol=2.0e-11
        )
        np.testing.assert_allclose(
            cpu.source_direction,
            cuda.source_direction,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            np.linalg.norm(cpu.source_direction, axis=1),
            1.0,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            cpu.source_weight, cuda.source_weight, rtol=2.0e-13, atol=2.0e-15
        )
        for name, expected in cpu.audit.items():
            self.assertAlmostEqual(expected, cuda.audit[name], places=12)

    def test_modal_and_explicit_projection_match_cpu(self) -> None:
        green = LXeCylinderGreen(
            self.liquid,
            ModeTruncation(
                azimuthal_maximum=3,
                radial_modes=8,
                samples_per_expected_root=24,
            ),
        )
        surface_radius, surface_area = surface_radial_quadrature(
            self.liquid.radius_mm, 6
        )
        projector = CudaModalProjector(green, surface_radius, surface_area)
        snapshot = propagate_scalar_collision_orders(
            [self.entry],
            self.liquid,
            collision_orders=(2,),
            controls=self.controls,
            cubature=self.cubature,
        ).snapshots[2]
        expected_modes, expected_return = (
            green.surface_fourier_coefficients_from_sources(
                snapshot.source_xyz_mm,
                snapshot.source_weight,
                surface_radius,
            )
        )
        actual_modes, actual_return = projector.project_diffusion_sources(
            snapshot.source_xyz_mm, snapshot.source_weight
        )
        np.testing.assert_allclose(
            actual_modes, expected_modes, rtol=2.0e-11, atol=2.0e-14
        )
        self.assertAlmostEqual(actual_return, expected_return, places=12)

        expected_currents = surface_current_fourier_coefficients_from_sources(
            green,
            snapshot.source_xyz_mm,
            snapshot.source_weight,
            surface_radius,
        )
        actual_currents = projector.project_surface_currents(
            snapshot.source_xyz_mm, snapshot.source_weight
        )
        self.assertAlmostEqual(
            actual_currents.expected_return,
            expected_currents.expected_return,
            places=12,
        )
        for actual, expected in (
            (actual_currents.normal, expected_currents.normal),
            (actual_currents.radial, expected_currents.radial),
            (actual_currents.azimuthal, expected_currents.azimuthal),
        ):
            np.testing.assert_allclose(
                actual, expected, rtol=3.0e-11, atol=3.0e-14
            )

        expected_explicit = _explicit_exit_fourier_coefficients(
            snapshot.explicit_exits.position_xy_mm,
            snapshot.explicit_exits.weight,
            surface_radius,
            surface_area,
            green.truncation.azimuthal_maximum,
        )
        actual_explicit = projector.project_explicit_exits(
            snapshot.explicit_exits.position_xy_mm,
            snapshot.explicit_exits.weight,
        )
        np.testing.assert_allclose(
            actual_explicit, expected_explicit, rtol=2.0e-13, atol=2.0e-15
        )

    def test_complete_operator_matches_cpu(self) -> None:
        grid = LXePhaseSpaceGrid(
            radius_mm=self.liquid.radius_mm,
            position_radial_bins=2,
            position_phi_bins=4,
            direction_mu_bins=2,
            direction_phi_bins=2,
            direction_phi_relative_to_position=True,
            deposition="multilinear",
            radial_node_spacing="chebyshev",
            direction_mu_minimum=0.804,
        )

        def build(backend: str):
            return build_lxe_response_operator(
                LXeCylinderGreen(
                    self.liquid,
                    ModeTruncation(
                        azimuthal_maximum=3,
                        radial_modes=8,
                        samples_per_expected_root=24,
                    ),
                ),
                grid,
                surface_radial_order=6,
                first_scatter_order=4,
                explicit_collision_order=2,
                collision_sample_power=6,
                collision_maximum_events=16,
                processes=1,
                compute_backend=backend,
            )

        cpu = build("cpu")
        cuda = build("cuda")
        nan_mismatch = np.isnan(cuda.coefficients) != np.isnan(cpu.coefficients)
        self.assertFalse(
            np.any(nan_mismatch),
            msg=(
                f"CUDA NaNs={np.isnan(cuda.coefficients).sum()}, "
                f"CPU NaNs={np.isnan(cpu.coefficients).sum()}, "
                f"first mismatch={np.argwhere(nan_mismatch)[:1].tolist()}"
            ),
        )
        np.testing.assert_allclose(
            cuda.coefficients, cpu.coefficients, rtol=2.0e-11, atol=2.0e-14
        )
        np.testing.assert_allclose(
            cuda.expected_return, cpu.expected_return, rtol=2.0e-13, atol=2.0e-14
        )
        np.testing.assert_allclose(
            cuda.audit_values, cpu.audit_values, rtol=2.0e-13, atol=2.0e-14
        )
        self.assertEqual(cuda.metadata["compute_backend"], "cuda")
        self.assertEqual(cpu.metadata["compute_backend"], "cpu")


if __name__ == "__main__":
    unittest.main()
