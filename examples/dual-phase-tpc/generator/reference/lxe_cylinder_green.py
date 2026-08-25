#!/usr/bin/env python3
"""Position-dependent diffusion Green function for a finite LXe cylinder.

This module is the numerical counterpart of ``lxe_cylinder_green.wl``.  It
solves the diffusion equation by a Bessel--Fourier expansion and retains the
actual first-scatter position ``(x0, y0, z0)``.  It therefore does not reuse a
centre-entry return kernel for off-axis photons.

Coordinates are in millimetres.  The liquid surface is ``z=0`` and positive
``z`` points down into LXe.  ``surface_return_density`` is the escaping
probability current per square millimetre at the liquid surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing
from typing import Iterable

import numpy as np
from scipy import optimize, special

from .lxe_diffusion_return import (
    LXeDiffusionConfig,
    configured_top_extrapolation_length_mm,
    diffuse_fresnel_reflection_moments,
    extrapolation_length_mm,
)


@dataclass(frozen=True)
class ModeTruncation:
    """Bessel--Fourier truncation of the finite-cylinder Green function."""

    azimuthal_maximum: int = 12
    radial_modes: int = 80
    samples_per_expected_root: int = 120

    def validate(self) -> None:
        if self.azimuthal_maximum < 0:
            raise ValueError("azimuthal_maximum must be non-negative")
        if self.radial_modes <= 0:
            raise ValueError("radial_modes must be positive")
        if self.samples_per_expected_root < 20:
            raise ValueError("samples_per_expected_root must be at least 20")


@dataclass(frozen=True)
class LXeEntryRay:
    """One weighted ray immediately after refraction into LXe.

    ``liquid_direction`` uses diffusion coordinates, with positive z pointing
    down from the liquid surface.
    """

    entry_xy_mm: np.ndarray
    liquid_direction: np.ndarray
    weight: float

    def validate(self) -> None:
        entry = np.asarray(self.entry_xy_mm, dtype=float)
        direction = np.asarray(self.liquid_direction, dtype=float)
        if entry.shape != (2,) or direction.shape != (3,):
            raise ValueError("LXe entry position/direction has invalid shape")
        if (
            not math.isfinite(self.weight)
            or self.weight < 0.0
            or not np.all(np.isfinite(entry))
            or not np.all(np.isfinite(direction))
        ):
            raise ValueError("LXe entry ray must be finite and non-negative")
        if not math.isclose(
            float(np.linalg.norm(direction)),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise ValueError("liquid_direction must be a unit vector")
        if direction[2] <= 0.0:
            raise ValueError("liquid_direction must point into LXe")


def robin_eigen_equation(
    order: int, alpha: np.ndarray | float, coefficient: float
) -> np.ndarray | float:
    """Return ``alpha J_m'(alpha) + coefficient J_m(alpha)``."""

    return np.asarray(alpha) * special.jvp(order, alpha) + coefficient * special.jv(
        order, alpha
    )


def robin_roots(
    order: int,
    count: int,
    coefficient: float,
    *,
    samples_per_expected_root: int = 120,
) -> np.ndarray:
    """Find the first non-negative radial Robin eigenvalues."""

    if order < 0 or count <= 0:
        raise ValueError("order must be non-negative and count positive")
    include_zero = order == 0 and abs(coefficient) < 1.0e-14
    positive_count = count - int(include_zero)
    if positive_count <= 0:
        return np.asarray([0.0])

    maximum = math.pi * (positive_count + 0.5 * order + 4.0)
    roots = np.empty(0, dtype=float)

    def equation(alpha: np.ndarray | float) -> np.ndarray | float:
        return robin_eigen_equation(order, alpha, coefficient)

    for _ in range(8):
        sample_count = max(
            2000,
            samples_per_expected_root * (positive_count + order + 6),
        )
        grid = np.linspace(1.0e-9, maximum, sample_count + 1)
        values = equation(grid)
        brackets = np.flatnonzero(values[:-1] * values[1:] < 0.0)
        roots = np.asarray(
            [
                optimize.brentq(
                    equation,
                    float(grid[index]),
                    float(grid[index + 1]),
                )
                for index in brackets
            ],
            dtype=float,
        )
        if roots.size >= positive_count:
            break
        maximum *= 1.5
    if roots.size < positive_count:
        # The preceding vectorized scan should always bracket simple roots.
        # Keep this explicit rather than silently returning a short basis.
        raise RuntimeError(
            f"found only {roots.size} of {positive_count} positive "
            f"Robin roots for m={order}"
        )
    roots = roots[:positive_count]
    if include_zero:
        roots = np.concatenate([np.asarray([0.0]), roots])
    return roots


class LXeCylinderGreen:
    """Bessel--Fourier Green function for one physical LXe configuration."""

    def __init__(
        self,
        config: LXeDiffusionConfig,
        truncation: ModeTruncation = ModeTruncation(),
        *,
        mode_workers: int = 1,
        source_chunk_size: int = 4096,
    ) -> None:
        config.validate()
        truncation.validate()
        if mode_workers <= 0:
            raise ValueError("mode_workers must be positive")
        if source_chunk_size <= 0:
            raise ValueError("source_chunk_size must be positive")
        self.config = config
        self.truncation = truncation
        self.mode_workers = mode_workers
        self.source_chunk_size = source_chunk_size
        diffusion = config.transport_diffusion_mm
        top_first_moment, top_second_moment = (
            diffuse_fresnel_reflection_moments(
                config.n_lxe, config.n_gxe
            )
        )
        self.top_fresnel_first_moment = top_first_moment
        self.top_fresnel_second_moment = top_second_moment
        self.top_escape_fraction = 1.0 - top_first_moment
        self.top_reflectivity = top_first_moment
        self.top_extrapolation_length_mm = configured_top_extrapolation_length_mm(
            config
        )
        self.beta_top = diffusion / self.top_extrapolation_length_mm
        self.beta_side = self._boundary_coefficient(config.side_reflectivity)
        self.beta_bottom = self._boundary_coefficient(config.bottom_reflectivity)
        self.dimensionless_side_robin = self.beta_side * config.radius_mm / diffusion
        self._roots: dict[int, np.ndarray] = {}
        self._mode_norms: dict[int, np.ndarray] = {}

    def _boundary_coefficient(self, reflectivity: float) -> float:
        diffusion = self.config.transport_diffusion_mm
        length = extrapolation_length_mm(diffusion, reflectivity)
        if math.isinf(length):
            return 0.0
        return diffusion / length

    def top_boundary_audit(self) -> dict[str, float | str]:
        """Return the Fresnel moments used by the top Robin boundary."""

        return {
            "fresnel_first_moment": self.top_fresnel_first_moment,
            "fresnel_second_moment": self.top_fresnel_second_moment,
            "diffuse_escape_fraction": self.top_escape_fraction,
            "extrapolation_length_mm": self.top_extrapolation_length_mm,
            "robin_coefficient": self.beta_top,
            "boundary_model": self.config.top_boundary_model,
        }

    def _first_boundary_distance(
        self, entry_xy_mm: np.ndarray, direction: np.ndarray
    ) -> float:
        """Distance to the first cylindrical side or bottom intersection."""

        entry = np.asarray(entry_xy_mm, dtype=float)
        ray = np.asarray(direction, dtype=float)
        transverse2 = float(np.dot(ray[:2], ray[:2]))
        side_distance = math.inf
        if transverse2 > 1.0e-30:
            linear = 2.0 * float(np.dot(entry, ray[:2]))
            constant = float(np.dot(entry, entry)) - self.config.radius_mm**2
            discriminant = linear * linear - 4.0 * transverse2 * constant
            if discriminant < 0.0:
                raise RuntimeError("LXe ray misses the cylindrical boundary")
            side_distance = (-linear + math.sqrt(max(0.0, discriminant))) / (
                2.0 * transverse2
            )
        bottom_distance = self.config.depth_mm / float(ray[2])
        distance = min(side_distance, bottom_distance)
        if not math.isfinite(distance) or distance < 0.0:
            raise RuntimeError("invalid first LXe boundary distance")
        return distance

    def roots(self, order: int) -> np.ndarray:
        """Return cached dimensionless Robin roots for azimuthal order m."""

        if order not in self._roots:
            # Close over order because scipy.optimize.brentq expects f(x).
            coefficient = self.dimensionless_side_robin

            def equation(alpha: np.ndarray | float) -> np.ndarray | float:
                return robin_eigen_equation(order, alpha, coefficient)

            include_zero = order == 0 and abs(coefficient) < 1.0e-14
            positive_count = self.truncation.radial_modes - int(include_zero)
            maximum = math.pi * (positive_count + 0.5 * order + 4.0)
            roots = np.empty(0, dtype=float)
            for _ in range(8):
                sample_count = max(
                    2000,
                    self.truncation.samples_per_expected_root
                    * (positive_count + order + 6),
                )
                grid = np.linspace(1.0e-9, maximum, sample_count + 1)
                values = equation(grid)
                brackets = np.flatnonzero(values[:-1] * values[1:] < 0.0)
                roots = np.asarray(
                    [
                        optimize.brentq(
                            equation,
                            float(grid[index]),
                            float(grid[index + 1]),
                        )
                        for index in brackets[:positive_count]
                    ],
                    dtype=float,
                )
                if roots.size >= positive_count:
                    break
                maximum *= 1.5
            if roots.size < positive_count:
                raise RuntimeError(
                    f"found only {roots.size} of {positive_count} "
                    f"positive Robin roots for m={order}"
                )
            if include_zero:
                roots = np.concatenate([np.asarray([0.0]), roots])
            self._roots[order] = roots
        return self._roots[order]

    def radial_mode_norms(self, order: int) -> np.ndarray:
        """Return ``integral_0^R r J_m(alpha r/R)^2 dr``."""

        if order not in self._mode_norms:
            roots = self.roots(order)
            radius = self.config.radius_mm
            self._mode_norms[order] = (
                0.5
                * radius**2
                * (
                    special.jv(order, roots) ** 2
                    - special.jv(order - 1, roots) * special.jv(order + 1, roots)
                )
            )
        return self._mode_norms[order]

    def vertical_escape_factor(
        self, wave_number: np.ndarray, source_depth_mm: np.ndarray
    ) -> np.ndarray:
        """Return ``beta_top g_k(0,z0)`` without hyperbolic overflow."""

        config = self.config
        diffusion = config.transport_diffusion_mm
        gamma = np.sqrt(
            np.asarray(wave_number, dtype=float) ** 2
            + config.absorption_coefficient_per_mm / diffusion
        )
        depth = np.asarray(source_depth_mm, dtype=float)
        gamma, depth = np.broadcast_arrays(gamma, depth)
        a_top = self.beta_top / diffusion
        a_bottom = self.beta_bottom / diffusion
        numerator = (1.0 + a_bottom / gamma) * np.exp(-gamma * depth) + (
            1.0 - a_bottom / gamma
        ) * np.exp(-gamma * (2.0 * config.depth_mm - depth))
        denominator = (
            gamma
            + a_top * a_bottom / gamma
            + a_top
            + a_bottom
            + (-gamma - a_top * a_bottom / gamma + a_top + a_bottom)
            * np.exp(-2.0 * gamma * config.depth_mm)
        )
        return self.beta_top / diffusion * numerator / denominator

    def total_return_probability(
        self,
        source_radius_mm: np.ndarray | float,
        source_depth_mm: np.ndarray | float,
    ) -> np.ndarray:
        """Integrate the top-return current for arbitrary source points."""

        radius = self.config.radius_mm
        source_radius, source_depth = np.broadcast_arrays(
            np.asarray(source_radius_mm, dtype=float),
            np.asarray(source_depth_mm, dtype=float),
        )
        if (
            np.any(source_radius < 0.0)
            or np.any(source_radius > radius)
            or np.any(source_depth <= 0.0)
            or np.any(source_depth >= self.config.depth_mm)
        ):
            raise ValueError("source point lies outside the liquid")
        roots = self.roots(0)
        norms = self.radial_mode_norms(0)
        disk_integrals = np.empty_like(roots)
        zero = np.abs(roots) < 1.0e-14
        disk_integrals[zero] = 0.5 * radius**2
        disk_integrals[~zero] = radius**2 * special.jv(1, roots[~zero]) / roots[~zero]
        source_bessel = special.jv(0, source_radius[..., np.newaxis] * roots / radius)
        vertical = self.vertical_escape_factor(
            roots / radius, source_depth[..., np.newaxis]
        )
        return np.sum(
            source_bessel * vertical * disk_integrals / norms,
            axis=-1,
        )

    def surface_return_density(
        self,
        surface_xy_mm: np.ndarray,
        source_xyz_mm: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the position-dependent return density on ``z=0``."""

        source = np.asarray(source_xyz_mm, dtype=float)
        if source.shape != (3,):
            raise ValueError("source_xyz_mm must have shape (3,)")
        return self.surface_return_density_mixture(
            surface_xy_mm,
            source[np.newaxis, :],
            np.asarray([1.0]),
        )

    def surface_return_density_mixture(
        self,
        surface_xy_mm: np.ndarray,
        source_xyz_mm: np.ndarray,
        source_weights: np.ndarray,
    ) -> np.ndarray:
        """Evaluate many scatter sources after accumulating modal weights."""

        surface = np.asarray(surface_xy_mm, dtype=float)
        sources = np.asarray(source_xyz_mm, dtype=float)
        weights = np.asarray(source_weights, dtype=float)
        if (
            surface.ndim != 2
            or surface.shape[1] != 2
            or sources.ndim != 2
            or sources.shape[1] != 3
            or weights.shape != (sources.shape[0],)
        ):
            raise ValueError(
                "surface/source arrays must have shapes (T,2), (S,3), (S,)"
            )
        if np.any(weights < 0.0) or np.any(~np.isfinite(weights)):
            raise ValueError("source weights must be finite and non-negative")
        if len(sources) == 0 or float(weights.sum()) == 0.0:
            return np.zeros(len(surface), dtype=float)
        surface_radius = np.linalg.norm(surface, axis=-1)
        source_radius = np.linalg.norm(sources[:, :2], axis=1)
        if (
            np.any(surface_radius > self.config.radius_mm)
            or np.any(source_radius > self.config.radius_mm)
            or np.any(sources[:, 2] <= 0.0)
            or np.any(sources[:, 2] >= self.config.depth_mm)
        ):
            raise ValueError("surface or source lies outside the cylinder")
        surface_phi = np.arctan2(surface[..., 1], surface[..., 0])
        source_phi = np.arctan2(sources[:, 1], sources[:, 0])
        orders = range(self.truncation.azimuthal_maximum + 1)
        if self.mode_workers == 1:
            contributions = (
                self._surface_order_contribution(
                    order,
                    surface_radius,
                    surface_phi,
                    source_radius,
                    source_phi,
                    sources[:, 2],
                    weights,
                )
                for order in orders
            )
            return sum(
                contributions,
                start=np.zeros(surface_radius.shape, dtype=float),
            )
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError(
                "parallel mode evaluation requires the fork start method"
            )
        global _MODE_WORKER_CONTEXT
        _MODE_WORKER_CONTEXT = (
            self,
            surface_radius,
            surface_phi,
            source_radius,
            source_phi,
            sources[:, 2],
            weights,
        )
        context = multiprocessing.get_context("fork")
        pool = context.Pool(processes=self.mode_workers)
        try:
            contributions = pool.map(_surface_mode_worker, orders, chunksize=1)
        finally:
            pool.close()
            pool.join()
            _MODE_WORKER_CONTEXT = None
        return sum(
            contributions,
            start=np.zeros(surface_radius.shape, dtype=float),
        )

    def _surface_order_contribution(
        self,
        order: int,
        surface_radius: np.ndarray,
        surface_phi: np.ndarray,
        source_radius: np.ndarray,
        source_phi: np.ndarray,
        source_depth: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        radius = self.config.radius_mm
        roots = self.roots(order)
        norms = self.radial_mode_norms(order)
        surface_bessel = special.jv(
            order,
            surface_radius[..., np.newaxis] * roots / radius,
        )
        multiplicity = 1.0 if order == 0 else 2.0
        cosine_coefficients = np.zeros_like(roots)
        sine_coefficients = np.zeros_like(roots)
        for start in range(0, len(source_radius), self.source_chunk_size):
            stop = min(start + self.source_chunk_size, len(source_radius))
            source_bessel = special.jv(
                order,
                source_radius[start:stop, np.newaxis]
                * roots
                / radius,
            )
            vertical = self.vertical_escape_factor(
                roots[np.newaxis, :] / radius,
                source_depth[start:stop, np.newaxis],
            )
            weighted_source = (
                weights[start:stop, np.newaxis]
                * source_bessel
                * vertical
            )
            cosine_coefficients += np.sum(
                weighted_source
                * np.cos(order * source_phi[start:stop])[
                    :, np.newaxis
                ],
                axis=0,
            )
            sine_coefficients += np.sum(
                weighted_source
                * np.sin(order * source_phi[start:stop])[
                    :, np.newaxis
                ],
                axis=0,
            )
        scale = multiplicity / (2.0 * math.pi * norms)
        radial_cosine = np.sum(
            surface_bessel * (scale * cosine_coefficients)[np.newaxis, :],
            axis=-1,
        )
        radial_sine = np.sum(
            surface_bessel * (scale * sine_coefficients)[np.newaxis, :],
            axis=-1,
        )
        return (
            np.cos(order * surface_phi) * radial_cosine
            + np.sin(order * surface_phi) * radial_sine
        )

    def first_scatter_sources(
        self,
        entries: Iterable[LXeEntryRay],
        *,
        quadrature_order: int = 32,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Expand arbitrary entry rays into deterministic scatter sources."""

        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        entry_list = list(entries)
        for entry in entry_list:
            entry.validate()
        base_nodes, base_weights = np.polynomial.legendre.leggauss(quadrature_order)
        extinction = self.config.extinction_coefficient_per_mm
        scatter_fraction = self.config.scattering_coefficient_per_mm / extinction
        absorption_fraction = 1.0 - scatter_fraction
        sources: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        boundary_reach = 0.0
        ballistic_absorption = 0.0
        total_entry = 0.0
        for entry in entry_list:
            total_entry += entry.weight
            entry_xy = np.asarray(entry.entry_xy_mm, dtype=float)
            direction = np.asarray(entry.liquid_direction, dtype=float)
            boundary_distance = self._first_boundary_distance(entry_xy, direction)
            maximum_x = extinction * boundary_distance
            path_x = 0.5 * maximum_x * (base_nodes + 1.0)
            path_x_weights = 0.5 * maximum_x * base_weights
            path = path_x / extinction
            positions = np.column_stack(
                [
                    entry_xy[0] + path * direction[0],
                    entry_xy[1] + path * direction[1],
                    path * direction[2],
                ]
            )
            physical_weights = (
                entry.weight * scatter_fraction * path_x_weights * np.exp(-path_x)
            )
            sources.append(positions)
            weights.append(physical_weights)
            survival_to_boundary = math.exp(-maximum_x)
            boundary_reach += entry.weight * survival_to_boundary
            ballistic_absorption += (
                entry.weight * absorption_fraction * (1.0 - survival_to_boundary)
            )
        if sources:
            source_array = np.concatenate(sources, axis=0)
            weight_array = np.concatenate(weights)
        else:
            source_array = np.empty((0, 3), dtype=float)
            weight_array = np.empty(0, dtype=float)
        first_scatter_weight = float(weight_array.sum())
        audit = {
            "entry_weight": total_entry,
            "first_scatter_weight": first_scatter_weight,
            "ballistic_absorption": ballistic_absorption,
            "ballistic_boundary_reach": boundary_reach,
            "accounted_before_diffusion": (
                first_scatter_weight + ballistic_absorption + boundary_reach
            ),
        }
        return source_array, weight_array, audit

    def return_weights_on_surface_quadrature(
        self,
        entries: Iterable[LXeEntryRay],
        surface_xy_mm: np.ndarray,
        surface_area_weights_mm2: np.ndarray,
        *,
        first_scatter_order: int = 32,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Map LXe entry rays to weighted surface-exit quadrature nodes."""

        surface = np.asarray(surface_xy_mm, dtype=float)
        area_weights = np.asarray(surface_area_weights_mm2, dtype=float)
        if surface.ndim != 2 or surface.shape[1] != 2:
            raise ValueError("surface_xy_mm must have shape (N,2)")
        if area_weights.shape != (len(surface),):
            raise ValueError("surface area weights must have shape (N,)")
        sources, weights, audit = self.first_scatter_sources(
            entries, quadrature_order=first_scatter_order
        )
        density = self.surface_return_density_mixture(surface, sources, weights)
        node_weights = density * area_weights
        negative_weight = float(-node_weights[node_weights < 0.0].sum())
        if len(sources):
            expected_return = float(
                np.dot(
                    weights,
                    self.total_return_probability(
                        np.linalg.norm(sources[:, :2], axis=1),
                        sources[:, 2],
                    ),
                )
            )
        else:
            expected_return = 0.0
        audit = {
            **audit,
            "surface_return_expected": expected_return,
            "surface_return_raw": float(node_weights.sum()),
            "negative_surface_weight": negative_weight,
            "surface_return_positive": float(np.maximum(node_weights, 0.0).sum()),
        }
        return node_weights, audit

    def return_fourier_coefficients(
        self,
        entry: LXeEntryRay,
        surface_radius_mm: np.ndarray,
        *,
        first_scatter_order: int = 32,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Return the cylindrical surface-response coefficients for one ray.

        The returned complex array has shape ``(m_max + 1, n_surface_r)`` and
        reconstructs the top-return density as

        ``real(sum_m coefficients[m, r] * exp(i*m*phi)))``.

        Keeping the azimuthal dependence in this modal form makes rotations
        exact within the configured Fourier truncation and avoids storing a
        dense phase-bin by surface-node response matrix.
        """

        radii = np.asarray(surface_radius_mm, dtype=float)
        if radii.ndim != 1:
            raise ValueError("surface_radius_mm must be one-dimensional")
        if np.any(radii < 0.0) or np.any(radii > self.config.radius_mm):
            raise ValueError("surface radius lies outside the liquid cylinder")
        entry.validate()
        sources, weights, audit = self.first_scatter_sources(
            [entry], quadrature_order=first_scatter_order
        )
        coefficients, expected_return = self.surface_fourier_coefficients_from_sources(
            sources,
            weights,
            radii,
        )
        return coefficients, {
            **audit,
            "surface_return_expected": expected_return,
        }

    def surface_fourier_coefficients_from_sources(
        self,
        source_xyz_mm: np.ndarray,
        source_weight: np.ndarray,
        surface_radius_mm: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Return surface modes for already-scattered isotropic sources.

        ``first_scatter_sources`` and the explicit low-collision transport
        both terminate in the same state: a weighted collection of Rayleigh
        collision positions whose outgoing scalar intensity is isotropic.
        Keeping this operation public avoids applying an additional,
        fictitious first-flight distribution to the low-collision tail.
        """

        sources = np.asarray(source_xyz_mm, dtype=float)
        weights = np.asarray(source_weight, dtype=float)
        radii = np.asarray(surface_radius_mm, dtype=float)
        if sources.ndim != 2 or sources.shape[1] != 3:
            raise ValueError("source_xyz_mm must have shape (N,3)")
        if weights.shape != (len(sources),):
            raise ValueError("source weights do not align with source positions")
        if np.any(weights < 0.0) or np.any(~np.isfinite(weights)):
            raise ValueError("source weights must be finite and non-negative")
        if radii.ndim != 1:
            raise ValueError("surface_radius_mm must be one-dimensional")
        if np.any(radii < 0.0) or np.any(radii > self.config.radius_mm):
            raise ValueError("surface radius lies outside the liquid cylinder")
        maximum_order = self.truncation.azimuthal_maximum
        coefficients = np.zeros(
            (maximum_order + 1, len(radii)), dtype=np.complex128
        )
        if len(sources) == 0:
            return coefficients, 0.0

        source_radius = np.linalg.norm(sources[:, :2], axis=1)
        source_phi = np.arctan2(sources[:, 1], sources[:, 0])
        expected_return = float(
            np.dot(
                weights,
                self.total_return_probability(
                    source_radius,
                    sources[:, 2],
                ),
            )
        )
        cylinder_radius = self.config.radius_mm
        for order in range(maximum_order + 1):
            roots = self.roots(order)
            norms = self.radial_mode_norms(order)
            source_bessel = special.jv(
                order,
                source_radius[:, np.newaxis] * roots / cylinder_radius,
            )
            vertical = self.vertical_escape_factor(
                roots[np.newaxis, :] / cylinder_radius,
                sources[:, 2, np.newaxis],
            )
            source_modes = np.sum(
                weights[:, np.newaxis]
                * source_bessel
                * vertical
                * np.exp(-1j * order * source_phi)[:, np.newaxis],
                axis=0,
            )
            surface_bessel = special.jv(
                order,
                radii[:, np.newaxis] * roots / cylinder_radius,
            )
            multiplicity = 1.0 if order == 0 else 2.0
            scale = multiplicity / (2.0 * math.pi * norms)
            coefficients[order] = surface_bessel @ (scale * source_modes)
        return coefficients, expected_return

    def first_scatter_total_return(
        self,
        entry_xy_mm: np.ndarray,
        liquid_direction: np.ndarray,
        *,
        quadrature_order: int = 64,
    ) -> dict[str, float]:
        """Integrate the return probability after one exact ballistic flight.

        The input direction is already refracted into LXe and has positive z.
        A path that reaches the side or bottom before its first interaction is
        reported separately; it is not replaced by a centre-source kernel.
        """

        entry = LXeEntryRay(
            np.asarray(entry_xy_mm, dtype=float),
            np.asarray(liquid_direction, dtype=float),
            1.0,
        )
        sources, weights, audit = self.first_scatter_sources(
            [entry], quadrature_order=quadrature_order
        )
        return_probability = float(
            np.dot(
                weights,
                self.total_return_probability(
                    np.linalg.norm(sources[:, :2], axis=1),
                    sources[:, 2],
                ),
            )
        )
        return {
            **audit,
            "return_to_gxe": return_probability,
        }


_MODE_WORKER_CONTEXT: tuple | None = None


def _surface_mode_worker(order: int) -> np.ndarray:
    if _MODE_WORKER_CONTEXT is None:
        raise RuntimeError("mode worker context was not initialized")
    model, *arguments = _MODE_WORKER_CONTEXT
    return model._surface_order_contribution(order, *arguments)
