#!/usr/bin/env python3
"""Offline full-vector P1 closure for the diffuse LXe collision tail.

This is an isolated opt-in branch in the block builder.  It preserves the
existing scalar finite-cylinder Green function for the normal return current
and reconstructs the two tangential current components analytically from its
Fourier--Bessel modes.  The resulting three fixed angular channels are
suitable for a sparse tail-only A/B while the exact low-collision exits
continue to use their stored gas directions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import special

from .lxe_cylinder_green import LXeCylinderGreen
from .lxe_diffusion_return import LXeDiffusionConfig
from .numerics import fresnel_power, gauss_interval


RAYLEIGH_MILNE_TOP_LENGTH_MM = 1089.8684625


@dataclass(frozen=True)
class P1AngularChannels:
    """Gas directions and three current-normalized P1 angular channels.

    ``normal`` sums to one. ``radial`` and ``azimuthal`` each sum to zero.
    For one surface node the angle-resolved escaping current is

    ``J_n * normal + J_r * radial + J_phi * azimuthal``.
    """

    gas_direction: np.ndarray
    liquid_mu: np.ndarray
    relative_phi: np.ndarray
    normal: np.ndarray
    radial: np.ndarray
    azimuthal: np.ndarray
    normalization: float
    boundary_factor: float

    def validate(self) -> None:
        count = len(self.normal)
        if (
            self.gas_direction.shape != (count, 3)
            or self.liquid_mu.shape != (count,)
            or self.relative_phi.shape != (count,)
            or self.radial.shape != (count,)
            or self.azimuthal.shape != (count,)
        ):
            raise ValueError("P1 angular channels have inconsistent shapes")
        if not np.all(np.isfinite(self.gas_direction)):
            raise ValueError("P1 gas directions are not finite")
        if not np.allclose(
            np.linalg.norm(self.gas_direction, axis=1),
            1.0,
            rtol=0.0,
            atol=2.0e-13,
        ):
            raise ValueError("P1 gas directions are not normalized")
        if not math.isclose(float(self.normal.sum()), 1.0, abs_tol=2.0e-14):
            raise ValueError("normal P1 channel does not sum to one")
        if abs(float(self.radial.sum())) > 2.0e-14:
            raise ValueError("radial P1 channel does not sum to zero")
        if abs(float(self.azimuthal.sum())) > 2.0e-14:
            raise ValueError("azimuthal P1 channel does not sum to zero")


@dataclass(frozen=True)
class P1SurfaceCurrentModes:
    """Fourier coefficients of top normal, radial and azimuthal currents."""

    normal: np.ndarray
    radial: np.ndarray
    azimuthal: np.ndarray
    expected_return: float
    top_extrapolation_length_mm: float

    def validate(self) -> None:
        if (
            self.normal.ndim != 2
            or self.radial.shape != self.normal.shape
            or self.azimuthal.shape != self.normal.shape
        ):
            raise ValueError("P1 surface-current modes have inconsistent shapes")
        if not all(
            np.all(np.isfinite(value))
            for value in (self.normal, self.radial, self.azimuthal)
        ):
            raise ValueError("P1 surface-current modes are not finite")
        if self.expected_return < 0.0 or not math.isfinite(self.expected_return):
            raise ValueError("P1 expected return is invalid")


def _refract_upward(
    liquid_mu: float, relative_phi: float, n_lxe: float, n_gxe: float
) -> np.ndarray:
    """Refract an upward LXe direction into GXe in the local surface frame."""

    transverse_liquid = math.sqrt(max(0.0, 1.0 - liquid_mu * liquid_mu))
    transverse_gas = n_lxe / n_gxe * transverse_liquid
    if transverse_gas > 1.0 + 2.0e-13:
        raise RuntimeError("P1 escape quadrature contains a TIR direction")
    transverse_gas = min(1.0, transverse_gas)
    gas_mu = math.sqrt(max(0.0, 1.0 - transverse_gas * transverse_gas))
    return np.asarray(
        [
            transverse_gas * math.cos(relative_phi),
            transverse_gas * math.sin(relative_phi),
            gas_mu,
        ]
    )


def p1_escape_angular_channels(
    config: LXeDiffusionConfig,
    *,
    liquid_mu_order: int = 8,
    direction_phi_bins: int = 16,
) -> P1AngularChannels:
    """Generate the normal/radial/azimuthal P1 escape channels.

    Gauss--Legendre integration is restricted to the transmitting interval
    ``[mu_c, 1]``.  Fresnel transmission is evaluated for each node, and the
    liquid direction is refracted explicitly into the stored gas direction.
    """

    config.validate()
    if liquid_mu_order < 2 or direction_phi_bins < 2:
        raise ValueError("P1 angular orders must be at least two")
    diffusion = config.transport_diffusion_mm
    top_length = (
        config.top_boundary_length_mm
        if config.top_boundary_length_mm is not None
        else LXeCylinderGreen(config).top_extrapolation_length_mm
    )
    boundary_factor = top_length / (2.0 * diffusion)
    ratio = config.n_gxe / config.n_lxe
    critical_mu = math.sqrt(max(0.0, 1.0 - ratio * ratio))
    mu_nodes, mu_weights = gauss_interval(liquid_mu_order, critical_mu, 1.0)

    gas_direction: list[np.ndarray] = []
    liquid_mu: list[float] = []
    relative_phi: list[float] = []
    normal_raw: list[float] = []
    radial_raw: list[float] = []
    azimuthal_raw: list[float] = []
    for mu, mu_weight in zip(mu_nodes, mu_weights):
        _, transmission, _ = fresnel_power(
            config.n_lxe, config.n_gxe, float(mu)
        )
        transmission_unpolarized = 0.5 * float(np.sum(transmission))
        transverse = math.sqrt(max(0.0, 1.0 - float(mu) ** 2))
        integration_weight = (
            2.0
            * float(mu)
            * transmission_unpolarized
            * float(mu_weight)
            / direction_phi_bins
        )
        for index in range(direction_phi_bins):
            phi = 2.0 * math.pi * (index + 0.5) / direction_phi_bins
            gas_direction.append(
                _refract_upward(float(mu), phi, config.n_lxe, config.n_gxe)
            )
            liquid_mu.append(float(mu))
            relative_phi.append(phi)
            normal_raw.append(
                integration_weight * (2.0 * boundary_factor + 3.0 * float(mu))
            )
            tangent = integration_weight * 3.0 * transverse
            radial_raw.append(tangent * math.cos(phi))
            azimuthal_raw.append(tangent * math.sin(phi))

    normal = np.asarray(normal_raw)
    normalization = float(normal.sum())
    if not normalization > 0.0:
        raise RuntimeError("P1 escape quadrature has zero normalization")
    channels = P1AngularChannels(
        gas_direction=np.asarray(gas_direction),
        liquid_mu=np.asarray(liquid_mu),
        relative_phi=np.asarray(relative_phi),
        normal=normal / normalization,
        radial=np.asarray(radial_raw) / normalization,
        azimuthal=np.asarray(azimuthal_raw) / normalization,
        normalization=normalization,
        boundary_factor=boundary_factor,
    )
    channels.validate()
    return channels


def surface_current_fourier_coefficients_from_sources(
    green: LXeCylinderGreen,
    source_xyz_mm: np.ndarray,
    source_weight: np.ndarray,
    surface_radius_mm: np.ndarray,
) -> P1SurfaceCurrentModes:
    """Return analytic top-current modes for isotropic handoff sources.

    The existing scalar Green function gives the outward normal current
    ``J_n = beta_top * Phi``.  Because ``D / beta_top = z_b``, the tangential
    currents follow without another PDE solve:

    ``J_r = -z_b * d_r J_n`` and
    ``J_phi = -z_b/r * d_phi J_n``.
    """

    sources = np.asarray(source_xyz_mm, dtype=float)
    weights = np.asarray(source_weight, dtype=float)
    radii = np.asarray(surface_radius_mm, dtype=float)
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("source_xyz_mm must have shape (N,3)")
    if weights.shape != (len(sources),):
        raise ValueError("source weights do not align with source positions")
    if radii.ndim != 1 or np.any(radii <= 0.0):
        raise ValueError("surface radii must be a positive one-dimensional array")
    if np.any(weights < 0.0) or np.any(~np.isfinite(weights)):
        raise ValueError("source weights must be finite and non-negative")

    maximum_order = green.truncation.azimuthal_maximum
    shape = (maximum_order + 1, len(radii))
    normal = np.zeros(shape, dtype=np.complex128)
    radial = np.zeros(shape, dtype=np.complex128)
    azimuthal = np.zeros(shape, dtype=np.complex128)
    if len(sources) == 0:
        result = P1SurfaceCurrentModes(
            normal,
            radial,
            azimuthal,
            0.0,
            green.top_extrapolation_length_mm,
        )
        result.validate()
        return result

    source_radius = np.linalg.norm(sources[:, :2], axis=1)
    source_phi = np.arctan2(sources[:, 1], sources[:, 0])
    expected_return = float(
        np.dot(
            weights,
            green.total_return_probability(source_radius, sources[:, 2]),
        )
    )
    cylinder_radius = green.config.radius_mm
    top_length = green.top_extrapolation_length_mm
    for order in range(maximum_order + 1):
        roots = green.roots(order)
        norms = green.radial_mode_norms(order)
        source_bessel = special.jv(
            order,
            source_radius[:, np.newaxis] * roots / cylinder_radius,
        )
        vertical = green.vertical_escape_factor(
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
        argument = radii[:, np.newaxis] * roots / cylinder_radius
        surface_bessel = special.jv(order, argument)
        surface_derivative = (
            roots[np.newaxis, :]
            / cylinder_radius
            * special.jvp(order, argument)
        )
        multiplicity = 1.0 if order == 0 else 2.0
        amplitude = multiplicity * source_modes / (2.0 * math.pi * norms)
        normal[order] = surface_bessel @ amplitude
        radial[order] = -top_length * (surface_derivative @ amplitude)
        if order:
            azimuthal[order] = (
                -top_length * 1j * order * normal[order] / radii
            )

    result = P1SurfaceCurrentModes(
        normal=normal,
        radial=radial,
        azimuthal=azimuthal,
        expected_return=expected_return,
        top_extrapolation_length_mm=top_length,
    )
    result.validate()
    return result


def reconstruct_surface_currents(
    modes: P1SurfaceCurrentModes, surface_phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct real current densities on a radius-by-phi product grid."""

    modes.validate()
    phi = np.asarray(surface_phi, dtype=float)
    if phi.ndim != 1:
        raise ValueError("surface_phi must be one-dimensional")
    orders = np.arange(modes.normal.shape[0], dtype=float)
    basis = np.exp(1j * np.outer(orders, phi))
    return tuple(
        np.einsum("mr,mp->rp", value, basis, optimize=True).real
        for value in (modes.normal, modes.radial, modes.azimuthal)
    )


def full_vector_p1_joint_density(
    normal_current: np.ndarray,
    radial_current: np.ndarray,
    azimuthal_current: np.ndarray,
    channels: P1AngularChannels,
    *,
    boundary_factor: float | None = None,
    apply_tangent_limiter: bool = False,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Construct angle-resolved surface current and audit P1 positivity.

    Raw values are always audited before an optional tangent-only limiter.
    The function never clips individual directions and never renormalizes a
    negative angular row.  The caller can retain both the raw and limited
    products for an explicit A/B.
    """

    channels.validate()
    normal = np.asarray(normal_current, dtype=float)
    radial = np.asarray(radial_current, dtype=float)
    azimuthal = np.asarray(azimuthal_current, dtype=float)
    if normal.shape != radial.shape or normal.shape != azimuthal.shape:
        raise ValueError("surface-current arrays must share one shape")
    if np.any(~np.isfinite(normal + radial + azimuthal)):
        raise ValueError("surface currents must be finite")
    factor = channels.boundary_factor if boundary_factor is None else boundary_factor
    if factor <= 1.5:
        raise ValueError("P1 boundary factor must exceed 3/2")

    raw = (
        normal[..., np.newaxis] * channels.normal
        + radial[..., np.newaxis] * channels.radial
        + azimuthal[..., np.newaxis] * channels.azimuthal
    )
    negative_raw = -np.minimum(raw, 0.0)
    tangent_norm = np.hypot(radial, azimuthal)
    # Phi = 2 A J_n.  Enforce Phi >= 3 sqrt(J_n^2 + |J_t|^2).
    tangent_bound_squared = np.maximum(
        (2.0 * factor * normal / 3.0) ** 2 - normal**2,
        0.0,
    )
    tangent_bound = np.sqrt(tangent_bound_squared)
    scale = np.ones_like(normal)
    nonzero = tangent_norm > 0.0
    scale[nonzero] = np.minimum(1.0, tangent_bound[nonzero] / tangent_norm[nonzero])
    limited_radial = radial * scale
    limited_azimuthal = azimuthal * scale
    if apply_tangent_limiter:
        result = (
            normal[..., np.newaxis] * channels.normal
            + limited_radial[..., np.newaxis] * channels.radial
            + limited_azimuthal[..., np.newaxis] * channels.azimuthal
        )
    else:
        result = raw
    audit = {
        "sample_count": int(raw.size),
        "negative_sample_count_raw": int(np.count_nonzero(raw < 0.0)),
        "negative_current_raw": float(negative_raw.sum()),
        "minimum_current_raw": float(raw.min(initial=0.0)),
        "limited_surface_node_count": int(np.count_nonzero(scale < 1.0)),
        "maximum_tangent_scale_change": float(np.max(1.0 - scale, initial=0.0)),
        "normal_sum": float(normal.sum()),
        "raw_angle_sum_error": float(
            np.max(np.abs(raw.sum(axis=-1) - normal), initial=0.0)
        ),
        "output_angle_sum_error": float(
            np.max(np.abs(result.sum(axis=-1) - normal), initial=0.0)
        ),
    }
    return result, audit
