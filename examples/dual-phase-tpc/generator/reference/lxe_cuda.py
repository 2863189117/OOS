#!/usr/bin/env python3
"""Optional CUDA acceleration for the intrinsic LXe response generator.

The production solver already has a native CUDA runtime, but the expensive
offline construction of the intrinsic LXe block is Python/NumPy code.  This
module keeps CuPy optional and local to that construction step.  It accelerates
the two dominant operations:

* deterministic Sobol path propagation through the explicit collision prefix;
* projection of the surviving diffusion sources and explicit exits onto the
  cylindrical Fourier--Bessel surface basis.

No CuPy array crosses the public generator boundary.  The resulting arrays
therefore have exactly the same NumPy/HDF5 contract as the CPU implementation.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy import special

from .lxe_cylinder_green import LXeCylinderGreen, LXeEntryRay
from .lxe_diffusion_return import LXeDiffusionConfig
from .lxe_p1_vector_tail import P1SurfaceCurrentModes
from .lxe_scalar_collision import (
    ScalarCollisionConfig,
    ScalarCollisionExpansion,
    ScalarCollisionSnapshot,
    ScalarExitEvents,
)


def _cupy():
    """Import CuPy lazily and give a useful error on CPU-only installations."""

    try:
        import cupy as cp
    except ImportError as error:  # pragma: no cover - depends on host hardware
        raise RuntimeError(
            "the CUDA LXe backend requires CuPy; install the wheel matching "
            "the host CUDA toolkit (for example cupy-cuda12x)"
        ) from error
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except cp.cuda.runtime.CUDARuntimeError as error:  # pragma: no cover
        raise RuntimeError(
            "CuPy is installed but no usable CUDA device exists"
        ) from error
    if device_count <= 0:  # pragma: no cover
        raise RuntimeError("CuPy is installed but no CUDA device was found")
    return cp


def cuda_available() -> bool:
    """Return whether a usable CuPy CUDA runtime is available."""

    try:
        _cupy()
    except RuntimeError:
        return False
    return True


def _as_numpy(cp, value):
    return cp.asnumpy(value)


def _orthonormal_basis(cp, direction):
    helper = cp.zeros_like(direction)
    use_z = cp.abs(direction[:, 2]) < 0.9
    helper[use_z, 2] = 1.0
    helper[~use_z, 1] = 1.0
    first = cp.cross(helper, direction)
    first /= cp.linalg.norm(first, axis=1)[:, cp.newaxis]
    second = cp.cross(direction, first)
    return first, second


def _rayleigh_directions(cp, incident, unit_mu, unit_phi):
    mu = 2.0 * cp.sinh(cp.arcsinh(4.0 * unit_mu - 2.0) / 3.0)
    transverse = cp.sqrt(cp.maximum(0.0, 1.0 - mu * mu))
    phi = 2.0 * math.pi * unit_phi
    first, second = _orthonormal_basis(cp, incident)
    result = (
        mu[:, cp.newaxis] * incident
        + (transverse * cp.cos(phi))[:, cp.newaxis] * first
        + (transverse * cp.sin(phi))[:, cp.newaxis] * second
    )
    result /= cp.linalg.norm(result, axis=1)[:, cp.newaxis]
    return result


def _lambertian_directions(cp, inward_normal, unit_mu_squared, unit_phi):
    normal = inward_normal / cp.linalg.norm(inward_normal, axis=1)[:, cp.newaxis]
    mu = cp.sqrt(unit_mu_squared)
    transverse = cp.sqrt(cp.maximum(0.0, 1.0 - mu * mu))
    phi = 2.0 * math.pi * unit_phi
    first, second = _orthonormal_basis(cp, normal)
    result = (
        mu[:, cp.newaxis] * normal
        + (transverse * cp.cos(phi))[:, cp.newaxis] * first
        + (transverse * cp.sin(phi))[:, cp.newaxis] * second
    )
    result /= cp.linalg.norm(result, axis=1)[:, cp.newaxis]
    return result


def _boundary_distance_and_kind(cp, position, direction, radius_mm, depth_mm):
    count = len(position)
    distances = cp.full((count, 3), cp.inf, dtype=cp.float64)
    upward = direction[:, 2] < -1.0e-15
    distances[upward, 0] = -position[upward, 2] / direction[upward, 2]
    downward = direction[:, 2] > 1.0e-15
    distances[downward, 2] = (depth_mm - position[downward, 2]) / direction[downward, 2]
    transverse2 = cp.sum(direction[:, :2] ** 2, axis=1)
    transverse = transverse2 > 1.0e-30
    dot = cp.sum(position[:, :2] * direction[:, :2], axis=1)
    radial2 = cp.sum(position[:, :2] ** 2, axis=1)
    discriminant = dot * dot + transverse2 * (radius_mm * radius_mm - radial2)
    valid = transverse & (discriminant >= 0.0)
    distances[valid, 1] = (
        -dot[valid] + cp.sqrt(cp.maximum(0.0, discriminant[valid]))
    ) / transverse2[valid]
    distances[distances <= 1.0e-12] = cp.inf
    kind = cp.argmin(distances, axis=1)
    distance = distances[cp.arange(count), kind]
    if bool(cp.any(~cp.isfinite(distance))):
        raise RuntimeError("a liquid path has no finite boundary intersection")
    return distance, kind


def _fresnel_power(cp, n_incident: float, n_transmitted: float, cosine):
    """Vectorized unpolarized Fresnel power matching ``numerics.fresnel_power``."""

    ci = cp.clip(cosine, 0.0, 1.0)
    ratio = n_incident / n_transmitted
    sin2_t = ratio * ratio * cp.maximum(0.0, 1.0 - ci * ci)
    transmitted = sin2_t < 1.0
    ct = cp.sqrt(cp.maximum(0.0, 1.0 - sin2_t))
    safe_ct = cp.where(transmitted, ct, 0.0)
    rs = (n_incident * ci - n_transmitted * safe_ct) / (
        n_incident * ci + n_transmitted * safe_ct
    )
    rp = (n_transmitted * ci - n_incident * safe_ct) / (
        n_transmitted * ci + n_incident * safe_ct
    )
    reflection = cp.where(transmitted, 0.5 * (rs * rs + rp * rp), 1.0)
    transmission = cp.where(transmitted, 1.0 - reflection, 0.0)
    return reflection, transmission, ct, transmitted


def propagate_scalar_collision_orders_cuda(
    entries: Iterable[LXeEntryRay],
    liquid: LXeDiffusionConfig,
    collision_orders: Sequence[int] = (1, 3, 5, 7),
    controls: ScalarCollisionConfig = ScalarCollisionConfig(),
    *,
    cubature: np.ndarray | None = None,
) -> ScalarCollisionExpansion:
    """CUDA equivalent of ``propagate_scalar_collision_orders``.

    The operator builder currently calls this with one entry ray and one
    requested collision order.  Supporting multiple weighted entries here is
    nevertheless inexpensive and keeps the numerical contract identical.
    """

    cp = _cupy()
    liquid.validate()
    controls.validate()
    orders = tuple(sorted(set(int(value) for value in collision_orders)))
    if not orders or orders[0] <= 0:
        raise ValueError("collision orders must be positive")
    maximum_order = orders[-1]
    if controls.maximum_events < maximum_order:
        raise ValueError("maximum_events must not be below maximum collision order")
    entry_list = list(entries)
    if not entry_list:
        raise ValueError("at least one entry ray is required")
    for entry in entry_list:
        entry.validate()

    dimension = 1 + 3 * controls.maximum_events
    if cubature is None:
        from scipy.stats import qmc

        cubature_values_np = qmc.Sobol(d=dimension, scramble=False).random_base2(
            controls.sample_power
        )
    else:
        cubature_values_np = np.asarray(cubature, dtype=float)
        if (
            cubature_values_np.ndim != 2
            or cubature_values_np.shape[1] < dimension
            or len(cubature_values_np) == 0
        ):
            raise ValueError("cubature must have shape (N, >= 1 + 3*maximum_events)")
        if np.any(cubature_values_np < 0.0) or np.any(cubature_values_np > 1.0):
            raise ValueError("cubature values must lie in [0, 1]")
    cubature_values = cp.asarray(cubature_values_np, dtype=cp.float64)
    sample_count = len(cubature_values_np)

    entry_weights = np.asarray([entry.weight for entry in entry_list], dtype=float)
    entry_total = float(entry_weights.sum())
    if entry_total <= 0.0:
        raise ValueError("entry rays have zero total weight")
    cumulative = np.cumsum(entry_weights)
    selected = np.searchsorted(
        cumulative, cubature_values_np[:, 0] * entry_total, side="right"
    )
    selected = np.minimum(selected, len(entry_list) - 1)
    position = cp.asarray(
        [
            [entry_list[index].entry_xy_mm[0], entry_list[index].entry_xy_mm[1], 0.0]
            for index in selected
        ],
        dtype=cp.float64,
    )
    direction = cp.asarray(
        [entry_list[index].liquid_direction for index in selected], dtype=cp.float64
    )
    weight = cp.full(sample_count, entry_total / sample_count, dtype=cp.float64)
    position[:, 2] = controls.boundary_epsilon_mm
    collision_count = cp.zeros(sample_count, dtype=cp.int16)
    active = cp.ones(sample_count, dtype=cp.bool_)

    exit_position = []
    exit_liquid_direction = []
    exit_gas_direction = []
    exit_weight = []
    exit_collision_count = []
    loss_weight = []
    loss_collision_count = []
    side_incident_weight = []
    side_reflected_weight = []
    side_collision_count = []
    bottom_incident_weight = []
    bottom_reflected_weight = []
    bottom_collision_count = []
    snapshot_positions = {order: [] for order in orders}
    snapshot_directions = {order: [] for order in orders}
    snapshot_weights = {order: [] for order in orders}

    tiny = np.finfo(float).tiny
    epsilon = controls.boundary_epsilon_mm
    for event in range(controls.maximum_events):
        active_indices = cp.flatnonzero(active)
        if int(active_indices.size) == 0:
            break
        u_free = cp.clip(
            cubature_values[active_indices, 1 + 3 * event], tiny, 1.0 - tiny
        )
        u_direction_1 = cubature_values[active_indices, 2 + 3 * event]
        u_direction_2 = cubature_values[active_indices, 3 + 3 * event]
        scatter_distance = -liquid.rayleigh_length_mm * cp.log1p(-u_free)
        boundary_distance, boundary_kind = _boundary_distance_and_kind(
            cp,
            position[active_indices],
            direction[active_indices],
            liquid.radius_mm,
            liquid.depth_mm,
        )
        scatter = scatter_distance < boundary_distance
        travel = cp.where(scatter, scatter_distance, boundary_distance)
        before_weight = weight[active_indices].copy()
        survival = cp.exp(-travel / liquid.absorption_length_mm)
        absorbed = before_weight * (1.0 - survival)
        if bool(cp.any(absorbed > 0.0)):
            loss_weight.append(absorbed)
            loss_collision_count.append(collision_count[active_indices].copy())
        weight[active_indices] *= survival
        position[active_indices] += travel[:, cp.newaxis] * direction[active_indices]

        scatter_indices = active_indices[scatter]
        if int(scatter_indices.size):
            local = cp.flatnonzero(scatter)
            direction[scatter_indices] = _rayleigh_directions(
                cp,
                direction[scatter_indices],
                u_direction_1[local],
                u_direction_2[local],
            )
            collision_count[scatter_indices] += 1
            for order in orders:
                selected_order = scatter_indices[
                    collision_count[scatter_indices] == order
                ]
                if int(selected_order.size):
                    snapshot_positions[order].append(position[selected_order].copy())
                    snapshot_directions[order].append(direction[selected_order].copy())
                    snapshot_weights[order].append(weight[selected_order].copy())
                    if order == maximum_order:
                        active[selected_order] = False

        boundary_indices = active_indices[~scatter]
        if int(boundary_indices.size) == 0:
            continue
        local_boundary = cp.flatnonzero(~scatter)
        kinds = boundary_kind[~scatter]

        top_local = cp.flatnonzero(kinds == 0)
        if int(top_local.size):
            indices = boundary_indices[top_local]
            reflection, transmission, cos_transmitted, has_transmission = (
                _fresnel_power(cp, liquid.n_lxe, liquid.n_gxe, -direction[indices, 2])
            )
            transmitted_weight = weight[indices] * transmission
            keep = transmitted_weight > controls.weight_cutoff
            if bool(cp.any(keep)):
                kept_indices = indices[keep]
                eta = liquid.n_lxe / liquid.n_gxe
                gas_direction = cp.column_stack(
                    [
                        eta * direction[kept_indices, 0],
                        eta * direction[kept_indices, 1],
                        cos_transmitted[keep],
                    ]
                )
                gas_direction /= cp.linalg.norm(gas_direction, axis=1)[:, cp.newaxis]
                exit_position.append(position[kept_indices, :2].copy())
                exit_liquid_direction.append(direction[kept_indices].copy())
                exit_gas_direction.append(gas_direction)
                exit_weight.append(transmitted_weight[keep])
                exit_collision_count.append(collision_count[kept_indices].copy())
            weight[indices] *= reflection
            direction[indices, 2] *= -1.0
            position[indices, 2] = epsilon
            inactive = (weight[indices] <= controls.weight_cutoff) | (
                ~has_transmission & (reflection <= controls.weight_cutoff)
            )
            active[indices[inactive]] = False

        side_local = cp.flatnonzero(kinds == 1)
        if int(side_local.size):
            indices = boundary_indices[side_local]
            previous = weight[indices].copy()
            weight[indices] *= liquid.side_reflectivity
            side_incident_weight.append(previous)
            side_reflected_weight.append(weight[indices].copy())
            side_collision_count.append(collision_count[indices].copy())
            lost = previous - weight[indices]
            if bool(cp.any(lost > 0.0)):
                loss_weight.append(lost)
                loss_collision_count.append(collision_count[indices].copy())
            radius = cp.linalg.norm(position[indices, :2], axis=1)
            outward = position[indices, :2] / radius[:, cp.newaxis]
            inward_normal = cp.column_stack(
                [-outward[:, 0], -outward[:, 1], cp.zeros(len(indices))]
            )
            direction[indices] = _lambertian_directions(
                cp,
                inward_normal,
                u_direction_1[local_boundary[side_local]],
                u_direction_2[local_boundary[side_local]],
            )
            position[indices, :2] = outward * (liquid.radius_mm - epsilon)
            active[indices[weight[indices] <= controls.weight_cutoff]] = False

        bottom_local = cp.flatnonzero(kinds == 2)
        if int(bottom_local.size):
            indices = boundary_indices[bottom_local]
            previous = weight[indices].copy()
            weight[indices] *= liquid.bottom_reflectivity
            bottom_incident_weight.append(previous)
            bottom_reflected_weight.append(weight[indices].copy())
            bottom_collision_count.append(collision_count[indices].copy())
            lost = previous - weight[indices]
            if bool(cp.any(lost > 0.0)):
                loss_weight.append(lost)
                loss_collision_count.append(collision_count[indices].copy())
            inward_normal = cp.tile(cp.asarray([0.0, 0.0, -1.0]), (len(indices), 1))
            direction[indices] = _lambertian_directions(
                cp,
                inward_normal,
                u_direction_1[local_boundary[bottom_local]],
                u_direction_2[local_boundary[bottom_local]],
            )
            position[indices, 2] = liquid.depth_mm - epsilon
            active[indices[weight[indices] <= controls.weight_cutoff]] = False

    def concatenate(values, shape, dtype):
        if values:
            return cp.concatenate(values)
        return cp.empty(shape, dtype=dtype)

    all_exit_position = concatenate(exit_position, (0, 2), cp.float64)
    all_exit_liquid = concatenate(exit_liquid_direction, (0, 3), cp.float64)
    all_exit_gas = concatenate(exit_gas_direction, (0, 3), cp.float64)
    all_exit_weight = concatenate(exit_weight, 0, cp.float64)
    all_exit_collisions = concatenate(exit_collision_count, 0, cp.int16)
    all_loss_weight = concatenate(loss_weight, 0, cp.float64)
    all_loss_collisions = concatenate(loss_collision_count, 0, cp.int16)

    def boundary_events(incident, reflected, collisions):
        return (
            concatenate(incident, 0, cp.float64),
            concatenate(reflected, 0, cp.float64),
            concatenate(collisions, 0, cp.int16),
        )

    side_incident, side_reflected, side_collisions = boundary_events(
        side_incident_weight, side_reflected_weight, side_collision_count
    )
    bottom_incident, bottom_reflected, bottom_collisions = boundary_events(
        bottom_incident_weight, bottom_reflected_weight, bottom_collision_count
    )

    result = {}
    for order in orders:
        source_position = concatenate(snapshot_positions[order], (0, 3), cp.float64)
        source_direction = concatenate(
            snapshot_directions[order], (0, 3), cp.float64
        )
        source_weight = concatenate(snapshot_weights[order], 0, cp.float64)
        exit_mask = all_exit_collisions < order
        loss_mask = all_loss_collisions < order
        explicit = ScalarExitEvents(
            position_xy_mm=_as_numpy(cp, all_exit_position[exit_mask]),
            liquid_direction=_as_numpy(cp, all_exit_liquid[exit_mask]),
            gas_direction=_as_numpy(cp, all_exit_gas[exit_mask]),
            weight=_as_numpy(cp, all_exit_weight[exit_mask]),
            preceding_collisions=_as_numpy(cp, all_exit_collisions[exit_mask]),
        )
        explicit_total = float(cp.sum(all_exit_weight[exit_mask]))
        source_total = float(cp.sum(source_weight))
        loss_total = float(cp.sum(all_loss_weight[loss_mask]))
        side_mask = side_collisions < order
        bottom_mask = bottom_collisions < order
        side_incident_total = float(cp.sum(side_incident[side_mask]))
        side_reflected_total = float(cp.sum(side_reflected[side_mask]))
        bottom_incident_total = float(cp.sum(bottom_incident[bottom_mask]))
        bottom_reflected_total = float(cp.sum(bottom_reflected[bottom_mask]))
        accounted = explicit_total + source_total + loss_total
        result[order] = ScalarCollisionSnapshot(
            collision_order=order,
            source_xyz_mm=_as_numpy(cp, source_position),
            source_direction=_as_numpy(cp, source_direction),
            source_weight=_as_numpy(cp, source_weight),
            explicit_exits=explicit,
            audit={
                "entry_weight": entry_total,
                "explicit_surface_escape": explicit_total,
                "diffusion_source_weight": source_total,
                "pre_switch_loss": loss_total,
                "pre_switch_accounted": accounted,
                "pre_switch_unresolved": entry_total - accounted,
                "side_boundary_incident": side_incident_total,
                "side_boundary_reflected": side_reflected_total,
                "side_boundary_absorbed": side_incident_total - side_reflected_total,
                "bottom_boundary_incident": bottom_incident_total,
                "bottom_boundary_reflected": bottom_reflected_total,
                "bottom_boundary_absorbed": (
                    bottom_incident_total - bottom_reflected_total
                ),
            },
        )
    unresolved = float(cp.sum(weight[active]))
    return ScalarCollisionExpansion(
        snapshots=result,
        sample_count=sample_count,
        entry_weight=entry_total,
        metadata={
            "sample_power": controls.sample_power,
            "maximum_events": controls.maximum_events,
            "collision_orders": list(orders),
            "active_weight_after_event_limit": unresolved,
            "compute_backend": "cuda",
        },
    )


_SOURCE_MODE_KERNEL = r"""
extern "C" __global__
void lxe_source_modes(
    const double* source_xyz,
    const double* source_weight,
    const long long source_count,
    const double* roots,
    const int radial_modes,
    const int order_count,
    const double cylinder_radius,
    const double cylinder_depth,
    const double diffusion,
    const double absorption_coefficient,
    const double beta_top,
    const double beta_bottom,
    double* output_real,
    double* output_imag)
{
    const int mode_index = blockDim.x * blockIdx.x + threadIdx.x;
    const int mode_count = radial_modes * order_count;
    if (mode_index >= mode_count) return;
    const int order = mode_index / radial_modes;
    const double root = roots[mode_index];
    const double wave_number = root / cylinder_radius;
    const double gamma = sqrt(
        wave_number * wave_number + absorption_coefficient / diffusion);
    const double a_top = beta_top / diffusion;
    const double a_bottom = beta_bottom / diffusion;
    const double denominator =
        gamma + a_top * a_bottom / gamma + a_top + a_bottom
        + (-gamma - a_top * a_bottom / gamma + a_top + a_bottom)
          * exp(-2.0 * gamma * cylinder_depth);
    double real_sum = 0.0;
    double imag_sum = 0.0;
    for (long long source = 0; source < source_count; ++source) {
        const double x = source_xyz[3 * source];
        const double y = source_xyz[3 * source + 1];
        const double depth = source_xyz[3 * source + 2];
        const double radius = hypot(x, y);
        const double phi = atan2(y, x);
        const double numerator =
            (1.0 + a_bottom / gamma) * exp(-gamma * depth)
            + (1.0 - a_bottom / gamma)
              * exp(-gamma * (2.0 * cylinder_depth - depth));
        const double vertical = beta_top / diffusion * numerator / denominator;
        const double argument = radius * root / cylinder_radius;
        // CUDA's device jn(m, 0) returns NaN for some m >= 2, whereas the
        // mathematical limit (and scipy.special.jv) is exactly zero.
        const double radial = fabs(argument) < 1.0e-12
            ? (order == 0 ? 1.0 : 0.0)
            : jn(order, argument);
        const double value = source_weight[source] * radial * vertical;
        real_sum += value * cos(order * phi);
        imag_sum -= value * sin(order * phi);
    }
    output_real[mode_index] = real_sum;
    output_imag[mode_index] = imag_sum;
}
"""


class CudaModalProjector:
    """Reusable CUDA projection state for all rows of one LXe operator."""

    def __init__(
        self,
        green: LXeCylinderGreen,
        surface_radius_mm: np.ndarray,
        surface_ring_area_mm2: np.ndarray,
    ) -> None:
        self.cp = _cupy()
        self.green = green
        self.surface_radius_mm = np.asarray(surface_radius_mm, dtype=float)
        self.surface_ring_area_mm2 = np.asarray(surface_ring_area_mm2, dtype=float)
        self.maximum_order = green.truncation.azimuthal_maximum
        self.radial_modes = green.truncation.radial_modes
        roots = np.stack(
            [green.roots(order) for order in range(self.maximum_order + 1)]
        )
        norms = np.stack(
            [green.radial_mode_norms(order) for order in range(self.maximum_order + 1)]
        )
        surface_bessel = np.stack(
            [
                special.jv(
                    order,
                    self.surface_radius_mm[:, np.newaxis]
                    * roots[order]
                    / green.config.radius_mm,
                )
                for order in range(self.maximum_order + 1)
            ]
        )
        surface_derivative = np.stack(
            [
                roots[order][np.newaxis, :]
                / green.config.radius_mm
                * special.jvp(
                    order,
                    self.surface_radius_mm[:, np.newaxis]
                    * roots[order]
                    / green.config.radius_mm,
                )
                for order in range(self.maximum_order + 1)
            ]
        )
        multiplicity = np.ones(self.maximum_order + 1)
        multiplicity[1:] = 2.0
        scale = multiplicity[:, np.newaxis] / (2.0 * math.pi * norms)
        disk_integrals = np.empty(self.radial_modes, dtype=float)
        roots_zero = roots[0]
        zero = np.abs(roots_zero) < 1.0e-14
        disk_integrals[zero] = 0.5 * green.config.radius_mm**2
        disk_integrals[~zero] = (
            green.config.radius_mm**2
            * special.jv(1, roots_zero[~zero])
            / roots_zero[~zero]
        )
        self.roots = self.cp.asarray(roots)
        self.surface_bessel = self.cp.asarray(surface_bessel)
        self.surface_derivative = self.cp.asarray(surface_derivative)
        self.surface_radius = self.cp.asarray(self.surface_radius_mm)
        self.scale = self.cp.asarray(scale)
        self.return_scale = self.cp.asarray(disk_integrals / norms[0])
        self.ring_area = self.cp.asarray(self.surface_ring_area_mm2)
        self.kernel = self.cp.RawKernel(_SOURCE_MODE_KERNEL, "lxe_source_modes")

    def _project_source_modes(
        self, source_xyz_mm: np.ndarray, source_weight: np.ndarray
    ):
        """Return device source modes and their expected scalar return."""

        cp = self.cp
        sources = cp.asarray(source_xyz_mm, dtype=cp.float64)
        weights = cp.asarray(source_weight, dtype=cp.float64)
        shape = (self.maximum_order + 1, self.radial_modes)
        if len(sources) == 0:
            return cp.zeros(shape, dtype=cp.complex128), 0.0
        output_real = cp.empty(shape, dtype=cp.float64)
        output_imag = cp.empty(shape, dtype=cp.float64)
        count = int(output_real.size)
        threads = 128
        self.kernel(
            ((count + threads - 1) // threads,),
            (threads,),
            (
                sources,
                weights,
                np.int64(len(sources)),
                self.roots,
                np.int32(self.radial_modes),
                np.int32(self.maximum_order + 1),
                np.float64(self.green.config.radius_mm),
                np.float64(self.green.config.depth_mm),
                np.float64(self.green.config.transport_diffusion_mm),
                np.float64(self.green.config.absorption_coefficient_per_mm),
                np.float64(self.green.beta_top),
                np.float64(self.green.beta_bottom),
                output_real,
                output_imag,
            ),
        )
        source_modes = output_real + 1j * output_imag
        expected_return = float(cp.dot(output_real[0], self.return_scale))
        return source_modes, expected_return

    def project_diffusion_sources(
        self, source_xyz_mm: np.ndarray, source_weight: np.ndarray
    ) -> tuple[np.ndarray, float]:
        cp = self.cp
        if len(source_xyz_mm) == 0:
            return (
                np.zeros(
                    (self.maximum_order + 1, len(self.surface_radius_mm)),
                    dtype=np.complex128,
                ),
                0.0,
            )
        source_modes, expected_return = self._project_source_modes(
            source_xyz_mm, source_weight
        )
        coefficients = cp.einsum(
            "msk,mk->ms",
            self.surface_bessel,
            self.scale * source_modes,
            optimize=True,
        )
        return cp.asnumpy(coefficients), expected_return

    def project_surface_currents(
        self, source_xyz_mm: np.ndarray, source_weight: np.ndarray
    ) -> P1SurfaceCurrentModes:
        """Project normal/radial/azimuthal P1 currents on the CUDA basis."""

        cp = self.cp
        source_modes, expected_return = self._project_source_modes(
            source_xyz_mm, source_weight
        )
        scaled = self.scale * source_modes
        normal = cp.einsum(
            "msk,mk->ms", self.surface_bessel, scaled, optimize=True
        )
        top_length = self.green.top_extrapolation_length_mm
        radial = -top_length * cp.einsum(
            "msk,mk->ms", self.surface_derivative, scaled, optimize=True
        )
        orders = cp.arange(self.maximum_order + 1, dtype=cp.float64)
        azimuthal = (
            -top_length
            * 1j
            * orders[:, cp.newaxis]
            * normal
            / self.surface_radius[cp.newaxis, :]
        )
        return P1SurfaceCurrentModes(
            normal=cp.asnumpy(normal),
            radial=cp.asnumpy(radial),
            azimuthal=cp.asnumpy(azimuthal),
            expected_return=expected_return,
            top_extrapolation_length_mm=top_length,
        )

    def project_explicit_exits(
        self, position_xy_mm: np.ndarray, weight: np.ndarray
    ) -> np.ndarray:
        cp = self.cp
        position = cp.asarray(position_xy_mm, dtype=cp.float64)
        values = cp.asarray(weight, dtype=cp.float64)
        result = cp.zeros(
            (self.maximum_order + 1, len(self.surface_radius_mm)),
            dtype=cp.complex128,
        )
        if len(values) == 0:
            return cp.asnumpy(result)
        radius = cp.linalg.norm(position, axis=1)
        phi = cp.arctan2(position[:, 1], position[:, 0])
        node_u = cp.asarray(self.surface_radius_mm**2)
        value_u = radius * radius
        upper = cp.clip(
            cp.searchsorted(node_u, value_u, side="right"), 1, len(node_u) - 1
        )
        lower = upper - 1
        denominator = node_u[upper] - node_u[lower]
        upper_weight = cp.where(
            denominator > 0.0, (value_u - node_u[lower]) / denominator, 0.0
        )
        below = value_u <= node_u[0]
        above = value_u >= node_u[-1]
        lower[below] = 0
        upper[below] = 0
        upper_weight[below] = 0.0
        lower[above] = len(node_u) - 1
        upper[above] = len(node_u) - 1
        upper_weight[above] = 0.0
        orders = cp.arange(self.maximum_order + 1, dtype=cp.float64)
        multiplicity = cp.where(orders == 0.0, 1.0, 2.0)
        phase = cp.exp(-1j * orders[:, cp.newaxis] * phi[cp.newaxis, :])
        for indices, radial_weight in (
            (lower, 1.0 - upper_weight),
            (upper, upper_weight),
        ):
            modal = (
                multiplicity[:, cp.newaxis]
                * values[cp.newaxis, :]
                * radial_weight[cp.newaxis, :]
                * phase
                / self.ring_area[indices][cp.newaxis, :]
            )
            for order in range(self.maximum_order + 1):
                real = cp.bincount(
                    indices,
                    weights=modal[order].real,
                    minlength=len(self.surface_radius_mm),
                )
                imaginary = cp.bincount(
                    indices,
                    weights=modal[order].imag,
                    minlength=len(self.surface_radius_mm),
                )
                result[order] += real + 1j * imaginary
        return cp.asnumpy(result)
