#!/usr/bin/env python3
r"""Low-order deterministic scalar Rayleigh transport in LXe.

The production LXe Green function treats the first Rayleigh flight exactly
and switches immediately to P1 diffusion.  This module delays that switch:
the first ``N`` Rayleigh collisions are propagated with a deterministic Sobol
cubature and the scalar unpolarized Rayleigh phase function

.. math::

   p(\Omega'\mid\Omega)
   = \frac{3}{16\pi}\left[1+(\Omega'\cdot\Omega)^2\right].

At requested collision orders the surviving particles are returned as
weighted diffusion sources.  Liquid-surface transmission before that order is
retained explicitly, including the transmitted direction.  Absorption is
integrated continuously along every flight.  Fresnel reflection at the liquid
surface is split deterministically into transmitted and reflected weights;
there is no Bernoulli boundary choice.

The Sobol points are a numerical cubature of path space, not photon-counting
Monte Carlo.  Repeating with increasing powers of two provides a direct
convergence test without Poisson noise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import qmc

from .numerics import fresnel_power
from .lxe_cylinder_green import LXeEntryRay
from .lxe_diffusion_return import LXeDiffusionConfig


@dataclass(frozen=True)
class ScalarCollisionConfig:
    """Controls for the low-order scalar collision expansion."""

    sample_power: int = 16
    maximum_events: int = 64
    weight_cutoff: float = 1.0e-15
    boundary_epsilon_mm: float = 1.0e-7

    def validate(self) -> None:
        if self.sample_power < 5:
            raise ValueError("sample_power must be at least five")
        if self.maximum_events <= 0:
            raise ValueError("maximum_events must be positive")
        if not math.isfinite(self.weight_cutoff) or self.weight_cutoff < 0.0:
            raise ValueError("weight_cutoff must be finite and non-negative")
        if (
            not math.isfinite(self.boundary_epsilon_mm)
            or self.boundary_epsilon_mm <= 0.0
        ):
            raise ValueError("boundary_epsilon_mm must be finite and positive")


@dataclass(frozen=True)
class ScalarExitEvents:
    """Explicit LXe-to-GXe transmissions before the diffusion switch."""

    position_xy_mm: np.ndarray
    liquid_direction: np.ndarray
    gas_direction: np.ndarray
    weight: np.ndarray
    preceding_collisions: np.ndarray


@dataclass(frozen=True)
class ScalarCollisionSnapshot:
    """Path-space state immediately after one requested collision order.

    ``source_direction`` is the sampled outgoing direction after the Rayleigh
    collision at ``source_xyz_mm``.  The legacy P1 tail intentionally ignores
    it, while SP_N or angle-resolved tails need it to compare alternative
    closures from an identical hand-off state.
    """

    collision_order: int
    source_xyz_mm: np.ndarray
    source_direction: np.ndarray
    source_weight: np.ndarray
    explicit_exits: ScalarExitEvents
    audit: dict[str, float]


@dataclass(frozen=True)
class ScalarCollisionExpansion:
    """All requested collision-order snapshots from one common cubature."""

    snapshots: dict[int, ScalarCollisionSnapshot]
    sample_count: int
    entry_weight: float
    metadata: dict[str, float | int | list[int]]


def scalar_rayleigh_mu(unit_interval: np.ndarray | float) -> np.ndarray:
    """Invert the unpolarized scalar Rayleigh CDF.

    The returned cosine follows ``p(mu)=3/8*(1+mu**2)`` on ``[-1, 1]``.
    The inverse is the real solution of ``mu**3 + 3*mu = 8*u - 4``.
    """

    value = np.asarray(unit_interval, dtype=float)
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("unit-interval values must lie in [0, 1]")
    return 2.0 * np.sinh(np.arcsinh(4.0 * value - 2.0) / 3.0)


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two transverse unit vectors for each direction row."""

    values = np.asarray(direction, dtype=float)
    helper = np.zeros_like(values)
    use_z = np.abs(values[:, 2]) < 0.9
    helper[use_z, 2] = 1.0
    helper[~use_z, 1] = 1.0
    first = np.cross(helper, values)
    first /= np.linalg.norm(first, axis=1)[:, np.newaxis]
    second = np.cross(values, first)
    return first, second


def scalar_rayleigh_directions(
    incident_direction: np.ndarray,
    unit_mu: np.ndarray,
    unit_phi: np.ndarray,
) -> np.ndarray:
    """Map scalar Rayleigh angular cubature values to global directions."""

    incident = np.asarray(incident_direction, dtype=float)
    if incident.ndim != 2 or incident.shape[1] != 3:
        raise ValueError("incident_direction must have shape (N,3)")
    unit_mu = np.asarray(unit_mu, dtype=float)
    unit_phi = np.asarray(unit_phi, dtype=float)
    if unit_mu.shape != (len(incident),) or unit_phi.shape != (len(incident),):
        raise ValueError("angular values must align with incident directions")
    mu = scalar_rayleigh_mu(unit_mu)
    transverse = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    phi = 2.0 * math.pi * unit_phi
    first, second = _orthonormal_basis(incident)
    result = (
        mu[:, np.newaxis] * incident
        + (transverse * np.cos(phi))[:, np.newaxis] * first
        + (transverse * np.sin(phi))[:, np.newaxis] * second
    )
    result /= np.linalg.norm(result, axis=1)[:, np.newaxis]
    return result


def _lambertian_directions(
    inward_normal: np.ndarray,
    unit_mu_squared: np.ndarray,
    unit_phi: np.ndarray,
) -> np.ndarray:
    """Sample cosine-weighted directions about inward-facing normals."""

    normal = np.asarray(inward_normal, dtype=float)
    normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]
    mu = np.sqrt(np.asarray(unit_mu_squared, dtype=float))
    transverse = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    phi = 2.0 * math.pi * np.asarray(unit_phi, dtype=float)
    first, second = _orthonormal_basis(normal)
    result = (
        mu[:, np.newaxis] * normal
        + (transverse * np.cos(phi))[:, np.newaxis] * first
        + (transverse * np.sin(phi))[:, np.newaxis] * second
    )
    result /= np.linalg.norm(result, axis=1)[:, np.newaxis]
    return result


def _boundary_distance_and_kind(
    position: np.ndarray,
    direction: np.ndarray,
    radius_mm: float,
    depth_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distance and kind (0 top, 1 side, 2 bottom) of first boundary."""

    count = len(position)
    distances = np.full((count, 3), np.inf, dtype=float)
    upward = direction[:, 2] < -1.0e-15
    distances[upward, 0] = -position[upward, 2] / direction[upward, 2]
    downward = direction[:, 2] > 1.0e-15
    distances[downward, 2] = (
        depth_mm - position[downward, 2]
    ) / direction[downward, 2]

    transverse2 = np.sum(direction[:, :2] ** 2, axis=1)
    transverse = transverse2 > 1.0e-30
    dot = np.sum(position[:, :2] * direction[:, :2], axis=1)
    radial2 = np.sum(position[:, :2] ** 2, axis=1)
    discriminant = dot * dot + transverse2 * (radius_mm * radius_mm - radial2)
    valid = transverse & (discriminant >= 0.0)
    distances[valid, 1] = (
        -dot[valid] + np.sqrt(np.maximum(0.0, discriminant[valid]))
    ) / transverse2[valid]
    distances[distances <= 1.0e-12] = np.inf
    kind = np.argmin(distances, axis=1)
    distance = distances[np.arange(count), kind]
    if np.any(~np.isfinite(distance)):
        raise RuntimeError("a liquid path has no finite boundary intersection")
    return distance, kind


def _select_weighted_entries(
    entries: Sequence[LXeEntryRay],
    selector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Use inverse-CDF cubature to obtain equally weighted entry states."""

    if not entries:
        raise ValueError("at least one entry ray is required")
    for entry in entries:
        entry.validate()
    entry_weight = np.asarray([entry.weight for entry in entries], dtype=float)
    total = float(entry_weight.sum())
    if total <= 0.0:
        raise ValueError("entry rays have zero total weight")
    cumulative = np.cumsum(entry_weight)
    indices = np.searchsorted(cumulative, selector * total, side="right")
    indices = np.minimum(indices, len(entries) - 1)
    position = np.asarray(
        [
            [entries[index].entry_xy_mm[0], entries[index].entry_xy_mm[1], 0.0]
            for index in indices
        ],
        dtype=float,
    )
    direction = np.asarray(
        [entries[index].liquid_direction for index in indices],
        dtype=float,
    )
    weight = np.full(len(selector), total / len(selector), dtype=float)
    return position, direction, weight, total


def propagate_scalar_collision_orders(
    entries: Iterable[LXeEntryRay],
    liquid: LXeDiffusionConfig,
    collision_orders: Sequence[int] = (1, 3, 5, 7),
    controls: ScalarCollisionConfig = ScalarCollisionConfig(),
    *,
    cubature: np.ndarray | None = None,
) -> ScalarCollisionExpansion:
    """Propagate common deterministic path samples to several collision orders."""

    liquid.validate()
    controls.validate()
    orders = tuple(sorted(set(int(value) for value in collision_orders)))
    if not orders or orders[0] <= 0:
        raise ValueError("collision orders must be positive")
    maximum_order = orders[-1]
    if controls.maximum_events < maximum_order:
        raise ValueError("maximum_events must not be below maximum collision order")

    entry_list = list(entries)
    dimension = 1 + 3 * controls.maximum_events
    if cubature is None:
        sample_count = 1 << controls.sample_power
        sobol = qmc.Sobol(d=dimension, scramble=False)
        cubature_values = sobol.random_base2(controls.sample_power)
    else:
        cubature_values = np.asarray(cubature, dtype=float)
        if (
            cubature_values.ndim != 2
            or cubature_values.shape[1] < dimension
            or len(cubature_values) == 0
        ):
            raise ValueError(
                "cubature must have shape (N, >= 1 + 3*maximum_events)"
            )
        if np.any(cubature_values < 0.0) or np.any(cubature_values > 1.0):
            raise ValueError("cubature values must lie in [0, 1]")
        sample_count = len(cubature_values)
    tiny = np.finfo(float).tiny
    position, direction, weight, entry_total = _select_weighted_entries(
        entry_list, cubature_values[:, 0]
    )
    position[:, 2] = controls.boundary_epsilon_mm
    collision_count = np.zeros(sample_count, dtype=np.int16)
    active = np.ones(sample_count, dtype=bool)

    exit_position: list[np.ndarray] = []
    exit_liquid_direction: list[np.ndarray] = []
    exit_gas_direction: list[np.ndarray] = []
    exit_weight: list[np.ndarray] = []
    exit_collision_count: list[np.ndarray] = []
    loss_weight: list[np.ndarray] = []
    loss_collision_count: list[np.ndarray] = []
    side_incident_weight: list[np.ndarray] = []
    side_reflected_weight: list[np.ndarray] = []
    side_collision_count: list[np.ndarray] = []
    bottom_incident_weight: list[np.ndarray] = []
    bottom_reflected_weight: list[np.ndarray] = []
    bottom_collision_count: list[np.ndarray] = []
    snapshot_positions: dict[int, list[np.ndarray]] = {
        order: [] for order in orders
    }
    snapshot_directions: dict[int, list[np.ndarray]] = {
        order: [] for order in orders
    }
    snapshot_weights: dict[int, list[np.ndarray]] = {
        order: [] for order in orders
    }

    scattering_length = liquid.rayleigh_length_mm
    absorption_length = liquid.absorption_length_mm
    epsilon = controls.boundary_epsilon_mm

    for event in range(controls.maximum_events):
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            break
        u_free = np.clip(
            cubature_values[active_indices, 1 + 3 * event],
            tiny,
            1.0 - tiny,
        )
        u_direction_1 = cubature_values[active_indices, 2 + 3 * event]
        u_direction_2 = cubature_values[active_indices, 3 + 3 * event]
        scatter_distance = -scattering_length * np.log1p(-u_free)
        boundary_distance, boundary_kind = _boundary_distance_and_kind(
            position[active_indices],
            direction[active_indices],
            liquid.radius_mm,
            liquid.depth_mm,
        )
        scatter = scatter_distance < boundary_distance
        travel = np.where(scatter, scatter_distance, boundary_distance)
        before_weight = weight[active_indices].copy()
        survival = np.exp(-travel / absorption_length)
        absorbed = before_weight * (1.0 - survival)
        if np.any(absorbed > 0.0):
            loss_weight.append(absorbed)
            loss_collision_count.append(collision_count[active_indices].copy())
        weight[active_indices] *= survival
        position[active_indices] += travel[:, np.newaxis] * direction[active_indices]

        scatter_indices = active_indices[scatter]
        if len(scatter_indices):
            local = np.flatnonzero(scatter)
            direction[scatter_indices] = scalar_rayleigh_directions(
                direction[scatter_indices],
                u_direction_1[local],
                u_direction_2[local],
            )
            collision_count[scatter_indices] += 1
            for order in orders:
                selected = scatter_indices[collision_count[scatter_indices] == order]
                if len(selected):
                    snapshot_positions[order].append(position[selected].copy())
                    snapshot_directions[order].append(direction[selected].copy())
                    snapshot_weights[order].append(weight[selected].copy())
                    if order == maximum_order:
                        active[selected] = False

        boundary_indices = active_indices[~scatter]
        if len(boundary_indices) == 0:
            continue
        local_boundary = np.flatnonzero(~scatter)
        kinds = boundary_kind[~scatter]

        top_local = np.flatnonzero(kinds == 0)
        if len(top_local):
            indices = boundary_indices[top_local]
            cos_incident = -direction[indices, 2]
            reflection = np.empty(len(indices), dtype=float)
            transmission = np.empty(len(indices), dtype=float)
            cos_transmitted = np.empty(len(indices), dtype=float)
            has_transmission = np.ones(len(indices), dtype=bool)
            for row, cosine in enumerate(cos_incident):
                reflected, transmitted, gas_cosine = fresnel_power(
                    liquid.n_lxe, liquid.n_gxe, float(cosine)
                )
                reflection[row] = float(np.mean(reflected))
                transmission[row] = float(np.mean(transmitted))
                if gas_cosine is None:
                    has_transmission[row] = False
                    cos_transmitted[row] = 0.0
                else:
                    cos_transmitted[row] = gas_cosine
            transmitted_weight = weight[indices] * transmission
            keep = transmitted_weight > controls.weight_cutoff
            if np.any(keep):
                kept_indices = indices[keep]
                eta = liquid.n_lxe / liquid.n_gxe
                gas_direction = np.column_stack(
                    [
                        eta * direction[kept_indices, 0],
                        eta * direction[kept_indices, 1],
                        cos_transmitted[keep],
                    ]
                )
                gas_direction /= np.linalg.norm(gas_direction, axis=1)[:, np.newaxis]
                exit_position.append(position[kept_indices, :2].copy())
                exit_liquid_direction.append(direction[kept_indices].copy())
                exit_gas_direction.append(gas_direction)
                exit_weight.append(transmitted_weight[keep])
                exit_collision_count.append(collision_count[kept_indices].copy())
            weight[indices] *= reflection
            direction[indices, 2] *= -1.0
            position[indices, 2] = epsilon
            inactive = (
                (weight[indices] <= controls.weight_cutoff)
                | (~has_transmission & (reflection <= controls.weight_cutoff))
            )
            active[indices[inactive]] = False

        side_local = np.flatnonzero(kinds == 1)
        if len(side_local):
            indices = boundary_indices[side_local]
            previous = weight[indices].copy()
            weight[indices] *= liquid.side_reflectivity
            side_incident_weight.append(previous)
            side_reflected_weight.append(weight[indices].copy())
            side_collision_count.append(collision_count[indices].copy())
            lost = previous - weight[indices]
            if np.any(lost > 0.0):
                loss_weight.append(lost)
                loss_collision_count.append(collision_count[indices].copy())
            radius = np.linalg.norm(position[indices, :2], axis=1)
            outward = position[indices, :2] / radius[:, np.newaxis]
            inward_normal = np.column_stack(
                [-outward[:, 0], -outward[:, 1], np.zeros(len(indices))]
            )
            direction[indices] = _lambertian_directions(
                inward_normal,
                u_direction_1[local_boundary[side_local]],
                u_direction_2[local_boundary[side_local]],
            )
            position[indices, :2] = outward * (liquid.radius_mm - epsilon)
            active[indices[weight[indices] <= controls.weight_cutoff]] = False

        bottom_local = np.flatnonzero(kinds == 2)
        if len(bottom_local):
            indices = boundary_indices[bottom_local]
            previous = weight[indices].copy()
            weight[indices] *= liquid.bottom_reflectivity
            bottom_incident_weight.append(previous)
            bottom_reflected_weight.append(weight[indices].copy())
            bottom_collision_count.append(collision_count[indices].copy())
            lost = previous - weight[indices]
            if np.any(lost > 0.0):
                loss_weight.append(lost)
                loss_collision_count.append(collision_count[indices].copy())
            inward_normal = np.tile(
                np.asarray([0.0, 0.0, -1.0]), (len(indices), 1)
            )
            direction[indices] = _lambertian_directions(
                inward_normal,
                u_direction_1[local_boundary[bottom_local]],
                u_direction_2[local_boundary[bottom_local]],
            )
            position[indices, 2] = liquid.depth_mm - epsilon
            active[indices[weight[indices] <= controls.weight_cutoff]] = False

    if exit_weight:
        all_exit_position = np.concatenate(exit_position)
        all_exit_liquid = np.concatenate(exit_liquid_direction)
        all_exit_gas = np.concatenate(exit_gas_direction)
        all_exit_weight = np.concatenate(exit_weight)
        all_exit_collisions = np.concatenate(exit_collision_count)
    else:
        all_exit_position = np.empty((0, 2), dtype=float)
        all_exit_liquid = np.empty((0, 3), dtype=float)
        all_exit_gas = np.empty((0, 3), dtype=float)
        all_exit_weight = np.empty(0, dtype=float)
        all_exit_collisions = np.empty(0, dtype=np.int16)
    if loss_weight:
        all_loss_weight = np.concatenate(loss_weight)
        all_loss_collisions = np.concatenate(loss_collision_count)
    else:
        all_loss_weight = np.empty(0, dtype=float)
        all_loss_collisions = np.empty(0, dtype=np.int16)

    def boundary_events(
        incident: list[np.ndarray],
        reflected: list[np.ndarray],
        collisions: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not incident:
            return (
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=np.int16),
            )
        return (
            np.concatenate(incident),
            np.concatenate(reflected),
            np.concatenate(collisions),
        )

    side_incident, side_reflected, side_collisions = boundary_events(
        side_incident_weight,
        side_reflected_weight,
        side_collision_count,
    )
    bottom_incident, bottom_reflected, bottom_collisions = boundary_events(
        bottom_incident_weight,
        bottom_reflected_weight,
        bottom_collision_count,
    )

    result: dict[int, ScalarCollisionSnapshot] = {}
    for order in orders:
        if not snapshot_positions[order]:
            source_position = np.empty((0, 3), dtype=float)
            source_direction = np.empty((0, 3), dtype=float)
            source_weight = np.empty(0, dtype=float)
        else:
            source_position = np.concatenate(snapshot_positions[order])
            source_direction = np.concatenate(snapshot_directions[order])
            source_weight = np.concatenate(snapshot_weights[order])
        exit_mask = all_exit_collisions < order
        loss_mask = all_loss_collisions < order
        explicit = ScalarExitEvents(
            position_xy_mm=all_exit_position[exit_mask],
            liquid_direction=all_exit_liquid[exit_mask],
            gas_direction=all_exit_gas[exit_mask],
            weight=all_exit_weight[exit_mask],
            preceding_collisions=all_exit_collisions[exit_mask],
        )
        explicit_total = float(explicit.weight.sum())
        source_total = float(source_weight.sum())
        loss_total = float(all_loss_weight[loss_mask].sum())
        side_mask = side_collisions < order
        bottom_mask = bottom_collisions < order
        side_incident_total = float(side_incident[side_mask].sum())
        side_reflected_total = float(side_reflected[side_mask].sum())
        bottom_incident_total = float(bottom_incident[bottom_mask].sum())
        bottom_reflected_total = float(bottom_reflected[bottom_mask].sum())
        accounted = explicit_total + source_total + loss_total
        result[order] = ScalarCollisionSnapshot(
            collision_order=order,
            source_xyz_mm=source_position,
            source_direction=source_direction,
            source_weight=source_weight,
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
                "side_boundary_absorbed": (
                    side_incident_total - side_reflected_total
                ),
                "bottom_boundary_incident": bottom_incident_total,
                "bottom_boundary_reflected": bottom_reflected_total,
                "bottom_boundary_absorbed": (
                    bottom_incident_total - bottom_reflected_total
                ),
            },
        )
    unresolved = float(weight[active].sum())
    return ScalarCollisionExpansion(
        snapshots=result,
        sample_count=sample_count,
        entry_weight=entry_total,
        metadata={
            "sample_power": controls.sample_power,
            "maximum_events": controls.maximum_events,
            "collision_orders": list(orders),
            "active_weight_after_event_limit": unresolved,
        },
    )
