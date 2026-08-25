#!/usr/bin/env python3
"""Precompute the position-resolved LXe return response in modal form.

The liquid diffusion problem is linear and cylindrically symmetric.  Instead
of reevaluating its Bessel--Fourier Green function for every source and every
LXe/GXe renewal cycle, this module tabulates the response of each radial
position, relative direction-azimuth, and direction-cosine phase bin.  The
absolute entry-position azimuth remains an exact Fourier rotation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import qmc

from .numerics import gauss_interval, json_compatible
from .lxe_cylinder_green import (
    LXeCylinderGreen,
    LXeEntryRay,
    ModeTruncation,
)
from .lxe_diffusion_return import LXeDiffusionConfig
from .lxe_scalar_collision import (
    ScalarCollisionConfig,
    propagate_scalar_collision_orders,
)
from .phase_space_grid import LXePhaseSpaceGrid


SCHEMA_VERSION = 2
DEFAULT_AUDIT_NAMES = (
    "first_scatter_weight",
    "ballistic_absorption",
    "ballistic_boundary_reach",
    "accounted_before_diffusion",
)
LOW_ORDER_AUDIT_NAMES = (
    "first_scatter_weight",
    "ballistic_absorption",
    "ballistic_boundary_reach",
    "accounted_before_diffusion",
    "explicit_surface_escape",
    "side_boundary_incident",
    "side_boundary_reflected",
    "side_boundary_absorbed",
    "bottom_boundary_incident",
    "bottom_boundary_reflected",
    "bottom_boundary_absorbed",
)
# Compatibility for callers that only need to interpret older operators.
AUDIT_NAMES = DEFAULT_AUDIT_NAMES


def _canonical_json(value: object) -> str:
    return json.dumps(
        json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def response_operator_hash(
    lxe_config: LXeDiffusionConfig,
    truncation: ModeTruncation,
    phase_grid: LXePhaseSpaceGrid,
    surface_radial_order: int,
    first_scatter_order: int,
    explicit_collision_order: int = 0,
    collision_sample_power: int = 0,
    collision_maximum_events: int = 0,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "lxe_config": asdict(lxe_config),
        "mode_truncation": asdict(truncation),
        "phase_space_grid": asdict(phase_grid),
        "surface_radial_order": surface_radial_order,
        "first_scatter_order": first_scatter_order,
        "explicit_collision_order": explicit_collision_order,
        "collision_sample_power": collision_sample_power,
        "collision_maximum_events": collision_maximum_events,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def surface_radial_quadrature(
    radius_mm: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return area-uniform radial nodes and complete-ring area weights."""

    if order <= 0:
        raise ValueError("surface radial order must be positive")
    u, weights = gauss_interval(order, 0.0, 1.0)
    return radius_mm * np.sqrt(u), math.pi * radius_mm**2 * weights


@dataclass
class LXeResponseOperator:
    """Rotation-covariant phase-space to liquid-surface response."""

    coefficients: np.ndarray
    expected_return: np.ndarray
    audit_values: np.ndarray
    surface_radius_mm: np.ndarray
    surface_ring_area_mm2: np.ndarray
    phase_grid: LXePhaseSpaceGrid
    metadata: dict

    @property
    def audit_names(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("audit_names", DEFAULT_AUDIT_NAMES))

    def validate(self) -> None:
        self.phase_grid.validate()
        if not self.phase_grid.direction_phi_relative_to_position:
            raise ValueError(
                "the compact LXe response operator requires direction azimuth "
                "relative to the entry-position radial direction"
            )
        expected_shape = (
            self.phase_grid.direction_phi_bins,
            self.phase_grid.position_radial_bins,
            self.phase_grid.direction_mu_bins,
        )
        maximum_order = int(
            self.metadata["mode_truncation"]["azimuthal_maximum"]
        )
        radial_order = int(self.metadata["surface_radial_order"])
        if self.coefficients.shape != (
            *expected_shape,
            maximum_order + 1,
            radial_order,
        ):
            raise ValueError("LXe response coefficient array has wrong shape")
        if self.expected_return.shape != expected_shape:
            raise ValueError("LXe expected-return array has wrong shape")
        if self.audit_values.shape != (*expected_shape, len(self.audit_names)):
            raise ValueError("LXe response audit array has wrong shape")
        if self.surface_radius_mm.shape != (radial_order,):
            raise ValueError("LXe response radial nodes have wrong shape")
        if self.surface_ring_area_mm2.shape != (radial_order,):
            raise ValueError("LXe response radial areas have wrong shape")

    def save(self, directory: Path) -> None:
        self.validate()
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "coefficients.npy", self.coefficients)
        np.save(directory / "expected_return.npy", self.expected_return)
        np.save(directory / "audit_values.npy", self.audit_values)
        np.save(directory / "surface_radius_mm.npy", self.surface_radius_mm)
        np.save(
            directory / "surface_ring_area_mm2.npy",
            self.surface_ring_area_mm2,
        )
        (directory / "metadata.json").write_text(
            json.dumps(
                json_compatible(self.metadata),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        expected_hash: str | None = None,
        mmap_mode: str | None = "r",
    ) -> "LXeResponseOperator":
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        if (
            expected_hash is not None
            and metadata["response_operator_hash"] != expected_hash
        ):
            raise ValueError("LXe response operator configuration hash mismatch")
        operator = cls(
            coefficients=np.load(
                directory / "coefficients.npy", mmap_mode=mmap_mode
            ),
            expected_return=np.load(
                directory / "expected_return.npy", mmap_mode=mmap_mode
            ),
            audit_values=np.load(
                directory / "audit_values.npy", mmap_mode=mmap_mode
            ),
            surface_radius_mm=np.load(
                directory / "surface_radius_mm.npy", mmap_mode=mmap_mode
            ),
            surface_ring_area_mm2=np.load(
                directory / "surface_ring_area_mm2.npy", mmap_mode=mmap_mode
            ),
            phase_grid=LXePhaseSpaceGrid(**metadata["phase_space_grid"]),
            metadata=metadata,
        )
        operator.validate()
        return operator

    def apply(
        self,
        phase_weights: np.ndarray,
        *,
        surface_phi_bins: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Apply the precomputed liquid response to a source batch."""

        self.validate()
        if surface_phi_bins <= 0:
            raise ValueError("surface_phi_bins must be positive")
        values = np.asarray(phase_weights, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.phase_grid.size:
            raise ValueError("phase weights do not match the LXe response grid")
        if np.any(values < 0.0) or np.any(~np.isfinite(values)):
            raise ValueError("phase weights must be finite and non-negative")

        grid = self.phase_grid
        batch = values.shape[0]
        phase = values.reshape(
            batch,
            grid.position_radial_bins,
            grid.position_phi_bins,
            grid.direction_mu_bins,
            grid.direction_phi_bins,
        )
        maximum_order = self.coefficients.shape[3] - 1
        orders = np.arange(maximum_order + 1, dtype=float)
        position_phi = grid.position_phi_values
        rotation = np.exp(-1j * np.outer(orders, position_phi))
        phase_modes = np.einsum(
            "brpud,mp->brudm",
            phase,
            rotation,
            optimize=True,
        )
        modal_surface = np.einsum(
            "brudm,drums->bms",
            phase_modes,
            self.coefficients,
            optimize=True,
        )
        surface_phi = (
            2.0
            * math.pi
            * (np.arange(surface_phi_bins, dtype=float) + 0.5)
            / surface_phi_bins
        )
        surface_basis = np.exp(1j * np.outer(orders, surface_phi))
        density = np.einsum(
            "bms,mp->bsp",
            modal_surface,
            surface_basis,
            optimize=True,
        ).real
        raw_weights = (
            density
            * self.surface_ring_area_mm2[np.newaxis, :, np.newaxis]
            / surface_phi_bins
        ).reshape(batch, -1)

        phase_without_position_phi = phase.sum(axis=2)
        expected = np.einsum(
            "brud,dru->b",
            phase_without_position_phi,
            self.expected_return,
            optimize=True,
        )
        audit = np.einsum(
            "brud,drux->bx",
            phase_without_position_phi,
            self.audit_values,
            optimize=True,
        )
        return raw_weights, {
            "surface_return_expected": expected,
            "surface_return_raw": raw_weights.sum(axis=1),
            "negative_surface_weight": -np.minimum(raw_weights, 0.0).sum(axis=1),
            **{
                name: audit[:, index]
                for index, name in enumerate(self.audit_names)
            },
        }

    def validate_surface_quadrature(
        self,
        surface_xy_mm: np.ndarray,
        surface_area_mm2: np.ndarray,
    ) -> int:
        """Validate a disk product quadrature and return its phi-bin count."""

        points = np.asarray(surface_xy_mm, dtype=float)
        areas = np.asarray(surface_area_mm2, dtype=float)
        radial_order = len(self.surface_radius_mm)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or areas.shape != (len(points),)
            or len(points) % radial_order
        ):
            raise ValueError("surface quadrature does not match the LXe operator")
        phi_bins = len(points) // radial_order
        radii = np.linalg.norm(points, axis=1).reshape(radial_order, phi_bins)
        area_grid = areas.reshape(radial_order, phi_bins)
        if not np.allclose(
            radii,
            self.surface_radius_mm[:, np.newaxis],
            rtol=0.0,
            atol=2.0e-10,
        ):
            raise ValueError("surface radial nodes differ from the LXe operator")
        if not np.allclose(
            area_grid,
            self.surface_ring_area_mm2[:, np.newaxis] / phi_bins,
            rtol=0.0,
            atol=2.0e-8,
        ):
            raise ValueError("surface area weights differ from the LXe operator")
        return phi_bins


_WORKER_CONTEXT: tuple | None = None


def _explicit_exit_fourier_coefficients(
    position_xy_mm: np.ndarray,
    weight: np.ndarray,
    surface_radius_mm: np.ndarray,
    surface_ring_area_mm2: np.ndarray,
    maximum_order: int,
) -> np.ndarray:
    """Project early liquid-surface exits onto the surface modal grid.

    The low-order collision expansion retains the exact exit position and
    transmitted weight.  The existing GXe surface kernel uses the conditional
    P1/Fresnel escape-angle basis, so only the spatial surface current is
    projected here.  Linear interpolation in ``r**2`` preserves total current
    and avoids a nearest-ring discontinuity.
    """

    position = np.asarray(position_xy_mm, dtype=float)
    values = np.asarray(weight, dtype=float)
    radius_nodes = np.asarray(surface_radius_mm, dtype=float)
    ring_area = np.asarray(surface_ring_area_mm2, dtype=float)
    result = np.zeros(
        (maximum_order + 1, len(radius_nodes)), dtype=np.complex128
    )
    if len(values) == 0:
        return result
    if position.shape != (len(values), 2):
        raise ValueError("explicit exit positions do not align with weights")
    radius = np.linalg.norm(position, axis=1)
    phi = np.arctan2(position[:, 1], position[:, 0])
    node_u = radius_nodes * radius_nodes
    value_u = radius * radius
    upper = np.searchsorted(node_u, value_u, side="right")
    upper = np.clip(upper, 1, len(node_u) - 1)
    lower = upper - 1
    denominator = node_u[upper] - node_u[lower]
    upper_weight = np.divide(
        value_u - node_u[lower],
        denominator,
        out=np.zeros_like(value_u),
        where=denominator > 0.0,
    )
    below = value_u <= node_u[0]
    above = value_u >= node_u[-1]
    lower[below] = 0
    upper[below] = 0
    upper_weight[below] = 0.0
    lower[above] = len(node_u) - 1
    upper[above] = len(node_u) - 1
    upper_weight[above] = 0.0
    for indices, radial_weight in (
        (lower, 1.0 - upper_weight),
        (upper, upper_weight),
    ):
        contribution = values * radial_weight
        for order in range(maximum_order + 1):
            multiplicity = 1.0 if order == 0 else 2.0
            modal = (
                multiplicity
                * contribution
                * np.exp(-1j * order * phi)
                / ring_area[indices]
            )
            np.add.at(result[order], indices, modal)
    return result


def _operator_row(
    task: tuple[int, int, int],
    green: LXeCylinderGreen,
    phase_grid: LXePhaseSpaceGrid,
    surface_radius: np.ndarray,
    surface_area: np.ndarray,
    first_scatter_order: int,
    explicit_collision_order: int,
    collision_controls: ScalarCollisionConfig | None,
    collision_cubature: np.ndarray | None,
) -> tuple[int, int, int, np.ndarray, float, np.ndarray]:
    direction_phi_index, radial_index, mu_index = task
    radius = float(phase_grid.position_radius_values[radial_index])
    mu = float(phase_grid.direction_mu_values[mu_index])
    phi = float(phase_grid.direction_phi_values[direction_phi_index])
    transverse = math.sqrt(max(0.0, 1.0 - mu * mu))
    entry = LXeEntryRay(
        entry_xy_mm=np.asarray([radius, 0.0]),
        liquid_direction=np.asarray(
            [transverse * math.cos(phi), transverse * math.sin(phi), mu]
        ),
        weight=1.0,
    )
    if explicit_collision_order <= 0:
        coefficients, audit = green.return_fourier_coefficients(
            entry,
            surface_radius,
            first_scatter_order=first_scatter_order,
        )
        audit_names = DEFAULT_AUDIT_NAMES
        expected_return = float(audit["surface_return_expected"])
    else:
        if collision_controls is None or collision_cubature is None:
            raise RuntimeError("low-order collision controls were not initialized")
        expansion = propagate_scalar_collision_orders(
            [entry],
            green.config,
            collision_orders=(explicit_collision_order,),
            controls=collision_controls,
            cubature=collision_cubature,
        )
        snapshot = expansion.snapshots[explicit_collision_order]
        unresolved = float(snapshot.audit["pre_switch_unresolved"])
        if abs(unresolved) > 3.0e-9:
            raise RuntimeError(
                "low-order collision expansion did not close before the "
                f"event limit: unresolved weight {unresolved}"
            )
        diffusion_coefficients, diffusion_return = (
            green.surface_fourier_coefficients_from_sources(
                snapshot.source_xyz_mm,
                snapshot.source_weight,
                surface_radius,
            )
        )
        explicit_coefficients = _explicit_exit_fourier_coefficients(
            snapshot.explicit_exits.position_xy_mm,
            snapshot.explicit_exits.weight,
            surface_radius,
            surface_area,
            green.truncation.azimuthal_maximum,
        )
        coefficients = diffusion_coefficients + explicit_coefficients
        explicit_return = float(snapshot.explicit_exits.weight.sum())
        expected_return = diffusion_return + explicit_return
        if expected_return < 0.0 or expected_return > 1.0 + 3.0e-9:
            raise RuntimeError(
                f"low-order LXe return lies outside [0,1]: {expected_return}"
            )
        low = snapshot.audit
        audit = {
            "first_scatter_weight": low["diffusion_source_weight"],
            "ballistic_absorption": low["pre_switch_loss"],
            "ballistic_boundary_reach": (
                low["side_boundary_incident"]
                + low["bottom_boundary_incident"]
            ),
            "accounted_before_diffusion": low["pre_switch_accounted"],
            "explicit_surface_escape": explicit_return,
            "side_boundary_incident": low["side_boundary_incident"],
            "side_boundary_reflected": low["side_boundary_reflected"],
            "side_boundary_absorbed": low["side_boundary_absorbed"],
            "bottom_boundary_incident": low["bottom_boundary_incident"],
            "bottom_boundary_reflected": low["bottom_boundary_reflected"],
            "bottom_boundary_absorbed": low["bottom_boundary_absorbed"],
        }
        audit_names = LOW_ORDER_AUDIT_NAMES
    audit_values = np.asarray([audit[name] for name in audit_names])
    return (
        direction_phi_index,
        radial_index,
        mu_index,
        coefficients,
        expected_return,
        audit_values,
    )


def _operator_row_worker(task: tuple[int, int, int]):
    if _WORKER_CONTEXT is None:
        raise RuntimeError("LXe response worker context was not initialized")
    return _operator_row(task, *_WORKER_CONTEXT)


def build_lxe_response_operator(
    green: LXeCylinderGreen,
    phase_grid: LXePhaseSpaceGrid,
    *,
    surface_radial_order: int,
    first_scatter_order: int,
    explicit_collision_order: int = 0,
    collision_sample_power: int = 12,
    collision_maximum_events: int = 64,
    processes: int = 1,
    coefficient_dtype: str = "complex128",
    progress_interval_s: float = 30.0,
) -> LXeResponseOperator:
    """Build a reusable phase-space to surface-return operator."""

    phase_grid.validate()
    if not phase_grid.direction_phi_relative_to_position:
        raise ValueError(
            "precomputation requires a radial-relative direction-phi grid"
        )
    if first_scatter_order < 2:
        raise ValueError("first-scatter order must be at least two")
    if explicit_collision_order < 0:
        raise ValueError("explicit collision order must be non-negative")
    if explicit_collision_order > collision_maximum_events:
        raise ValueError(
            "collision maximum events must not be below the explicit order"
        )
    if processes <= 0:
        raise ValueError("processes must be positive")
    if coefficient_dtype not in {"complex64", "complex128"}:
        raise ValueError("coefficient dtype must be complex64 or complex128")

    surface_radius, surface_area = surface_radial_quadrature(
        phase_grid.radius_mm, surface_radial_order
    )
    collision_controls = None
    collision_cubature = None
    audit_names = DEFAULT_AUDIT_NAMES
    if explicit_collision_order:
        collision_controls = ScalarCollisionConfig(
            sample_power=collision_sample_power,
            maximum_events=collision_maximum_events,
        )
        collision_controls.validate()
        collision_cubature = qmc.Sobol(
            d=1 + 3 * collision_maximum_events,
            scramble=False,
        ).random_base2(collision_sample_power)
        audit_names = LOW_ORDER_AUDIT_NAMES
    maximum_order = green.truncation.azimuthal_maximum
    shape = (
        phase_grid.direction_phi_bins,
        phase_grid.position_radial_bins,
        phase_grid.direction_mu_bins,
    )
    coefficients = np.empty(
        (*shape, maximum_order + 1, surface_radial_order),
        dtype=np.dtype(coefficient_dtype),
    )
    expected_return = np.empty(shape, dtype=np.float64)
    audit_values = np.empty((*shape, len(audit_names)), dtype=np.float64)
    tasks = [
        (direction_phi, radial, mu)
        for direction_phi in range(phase_grid.direction_phi_bins)
        for radial in range(phase_grid.position_radial_bins)
        for mu in range(phase_grid.direction_mu_bins)
    ]

    # Populate roots before forking so workers share the read-only arrays.
    for order in range(maximum_order + 1):
        green.roots(order)
        green.radial_mode_norms(order)

    if processes == 1:
        rows = (
            _operator_row(
                task,
                green,
                phase_grid,
                surface_radius,
                surface_area,
                first_scatter_order,
                explicit_collision_order,
                collision_controls,
                collision_cubature,
            )
            for task in tasks
        )
        pool = None
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("parallel LXe response build requires fork")
        global _WORKER_CONTEXT
        _WORKER_CONTEXT = (
            green,
            phase_grid,
            surface_radius,
            surface_area,
            first_scatter_order,
            explicit_collision_order,
            collision_controls,
            collision_cubature,
        )
        pool = multiprocessing.get_context("fork").Pool(processes=processes)
        rows = pool.imap_unordered(
            _operator_row_worker,
            tasks,
            chunksize=max(1, min(16, len(tasks) // (processes * 32))),
        )

    started = time.monotonic()
    last_report = started
    completed = 0
    try:
        for direction_phi, radial, mu, modes, expected, audit in rows:
            coefficients[direction_phi, radial, mu] = modes
            expected_return[direction_phi, radial, mu] = expected
            audit_values[direction_phi, radial, mu] = audit
            completed += 1
            now = time.monotonic()
            if completed == len(tasks) or now - last_report >= progress_interval_s:
                print(
                    f"LXe response states: {completed}/{len(tasks)}",
                    flush=True,
                )
                last_report = now
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _WORKER_CONTEXT = None

    digest = response_operator_hash(
        green.config,
        green.truncation,
        phase_grid,
        surface_radial_order,
        first_scatter_order,
        explicit_collision_order,
        collision_sample_power,
        collision_maximum_events,
    )
    operator = LXeResponseOperator(
        coefficients=coefficients,
        expected_return=expected_return,
        audit_values=audit_values,
        surface_radius_mm=surface_radius,
        surface_ring_area_mm2=surface_area,
        phase_grid=phase_grid,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "response_operator_hash": digest,
            "lxe_config": asdict(green.config),
            "mode_truncation": asdict(green.truncation),
            "phase_space_grid": asdict(phase_grid),
            "surface_radial_order": surface_radial_order,
            "first_scatter_order": first_scatter_order,
            "explicit_collision_order": explicit_collision_order,
            "collision_sample_power": (
                collision_sample_power if explicit_collision_order else 0
            ),
            "collision_sample_count": (
                1 << collision_sample_power if explicit_collision_order else 0
            ),
            "collision_maximum_events": (
                collision_maximum_events if explicit_collision_order else 0
            ),
            "early_escape_angular_projection": (
                "conditional P1/Fresnel surface basis"
                if explicit_collision_order
                else None
            ),
            "audit_names": list(audit_names),
            "coefficient_dtype": coefficient_dtype,
            "coefficient_count": int(coefficients.size),
            "coefficient_bytes": int(coefficients.nbytes),
            "state_count": len(tasks),
            "build_seconds": time.monotonic() - started,
            "processes": processes,
        },
    )
    operator.validate()
    return operator


def _load_phase_grid(path: Path) -> LXePhaseSpaceGrid:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return LXePhaseSpaceGrid(**metadata["phase_space_grid"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--phase-operator-directory", type=Path)
    result.add_argument(
        "--lxe-field-cage-radius-mm",
        "--radius-mm",
        dest="lxe_field_cage_radius_mm",
        type=float,
        default=1000.0,
        help="inner field-cage radius bounding the LXe diffusion domain",
    )
    result.add_argument("--position-radial-bins", type=int, default=48)
    result.add_argument("--position-phi-bins", type=int, default=72)
    result.add_argument("--direction-mu-bins", type=int, default=4)
    result.add_argument("--direction-phi-bins", type=int, default=16)
    result.add_argument(
        "--phase-deposition",
        choices=("nearest", "multilinear"),
        default="nearest",
        help=(
            "build the response at cell centres or at conservative "
            "multilinear interpolation nodes"
        ),
    )
    result.add_argument(
        "--radial-node-spacing",
        choices=("linear", "chebyshev"),
        default="chebyshev",
    )
    result.add_argument(
        "--direction-mu-minimum",
        type=float,
        help=(
            "minimum liquid direction cosine represented by multilinear "
            "nodes; defaults to the GXe-to-LXe Snell limit"
        ),
    )
    result.add_argument("--surface-radial-order", type=int, default=40)
    result.add_argument("--first-scatter-order", type=int, default=8)
    result.add_argument(
        "--explicit-collision-order",
        type=int,
        default=7,
        help=(
            "number of Rayleigh collisions propagated explicitly, including "
            "side/bottom boundary competition, before the diffusion switch; "
            "zero selects the first-flight-only operator"
        ),
    )
    result.add_argument(
        "--collision-sample-power",
        type=int,
        default=12,
        help="Sobol cubature sample count is 2**this value",
    )
    result.add_argument(
        "--collision-maximum-events",
        type=int,
        default=64,
        help=(
            "maximum free-flight or boundary events before declaring "
            "unresolved pre-switch weight"
        ),
    )
    result.add_argument("--processes", type=int, default=24)
    result.add_argument(
        "--coefficient-dtype",
        choices=("complex64", "complex128"),
        default="complex128",
    )
    result.add_argument("--lxe-depth-mm", type=float, default=2000.0)
    result.add_argument(
        "--rayleigh-length-mm", type=float, default=341.51442280354416
    )
    result.add_argument("--lxe-absorption-length-mm", type=float, default=70000.0)
    result.add_argument("--lxe-side-reflectivity", type=float, default=0.95)
    result.add_argument("--lxe-bottom-reflectivity", type=float, default=0.0)
    result.add_argument("--n-lxe", type=float, default=1.6829)
    result.add_argument("--n-gxe", type=float, default=1.000702)
    result.add_argument("--azimuthal-maximum", type=int, default=32)
    result.add_argument("--radial-modes", type=int, default=120)
    result.add_argument("--root-samples", type=int, default=120)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.phase_operator_directory is None:
        direction_mu_minimum = arguments.direction_mu_minimum
        if direction_mu_minimum is None:
            direction_mu_minimum = (
                math.sqrt(
                    max(
                        0.0,
                        1.0 - (arguments.n_gxe / arguments.n_lxe) ** 2,
                    )
                )
                if arguments.phase_deposition == "multilinear"
                else 0.0
            )
        phase_grid = LXePhaseSpaceGrid(
            radius_mm=arguments.lxe_field_cage_radius_mm,
            position_radial_bins=arguments.position_radial_bins,
            position_phi_bins=arguments.position_phi_bins,
            direction_mu_bins=arguments.direction_mu_bins,
            direction_phi_bins=arguments.direction_phi_bins,
            direction_phi_relative_to_position=True,
            deposition=arguments.phase_deposition,
            radial_node_spacing=arguments.radial_node_spacing,
            direction_mu_minimum=direction_mu_minimum,
        )
    else:
        phase_grid = _load_phase_grid(arguments.phase_operator_directory)
    lxe_config = LXeDiffusionConfig(
        radius_mm=phase_grid.radius_mm,
        depth_mm=arguments.lxe_depth_mm,
        rayleigh_length_mm=arguments.rayleigh_length_mm,
        absorption_length_mm=arguments.lxe_absorption_length_mm,
        n_lxe=arguments.n_lxe,
        n_gxe=arguments.n_gxe,
        side_reflectivity=arguments.lxe_side_reflectivity,
        bottom_reflectivity=arguments.lxe_bottom_reflectivity,
    )
    truncation = ModeTruncation(
        azimuthal_maximum=arguments.azimuthal_maximum,
        radial_modes=arguments.radial_modes,
        samples_per_expected_root=arguments.root_samples,
    )
    green = LXeCylinderGreen(lxe_config, truncation, mode_workers=1)
    operator = build_lxe_response_operator(
        green,
        phase_grid,
        surface_radial_order=arguments.surface_radial_order,
        first_scatter_order=arguments.first_scatter_order,
        explicit_collision_order=arguments.explicit_collision_order,
        collision_sample_power=arguments.collision_sample_power,
        collision_maximum_events=arguments.collision_maximum_events,
        processes=arguments.processes,
        coefficient_dtype=arguments.coefficient_dtype,
    )
    operator.save(arguments.output)
    print(json.dumps(operator.metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
