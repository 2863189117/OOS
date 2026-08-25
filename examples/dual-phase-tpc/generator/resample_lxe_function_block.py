#!/usr/bin/env python3
"""Resample a factorized LXe block onto a finer surface-azimuth basis.

The expensive LXe collision/diffusion calculation is stored as Fourier--Bessel
coefficients and is independent of the azimuthal sampling used to couple the
liquid surface to a particular geometry.  This utility preserves that modal
payload exactly and rebuilds only the geometry-dependent egress basis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from generator.build_lxe_function_block import (
        diffuse_escape_quadrature,
        egress_basis,
        geometry_lxe_contract,
        sha256,
    )
    from generator.reference.lxe_diffusion_return import LXeDiffusionConfig
    from generator.write_intrinsic_lxe_block import write_factorized_block
else:
    from .build_lxe_function_block import (
        diffuse_escape_quadrature,
        egress_basis,
        geometry_lxe_contract,
        sha256,
    )
    from .reference.lxe_diffusion_return import LXeDiffusionConfig
    from .write_intrinsic_lxe_block import write_factorized_block


@dataclass(frozen=True)
class FactorizedLXeBlock:
    """In-memory portion of a block needed for egress resampling."""

    coefficients: np.ndarray
    expected_return: np.ndarray
    audit_values: np.ndarray
    surface_radius_mm: np.ndarray
    surface_ring_area_mm2: np.ndarray
    angular_weight: np.ndarray
    phase_grid: dict[str, Any]
    audit_names: list[str]
    producer: dict[str, Any]
    surface_phi_bins: int | None
    surface_point_count: int


@dataclass(frozen=True)
class RaggedSurface:
    """Geometry-owned ragged LXe surface quadrature."""

    ring_offsets: np.ndarray
    phi_rad: np.ndarray
    area_mm2: np.ndarray
    surface_element: np.ndarray
    barycentric: np.ndarray
    side: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    ring_phi_count: np.ndarray
    geometry_discretization: dict[str, Any]


def read_json(handle: h5py.File, path: str) -> Any:
    """Decode one canonical byte-array JSON dataset."""

    try:
        payload = bytes(np.asarray(handle[path], dtype=np.uint8))
    except KeyError as error:
        raise ValueError(f"input block is missing {path}") from error
    try:
        return json.loads(payload.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"input block contains invalid JSON at {path}") from error


def _required_positive_integer(metadata: dict[str, Any], name: str) -> int:
    try:
        value = int(metadata[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"input metadata requires a positive {name}") from error
    if value <= 0:
        raise ValueError(f"input metadata requires a positive {name}")
    return value


def load_factorized_block(path: Path) -> FactorizedLXeBlock:
    """Load and cross-check the immutable modal payload of ``path``."""

    try:
        with h5py.File(path, "r") as handle:
            real = np.asarray(handle["/function/coefficients_real"], dtype=float)
            imaginary = np.asarray(
                handle["/function/coefficients_imag"], dtype=float
            )
            if real.shape != imaginary.shape:
                raise ValueError("real and imaginary coefficient shapes differ")
            coefficients = real + 1j * imaginary
            expected_return = np.asarray(
                handle["/function/expected_return"], dtype=float
            )
            audit_values = np.asarray(
                handle["/function/audit_values"], dtype=float
            )
            surface_radius_mm = np.asarray(
                handle["/function/surface_radius_mm"], dtype=float
            )
            surface_ring_area_mm2 = np.asarray(
                handle["/function/surface_ring_area_mm2"], dtype=float
            )
            angular_weight = np.asarray(
                handle["/function/angular_weight"], dtype=float
            )
            egress_count = int(
                handle["/nonlocal/egress/surface_element"].shape[0]
            )
            phase_grid = read_json(handle, "/metadata/phase_grid_json")
            audit_names = read_json(handle, "/metadata/audit_names_json")
            producer = read_json(handle, "/metadata/generator_json")
    except (KeyError, OSError) as error:
        raise ValueError(f"cannot read factorized LXe block {path}") from error

    if coefficients.ndim not in (5, 6):
        raise ValueError("factorized coefficients must be rank five or six")
    if not isinstance(phase_grid, dict) or not isinstance(producer, dict):
        raise ValueError("phase-grid and generator metadata must be JSON objects")
    if not isinstance(audit_names, list) or not all(
        isinstance(name, str) for name in audit_names
    ):
        raise ValueError("audit_names_json must be a list of strings")
    denominator = int(angular_weight.size)
    if denominator == 0 or egress_count % denominator:
        raise ValueError("input egress basis does not match its modal quadrature")
    if coefficients.ndim == 6:
        if coefficients.shape[-1] != denominator:
            raise ValueError(
                "joint coefficient angle axis does not match angular_weight"
            )
        if producer.get("coefficient_layout") != "joint_surface_angle_v1":
            raise ValueError(
                "rank-six input requires coefficient_layout=joint_surface_angle_v1"
            )
    surface_point_count = egress_count // denominator
    schema = producer.get("schema")
    if schema == "oos.nonlocal.function.v1":
        radial_denominator = int(surface_radius_mm.size)
        if radial_denominator == 0 or surface_point_count % radial_denominator:
            raise ValueError(
                "input v1 egress basis does not match its modal quadrature"
            )
        surface_phi_bins: int | None = (
            surface_point_count // radial_denominator
        )
    elif schema == "oos.nonlocal.function.v2":
        if producer.get("surface_layout") != "ragged_ring_v1":
            raise ValueError("unsupported v2 surface layout")
        surface_phi_bins = None
    else:
        raise ValueError(f"unsupported input function schema {schema!r}")
    for name, actual in (
        ("surface_phi_bins", surface_phi_bins),
        ("angular_count", angular_weight.size),
        ("egress_count", egress_count),
        ("surface_point_count", surface_point_count),
    ):
        if name in producer and actual is not None and int(producer[name]) != actual:
            raise ValueError(f"input metadata {name} disagrees with its arrays")
    if "lxe_config" not in producer or not isinstance(
        producer["lxe_config"], dict
    ):
        raise ValueError("input metadata is missing lxe_config")
    for name in ("radius_mm", "depth_mm"):
        try:
            value = float(producer["lxe_config"][name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"input lxe_config is missing {name}") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"input lxe_config has invalid {name}")
    _required_positive_integer(producer, "return_mu_order")
    _required_positive_integer(producer, "return_direction_phi")

    return FactorizedLXeBlock(
        coefficients=coefficients,
        expected_return=expected_return,
        audit_values=audit_values,
        surface_radius_mm=surface_radius_mm,
        surface_ring_area_mm2=surface_ring_area_mm2,
        angular_weight=angular_weight,
        phase_grid=phase_grid,
        audit_names=audit_names,
        producer=producer,
        surface_phi_bins=surface_phi_bins,
        surface_point_count=surface_point_count,
    )


def _geometry_ragged_surface(
    path: Path,
    surface_radius_mm: np.ndarray,
    surface_ring_area_mm2: np.ndarray,
    *,
    surface_id: int,
    gxe_domain_id: int,
    maximum_order: int,
    tolerance_mm: float,
) -> RaggedSurface:
    """Read the complete egress quadrature from analytic geometry.

    No point, area, or surface-element identifier is synthesized here.  The
    modal radii are used only to validate and group geometry-owned elements.
    """

    try:
        with h5py.File(path, "r") as handle:
            primitive_surface = np.asarray(
                handle["/analytic/surface_id"], dtype=np.uint32
            )
            primitive_minus = np.asarray(
                handle["/analytic/minus_domain_id"], dtype=np.int32
            )
            primitive_plus = np.asarray(
                handle["/analytic/plus_domain_id"], dtype=np.int32
            )
            candidates = np.flatnonzero(
                (primitive_surface == np.uint32(surface_id))
                & (
                    (primitive_minus == gxe_domain_id)
                    ^ (primitive_plus == gxe_domain_id)
                )
            )
            if candidates.size != 1:
                raise ValueError(
                    "geometry must contain exactly one analytic LXe surface"
                )
            primitive = int(candidates[0])
            primitive_center = np.asarray(
                handle["/analytic/center_mm"], dtype=float
            )[primitive]
            primitive_axes = np.asarray(
                [
                    np.asarray(handle["/analytic/axis_x"], dtype=float)[
                        primitive
                    ],
                    np.asarray(handle["/analytic/axis_y"], dtype=float)[
                        primitive
                    ],
                    np.asarray(handle["/analytic/axis_z"], dtype=float)[
                        primitive
                    ],
                ]
            )
            owners = np.asarray(
                handle["/analytic/elements/primitive_index"], dtype=np.uint32
            )
            selected = np.flatnonzero(owners == primitive)
            if selected.size == 0:
                raise ValueError("analytic LXe surface has no quadrature points")
            center = np.asarray(
                handle["/analytic/elements/center_mm"], dtype=float
            )[selected]
            area = np.asarray(
                handle["/analytic/elements/area_mm2"], dtype=float
            )[selected]
            element = np.asarray(
                handle["/analytic/elements/surface_element"], dtype=np.uint64
            )[selected]
            element_normal = np.asarray(
                handle["/analytic/elements/normal"], dtype=float
            )[selected]
            tangent_value = np.asarray(
                handle["/analytic/axis_x"], dtype=float
            )[primitive]
            try:
                identity = read_json(
                    handle, "/metadata/analytic_generator_json"
                )
                discretization = dict(
                    identity.get(
                        "analytic_discretization",
                        identity.get("legacy_discretization", {}),
                    )
                )
                # The archived legacy builder stores these controls in a
                # nested object but keeps the newer LXe-egress controls at the
                # identity root.  Preserve both layouts in derived metadata.
                for name in (
                    "lxe_surface_target_arc_mm",
                    "lxe_surface_min_phi",
                    "lxe_surface_phi_multiple",
                ):
                    if name in identity:
                        discretization[name] = identity[name]
            except ValueError:
                discretization = {}
    except (KeyError, OSError) as error:
        raise ValueError(
            "ragged resampling requires analytic LXe geometry elements"
        ) from error

    radius_nodes = np.asarray(surface_radius_mm, dtype=float)
    ring_area = np.asarray(surface_ring_area_mm2, dtype=float)
    if not np.allclose(
        primitive_center, np.zeros(3), rtol=0.0, atol=tolerance_mm
    ) or not np.allclose(
        primitive_axes, np.eye(3), rtol=0.0, atol=2.0e-12
    ):
        raise ValueError(
            "ragged resampling currently requires an origin-centered LXe "
            "surface with canonical analytic axes"
        )
    if (
        radius_nodes.ndim != 1
        or ring_area.shape != radius_nodes.shape
        or np.any(~np.isfinite(radius_nodes))
        or np.any(~np.isfinite(ring_area))
    ):
        raise ValueError("modal surface quadrature is invalid")
    if center.shape != (selected.size, 3):
        raise ValueError("geometry LXe centers have the wrong shape")
    if np.unique(element).size != element.size:
        raise ValueError("geometry LXe surface-element IDs are not unique")
    point_radius = np.linalg.norm(center[:, :2], axis=1)
    radial_distance = np.abs(
        point_radius[:, np.newaxis] - radius_nodes[np.newaxis, :]
    )
    ring_index = np.argmin(radial_distance, axis=1)
    nearest_distance = radial_distance[np.arange(len(center)), ring_index]
    if np.any(nearest_distance > tolerance_mm):
        bad = int(np.argmax(nearest_distance))
        raise ValueError(
            f"geometry LXe point {bad} is not on a modal radius "
            f"(distance={nearest_distance[bad]:.6g} mm)"
        )

    ordered: list[np.ndarray] = []
    ring_counts: list[int] = []
    expected_minimum = 2 * maximum_order + 1
    for ring in range(radius_nodes.size):
        members = np.flatnonzero(ring_index == ring)
        if members.size < expected_minimum:
            raise ValueError(
                f"geometry ring {ring} has {members.size} phi points; "
                f"at least 2*M+1={expected_minimum} are required"
            )
        phi = np.mod(
            np.arctan2(center[members, 1], center[members, 0]),
            2.0 * math.pi,
        )
        members = members[np.argsort(phi)]
        phi = np.mod(
            np.arctan2(center[members, 1], center[members, 0]),
            2.0 * math.pi,
        )
        expected_phi = (
            2.0
            * math.pi
            * (np.arange(members.size, dtype=float) + 0.5)
            / members.size
        )
        if not np.allclose(phi, expected_phi, rtol=0.0, atol=2.0e-12):
            raise ValueError(
                f"geometry ring {ring} is not a midpoint azimuth quadrature"
            )
        if not math.isclose(
            float(area[members].sum()),
            float(ring_area[ring]),
            rel_tol=2.0e-12,
            abs_tol=max(5.0e-8, tolerance_mm),
        ):
            raise ValueError(
                f"geometry ring {ring} area disagrees with modal quadrature"
            )
        if not np.allclose(
            area[members],
            ring_area[ring] / members.size,
            rtol=0.0,
            atol=max(2.0e-10, tolerance_mm),
        ):
            raise ValueError(
                f"geometry ring {ring} does not use area-uniform sectors"
            )
        ordered.append(members)
        ring_counts.append(int(members.size))

    permutation = np.concatenate(ordered)
    counts = np.asarray(ring_counts, dtype=np.uint64)
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.uint64), np.cumsum(counts)]
    )
    center = center[permutation]
    area = area[permutation]
    element = element[permutation]
    element_normal = element_normal[permutation]
    phi = np.mod(np.arctan2(center[:, 1], center[:, 0]), 2.0 * math.pi)

    target_arc = discretization.get("lxe_surface_target_arc_mm")
    if target_arc is not None:
        phi_multiple = int(
            discretization.get("lxe_surface_phi_multiple", 1)
        )
        if phi_multiple <= 0 or np.any(counts % phi_multiple):
            raise ValueError(
                "geometry ragged-ring counts violate lxe_surface_phi_multiple"
            )
        minimum_phi = int(discretization.get("lxe_surface_min_phi", 1))
        ring_outer_radius = np.sqrt(np.cumsum(ring_area) / math.pi)
        requested = np.maximum(
            minimum_phi,
            np.ceil(
                2.0 * math.pi * ring_outer_radius / float(target_arc)
            ).astype(np.int64),
        )
        expected_counts = (
            (requested + phi_multiple - 1) // phi_multiple
        ) * phi_multiple
        if not np.array_equal(counts, expected_counts.astype(np.uint64)):
            raise ValueError(
                "geometry ring counts disagree with its target-arc metadata"
            )
    if primitive_minus[primitive] == gxe_domain_id:
        side_value = 0
        normal = -element_normal
    elif primitive_plus[primitive] == gxe_domain_id:
        side_value = 1
        normal = element_normal
    else:  # pragma: no cover - guarded by candidate selection
        raise ValueError("analytic LXe surface is not adjacent to GXe")
    normal_norm = np.linalg.norm(normal, axis=1)
    if np.any(normal_norm == 0.0):
        raise ValueError("geometry LXe surface has a zero normal")
    normal = normal / normal_norm[:, np.newaxis]
    tangent_value = tangent_value / np.linalg.norm(tangent_value)
    tangent = np.tile(tangent_value, (len(center), 1))
    return RaggedSurface(
        ring_offsets=offsets,
        phi_rad=phi,
        area_mm2=area,
        surface_element=element,
        barycentric=np.tile([1.0, 0.0, 0.0], (len(center), 1)),
        side=np.full(len(center), side_value, dtype=np.uint64),
        tangent=tangent,
        normal=normal,
        ring_phi_count=counts,
        geometry_discretization=discretization,
    )


def _egress_from_ragged_surface(
    surface: RaggedSurface,
    gas_direction: np.ndarray,
    angular_stokes: np.ndarray,
    *,
    rotate_with_surface_phi: bool = False,
) -> tuple[np.ndarray, ...]:
    """Expand geometry points only over the independent angle quadrature."""

    angular_count = len(gas_direction)
    point_count = len(surface.surface_element)
    count = point_count * angular_count
    result_direction = np.empty((count, 3), dtype=float)
    result_reference = np.empty((count, 3), dtype=float)
    for point_index in range(point_count):
        bitangent = np.cross(
            surface.normal[point_index], surface.tangent[point_index]
        )
        basis = np.column_stack(
            [
                surface.tangent[point_index],
                bitangent,
                surface.normal[point_index],
            ]
        )
        for angle_index, direction in enumerate(gas_direction):
            output = point_index * angular_count + angle_index
            world_direction = direction
            if rotate_with_surface_phi:
                phi = float(surface.phi_rad[point_index])
                cosine = math.cos(phi)
                sine = math.sin(phi)
                world_direction = np.asarray(
                    [
                        cosine * direction[0] - sine * direction[1],
                        sine * direction[0] + cosine * direction[1],
                        direction[2],
                    ]
                )
            result_direction[output] = basis.T @ world_direction
            azimuth = math.atan2(
                float(world_direction[1]), float(world_direction[0])
            )
            s_axis = np.asarray(
                [-math.sin(azimuth), math.cos(azimuth), 0.0]
            )
            result_reference[output] = basis.T @ s_axis
    return (
        np.repeat(surface.surface_element, angular_count),
        np.repeat(surface.barycentric, angular_count, axis=0),
        np.repeat(surface.side, angular_count),
        result_direction,
        np.tile(angular_stokes, (point_count, 1)),
        result_reference,
    )


def cesaro_modal_weights(
    order_count: int,
    *,
    kind: str,
    cesaro_order: float = 2.0,
) -> np.ndarray:
    """Return a supported positivity-preserving modal kernel through ``M``."""

    if order_count <= 0:
        raise ValueError("order_count must be positive")
    if kind == "none":
        return np.ones(order_count, dtype=float)
    if kind not in {"caratheodory", "fejer", "cesaro"}:
        raise ValueError(f"unknown modal filter {kind!r}")
    if kind == "caratheodory":
        # Caratheodory--Fejer extremal nonnegative trigonometric kernel:
        # K(phi) = |sum_j a_j exp(i*j*phi)|^2 / sum_j a_j^2,
        # a_j = sin((j+1)*pi/(M+2)).  Its Fourier multipliers are the
        # normalized autocorrelation of a.  K is nonnegative by construction,
        # w_0=1, and w_1 is maximal among degree-M nonnegative kernels.
        maximum = order_count - 1
        index = np.arange(order_count, dtype=float)
        factor = np.sin((index + 1.0) * math.pi / (maximum + 2.0))
        denominator = float(np.dot(factor, factor))
        result = np.asarray(
            [
                np.dot(factor[: order_count - order], factor[order:])
                / denominator
                for order in range(order_count)
            ],
            dtype=float,
        )
        result[0] = 1.0
        return result
    if kind == "fejer":
        return (
            np.arange(order_count, 0, -1, dtype=float) / order_count
        )
    alpha = float(cesaro_order)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("cesaro_order must be finite and positive")
    maximum = order_count - 1
    denominator = (
        math.lgamma(maximum + alpha + 1.0)
        - math.lgamma(maximum + 1.0)
    )
    result = np.empty(order_count, dtype=float)
    for order in range(order_count):
        remaining = maximum - order
        result[order] = math.exp(
            math.lgamma(remaining + alpha + 1.0)
            - math.lgamma(remaining + 1.0)
            - denominator
        )
    result[0] = 1.0
    return result


def normalize_modal_m0(
    coefficients: np.ndarray,
    expected_return: np.ndarray,
    ring_area_mm2: np.ndarray,
    *,
    tolerance: float = 1.0e-13,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize every phase row so its m=0 integral is exactly expected."""

    modes = np.asarray(coefficients, dtype=np.complex128).copy()
    expected = np.asarray(expected_return, dtype=float)
    ring_area = np.asarray(ring_area_mm2, dtype=float)
    raw = np.einsum("...s,s->...", modes[..., 0, :].real, ring_area)
    if raw.shape != expected.shape:
        raise ValueError("m=0 rows do not match expected_return")
    active = expected > tolerance
    if np.any(active & (raw <= tolerance)):
        bad = np.argwhere(active & (raw <= tolerance))[0].tolist()
        raise ValueError(f"cannot normalize nonpositive m=0 row {bad}")
    scale = np.zeros_like(expected)
    scale[active] = expected[active] / raw[active]
    zero = ~active
    if np.any(zero & (np.abs(raw) > tolerance)):
        raise ValueError("zero-return state has a nonzero m=0 integral")
    modes *= scale[..., np.newaxis, np.newaxis]

    # One residual correction makes the conservation statement independent of
    # summation order and leaves every high-order conditional shape untouched.
    normalized = np.einsum(
        "...s,s->...", modes[..., 0, :].real, ring_area
    )
    residual = expected - normalized
    anchor = int(np.argmax(ring_area))
    anchor_values = modes[..., 0, anchor]
    modes[..., 0, anchor] = (
        anchor_values.real + residual / ring_area[anchor]
    ) + 1j * anchor_values.imag
    final = np.einsum("...s,s->...", modes[..., 0, :].real, ring_area)
    error = np.abs(final - expected)
    allowed = max(tolerance, 16.0 * np.finfo(float).eps)
    if float(error.max(initial=0.0)) > allowed:
        raise RuntimeError("strict m=0 normalization failed")
    relative_before = np.divide(
        raw - expected,
        expected,
        out=np.zeros_like(raw),
        where=active,
    )
    active_scale = scale[active]
    return modes, {
        "method": "per_phase_row_scale_plus_residual",
        "anchor_ring": anchor,
        "maximum_relative_error_before": float(
            np.max(np.abs(relative_before), initial=0.0)
        ),
        "maximum_absolute_error_after": float(error.max(initial=0.0)),
        "scale_minimum": float(active_scale.min()) if active_scale.size else 1.0,
        "scale_maximum": float(active_scale.max()) if active_scale.size else 1.0,
    }


def audit_joint_modal_m0(
    coefficients: np.ndarray,
    expected_return: np.ndarray,
    ring_area_mm2: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Verify rank-six angle-summed m=0 conservation without modifying it."""

    modes = np.asarray(coefficients, dtype=np.complex128)
    expected = np.asarray(expected_return, dtype=float)
    ring_area = np.asarray(ring_area_mm2, dtype=float)
    if modes.ndim != 6:
        raise ValueError("joint m=0 audit requires rank-six coefficients")
    raw = np.einsum("...sa,s->...", modes[..., 0, :, :].real, ring_area)
    if raw.shape != expected.shape:
        raise ValueError("angle-summed m=0 rows do not match expected_return")
    error = np.abs(raw - expected)
    allowed = tolerance + tolerance * np.abs(expected)
    if np.any(error > allowed):
        bad = np.argwhere(error > allowed)[0].tolist()
        raise ValueError(f"joint angle-summed m=0 mismatch in phase row {bad}")
    return {
        "method": "angle_summed_audit_no_coefficient_change",
        "maximum_absolute_error": float(error.max(initial=0.0)),
        "tolerance": float(tolerance),
    }


def dense_phi_positivity_audit(
    coefficients: np.ndarray,
    ring_area_mm2: np.ndarray,
    *,
    phi_bins: int,
    tolerance: float = 1.0e-13,
    chunk_rows: int = 512,
) -> dict[str, Any]:
    """Audit each phase-row/ring Fourier polynomial on a dense phi grid."""

    modes = np.asarray(coefficients, dtype=np.complex128)
    order_count = modes.shape[-2]
    minimum_bins = 2 * order_count - 1
    if phi_bins < minimum_bins:
        raise ValueError(
            f"positivity phi grid needs at least 2*M+1={minimum_bins} bins"
        )
    rows = np.moveaxis(modes, -2, -1).reshape(-1, order_count)
    ring_count = modes.shape[-1]
    area = np.tile(np.asarray(ring_area_mm2, dtype=float), rows.shape[0] // ring_count)
    minimum = math.inf
    negative_samples = 0
    rows_with_negative = 0
    maximum_negative_integral = 0.0
    for begin in range(0, rows.shape[0], chunk_rows):
        values = rows[begin : begin + chunk_rows]
        spectrum = np.zeros((len(values), phi_bins), dtype=np.complex128)
        spectrum[:, 0] = values[:, 0].real
        if order_count > 1:
            spectrum[:, 1:order_count] = 0.5 * values[:, 1:]
            spectrum[:, -(order_count - 1) :] = (
                0.5 * np.conj(values[:, 1:][:, ::-1])
            )
        density = np.fft.ifft(spectrum, axis=1).real * phi_bins
        minimum = min(minimum, float(density.min(initial=math.inf)))
        negative = density < -tolerance
        negative_samples += int(np.count_nonzero(negative))
        rows_with_negative += int(np.count_nonzero(np.any(negative, axis=1)))
        negative_integral = (
            np.maximum(-density, 0.0).mean(axis=1)
            * area[begin : begin + len(values)]
        )
        maximum_negative_integral = max(
            maximum_negative_integral,
            float(negative_integral.max(initial=0.0)),
        )
    return {
        "dense_phi_bins": int(phi_bins),
        "tolerance": float(tolerance),
        "minimum_density_per_mm2": float(minimum),
        "negative_sample_count": negative_samples,
        "sample_count": int(rows.shape[0] * phi_bins),
        "row_ring_count": int(rows.shape[0]),
        "rows_with_negative": rows_with_negative,
        "maximum_negative_ring_integral": maximum_negative_integral,
    }


def positive_ring_contraction(
    coefficients: np.ndarray,
    expected_return: np.ndarray,
    ring_area_mm2: np.ndarray,
    *,
    phi_bins: int,
    tolerance: float = 1.0e-13,
    chunk_rows: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove Fourier undershoot while preserving every radial-ring marginal.

    Write each ring kernel as ``q(phi) = mu + g(phi)``, where ``mu`` is its
    stored m=0 coefficient.  If a certified lower bound is negative, all
    nonzero modes in only that phase-row/ring are contracted by
    ``mu / (mu - q_min)``.  The result is nonnegative while keeping ``mu`` --
    and therefore every radial marginal and total return probability --
    unchanged.  Unlike Fejer/Cesaro filtering, this does not preferentially
    damp higher retained orders.

    The certificate combines a dense periodic grid with the global derivative
    bound ``|q'| <= sum_m m |c_m|``.  Since any angle is at most ``pi/N`` from
    a grid node, ``min_grid(q) - pi/N * sum_m m |c_m|`` is a conservative
    lower bound between nodes as well.
    """

    modes = np.asarray(coefficients, dtype=np.complex128)
    expected = np.asarray(expected_return, dtype=float)
    area = np.asarray(ring_area_mm2, dtype=float)
    if modes.ndim < 2 or modes.shape[:-2] != expected.shape:
        raise ValueError("modal rows do not match expected_return")
    if area.shape != (modes.shape[-1],):
        raise ValueError("ring areas do not match modal coefficients")
    if phi_bins < 2 * modes.shape[-2] - 1:
        raise ValueError(
            "positive ring contraction needs at least 2*M+1 azimuth samples"
        )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "positive ring contraction tolerance must be nonnegative"
        )

    order_count = modes.shape[-2]
    ring_count = modes.shape[-1]
    row_ring = np.moveaxis(modes, -2, -1).reshape(-1, order_count)
    sampled_minimum = np.empty(row_ring.shape[0], dtype=float)
    for begin in range(0, row_ring.shape[0], chunk_rows):
        values = row_ring[begin : begin + chunk_rows]
        spectrum = np.zeros((len(values), phi_bins), dtype=np.complex128)
        spectrum[:, 0] = values[:, 0].real
        if order_count > 1:
            spectrum[:, 1:order_count] = 0.5 * values[:, 1:]
            spectrum[:, -(order_count - 1) :] = (
                0.5 * np.conj(values[:, 1:][:, ::-1])
            )
        density = np.fft.ifft(spectrum, axis=1).real * phi_bins
        sampled_minimum[begin : begin + len(values)] = density.min(axis=1)

    derivative_bound = np.sum(
        np.arange(order_count, dtype=float)[np.newaxis, :]
        * np.abs(row_ring),
        axis=1,
    )
    certified_minimum = (
        sampled_minimum - math.pi * derivative_bound / phi_bins
    )
    mean = row_ring[:, 0].real
    if np.any(mean < -tolerance):
        bad = int(np.argmin(mean))
        raise ValueError(f"negative m=0 density in row-ring {bad}")
    contraction = np.ones_like(mean)
    negative = certified_minimum < 0.0
    positive_mean = mean > tolerance
    selected = negative & positive_mean
    contraction[selected] = mean[selected] / (
        mean[selected] - certified_minimum[selected]
    )
    # A zero-integral nonnegative periodic density must vanish identically.
    contraction[negative & ~positive_mean] = 0.0
    contraction = np.clip(contraction, 0.0, 1.0)

    projected_rows = row_ring.copy()
    projected_rows[:, 1:] *= contraction[:, np.newaxis]
    projected = np.moveaxis(
        projected_rows.reshape(expected.shape + (ring_count, order_count)),
        -1,
        -2,
    )
    m0_before = modes[..., 0, :]
    m0_after = projected[..., 0, :]
    if not np.array_equal(m0_after, m0_before):
        raise RuntimeError("positive ring contraction changed m=0")

    contracted = contraction < 1.0
    return projected, {
        "method": "mean_preserving_ringwise_nonzero_mode_contraction",
        "dense_phi_bins": int(phi_bins),
        "derivative_bound_certificate": True,
        "contracted_row_ring_count": int(np.count_nonzero(contracted)),
        "row_ring_count": int(contraction.size),
        "minimum_contraction": float(contraction.min(initial=1.0)),
        "mean_contraction_among_modified": float(
            contraction[contracted].mean() if np.any(contracted) else 1.0
        ),
        "minimum_sampled_density_before_per_mm2": float(
            sampled_minimum.min(initial=math.inf)
        ),
        "minimum_certified_density_before_per_mm2": float(
            certified_minimum.min(initial=math.inf)
        ),
        "m0_preserved_bitwise": True,
    }


def _resampler_digest() -> str:
    return sha256(Path(__file__))


def _cache_key(
    *,
    input_sha256: str,
    geometry_contract_sha256: str,
    surface_layout: dict[str, Any],
    surface_id: int,
    gxe_domain_id: int,
    geometry_tolerance_mm: float,
    modal_filter: str,
    cesaro_order: float,
    positivity_phi_bins: int,
    positivity_tolerance: float,
) -> str:
    material = {
        "schema": "oos.dual-phase-tpc.lxe-egress-resampler.v2",
        "input_sha256": input_sha256,
        "geometry_contract_sha256": geometry_contract_sha256,
        "surface_layout": surface_layout,
        "surface_id": surface_id,
        "gxe_domain_id": gxe_domain_id,
        "geometry_tolerance_mm": geometry_tolerance_mm,
        "modal_filter": modal_filter,
        "cesaro_order": cesaro_order,
        "strict_m0_normalization": surface_layout["kind"] == "ragged_ring_v1",
        "positivity_phi_bins": positivity_phi_bins,
        "positivity_tolerance": positivity_tolerance,
        "resampler_sha256": _resampler_digest(),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resampled_metadata(
    source: FactorizedLXeBlock,
    *,
    input_sha256: str,
    geometry_sha256: str,
    geometry_contract: dict[str, Any],
    target_layout: dict[str, Any],
    cache_key_sha256: str,
    derived_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    derived = {
        "angular_count",
        "cache_key_sha256",
        "contraction_bound",
        "egress_count",
        "execution",
        "geometry_contract",
        "geometry_contract_sha256",
        "geometry_sha256",
        "schema",
        "state_count",
        "surface_layout",
        "surface_point_count",
        "surface_phi_bins",
        "surface_relative",
    }
    metadata = {
        key: value for key, value in source.producer.items() if key not in derived
    }
    history = metadata.get("egress_resampling_history", [])
    if not isinstance(history, list):
        history = []
    history = [
        *history,
        {
            "input_sha256": input_sha256,
            "source_schema": source.producer.get("schema"),
            "source_surface_phi_bins": source.surface_phi_bins,
            "source_surface_point_count": source.surface_point_count,
            "target_surface_layout": target_layout,
            "geometry_contract_sha256": geometry_contract[
                "fingerprint_sha256"
            ],
        },
    ]
    metadata.update(
        {
            "cache_key_sha256": cache_key_sha256,
            "geometry_sha256": geometry_sha256,
            "geometry_contract_sha256": geometry_contract[
                "fingerprint_sha256"
            ],
            "geometry_contract": geometry_contract,
            "resampler_sha256": _resampler_digest(),
            "resampled_from_sha256": input_sha256,
            "resampled_from_surface_phi_bins": source.surface_phi_bins,
            "egress_resampling_history": history,
            **(derived_metadata or {}),
        }
    )
    return metadata


def resample_lxe_function_block(
    input_path: Path,
    geometry_path: Path,
    output_path: Path,
    *,
    surface_phi_bins: int | None = None,
    ragged_rings: bool = False,
    modal_filter: str | None = None,
    cesaro_order: float = 2.0,
    positivity_phi_bins: int = 8192,
    positivity_tolerance: float = 1.0e-13,
    surface_id: int = 1,
    gxe_domain_id: int = 0,
    geometry_tolerance_mm: float = 1.0e-8,
    force: bool = False,
) -> dict[str, Any]:
    """Preserve modal coefficients and rebuild the target egress basis."""

    input_path = Path(input_path)
    geometry_path = Path(geometry_path)
    output_path = Path(output_path)
    if ragged_rings:
        if surface_phi_bins is not None:
            raise ValueError(
                "ragged_rings and surface_phi_bins are mutually exclusive"
            )
    elif surface_phi_bins is None or surface_phi_bins <= 0:
        raise ValueError("surface_phi_bins must be positive")
    if modal_filter is None:
        modal_filter = "none"
    if modal_filter not in {
        "none",
        "caratheodory",
        "positive-ring",
        "fejer",
        "cesaro",
    }:
        raise ValueError(f"unknown modal filter {modal_filter!r}")
    if not ragged_rings and modal_filter != "none":
        raise ValueError("modal filtering is available only for ragged blocks")
    if positivity_phi_bins <= 0:
        raise ValueError("positivity_phi_bins must be positive")
    if (
        not math.isfinite(positivity_tolerance)
        or positivity_tolerance < 0.0
    ):
        raise ValueError("positivity_tolerance must be finite and nonnegative")
    if not math.isfinite(geometry_tolerance_mm) or geometry_tolerance_mm <= 0.0:
        raise ValueError("geometry_tolerance_mm must be finite and positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists() and not force:
        raise FileExistsError(f"output exists: {output_path}")

    source = load_factorized_block(input_path)
    joint_angular = source.coefficients.ndim == 6
    if joint_angular and modal_filter != "none":
        raise ValueError(
            "rank-six joint coefficients currently require modal_filter=none"
        )
    if ragged_rings and source.surface_phi_bins is None:
        raise ValueError(
            "ragged resampling currently requires a v1 input block; "
            "re-filtering an existing v2 block is intentionally rejected"
        )
    if not ragged_rings:
        if source.surface_phi_bins is None:
            raise ValueError("uniform resampling requires a v1 input block")
        assert surface_phi_bins is not None
        if surface_phi_bins <= source.surface_phi_bins:
            raise ValueError(
                "surface_phi_bins must be greater than the input basis "
                f"({source.surface_phi_bins})"
            )

    geometry_contract = geometry_lxe_contract(
        geometry_path,
        surface_id=surface_id,
        gxe_domain_id=gxe_domain_id,
    )
    lxe_config = source.producer["lxe_config"]
    for name in ("radius_mm", "depth_mm"):
        source_value = float(lxe_config[name])
        target_value = float(geometry_contract[name])
        if not math.isclose(
            source_value,
            target_value,
            rel_tol=0.0,
            abs_tol=geometry_tolerance_mm,
        ):
            raise ValueError(
                f"target geometry {name}={target_value:.12g} mm disagrees "
                f"with the input LXe block ({source_value:.12g} mm)"
            )

    lxe = LXeDiffusionConfig(**lxe_config)
    direction, angular_weight, angular_stokes = diffuse_escape_quadrature(
        lxe,
        liquid_mu_order=_required_positive_integer(
            source.producer, "return_mu_order"
        ),
        direction_phi=_required_positive_integer(
            source.producer, "return_direction_phi"
        ),
    )
    if angular_weight.shape != source.angular_weight.shape or not np.allclose(
        angular_weight,
        source.angular_weight,
        rtol=0.0,
        atol=2.0e-15,
    ):
        raise ValueError(
            "escape quadrature reconstructed from metadata differs from input"
        )

    coefficients = source.coefficients
    ragged_surface: RaggedSurface | None = None
    derived_metadata: dict[str, Any] = {}
    if ragged_rings:
        maximum_order = source.coefficients.shape[3] - 1
        ragged_surface = _geometry_ragged_surface(
            geometry_path,
            source.surface_radius_mm,
            source.surface_ring_area_mm2,
            surface_id=surface_id,
            gxe_domain_id=gxe_domain_id,
            maximum_order=maximum_order,
            tolerance_mm=geometry_tolerance_mm,
        )
        egress = _egress_from_ragged_surface(
            ragged_surface,
            direction,
            angular_stokes,
            rotate_with_surface_phi=joint_angular,
        )
        if joint_angular:
            coefficients = source.coefficients
            normalization_audit = audit_joint_modal_m0(
                coefficients,
                source.expected_return,
                source.surface_ring_area_mm2,
            )
        else:
            coefficients, normalization_audit = normalize_modal_m0(
                source.coefficients,
                source.expected_return,
                source.surface_ring_area_mm2,
            )
        modal_weight = cesaro_modal_weights(
            source.coefficients.shape[3],
            kind=(
                "none" if modal_filter == "positive-ring" else modal_filter
            ),
            cesaro_order=cesaro_order,
        )
        if not joint_angular:
            coefficients *= modal_weight[
                np.newaxis, np.newaxis, np.newaxis, :, np.newaxis
            ]
        positivity_correction: dict[str, Any] | None = None
        if modal_filter == "positive-ring":
            coefficients, positivity_correction = positive_ring_contraction(
                coefficients,
                source.expected_return,
                source.surface_ring_area_mm2,
                phi_bins=positivity_phi_bins,
                tolerance=positivity_tolerance,
            )
        # Filtering is stored directly in the modal coefficients.  w_0=1 is
        # asserted so the strict return normalization cannot be weakened by a
        # runtime-dependent convention.
        if modal_weight[0] != 1.0:
            raise RuntimeError("modal filter does not preserve m=0")
        positivity_coefficients = (
            np.moveaxis(coefficients, -1, -3)
            if joint_angular
            else coefficients
        )
        positivity_audit = dense_phi_positivity_audit(
            positivity_coefficients,
            source.surface_ring_area_mm2,
            phi_bins=positivity_phi_bins,
            tolerance=positivity_tolerance,
        )
        if (
            modal_filter != "none"
            and positivity_audit["negative_sample_count"] != 0
        ):
            raise ValueError(
                f"{modal_filter} modal density failed the dense positivity audit"
            )
        target_layout = {
            "kind": "ragged_ring_v1",
            "surface_point_count": int(ragged_surface.ring_offsets[-1]),
            "ring_phi_count": ragged_surface.ring_phi_count.tolist(),
        }
        derived_metadata = {
            "modal_filter": {
                "kind": modal_filter,
                "cesaro_order": (
                    cesaro_order
                    if modal_filter == "cesaro"
                    else (1.0 if modal_filter == "fejer" else None)
                ),
                "weights": modal_weight.tolist(),
                "applied_to_stored_coefficients": True,
            },
            "positivity_correction": positivity_correction,
            "strict_m0_normalization": normalization_audit,
            "dense_phi_positivity_audit": positivity_audit,
            "nonnegative_density_on_audit_grid": (
                positivity_audit["negative_sample_count"] == 0
            ),
            "ragged_surface_quadrature": {
                "geometry_is_unique_egress_source": True,
                "ring_count": int(len(ragged_surface.ring_phi_count)),
                "surface_point_count": int(ragged_surface.ring_offsets[-1]),
                "minimum_ring_phi": int(ragged_surface.ring_phi_count.min()),
                "maximum_ring_phi": int(ragged_surface.ring_phi_count.max()),
                "maximum_outer_arc_mm": float(
                    np.max(
                        2.0
                        * math.pi
                        * np.sqrt(
                            np.cumsum(source.surface_ring_area_mm2) / math.pi
                        )
                        / ragged_surface.ring_phi_count
                    )
                ),
                "target_arc_mm": ragged_surface.geometry_discretization.get(
                    "lxe_surface_target_arc_mm"
                ),
                "minimum_phi_requested": (
                    ragged_surface.geometry_discretization.get(
                        "lxe_surface_min_phi"
                    )
                ),
                "phi_multiple": ragged_surface.geometry_discretization.get(
                    "lxe_surface_phi_multiple"
                ),
                "minimum_phi_required_by_modes": 2 * maximum_order + 1,
            },
        }
    else:
        assert surface_phi_bins is not None
        egress = egress_basis(
            geometry_path,
            source.surface_radius_mm,
            direction,
            angular_stokes,
            surface_phi_bins=surface_phi_bins,
            surface_id=surface_id,
            gxe_domain_id=gxe_domain_id,
            tolerance_mm=geometry_tolerance_mm,
            rotate_with_surface_phi=joint_angular,
        )
        target_layout = {
            "kind": "uniform_phi_v1",
            "surface_phi_bins": surface_phi_bins,
        }
    input_digest = sha256(input_path)
    geometry_digest = sha256(geometry_path)
    key = _cache_key(
        input_sha256=input_digest,
        geometry_contract_sha256=geometry_contract["fingerprint_sha256"],
        surface_layout=target_layout,
        surface_id=surface_id,
        gxe_domain_id=gxe_domain_id,
        geometry_tolerance_mm=geometry_tolerance_mm,
        modal_filter=modal_filter,
        cesaro_order=cesaro_order,
        positivity_phi_bins=positivity_phi_bins,
        positivity_tolerance=positivity_tolerance,
    )
    metadata = _resampled_metadata(
        source,
        input_sha256=input_digest,
        geometry_sha256=geometry_digest,
        geometry_contract=geometry_contract,
        target_layout=target_layout,
        cache_key_sha256=key,
        derived_metadata=derived_metadata,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_factorized_block(
            temporary,
            coefficients=coefficients,
            expected_return=source.expected_return,
            audit_values=source.audit_values,
            surface_radius_mm=source.surface_radius_mm,
            surface_ring_area_mm2=source.surface_ring_area_mm2,
            angular_weight=angular_weight,
            surface_element=egress[0],
            barycentric=egress[1],
            side=egress[2],
            direction_local=egress[3],
            stokes=egress[4],
            reference_axis_local=egress[5],
            phase_grid=source.phase_grid,
            audit_names=source.audit_names,
            surface_ring_offsets=(
                ragged_surface.ring_offsets
                if ragged_surface is not None
                else None
            ),
            surface_phi_rad=(
                ragged_surface.phi_rad
                if ragged_surface is not None
                else None
            ),
            surface_area_mm2=(
                ragged_surface.area_mm2
                if ragged_surface is not None
                else None
            ),
            metadata=metadata,
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "input": str(input_path.resolve()),
        "input_sha256": input_digest,
        "geometry": str(geometry_path.resolve()),
        "geometry_sha256": geometry_digest,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "source_surface_phi_bins": source.surface_phi_bins,
        "surface_phi_bins": surface_phi_bins,
        "surface_layout": target_layout,
        "egress_count": int(egress[0].size),
        "cache_key_sha256": key,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--input", type=Path, required=True, help="existing factorized LXe HDF5 block"
    )
    result.add_argument(
        "--geometry", type=Path, required=True, help="target OOS geometry HDF5"
    )
    result.add_argument("--output", type=Path, required=True)
    layout = result.add_mutually_exclusive_group(required=True)
    layout.add_argument(
        "--surface-phi-bins",
        type=int,
        help="target azimuthal count; must be finer than the input block",
    )
    layout.add_argument(
        "--ragged-rings",
        action="store_true",
        help="use the geometry-owned ragged-ring LXe quadrature and write v2",
    )
    result.add_argument(
        "--modal-filter",
        choices=(
            "none",
            "caratheodory",
            "positive-ring",
            "fejer",
            "cesaro",
        ),
        default=None,
        help=(
            "stored high-order modal taper for ragged blocks "
            "(default: none; final physical-source PMT responses are "
            "certified nonnegative before they are written)"
        ),
    )
    result.add_argument(
        "--cesaro-order",
        type=float,
        default=2.0,
        help="generalized Cesaro alpha (Fejer always uses alpha=1)",
    )
    result.add_argument(
        "--positivity-phi-bins",
        type=int,
        default=8192,
        help="dense azimuth grid used to audit filtered modal positivity",
    )
    result.add_argument(
        "--positivity-tolerance",
        type=float,
        default=1.0e-13,
    )
    result.add_argument("--surface-id", type=int, default=1)
    result.add_argument("--gxe-domain-id", type=int, default=0)
    result.add_argument("--geometry-tolerance-mm", type=float, default=1.0e-8)
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    arguments = parser().parse_args()
    try:
        summary = resample_lxe_function_block(
            arguments.input,
            arguments.geometry,
            arguments.output,
            surface_phi_bins=arguments.surface_phi_bins,
            ragged_rings=arguments.ragged_rings,
            modal_filter=arguments.modal_filter,
            cesaro_order=arguments.cesaro_order,
            positivity_phi_bins=arguments.positivity_phi_bins,
            positivity_tolerance=arguments.positivity_tolerance,
            surface_id=arguments.surface_id,
            gxe_domain_id=arguments.gxe_domain_id,
            geometry_tolerance_mm=arguments.geometry_tolerance_mm,
            force=arguments.force,
        )
    except (FileExistsError, ValueError) as error:
        parser().error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
