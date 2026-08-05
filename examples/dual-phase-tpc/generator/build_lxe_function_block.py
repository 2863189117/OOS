#!/usr/bin/env python3
"""Build the self-contained factorized LXe block for the dual-phase example.

The expensive explicit-seven-collision plus diffusion calculation remains an
example-owned offline Python generator.  Its output is a canonical,
surface-relative HDF5 function block consumed by the native C++ plugin.  No
PMT channel or external destination is stored in this file.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import h5py
import numpy as np
from scipy.spatial import cKDTree

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from generator.reference.lxe_cylinder_green import (
        LXeCylinderGreen,
        ModeTruncation,
    )
    from generator.reference.lxe_diffusion_return import (
        LXeDiffusionConfig,
        diffuse_fresnel_reflection_moments,
    )
    from generator.reference.lxe_response_operator import (
        build_lxe_response_operator,
    )
    from generator.reference.numerics import fresnel_power, gauss_interval
    from generator.reference.phase_space_grid import LXePhaseSpaceGrid
    from generator.write_intrinsic_lxe_block import write_factorized_block
else:
    from .reference.lxe_cylinder_green import (
        LXeCylinderGreen,
        ModeTruncation,
    )
    from .reference.lxe_diffusion_return import (
        LXeDiffusionConfig,
        diffuse_fresnel_reflection_moments,
    )
    from .reference.lxe_response_operator import build_lxe_response_operator
    from .reference.numerics import fresnel_power, gauss_interval
    from .reference.phase_space_grid import LXePhaseSpaceGrid
    from .write_intrinsic_lxe_block import write_factorized_block


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted((root / "reference").glob("*.py")) + [
        Path(__file__).resolve(),
        root / "write_intrinsic_lxe_block.py",
    ]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _update_array_digest(
    digest: "hashlib._Hash", name: str, values: np.ndarray
) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode())
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())


def geometry_lxe_contract(
    path: Path, *, surface_id: int, gxe_domain_id: int
) -> dict[str, object]:
    """Return the liquid-only geometry identity used by the LXe block."""

    with h5py.File(path, "r") as handle:
        generator = json.loads(
            bytes(handle["/metadata/generator_json"][:]).decode()
        )
        config = generator["config"]
        radius_mm = float(config["active_radius_mm"])
        depth_mm = float(config["lxe_depth_mm"])
        digest = hashlib.sha256()
        digest.update(b"oos.dual-phase-tpc.lxe-geometry.v1\n")
        digest.update(
            json.dumps(
                {
                    "radius_mm": radius_mm,
                    "depth_mm": depth_mm,
                    "surface_id": surface_id,
                    "gxe_domain_id": gxe_domain_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        if "/analytic/elements/primitive_index" in handle:
            primitive_surface = np.asarray(
                handle["/analytic/surface_id"], dtype=np.uint32
            )
            primitive_minus = np.asarray(
                handle["/analytic/minus_domain_id"], dtype=np.int32
            )
            primitive_plus = np.asarray(
                handle["/analytic/plus_domain_id"], dtype=np.int32
            )
            selected_primitives = np.flatnonzero(
                (primitive_surface == surface_id)
                & (
                    (primitive_minus == gxe_domain_id)
                    ^ (primitive_plus == gxe_domain_id)
                )
            )
            if selected_primitives.size != 1:
                raise RuntimeError(
                    "geometry must declare exactly one analytic LXe surface"
                )
            primitive = int(selected_primitives[0])
            kind = int(handle["/analytic/kind"][primitive])
            parameters = np.asarray(
                handle["/analytic/parameters"][primitive], dtype=np.float64
            )
            if kind != 1 or not math.isclose(
                float(parameters[0]), radius_mm, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise RuntimeError(
                    "analytic LXe disk radius disagrees with geometry metadata"
                )
            for name in (
                "kind",
                "center_mm",
                "axis_x",
                "axis_y",
                "axis_z",
                "parameters",
                "normal_sign",
                "surface_id",
                "minus_domain_id",
                "plus_domain_id",
            ):
                _update_array_digest(
                    digest,
                    f"primitive/{name}",
                    np.asarray(handle[f"/analytic/{name}"][primitive]),
                )
            owner = np.asarray(
                handle["/analytic/elements/primitive_index"], dtype=np.uint32
            )
            selected = np.flatnonzero(owner == primitive)
            if selected.size == 0:
                raise RuntimeError("analytic LXe surface has no elements")
            element_id = np.asarray(
                handle["/analytic/elements/surface_element"], dtype=np.uint64
            )[selected]
            order = np.argsort(element_id)
            selected = selected[order]
            if np.unique(element_id).size != element_id.size:
                raise RuntimeError("analytic LXe surface elements are not unique")
            for name in (
                "coordinates",
                "bounds",
                "center_mm",
                "normal",
                "surface_element",
            ):
                _update_array_digest(
                    digest,
                    f"elements/{name}",
                    np.asarray(handle[f"/analytic/elements/{name}"])[selected],
                )
            representation = "analytic"
        else:
            surface = np.asarray(
                handle["/geometry/surface_id"], dtype=np.uint32
            )
            minus = np.asarray(
                handle["/geometry/minus_domain_id"], dtype=np.int32
            )
            plus = np.asarray(
                handle["/geometry/plus_domain_id"], dtype=np.int32
            )
            selected = np.flatnonzero(
                (surface == surface_id)
                & ((minus == gxe_domain_id) ^ (plus == gxe_domain_id))
            )
            if selected.size == 0:
                raise RuntimeError("triangle geometry has no LXe surface")
            triangles = np.asarray(
                handle["/geometry/triangles"], dtype=np.uint32
            )[selected]
            vertices = np.asarray(
                handle["/geometry/vertices"], dtype=np.float64
            )
            _update_array_digest(digest, "triangle/index", selected)
            _update_array_digest(digest, "triangle/vertices", vertices[triangles])
            _update_array_digest(digest, "triangle/minus", minus[selected])
            _update_array_digest(digest, "triangle/plus", plus[selected])
            representation = "triangles"
    return {
        "schema": "oos.dual-phase-tpc.lxe-geometry.v1",
        "fingerprint_sha256": digest.hexdigest(),
        "radius_mm": radius_mm,
        "depth_mm": depth_mm,
        "surface_id": surface_id,
        "gxe_domain_id": gxe_domain_id,
        "representation": representation,
    }


def apply_test_profile(arguments: argparse.Namespace) -> None:
    arguments.position_radial_bins = 4
    arguments.position_phi_bins = 8
    arguments.direction_mu_bins = 2
    arguments.direction_phi_bins = 4
    arguments.surface_radial_order = 40
    arguments.surface_phi_bins = 80
    arguments.return_mu_order = 2
    arguments.return_direction_phi = 4
    arguments.first_scatter_order = 4
    arguments.explicit_collision_order = 2
    arguments.collision_sample_power = 6
    arguments.collision_maximum_events = 16
    arguments.azimuthal_maximum = 8
    arguments.radial_modes = 24
    arguments.root_samples = 24


def refract_direction(
    direction: np.ndarray,
    normal_toward_incident: np.ndarray,
    n_incident: float,
    n_transmitted: float,
) -> np.ndarray | None:
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    normal = np.asarray(normal_toward_incident, dtype=float)
    normal /= np.linalg.norm(normal)
    cosine = float(-np.dot(direction, normal))
    ratio = n_incident / n_transmitted
    discriminant = 1.0 - ratio * ratio * (1.0 - cosine * cosine)
    if discriminant < 0.0:
        return None
    result = (
        ratio * direction
        + (ratio * cosine - math.sqrt(discriminant)) * normal
    )
    return result / np.linalg.norm(result)


def diffuse_escape_quadrature(
    config: LXeDiffusionConfig,
    *,
    liquid_mu_order: int,
    direction_phi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gas directions, scalar weights, and unit-intensity Stokes rows."""

    ratio = config.n_gxe / config.n_lxe
    critical_mu = math.sqrt(max(0.0, 1.0 - ratio * ratio))
    first, second = diffuse_fresnel_reflection_moments(
        config.n_lxe, config.n_gxe
    )
    boundary_factor = (1.0 + second) / (1.0 - first)
    mu_nodes, mu_weights = gauss_interval(
        liquid_mu_order, critical_mu, 1.0
    )
    directions: list[np.ndarray] = []
    polarization: list[np.ndarray] = []
    for mu, mu_weight in zip(mu_nodes, mu_weights):
        _, transmission, _ = fresnel_power(
            config.n_lxe, config.n_gxe, float(mu)
        )
        transverse = math.sqrt(max(0.0, 1.0 - float(mu) ** 2))
        radiance = 2.0 * boundary_factor + 3.0 * float(mu)
        base = (
            2.0
            * float(mu)
            * radiance
            * float(mu_weight)
            / direction_phi
        )
        for index in range(direction_phi):
            phi = 2.0 * math.pi * (index + 0.5) / direction_phi
            liquid = np.asarray(
                [
                    transverse * math.cos(phi),
                    transverse * math.sin(phi),
                    float(mu),
                ]
            )
            gas = refract_direction(
                liquid,
                np.asarray([0.0, 0.0, -1.0]),
                config.n_lxe,
                config.n_gxe,
            )
            if gas is None:
                continue
            directions.append(gas)
            polarization.append(0.5 * transmission * base)
    pair = np.asarray(polarization, dtype=float)
    total = float(pair.sum())
    if not total > 0.0:
        raise RuntimeError("empty LXe escape quadrature")
    pair /= total
    weights = pair.sum(axis=1)
    stokes = np.zeros((len(pair), 4), dtype=float)
    stokes[:, 0] = 1.0
    stokes[:, 1] = np.divide(
        pair[:, 0] - pair[:, 1],
        weights,
        out=np.zeros_like(weights),
        where=weights > 0.0,
    )
    return np.asarray(directions), weights, stokes


def barycentric(
    point: np.ndarray, triangle: np.ndarray
) -> tuple[np.ndarray, float]:
    a, b, c = triangle
    edge0 = b - a
    edge1 = c - a
    offset = point - a
    d00 = float(np.dot(edge0, edge0))
    d01 = float(np.dot(edge0, edge1))
    d11 = float(np.dot(edge1, edge1))
    d20 = float(np.dot(offset, edge0))
    d21 = float(np.dot(offset, edge1))
    denominator = d00 * d11 - d01 * d01
    if denominator <= 0.0:
        return np.full(3, np.nan), math.inf
    second = (d11 * d20 - d01 * d21) / denominator
    third = (d00 * d21 - d01 * d20) / denominator
    values = np.asarray([1.0 - second - third, second, third])
    normal = np.cross(edge0, edge1)
    normal /= np.linalg.norm(normal)
    distance = abs(float(np.dot(point - a, normal)))
    return values, distance


def locate_surface_points(
    geometry: Path,
    points: np.ndarray,
    *,
    surface_id: int,
    gxe_domain_id: int,
    tolerance_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map local surface points to intrinsic analytic elements or triangles.

    Exact analytic elements are the canonical nonlocal-surface basis when
    present. Triangle ordinals provide the corresponding generic mesh path.
    """

    with h5py.File(geometry, "r") as handle:
        if "/analytic/elements/surface_element" in handle:
            primitive_surface = np.asarray(
                handle["/analytic/surface_id"], dtype=np.uint32
            )
            primitive_minus = np.asarray(
                handle["/analytic/minus_domain_id"], dtype=np.int32
            )
            primitive_plus = np.asarray(
                handle["/analytic/plus_domain_id"], dtype=np.int32
            )
            primitive_axis_x = np.asarray(
                handle["/analytic/axis_x"], dtype=float
            )
            element_primitive = np.asarray(
                handle["/analytic/elements/primitive_index"],
                dtype=np.uint32,
            )
            selected = (
                primitive_surface[element_primitive]
                == np.uint32(surface_id)
            )
            if np.any(selected):
                centers = np.asarray(
                    handle["/analytic/elements/center_mm"], dtype=float
                )[selected]
                elements = np.asarray(
                    handle["/analytic/elements/surface_element"],
                    dtype=np.uint64,
                )[selected]
                normals = np.asarray(
                    handle["/analytic/elements/normal"], dtype=float
                )[selected]
                owners = element_primitive[selected]
                tree = cKDTree(centers)
                distance, nearest = tree.query(points, k=1)
                if np.any(distance > tolerance_mm):
                    bad = int(np.argmax(distance))
                    raise RuntimeError(
                        f"surface point {bad} is not covered by an intrinsic "
                        f"analytic element (distance={distance[bad]:.6g} mm)"
                    )
                owners = owners[nearest]
                if np.all(primitive_minus[owners] == gxe_domain_id):
                    side = np.zeros(len(points), dtype=np.uint64)
                    side_normal = -normals[nearest]
                elif np.all(primitive_plus[owners] == gxe_domain_id):
                    side = np.ones(len(points), dtype=np.uint64)
                    side_normal = normals[nearest]
                else:
                    raise RuntimeError(
                        "analytic LXe elements do not share one GXe side"
                    )
                return (
                    elements[nearest],
                    np.tile(
                        np.asarray([1.0, 0.0, 0.0]), (len(points), 1)
                    ),
                    side,
                    primitive_axis_x[owners],
                    side_normal,
                )
        vertices = np.asarray(handle["/geometry/vertices"], dtype=float)
        triangles = np.asarray(handle["/geometry/triangles"], dtype=np.uint32)
        surface = np.asarray(handle["/geometry/surface_id"], dtype=np.uint32)
        minus = np.asarray(handle["/geometry/minus_domain_id"], dtype=np.int32)
        plus = np.asarray(handle["/geometry/plus_domain_id"], dtype=np.int32)
    primitives = np.flatnonzero(surface == surface_id)
    if primitives.size == 0:
        raise ValueError(f"geometry has no surface_id={surface_id}")
    surface_triangles = vertices[triangles[primitives]]
    centers = surface_triangles.mean(axis=1)
    tree = cKDTree(centers)
    candidates = min(64, len(surface_triangles))
    _, nearby = tree.query(points, k=candidates)
    if candidates == 1:
        nearby = nearby[:, np.newaxis]

    element = np.empty(len(points), dtype=np.uint64)
    coordinates = np.empty((len(points), 3), dtype=float)
    side = np.empty(len(points), dtype=np.uint64)
    tangent = np.empty((len(points), 3), dtype=float)
    side_normal = np.empty((len(points), 3), dtype=float)
    for index, point in enumerate(points):
        selected = None
        best_score = math.inf
        for candidate in np.atleast_1d(nearby[index]):
            values, distance = barycentric(
                point, surface_triangles[int(candidate)]
            )
            negativity = max(0.0, -float(values.min()))
            score = distance + 1000.0 * negativity
            if score < best_score:
                best_score = score
                selected = (int(candidate), values, distance)
            if negativity <= 1.0e-11 and distance <= tolerance_mm:
                selected = (int(candidate), values, distance)
                break
        assert selected is not None
        ordinal, values, distance = selected
        if values.min() < -1.0e-9 or distance > tolerance_mm:
            raise RuntimeError(
                f"surface point {index} is not covered by its triangle group"
            )
        values = np.maximum(values, 0.0)
        values /= values.sum()
        primitive = int(primitives[ordinal])
        triangle = surface_triangles[ordinal]
        geometric = np.cross(
            triangle[1] - triangle[0], triangle[2] - triangle[0]
        )
        geometric /= np.linalg.norm(geometric)
        if int(plus[primitive]) == gxe_domain_id:
            output_side = 1
            normal = geometric
        elif int(minus[primitive]) == gxe_domain_id:
            output_side = 0
            normal = -geometric
        else:
            raise RuntimeError(
                "LXe surface triangle is not adjacent to the GXe domain"
            )
        edge = triangle[1] - triangle[0]
        edge /= np.linalg.norm(edge)
        element[index] = ordinal
        coordinates[index] = values
        side[index] = output_side
        tangent[index] = edge
        side_normal[index] = normal
    return element, coordinates, side, tangent, side_normal


def egress_basis(
    geometry: Path,
    surface_radius_mm: np.ndarray,
    gas_direction: np.ndarray,
    angular_stokes: np.ndarray,
    *,
    surface_phi_bins: int,
    surface_id: int,
    gxe_domain_id: int,
    tolerance_mm: float,
) -> tuple[np.ndarray, ...]:
    surface_phi = (
        2.0
        * math.pi
        * (np.arange(surface_phi_bins, dtype=float) + 0.5)
        / surface_phi_bins
    )
    points = np.asarray(
        [
            [radius * math.cos(phi), radius * math.sin(phi), 0.0]
            for radius in surface_radius_mm
            for phi in surface_phi
        ]
    )
    element, bary, side, tangent, normal = locate_surface_points(
        geometry,
        points,
        surface_id=surface_id,
        gxe_domain_id=gxe_domain_id,
        tolerance_mm=tolerance_mm,
    )
    angular_count = len(gas_direction)
    count = len(points) * angular_count
    result_element = np.repeat(element, angular_count)
    result_bary = np.repeat(bary, angular_count, axis=0)
    result_side = np.repeat(side, angular_count)
    result_direction = np.empty((count, 3), dtype=float)
    result_reference = np.empty((count, 3), dtype=float)
    result_stokes = np.tile(angular_stokes, (len(points), 1))
    for point_index in range(len(points)):
        bitangent = np.cross(normal[point_index], tangent[point_index])
        basis = np.column_stack(
            [tangent[point_index], bitangent, normal[point_index]]
        )
        for angle_index, direction in enumerate(gas_direction):
            output = point_index * angular_count + angle_index
            result_direction[output] = basis.T @ direction
            azimuth = math.atan2(float(direction[1]), float(direction[0]))
            s_axis = np.asarray([-math.sin(azimuth), math.cos(azimuth), 0.0])
            if np.linalg.norm(s_axis) < 1.0e-14:
                s_axis = tangent[point_index]
            result_reference[output] = basis.T @ s_axis
    return (
        result_element,
        result_bary,
        result_side,
        result_direction,
        result_stokes,
        result_reference,
    )


def cache_key(
    arguments: argparse.Namespace, geometry_contract: dict[str, object]
) -> str:
    material = {
        "schema": "oos.dual-phase-tpc.lxe-function-generator.v2",
        "geometry_contract_sha256": geometry_contract[
            "fingerprint_sha256"
        ],
        "generator_sha256": source_digest(),
        "parameters": {
            key: value
            for key, value in vars(arguments).items()
            if key not in {
                "output",
                "force",
                "geometry",
                "processes",
                "test",
            }
        },
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode()
    ).hexdigest()


def stored_cache_key(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as handle:
            raw = bytes(
                np.asarray(handle["/metadata/generator_json"], dtype=np.uint8)
            )
        return json.loads(raw.decode()).get("cache_key_sha256")
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--geometry", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--surface-id", type=int, default=1)
    result.add_argument("--gxe-domain-id", type=int, default=0)
    result.add_argument("--geometry-tolerance-mm", type=float, default=1e-8)
    result.add_argument("--position-radial-bins", type=int, default=24)
    result.add_argument("--position-phi-bins", type=int, default=72)
    result.add_argument("--direction-mu-bins", type=int, default=4)
    result.add_argument("--direction-phi-bins", type=int, default=16)
    result.add_argument("--direction-mu-minimum", type=float, default=0.804)
    result.add_argument("--surface-radial-order", type=int, default=40)
    result.add_argument("--surface-phi-bins", type=int, default=80)
    result.add_argument("--return-mu-order", type=int, default=4)
    result.add_argument("--return-direction-phi", type=int, default=8)
    result.add_argument("--first-scatter-order", type=int, default=8)
    result.add_argument("--explicit-collision-order", type=int, default=7)
    result.add_argument("--collision-sample-power", type=int, default=12)
    result.add_argument("--collision-maximum-events", type=int, default=64)
    result.add_argument("--processes", type=int, default=24)
    result.add_argument("--rayleigh-length-mm", type=float,
                        default=341.51442280354416)
    result.add_argument("--absorption-length-mm", type=float, default=70000.0)
    result.add_argument("--side-reflectivity", type=float, default=0.95)
    result.add_argument("--bottom-reflectivity", type=float, default=0.0)
    result.add_argument("--n-lxe", type=float, default=1.6829)
    result.add_argument("--n-gxe", type=float, default=1.000702)
    result.add_argument("--azimuthal-maximum", type=int, default=32)
    result.add_argument("--radial-modes", type=int, default=120)
    result.add_argument("--root-samples", type=int, default=120)
    result.add_argument(
        "--test",
        action="store_true",
        help="use a coarse liquid grid while preserving the egress contract",
    )
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    arguments = parser().parse_args()
    geometry_contract = geometry_lxe_contract(
        arguments.geometry,
        surface_id=arguments.surface_id,
        gxe_domain_id=arguments.gxe_domain_id,
    )
    arguments.radius_mm = geometry_contract["radius_mm"]
    arguments.depth_mm = geometry_contract["depth_mm"]
    if arguments.test:
        apply_test_profile(arguments)
    key = cache_key(arguments, geometry_contract)
    if not arguments.force and stored_cache_key(arguments.output) == key:
        print(f"cache hit: {arguments.output}")
        print(f"cache key: {key}")
        return
    grid = LXePhaseSpaceGrid(
        radius_mm=arguments.radius_mm,
        position_radial_bins=arguments.position_radial_bins,
        position_phi_bins=arguments.position_phi_bins,
        direction_mu_bins=arguments.direction_mu_bins,
        direction_phi_bins=arguments.direction_phi_bins,
        direction_phi_relative_to_position=True,
        deposition="multilinear",
        radial_node_spacing="chebyshev",
        direction_mu_minimum=arguments.direction_mu_minimum,
    )
    lxe = LXeDiffusionConfig(
        radius_mm=arguments.radius_mm,
        depth_mm=arguments.depth_mm,
        rayleigh_length_mm=arguments.rayleigh_length_mm,
        absorption_length_mm=arguments.absorption_length_mm,
        n_lxe=arguments.n_lxe,
        n_gxe=arguments.n_gxe,
        side_reflectivity=arguments.side_reflectivity,
        bottom_reflectivity=arguments.bottom_reflectivity,
    )
    truncation = ModeTruncation(
        azimuthal_maximum=arguments.azimuthal_maximum,
        radial_modes=arguments.radial_modes,
        samples_per_expected_root=arguments.root_samples,
    )
    operator = build_lxe_response_operator(
        LXeCylinderGreen(lxe, truncation, mode_workers=1),
        grid,
        surface_radial_order=arguments.surface_radial_order,
        first_scatter_order=arguments.first_scatter_order,
        explicit_collision_order=arguments.explicit_collision_order,
        collision_sample_power=arguments.collision_sample_power,
        collision_maximum_events=arguments.collision_maximum_events,
        processes=arguments.processes,
        coefficient_dtype="complex128",
    )
    gas_direction, angular_weight, angular_stokes = (
        diffuse_escape_quadrature(
            lxe,
            liquid_mu_order=arguments.return_mu_order,
            direction_phi=arguments.return_direction_phi,
        )
    )
    egress = egress_basis(
        arguments.geometry,
        operator.surface_radius_mm,
        gas_direction,
        angular_stokes,
        surface_phi_bins=arguments.surface_phi_bins,
        surface_id=arguments.surface_id,
        gxe_domain_id=arguments.gxe_domain_id,
        tolerance_mm=arguments.geometry_tolerance_mm,
    )
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    write_factorized_block(
        temporary,
        coefficients=operator.coefficients,
        expected_return=operator.expected_return,
        audit_values=operator.audit_values,
        surface_radius_mm=operator.surface_radius_mm,
        surface_ring_area_mm2=operator.surface_ring_area_mm2,
        angular_weight=angular_weight,
        surface_element=egress[0],
        barycentric=egress[1],
        side=egress[2],
        direction_local=egress[3],
        stokes=egress[4],
        reference_axis_local=egress[5],
        phase_grid=asdict(grid),
        audit_names=list(operator.audit_names),
        metadata={
            **operator.metadata,
            "cache_key_sha256": key,
            "geometry_sha256": sha256(arguments.geometry),
            "geometry_contract_sha256": geometry_contract[
                "fingerprint_sha256"
            ],
            "geometry_contract": geometry_contract,
            "generator_sha256": source_digest(),
            "surface_phi_bins": arguments.surface_phi_bins,
            "return_mu_order": arguments.return_mu_order,
            "return_direction_phi": arguments.return_direction_phi,
        },
    )
    temporary.replace(arguments.output)
    print(f"built: {arguments.output}")
    print(f"cache key: {key}")


if __name__ == "__main__":
    main()
