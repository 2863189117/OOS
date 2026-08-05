#!/usr/bin/env python3
"""Build the dual-phase TPC mesh plus an analytic transport basis.

The fine constrained triangle mesh remains the canonical topology/enclosure
representation.  Transport is instead owned by exact disks, annuli, finite
cylinders, and generated quadtree/sector radiance elements.  Consequently
changing triangle quality does not change the optical operator, while every
discretization control remains explicit and reproducible.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as exception:  # pragma: no cover - operational message
    raise SystemExit(
        "build_analytic_geometry.py requires scipy for the "
        "quadtree nearest-aperture test"
    ) from exception

from build_geometry import (
    ABSORBING_BOUNDARY,
    AL_FLASHING,
    FIELD_CAGE_PTFE,
    GXE,
    GXE_LXE,
    LXE,
    OUTER_WALL_PTFE,
    OUTSIDE,
    PHOTOCATHODE,
    QUARTZ,
    QUARTZ_SIDE_ABSORBER,
    QUARTZ_WINDOW,
    TOP_PTFE,
    GeometryConfig,
    PROFILES as VALIDATION_PROFILES,
    file_sha256,
    pmt_layout,
    write_geometry,
)


DISK = 1
ANNULUS = 2
FINITE_CYLINDER = 3
PERFORATED_DISK = 5

PLANE_UV = 0
CYLINDER_PHI_Z = 1
ANNULUS_R2_PHI = 2

SOURCE_VISIBILITY_RAY_TRACED = 0
SOURCE_VISIBILITY_DIRECT = 1
SOURCE_VISIBILITY_PROJECTED_APERTURE = 2
SOURCE_INTEGRAL_NONE = 0
SOURCE_INTEGRAL_DIRECTIONAL_DISK = 1


@dataclass(frozen=True)
class AnalyticDiscretization:
    top_max_cell_mm: float = 40.0
    boundary_cell_mm: float = 5.0
    boundary_samples: int = 7
    hole_wall_phi: int = 4
    hole_wall_z: int = 1
    outer_wall_phi: int = 96
    outer_wall_max_z_mm: float = 20.0

    @classmethod
    def profile(cls, name: str) -> "AnalyticDiscretization":
        if name == "debug":
            return cls(160.0, 20.0, 5, 2, 1, 32, 30.0)
        if name == "coarse":
            return cls()
        if name == "production":
            return cls(10.0, 1.25, 7, 8, 2, 384, 10.0)
        if name == "fine":
            return cls(5.0, 0.625, 9, 12, 2, 576, 5.0)
        raise ValueError(f"unknown analytic profile {name!r}")


@dataclass(frozen=True)
class AnalyticElement:
    primitive_index: int
    coordinates: int
    bounds: tuple[float, float, float, float, float]
    center_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    area_mm2: float
    basis_id: int
    surface_element: int
    projected_aperture_primitive_index: int = np.iinfo(np.uint32).max
    projected_aperture_hole_index: int = np.iinfo(np.uint32).max
    source_quadrature: bool = True
    source_visibility: int = SOURCE_VISIBILITY_RAY_TRACED


@dataclass(frozen=True)
class Primitive:
    kind: int
    center_mm: tuple[float, float, float]
    parameters: tuple[float, float, float, float]
    normal_sign: float
    surface_id: int
    minus_domain_id: int
    plus_domain_id: int
    channel_id: int = -1
    surface_element: int = 0
    source_integral: int = SOURCE_INTEGRAL_NONE
    holes: tuple[tuple[float, float, float], ...] = ()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cell_relation(
    x: float,
    y: float,
    half: float,
    radius: float,
    pmt_tree: cKDTree,
    aperture_radius: float,
) -> int:
    center_radius = math.hypot(x, y)
    half_diagonal = math.sqrt(2.0) * half
    if center_radius - half_diagonal >= radius:
        return -1
    outer_full = center_radius + half_diagonal <= radius
    distances, _ = pmt_tree.query([x, y], k=min(4, int(pmt_tree.n)))
    distances = np.atleast_1d(distances)
    if np.any(distances + half_diagonal <= aperture_radius):
        return -1
    holes_clear = bool(
        np.all(distances - half_diagonal >= aperture_radius)
    )
    return 1 if outer_full and holes_clear else 0


def _sample_partial_top(
    x: float,
    y: float,
    half: float,
    radius: float,
    pmt_tree: cKDTree,
    aperture_radius: float,
    samples: int,
) -> tuple[tuple[float, float], float] | None:
    offsets = (np.arange(samples, dtype=float) + 0.5) / samples
    offsets = (2.0 * offsets - 1.0) * half
    xx, yy = np.meshgrid(x + offsets, y + offsets, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside_outer = np.sum(points * points, axis=1) <= radius * radius
    nearest, _ = pmt_tree.query(points)
    material = inside_outer & (nearest >= aperture_radius)
    count = int(material.sum())
    if count == 0:
        return None
    selected = points[material]
    area = (2.0 * half) ** 2 * count / points.shape[0]
    centroid = selected.mean(axis=0)
    return (float(centroid[0]), float(centroid[1])), float(area)


def _append_primitive(
    primitives: list[Primitive], value: Primitive
) -> int:
    primitives.append(value)
    return len(primitives) - 1


def _area_quadrature_elements(
    primitive_index: int,
    *,
    center_xy: tuple[float, float],
    z_mm: float,
    inner_radius_mm: float,
    outer_radius_mm: float,
    normal_z: float,
    radial_order: int = 4,
    phi_count: int = 16,
    surface_element_offset: int = 0,
    projected_aperture_primitive_index: int = np.iinfo(np.uint32).max,
    projected_aperture_hole_index: int = np.iinfo(np.uint32).max,
    source_quadrature: bool = True,
) -> list[AnalyticElement]:
    nodes, weights = np.polynomial.legendre.leggauss(radial_order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    u_edges = np.concatenate([[0.0], np.cumsum(weights)])
    u_edges[-1] = 1.0
    inner2 = inner_radius_mm**2
    span2 = outer_radius_mm**2 - inner2
    result: list[AnalyticElement] = []
    for radial_index, (node, weight) in enumerate(zip(nodes, weights)):
        radius2 = inner2 + span2 * float(node)
        radius = math.sqrt(max(0.0, radius2))
        lower2 = inner2 + span2 * float(u_edges[radial_index])
        upper2 = inner2 + span2 * float(u_edges[radial_index + 1])
        for phi_index in range(phi_count):
            phi_lower = 2.0 * math.pi * phi_index / phi_count
            phi_upper = 2.0 * math.pi * (phi_index + 1) / phi_count
            phi = 0.5 * (phi_lower + phi_upper)
            local = radial_index * phi_count + phi_index
            result.append(
                AnalyticElement(
                    primitive_index,
                    ANNULUS_R2_PHI,
                    (lower2, upper2, phi_lower, phi_upper, 0.0),
                    (
                        center_xy[0] + radius * math.cos(phi),
                        center_xy[1] + radius * math.sin(phi),
                        z_mm,
                    ),
                    (0.0, 0.0, normal_z),
                    math.pi
                    * span2
                    * float(weight)
                    / phi_count,
                    0,
                    surface_element_offset + local,
                    projected_aperture_primitive_index,
                    projected_aperture_hole_index,
                    source_quadrature,
                    (
                        SOURCE_VISIBILITY_PROJECTED_APERTURE
                        if projected_aperture_primitive_index
                        != np.iinfo(np.uint32).max
                        else SOURCE_VISIBILITY_DIRECT
                    ),
                )
            )
    return result


def build_analytic_transport(
    config: GeometryConfig,
    discretization: AnalyticDiscretization,
    pmt_xy: np.ndarray,
    channel_id: np.ndarray,
) -> tuple[list[Primitive], list[AnalyticElement]]:
    primitives: list[Primitive] = []
    elements: list[AnalyticElement] = []
    pmt_tree = cKDTree(pmt_xy)

    top_primitive = _append_primitive(
        primitives,
        Primitive(
            PERFORATED_DISK,
            (0.0, 0.0, config.height_mm),
            (config.active_radius_mm, 0.0, 0.0, 0.0),
            1.0,
            TOP_PTFE,
            GXE,
            OUTSIDE,
            holes=tuple(
                (float(x), float(y), config.aperture_radius_mm)
                for x, y in pmt_xy
            ),
        ),
    )
    hole_primitives = [
        _append_primitive(
            primitives,
            Primitive(
                FINITE_CYLINDER,
                (
                    float(center[0]),
                    float(center[1]),
                    config.height_mm + 0.5 * config.hole_depth_mm,
                ),
                (
                    config.aperture_radius_mm,
                    0.5 * config.hole_depth_mm,
                    0.0,
                    0.0,
                ),
                1.0,
                TOP_PTFE,
                GXE,
                OUTSIDE,
                surface_element=index + 1,
            ),
        )
        for index, center in enumerate(pmt_xy)
    ]
    outer_primitive = _append_primitive(
        primitives,
        Primitive(
            FINITE_CYLINDER,
            (0.0, 0.0, 0.5 * config.height_mm),
            (
                config.wall_radius_mm,
                0.5 * config.height_mm,
                0.0,
                0.0,
            ),
            1.0,
            OUTER_WALL_PTFE,
            GXE,
            OUTSIDE,
        ),
    )
    field_primitive = _append_primitive(
        primitives,
        Primitive(
            ANNULUS,
            (0.0, 0.0, 0.0),
            (
                config.active_radius_mm,
                config.field_cage_outer_radius_mm,
                0.0,
                0.0,
            ),
            -1.0,
            FIELD_CAGE_PTFE,
            GXE,
            OUTSIDE,
        ),
    )
    lxe_primitive = _append_primitive(
        primitives,
        Primitive(
            DISK,
            (0.0, 0.0, 0.0),
            (config.active_radius_mm, 0.0, 0.0, 0.0),
            -1.0,
            GXE_LXE,
            GXE,
            LXE,
            source_integral=SOURCE_INTEGRAL_DIRECTIONAL_DISK,
        ),
    )

    # Absorbing horizontal remainder on both ends of the GXe cylinder.
    absorbing_primitives: list[tuple[int, float, float, float]] = []
    for z, sign, inner in (
        (config.height_mm, 1.0, config.active_radius_mm),
        (0.0, -1.0, config.field_cage_outer_radius_mm),
    ):
        primitive_index = _append_primitive(
            primitives,
            Primitive(
                ANNULUS,
                (0.0, 0.0, z),
                (
                    inner,
                    config.wall_radius_mm,
                    0.0,
                    0.0,
                ),
                sign,
                ABSORBING_BOUNDARY,
                GXE,
                OUTSIDE,
            ),
        )
        absorbing_primitives.append((primitive_index, z, sign, inner))

    # Exact circular PMT boundaries.  These are terminal/specular surfaces,
    # not diffuse basis states.
    flashing_min_z, flashing_max_z = config.al_flashing_z_range_mm
    quartz_primitives: list[int] = []
    for index, (center, channel) in enumerate(zip(pmt_xy, channel_id)):
        quartz_primitive = _append_primitive(
            primitives,
            Primitive(
                DISK,
                (
                    float(center[0]),
                    float(center[1]),
                    config.quartz_bottom_z_mm,
                ),
                (config.aperture_radius_mm, 0.0, 0.0, 0.0),
                1.0,
                QUARTZ_WINDOW,
                GXE,
                QUARTZ,
                surface_element=index,
            ),
        )
        quartz_primitives.append(quartz_primitive)
        _append_primitive(
            primitives,
            Primitive(
                DISK,
                (
                    float(center[0]),
                    float(center[1]),
                    config.photocathode_z_mm,
                ),
                (config.aperture_radius_mm, 0.0, 0.0, 0.0),
                1.0,
                PHOTOCATHODE,
                QUARTZ,
                OUTSIDE,
                channel_id=int(channel),
                surface_element=index,
            ),
        )
        for lower, upper, surface in (
            (
                config.quartz_bottom_z_mm,
                flashing_min_z,
                QUARTZ_SIDE_ABSORBER,
            ),
            (flashing_min_z, flashing_max_z, AL_FLASHING),
            (
                flashing_max_z,
                config.photocathode_z_mm,
                QUARTZ_SIDE_ABSORBER,
            ),
        ):
            if upper <= lower:
                continue
            _append_primitive(
                primitives,
                Primitive(
                    FINITE_CYLINDER,
                    (
                        float(center[0]),
                        float(center[1]),
                        0.5 * (lower + upper),
                    ),
                    (
                        config.aperture_radius_mm,
                        0.5 * (upper - lower),
                        0.0,
                        0.0,
                    ),
                    1.0,
                    surface,
                    QUARTZ,
                    OUTSIDE,
                    surface_element=index,
                ),
            )

    # Adaptive quadtree cells describe the perforated top reflector without
    # depending on the validation triangulation.
    patch = 0
    stack = [(0.0, 0.0, config.active_radius_mm)]
    while stack:
        x, y, half = stack.pop()
        relation = _cell_relation(
            x,
            y,
            half,
            config.active_radius_mm,
            pmt_tree,
            config.aperture_radius_mm,
        )
        size = 2.0 * half
        refine = size > discretization.top_max_cell_mm or (
            relation == 0 and size > discretization.boundary_cell_mm
        )
        if refine:
            child_half = 0.5 * half
            for dx in (-child_half, child_half):
                for dy in (-child_half, child_half):
                    stack.append((x + dx, y + dy, child_half))
            continue
        if relation == 1:
            center = (x, y)
            area = size * size
        elif relation == 0:
            sampled = _sample_partial_top(
                x,
                y,
                half,
                config.active_radius_mm,
                pmt_tree,
                config.aperture_radius_mm,
                discretization.boundary_samples,
            )
            if sampled is None:
                continue
            center, area = sampled
        else:
            continue
        elements.append(
            AnalyticElement(
                top_primitive,
                PLANE_UV,
                (x - half, x + half, y - half, y + half, 0.0),
                (center[0], center[1], config.height_mm),
                (0.0, 0.0, 1.0),
                area,
                patch,
                patch,
                source_visibility=SOURCE_VISIBILITY_DIRECT,
            )
        )
        patch += 1

    dphi = 2.0 * math.pi / discretization.hole_wall_phi
    dz = config.hole_depth_mm / discretization.hole_wall_z
    for pmt_index, pmt in enumerate(pmt_xy):
        primitive_index = hole_primitives[pmt_index]
        for z_index in range(discretization.hole_wall_z):
            local_lower_z = (
                -0.5 * config.hole_depth_mm + z_index * dz
            )
            for phi_index in range(discretization.hole_wall_phi):
                phi_lower = phi_index * dphi
                phi_center = (phi_index + 0.5) * dphi
                radial = (math.cos(phi_center), math.sin(phi_center))
                elements.append(
                    AnalyticElement(
                        primitive_index,
                        CYLINDER_PHI_Z,
                        (
                            phi_lower,
                            phi_lower + dphi,
                            local_lower_z,
                            local_lower_z + dz,
                            0.0,
                        ),
                        (
                            float(pmt[0])
                            + config.aperture_radius_mm * radial[0],
                            float(pmt[1])
                            + config.aperture_radius_mm * radial[1],
                            config.height_mm + (z_index + 0.5) * dz,
                        ),
                        (radial[0], radial[1], 0.0),
                        config.aperture_radius_mm * dphi * dz,
                        patch,
                        patch,
                        projected_aperture_primitive_index=top_primitive,
                        projected_aperture_hole_index=pmt_index,
                        source_visibility=(
                            SOURCE_VISIBILITY_PROJECTED_APERTURE
                        ),
                    )
                )
                patch += 1

    outer_nz = max(
        1,
        math.ceil(
            config.height_mm / discretization.outer_wall_max_z_mm
        ),
    )
    outer_dz = config.height_mm / outer_nz
    outer_dphi = 2.0 * math.pi / discretization.outer_wall_phi
    for z_index in range(outer_nz):
        local_lower_z = -0.5 * config.height_mm + z_index * outer_dz
        for phi_index in range(discretization.outer_wall_phi):
            phi_lower = phi_index * outer_dphi
            phi_center = (phi_index + 0.5) * outer_dphi
            radial = (math.cos(phi_center), math.sin(phi_center))
            elements.append(
                AnalyticElement(
                    outer_primitive,
                    CYLINDER_PHI_Z,
                    (
                        phi_lower,
                        phi_lower + outer_dphi,
                        local_lower_z,
                        local_lower_z + outer_dz,
                        0.0,
                    ),
                    (
                        config.wall_radius_mm * radial[0],
                        config.wall_radius_mm * radial[1],
                        (z_index + 0.5) * outer_dz,
                    ),
                    (radial[0], radial[1], 0.0),
                    config.wall_radius_mm * outer_dphi * outer_dz,
                    patch,
                    patch,
                    source_visibility=SOURCE_VISIBILITY_DIRECT,
                )
            )
            patch += 1

    field_thickness = (
        config.field_cage_outer_radius_mm - config.active_radius_mm
    )
    if field_thickness > 0.0:
        radial_bins = max(
            1, math.ceil(field_thickness / discretization.boundary_cell_mm)
        )
        inner2 = config.active_radius_mm**2
        outer2 = config.field_cage_outer_radius_mm**2
        sector_area = math.pi * (outer2 - inner2) / (
            radial_bins * discretization.outer_wall_phi
        )
        for radial_index in range(radial_bins):
            lower2 = inner2 + (outer2 - inner2) * (
                radial_index / radial_bins
            )
            upper2 = inner2 + (outer2 - inner2) * (
                (radial_index + 1) / radial_bins
            )
            radius = math.sqrt(0.5 * (lower2 + upper2))
            for phi_index in range(discretization.outer_wall_phi):
                phi_lower = (
                    2.0
                    * math.pi
                    * phi_index
                    / discretization.outer_wall_phi
                )
                phi_upper = (
                    2.0
                    * math.pi
                    * (phi_index + 1)
                    / discretization.outer_wall_phi
                )
                phi_center = 0.5 * (phi_lower + phi_upper)
                elements.append(
                    AnalyticElement(
                        field_primitive,
                        ANNULUS_R2_PHI,
                        (
                            lower2,
                            upper2,
                            phi_lower,
                            phi_upper,
                            0.0,
                        ),
                        (
                            radius * math.cos(phi_center),
                            radius * math.sin(phi_center),
                            0.0,
                        ),
                        (0.0, 0.0, -1.0),
                        sector_area,
                        patch,
                        patch,
                        source_visibility=SOURCE_VISIBILITY_DIRECT,
                    )
                )
                patch += 1
    # The nonlocal LXe block emits on the same intrinsic area-uniform
    # 40-by-80 surface basis used by its Fourier--Bessel return operator.
    # These elements are local to the GXe--LXe surface; they do not refer to
    # the validation triangles and therefore remain stable if that mesh is
    # retriangulated.
    elements.extend(
        _area_quadrature_elements(
            lxe_primitive,
            center_xy=(0.0, 0.0),
            z_mm=0.0,
            inner_radius_mm=0.0,
            outer_radius_mm=config.active_radius_mm,
            normal_z=-1.0,
            radial_order=40,
            phi_count=80,
            source_quadrature=False,
        )
    )
    for pmt_index, (center, primitive_index) in enumerate(
        zip(pmt_xy, quartz_primitives)
    ):
        elements.extend(
            _area_quadrature_elements(
                primitive_index,
                center_xy=(float(center[0]), float(center[1])),
                z_mm=config.quartz_bottom_z_mm,
                inner_radius_mm=0.0,
                outer_radius_mm=config.aperture_radius_mm,
                normal_z=1.0,
                surface_element_offset=pmt_index * 64,
                projected_aperture_primitive_index=top_primitive,
                projected_aperture_hole_index=pmt_index,
            )
        )
    for primitive_index, z, sign, inner in absorbing_primitives:
        elements.extend(
            _area_quadrature_elements(
                primitive_index,
                center_xy=(0.0, 0.0),
                z_mm=z,
                inner_radius_mm=inner,
                outer_radius_mm=config.wall_radius_mm,
                normal_z=sign,
            )
        )
    return primitives, elements


def _write_analytic(
    handle: h5py.File,
    primitives: list[Primitive],
    elements: list[AnalyticElement],
) -> None:
    analytic = handle.create_group("analytic")
    count = len(primitives)
    analytic.create_dataset(
        "kind", data=np.asarray([value.kind for value in primitives],
                                dtype=np.uint8)
    )
    analytic.create_dataset(
        "center_mm",
        data=np.asarray([value.center_mm for value in primitives],
                        dtype=np.float64),
    )
    identity_axes = np.tile(np.eye(3, dtype=np.float64), (count, 1, 1))
    analytic.create_dataset("axis_x", data=identity_axes[:, 0])
    analytic.create_dataset("axis_y", data=identity_axes[:, 1])
    analytic.create_dataset("axis_z", data=identity_axes[:, 2])
    analytic.create_dataset(
        "parameters",
        data=np.asarray([value.parameters for value in primitives],
                        dtype=np.float64),
    )
    analytic.create_dataset(
        "normal_sign",
        data=np.asarray([value.normal_sign for value in primitives],
                        dtype=np.float64),
    )
    analytic.create_dataset(
        "source_integral",
        data=np.asarray(
            [value.source_integral for value in primitives],
            dtype=np.uint8,
        ),
    )
    analytic.create_dataset(
        "surface_id",
        data=np.asarray([value.surface_id for value in primitives],
                        dtype=np.uint32),
    )
    analytic.create_dataset(
        "surface_basis_id",
        data=np.full(count, np.iinfo(np.uint32).max, dtype=np.uint32),
    )
    analytic.create_dataset(
        "minus_domain_id",
        data=np.asarray(
            [value.minus_domain_id for value in primitives],
            dtype=np.int32,
        ),
    )
    analytic.create_dataset(
        "plus_domain_id",
        data=np.asarray(
            [value.plus_domain_id for value in primitives],
            dtype=np.int32,
        ),
    )
    analytic.create_dataset(
        "channel_id",
        data=np.asarray([value.channel_id for value in primitives],
                        dtype=np.int32),
    )
    analytic.create_dataset(
        "surface_element",
        data=np.asarray(
            [value.surface_element for value in primitives],
            dtype=np.uint64,
        ),
    )
    hole_offset = [0]
    hole_center: list[tuple[float, float]] = []
    hole_radius: list[float] = []
    for value in primitives:
        for x, y, radius in value.holes:
            hole_center.append((x, y))
            hole_radius.append(radius)
        hole_offset.append(len(hole_radius))
    analytic.create_dataset(
        "hole_offset", data=np.asarray(hole_offset, dtype=np.uint64)
    )
    analytic.create_dataset(
        "hole_center_uv_mm",
        data=np.asarray(hole_center, dtype=np.float64).reshape(-1, 2),
    )
    analytic.create_dataset(
        "hole_radius_mm",
        data=np.asarray(hole_radius, dtype=np.float64),
    )
    element_group = analytic.create_group("elements")
    element_group.create_dataset(
        "primitive_index",
        data=np.asarray(
            [value.primitive_index for value in elements], dtype=np.uint32
        ),
    )
    element_group.create_dataset(
        "coordinates",
        data=np.asarray(
            [value.coordinates for value in elements], dtype=np.uint8
        ),
    )
    element_group.create_dataset(
        "bounds",
        data=np.asarray([value.bounds for value in elements],
                        dtype=np.float64),
    )
    element_group.create_dataset(
        "center_mm",
        data=np.asarray([value.center_mm for value in elements],
                        dtype=np.float64),
    )
    element_group.create_dataset(
        "normal",
        data=np.asarray([value.normal for value in elements],
                        dtype=np.float64),
    )
    element_group.create_dataset(
        "area_mm2",
        data=np.asarray([value.area_mm2 for value in elements],
                        dtype=np.float64),
    )
    element_group.create_dataset(
        "surface_basis_id",
        data=np.asarray([value.basis_id for value in elements],
                        dtype=np.uint32),
    )
    element_group.create_dataset(
        "surface_element",
        data=np.asarray([value.surface_element for value in elements],
                        dtype=np.uint64),
    )
    element_group.create_dataset(
        "source_quadrature",
        data=np.asarray(
            [value.source_quadrature for value in elements],
            dtype=np.uint8,
        ),
    )
    element_group.create_dataset(
        "source_visibility",
        data=np.asarray(
            [value.source_visibility for value in elements],
            dtype=np.uint8,
        ),
    )
    element_group.create_dataset(
        "projected_aperture_primitive_index",
        data=np.asarray(
            [
                value.projected_aperture_primitive_index
                for value in elements
            ],
            dtype=np.uint32,
        ),
    )
    element_group.create_dataset(
        "projected_aperture_hole_index",
        data=np.asarray(
            [value.projected_aperture_hole_index for value in elements],
            dtype=np.uint32,
        ),
    )


def build(
    output: Path,
    config: GeometryConfig,
    discretization: AnalyticDiscretization,
    *,
    validation_profile: str,
    force: bool,
) -> dict:
    identity = {
        "schema": "oos.dual-phase-tpc.analytic-geometry.v1",
        "geometry": asdict(config),
        "analytic_discretization": asdict(discretization),
        "validation_profile": validation_profile,
        "generator_sha256": file_sha256(Path(__file__).resolve()),
    }
    if not force and output.is_file():
        try:
            with h5py.File(output, "r") as handle:
                previous = json.loads(
                    bytes(handle["/metadata/analytic_generator_json"][:])
                    .decode()
                )
                if previous == identity:
                    return {
                        "path": str(output.resolve()),
                        "sha256": file_sha256(output),
                        "states": int(
                            handle["/analytic/elements/area_mm2"].shape[0]
                        ),
                        "cache": "hit",
                    }
        except Exception:
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, base_name = tempfile.mkstemp(
        prefix=output.name + ".base.", dir=output.parent
    )
    os.close(descriptor)
    base = Path(base_name)
    descriptor, result_name = tempfile.mkstemp(
        prefix=output.name + ".tmp.", dir=output.parent
    )
    os.close(descriptor)
    result = Path(result_name)
    try:
        values = {
            **asdict(config),
            **VALIDATION_PROFILES[validation_profile],
        }
        validation_config = GeometryConfig(**values)
        write_geometry(
            base,
            validation_config,
            force=True,
        )
        shutil.copy2(base, result)
        pmt_xy, channel_id = pmt_layout(config)
        primitives, elements = build_analytic_transport(
            config, discretization, pmt_xy, channel_id
        )
        with h5py.File(result, "r+") as handle:
            surfaces = np.asarray(
                handle["/geometry/surface_id"][:], dtype=np.uint32
            )
            replaced = np.isin(
                surfaces,
                np.asarray(
                    [
                        TOP_PTFE,
                        GXE_LXE,
                        ABSORBING_BOUNDARY,
                        FIELD_CAGE_PTFE,
                        OUTER_WALL_PTFE,
                        QUARTZ_WINDOW,
                        PHOTOCATHODE,
                        QUARTZ_SIDE_ABSORBER,
                        AL_FLASHING,
                    ],
                    dtype=np.uint32,
                ),
            )
            handle["/geometry/triangle_transport"][...] = np.where(
                replaced, 0, 1
            ).astype(np.uint8)
            # The exact analytic LXe disk owns ray intersection and its
            # 40x80 intrinsic elements own only the return basis.  The
            # canonical triangulation independently partitions the disk for
            # exact-solid-angle adaptive source integration.
            handle["/geometry/triangle_source_quadrature"][...] = np.where(
                replaced,
                surfaces == GXE_LXE,
                1,
            ).astype(np.uint8)
            lxe_analytic_primitive = next(
                index
                for index, primitive in enumerate(primitives)
                if primitive.surface_id == GXE_LXE
                and primitive.kind == DISK
            )
            source_replacement = np.full(
                surfaces.shape,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
            source_replacement[surfaces == GXE_LXE] = (
                lxe_analytic_primitive
            )
            handle[
                "/geometry/triangle_source_analytic_primitive"
            ][...] = source_replacement
            _write_analytic(handle, primitives, elements)
            metadata = handle["/metadata"]
            metadata.create_dataset(
                "analytic_generator_json",
                data=np.frombuffer(
                    json.dumps(identity, sort_keys=True).encode(),
                    dtype=np.uint8,
                ),
            )
        result.replace(output)
    finally:
        base.unlink(missing_ok=True)
        result.unlink(missing_ok=True)
    return {
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "analytic_primitives": len(primitives),
        "diffuse_states": len(elements),
        "source_quadrature_elements": len(elements),
        "configuration_sha256": _sha256_json(identity),
        "cache": "miss",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--analytic-profile",
        choices=("debug", "coarse", "production", "fine"),
        default="coarse",
    )
    parser.add_argument(
        "--validation-profile",
        choices=tuple(VALIDATION_PROFILES),
        default="production",
    )
    parser.add_argument("--height-mm", type=float, default=60.0)
    parser.add_argument("--top-max-cell-mm", type=float)
    parser.add_argument("--boundary-cell-mm", type=float)
    parser.add_argument("--boundary-samples", type=int)
    parser.add_argument("--hole-wall-phi", type=int)
    parser.add_argument("--hole-wall-z", type=int)
    parser.add_argument("--outer-wall-phi", type=int)
    parser.add_argument("--outer-wall-max-z-mm", type=float)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    config = replace(GeometryConfig(), height_mm=arguments.height_mm)
    discretization = AnalyticDiscretization.profile(arguments.analytic_profile)
    overrides = {
        name: getattr(arguments, name)
        for name in (
            "top_max_cell_mm",
            "boundary_cell_mm",
            "boundary_samples",
            "hole_wall_phi",
            "hole_wall_z",
            "outer_wall_phi",
            "outer_wall_max_z_mm",
        )
        if getattr(arguments, name) is not None
    }
    discretization = replace(discretization, **overrides)
    print(
        json.dumps(
            build(
                arguments.output,
                config,
                discretization,
                validation_profile=arguments.validation_profile,
                force=arguments.force,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
