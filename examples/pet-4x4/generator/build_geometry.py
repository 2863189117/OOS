#!/usr/bin/env python3
"""Generate the closed PET-4x4 deterministic optical geometry."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import tempfile

import h5py
import numpy as np


PTFE = 0
QUARTZ_INTERFACE = 1
PHOTOCATHODE = 2
SILICON_INTERFACE = 3
CATHODE = 4
ANODE_WIRE = 5

LXE = 0
QUARTZ = 1
SILICON = 2
OUTSIDE = -1

NO_BASIS = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class GeometryConfig:
    lxe_half_x_mm: float = 34.9
    lxe_half_y_mm: float = 34.9
    lxe_half_z_mm: float = 24.9
    quartz_half_x_mm: float = 34.0
    quartz_half_y_mm: float = 34.0
    quartz_half_z_mm: float = 0.25
    quartz_center_z_mm: float = 17.75
    sipm_grid_size: int = 4
    sipm_pitch_mm: float = 16.25
    sipm_half_x_mm: float = 7.5
    sipm_half_y_mm: float = 7.5
    sipm_half_z_mm: float = 1.0
    sipm_center_z_mm: float = 22.9
    photocathode_width_mm: float = 5.95
    photocathode_gap_mm: float = 0.5
    photocathode_stack_depth_mm: float = 1.0
    photocathode_silicon_window_mm: float = 0.01
    anode_wire_count: int = 68
    anode_wire_pitch_mm: float = 1.0
    anode_wire_half_length_mm: float = 34.0
    anode_wire_width_mm: float = 0.005
    anode_wire_thickness_mm: float = 0.001
    anode_wire_quartz_gap_mm: float = 0.0005
    cathode_half_xy_mm: float = 34.9
    cathode_center_z_mm: float = -24.005
    cathode_thickness_mm: float = 0.01
    cathode_square_size_mm: float = 6.0
    cathode_gap_mm: float = 0.89
    cathode_edge_margin_mm: float = 0.895
    cathode_grid_size: int = 10
    ptfe_patch_size_mm: float = 20.0

    @property
    def quartz_lower_z_mm(self) -> float:
        return self.quartz_center_z_mm - self.quartz_half_z_mm

    @property
    def source_z_mm(self) -> float:
        return self.anode_wire_center_z_mm

    @property
    def anode_wire_y_start_mm(self) -> float:
        return -0.5 * self.anode_wire_pitch_mm * (
            self.anode_wire_count - 1
        )

    @property
    def anode_wire_top_z_mm(self) -> float:
        return self.quartz_lower_z_mm - self.anode_wire_quartz_gap_mm

    @property
    def anode_wire_bottom_z_mm(self) -> float:
        return self.anode_wire_top_z_mm - self.anode_wire_thickness_mm

    @property
    def anode_wire_center_z_mm(self) -> float:
        return 0.5 * (
            self.anode_wire_top_z_mm + self.anode_wire_bottom_z_mm
        )

    @property
    def source_distance_max_mm(self) -> float:
        return 0.006


PROFILES = {
    "smoke": 20.0,
    "coarse": 10.0,
    "production": 5.0,
    "fine": 2.5,
}


class Mesh:
    def __init__(self) -> None:
        self.vertices: list[tuple[float, float, float]] = []
        self.triangles: list[tuple[int, int, int]] = []
        self.surface_id: list[int] = []
        self.surface_basis_id: list[int] = []
        self.minus_domain_id: list[int] = []
        self.plus_domain_id: list[int] = []
        self.channel_id: list[int] = []
        self._vertex_by_position: dict[tuple[float, float, float], int] = {}
        self._next_basis: dict[int, int] = {}
        self._basis_by_key: dict[tuple[int, object], int] = {}

    def vertex(self, value: tuple[float, float, float]) -> int:
        key = tuple(round(float(component), 12) for component in value)
        if key not in self._vertex_by_position:
            self._vertex_by_position[key] = len(self.vertices)
            self.vertices.append(key)
        return self._vertex_by_position[key]

    def quad(
        self,
        points: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        *,
        surface: int,
        minus: int,
        plus: int,
        channel: int = -1,
        diffuse_basis: bool = False,
        diffuse_basis_key: object | None = None,
    ) -> None:
        indices = tuple(self.vertex(point) for point in points)
        basis = NO_BASIS
        if diffuse_basis:
            if diffuse_basis_key is None:
                basis = self._next_basis.get(surface, 0)
                self._next_basis[surface] = basis + 1
            else:
                key = (surface, diffuse_basis_key)
                if key not in self._basis_by_key:
                    basis = self._next_basis.get(surface, 0)
                    self._next_basis[surface] = basis + 1
                    self._basis_by_key[key] = basis
                basis = self._basis_by_key[key]
        for triangle in (
            (indices[0], indices[1], indices[2]),
            (indices[0], indices[2], indices[3]),
        ):
            self.triangles.append(triangle)
            self.surface_id.append(surface)
            self.surface_basis_id.append(basis)
            self.minus_domain_id.append(minus)
            self.plus_domain_id.append(plus)
            self.channel_id.append(channel)


def axis_edges(minimum: float, maximum: float, patch: float) -> np.ndarray:
    count = max(1, math.ceil((maximum - minimum) / patch))
    return np.linspace(minimum, maximum, count + 1)


def add_outward_box(
    mesh: Mesh,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    surface: int,
    minus: int,
    plus: int,
    diffuse_basis: bool = False,
) -> None:
    xmin, xmax = float(x_edges[0]), float(x_edges[-1])
    ymin, ymax = float(y_edges[0]), float(y_edges[-1])
    zmin, zmax = float(z_edges[0]), float(z_edges[-1])

    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
            mesh.quad(
                (
                    (xmax, y0, z0),
                    (xmax, y1, z0),
                    (xmax, y1, z1),
                    (xmax, y0, z1),
                ),
                surface=surface,
                minus=minus,
                plus=plus,
                diffuse_basis=diffuse_basis,
            )
            mesh.quad(
                (
                    (xmin, y1, z0),
                    (xmin, y0, z0),
                    (xmin, y0, z1),
                    (xmin, y1, z1),
                ),
                surface=surface,
                minus=minus,
                plus=plus,
                diffuse_basis=diffuse_basis,
            )

    for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
        for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
            mesh.quad(
                (
                    (x1, ymax, z0),
                    (x0, ymax, z0),
                    (x0, ymax, z1),
                    (x1, ymax, z1),
                ),
                surface=surface,
                minus=minus,
                plus=plus,
                diffuse_basis=diffuse_basis,
            )
            mesh.quad(
                (
                    (x0, ymin, z0),
                    (x1, ymin, z0),
                    (x1, ymin, z1),
                    (x0, ymin, z1),
                ),
                surface=surface,
                minus=minus,
                plus=plus,
                diffuse_basis=diffuse_basis,
            )

    for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
        for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
            mesh.quad(
                (
                    (x0, y0, zmax),
                    (x1, y0, zmax),
                    (x1, y1, zmax),
                    (x0, y1, zmax),
                ),
                surface=surface,
                minus=minus,
                plus=plus,
                diffuse_basis=diffuse_basis,
            )
            mesh.quad(
                (
                    (x0, y1, zmin),
                    (x1, y1, zmin),
                    (x1, y0, zmin),
                    (x0, y0, zmin),
                ),
                surface=surface,
                minus=minus,
                plus=plus,
                diffuse_basis=diffuse_basis,
            )


def sipm_centers(config: GeometryConfig) -> np.ndarray:
    start = -0.5 * config.sipm_pitch_mm * (config.sipm_grid_size - 1)
    return np.asarray(
        [
            (start + i * config.sipm_pitch_mm,
             start + j * config.sipm_pitch_mm)
            for i in range(config.sipm_grid_size)
            for j in range(config.sipm_grid_size)
        ],
        dtype=np.float64,
    )


def patch_bin(value: float, minimum: float, patch_size: float) -> int:
    return math.floor((value - minimum) / patch_size)


def add_sipm(
    mesh: Mesh,
    config: GeometryConfig,
    center_x: float,
    center_y: float,
    channel: int,
) -> None:
    half_active = 0.5 * config.photocathode_width_mm
    active_center = 0.5 * (
        config.photocathode_width_mm + config.photocathode_gap_mm
    )
    local_edges = np.asarray(
        [
            -config.sipm_half_x_mm,
            -active_center - half_active,
            -active_center + half_active,
            active_center - half_active,
            active_center + half_active,
            config.sipm_half_x_mm,
        ],
        dtype=np.float64,
    )
    x_edges = local_edges + center_x
    y_edges = local_edges + center_y
    zmin = config.sipm_center_z_mm - config.sipm_half_z_mm
    zmax = config.sipm_center_z_mm + config.sipm_half_z_mm
    photocathode_bottom = zmin + config.photocathode_silicon_window_mm
    photocathode_top = zmin + config.photocathode_stack_depth_mm

    for ix, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:])):
        for iy, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:])):
            mesh.quad(
                (
                    (x0, y1, zmin),
                    (x1, y1, zmin),
                    (x1, y0, zmin),
                    (x0, y0, zmin),
                ),
                surface=SILICON_INTERFACE,
                minus=SILICON,
                plus=LXE,
            )
            mesh.quad(
                (
                    (x0, y0, zmax),
                    (x1, y0, zmax),
                    (x1, y1, zmax),
                    (x0, y1, zmax),
                ),
                surface=SILICON_INTERFACE,
                minus=SILICON,
                plus=LXE,
            )

    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        mesh.quad(
            (
                (x_edges[-1], y0, zmin),
                (x_edges[-1], y1, zmin),
                (x_edges[-1], y1, zmax),
                (x_edges[-1], y0, zmax),
            ),
            surface=SILICON_INTERFACE,
            minus=SILICON,
            plus=LXE,
        )
        mesh.quad(
            (
                (x_edges[0], y1, zmin),
                (x_edges[0], y0, zmin),
                (x_edges[0], y0, zmax),
                (x_edges[0], y1, zmax),
            ),
            surface=SILICON_INTERFACE,
            minus=SILICON,
            plus=LXE,
        )
    for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
        mesh.quad(
            (
                (x1, y_edges[-1], zmin),
                (x0, y_edges[-1], zmin),
                (x0, y_edges[-1], zmax),
                (x1, y_edges[-1], zmax),
            ),
            surface=SILICON_INTERFACE,
            minus=SILICON,
            plus=LXE,
        )
        mesh.quad(
            (
                (x0, y_edges[0], zmin),
                (x1, y_edges[0], zmin),
                (x1, y_edges[0], zmax),
                (x0, y_edges[0], zmax),
            ),
            surface=SILICON_INTERFACE,
            minus=SILICON,
            plus=LXE,
        )

    for ix in (1, 3):
        x0 = float(x_edges[ix])
        x1 = float(x_edges[ix + 1])
        for iy in (1, 3):
            y0 = float(y_edges[iy])
            y1 = float(y_edges[iy + 1])
            mesh.quad(
                (
                    (x0, y1, photocathode_bottom),
                    (x1, y1, photocathode_bottom),
                    (x1, y0, photocathode_bottom),
                    (x0, y0, photocathode_bottom),
                ),
                surface=PHOTOCATHODE,
                minus=OUTSIDE,
                plus=SILICON,
                channel=channel,
            )
            mesh.quad(
                (
                    (x0, y0, photocathode_top),
                    (x1, y0, photocathode_top),
                    (x1, y1, photocathode_top),
                    (x0, y1, photocathode_top),
                ),
                surface=PHOTOCATHODE,
                minus=OUTSIDE,
                plus=SILICON,
                channel=channel,
            )
            mesh.quad(
                (
                    (x0, y1, photocathode_bottom),
                    (x0, y0, photocathode_bottom),
                    (x0, y0, photocathode_top),
                    (x0, y1, photocathode_top),
                ),
                surface=PHOTOCATHODE,
                minus=OUTSIDE,
                plus=SILICON,
                channel=channel,
            )
            mesh.quad(
                (
                    (x1, y0, photocathode_bottom),
                    (x1, y1, photocathode_bottom),
                    (x1, y1, photocathode_top),
                    (x1, y0, photocathode_top),
                ),
                surface=PHOTOCATHODE,
                minus=OUTSIDE,
                plus=SILICON,
                channel=channel,
            )
            mesh.quad(
                (
                    (x1, y1, photocathode_bottom),
                    (x0, y1, photocathode_bottom),
                    (x0, y1, photocathode_top),
                    (x1, y1, photocathode_top),
                ),
                surface=PHOTOCATHODE,
                minus=OUTSIDE,
                plus=SILICON,
                channel=channel,
            )
            mesh.quad(
                (
                    (x0, y0, photocathode_bottom),
                    (x1, y0, photocathode_bottom),
                    (x1, y0, photocathode_top),
                    (x0, y0, photocathode_top),
                ),
                surface=PHOTOCATHODE,
                minus=OUTSIDE,
                plus=SILICON,
                channel=channel,
            )


def cathode_holes(
    config: GeometryConfig,
) -> list[tuple[float, float, float, float]]:
    square = config.cathode_square_size_mm
    spacing = square + config.cathode_gap_mm
    first_center = (
        -config.cathode_half_xy_mm
        + config.cathode_edge_margin_mm
        + 0.5 * square
    )
    result = []
    for ix in range(config.cathode_grid_size):
        center_x = first_center + ix * spacing
        for iy in range(config.cathode_grid_size):
            center_y = first_center + iy * spacing
            result.append(
                (
                    center_x - 0.5 * square,
                    center_x + 0.5 * square,
                    center_y - 0.5 * square,
                    center_y + 0.5 * square,
                )
            )
    return result


def add_cathode_grid(mesh: Mesh, config: GeometryConfig) -> None:
    half = config.cathode_half_xy_mm
    half_z = 0.5 * config.cathode_thickness_mm
    zmin = config.cathode_center_z_mm - half_z
    zmax = config.cathode_center_z_mm + half_z
    holes = cathode_holes(config)
    x_edges = sorted(
        {
            *axis_edges(-half, half, config.ptfe_patch_size_mm),
            *(value for hole in holes for value in hole[:2]),
        }
    )
    y_edges = sorted(
        {
            *axis_edges(-half, half, config.ptfe_patch_size_mm),
            *(value for hole in holes for value in hole[2:]),
        }
    )
    patch = config.ptfe_patch_size_mm

    def inside_hole(x: float, y: float) -> bool:
        return any(
            x0 < x < x1 and y0 < y < y1
            for x0, x1, y0, y1 in holes
        )

    for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
        for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
            center_x = 0.5 * (x0 + x1)
            center_y = 0.5 * (y0 + y1)
            if inside_hole(center_x, center_y):
                continue
            bins = (
                patch_bin(center_x, -half, patch),
                patch_bin(center_y, -half, patch),
            )
            mesh.quad(
                (
                    (x0, y0, zmax),
                    (x1, y0, zmax),
                    (x1, y1, zmax),
                    (x0, y1, zmax),
                ),
                surface=CATHODE,
                minus=OUTSIDE,
                plus=LXE,
                diffuse_basis=True,
                diffuse_basis_key=("top", *bins),
            )
            mesh.quad(
                (
                    (x0, y1, zmin),
                    (x1, y1, zmin),
                    (x1, y0, zmin),
                    (x0, y0, zmin),
                ),
                surface=CATHODE,
                minus=OUTSIDE,
                plus=LXE,
                diffuse_basis=True,
                diffuse_basis_key=("bottom", *bins),
            )

    for hole_index, (x0, x1, y0, y1) in enumerate(holes):
        hole_y_edges = [
            value for value in y_edges if y0 <= value <= y1
        ]
        for ya, yb in zip(hole_y_edges[:-1], hole_y_edges[1:]):
            for wall, points in (
                (
                    "left",
                    (
                        (x0, ya, zmin),
                        (x0, yb, zmin),
                        (x0, yb, zmax),
                        (x0, ya, zmax),
                    ),
                ),
                (
                    "right",
                    (
                        (x1, yb, zmin),
                        (x1, ya, zmin),
                        (x1, ya, zmax),
                        (x1, yb, zmax),
                    ),
                ),
            ):
                mesh.quad(
                    points,
                    surface=CATHODE,
                    minus=OUTSIDE,
                    plus=LXE,
                    diffuse_basis=True,
                    diffuse_basis_key=("hole", hole_index, wall),
                )
        hole_x_edges = [
            value for value in x_edges if x0 <= value <= x1
        ]
        for xa, xb in zip(hole_x_edges[:-1], hole_x_edges[1:]):
            for wall, points in (
                (
                    "lower",
                    (
                        (xb, y0, zmin),
                        (xa, y0, zmin),
                        (xa, y0, zmax),
                        (xb, y0, zmax),
                    ),
                ),
                (
                    "upper",
                    (
                        (xa, y1, zmin),
                        (xb, y1, zmin),
                        (xb, y1, zmax),
                        (xa, y1, zmax),
                    ),
                ),
            ):
                mesh.quad(
                    points,
                    surface=CATHODE,
                    minus=OUTSIDE,
                    plus=LXE,
                    diffuse_basis=True,
                    diffuse_basis_key=("hole", hole_index, wall),
                )

def add_enclosure(mesh: Mesh, config: GeometryConfig) -> None:
    half_x = config.lxe_half_x_mm
    half_y = config.lxe_half_y_mm
    half_z = config.lxe_half_z_mm
    patch = config.ptfe_patch_size_mm
    cathode_half_z = 0.5 * config.cathode_thickness_mm
    cathode_zmin = config.cathode_center_z_mm - cathode_half_z
    cathode_zmax = config.cathode_center_z_mm + cathode_half_z
    holes = cathode_holes(config)
    x_edges = sorted(
        {
            *axis_edges(-half_x, half_x, patch),
            *(value for hole in holes for value in hole[:2]),
        }
    )
    y_edges = sorted(
        {
            *axis_edges(-half_y, half_y, patch),
            *(value for hole in holes for value in hole[2:]),
        }
    )
    z_edges = sorted(
        {
            *axis_edges(-half_z, half_z, patch),
            cathode_zmin,
            cathode_zmax,
        }
    )

    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        center_y = 0.5 * (y0 + y1)
        for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
            center_z = 0.5 * (z0 + z1)
            cathode_band = z0 == cathode_zmin and z1 == cathode_zmax
            if cathode_band:
                continue
            for face, points in (
                (
                    "xmax",
                    (
                        (half_x, y0, z0),
                        (half_x, y1, z0),
                        (half_x, y1, z1),
                        (half_x, y0, z1),
                    ),
                ),
                (
                    "xmin",
                    (
                        (-half_x, y1, z0),
                        (-half_x, y0, z0),
                        (-half_x, y0, z1),
                        (-half_x, y1, z1),
                    ),
                ),
            ):
                key = (
                    "outer_" + face,
                    patch_bin(center_y, -half_y, patch),
                    patch_bin(center_z, -half_z, patch),
                )
                mesh.quad(
                    points,
                    surface=PTFE,
                    minus=LXE,
                    plus=OUTSIDE,
                    diffuse_basis=True,
                    diffuse_basis_key=key,
                )

    for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
        center_x = 0.5 * (x0 + x1)
        for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
            center_z = 0.5 * (z0 + z1)
            cathode_band = z0 == cathode_zmin and z1 == cathode_zmax
            if cathode_band:
                continue
            for face, points in (
                (
                    "ymax",
                    (
                        (x1, half_y, z0),
                        (x0, half_y, z0),
                        (x0, half_y, z1),
                        (x1, half_y, z1),
                    ),
                ),
                (
                    "ymin",
                    (
                        (x0, -half_y, z0),
                        (x1, -half_y, z0),
                        (x1, -half_y, z1),
                        (x0, -half_y, z1),
                    ),
                ),
            ):
                key = (
                    "outer_" + face,
                    patch_bin(center_x, -half_x, patch),
                    patch_bin(center_z, -half_z, patch),
                )
                mesh.quad(
                    points,
                    surface=PTFE,
                    minus=LXE,
                    plus=OUTSIDE,
                    diffuse_basis=True,
                    diffuse_basis_key=key,
                )

    for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
        center_x = 0.5 * (x0 + x1)
        for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
            center_y = 0.5 * (y0 + y1)
            bins = (
                patch_bin(center_x, -half_x, patch),
                patch_bin(center_y, -half_y, patch),
            )
            mesh.quad(
                (
                    (x0, y0, half_z),
                    (x1, y0, half_z),
                    (x1, y1, half_z),
                    (x0, y1, half_z),
                ),
                surface=PTFE,
                minus=LXE,
                plus=OUTSIDE,
                diffuse_basis=True,
                diffuse_basis_key=("zmax", *bins),
            )
            mesh.quad(
                (
                    (x0, y1, -half_z),
                    (x1, y1, -half_z),
                    (x1, y0, -half_z),
                    (x0, y0, -half_z),
                ),
                surface=PTFE,
                minus=LXE,
                plus=OUTSIDE,
                diffuse_basis=True,
                diffuse_basis_key=("zmin", *bins),
            )


def build_mesh(config: GeometryConfig) -> tuple[Mesh, np.ndarray]:
    if not (
        0.0
        < config.photocathode_silicon_window_mm
        < config.photocathode_stack_depth_mm
    ):
        raise ValueError(
            "photocathode silicon window must be positive and thinner than "
            "the photocathode layer"
        )
    mesh = Mesh()
    add_enclosure(mesh, config)

    add_outward_box(
        mesh,
        x_edges=np.asarray(
            [-config.quartz_half_x_mm, config.quartz_half_x_mm]
        ),
        y_edges=np.asarray(
            [-config.quartz_half_y_mm, config.quartz_half_y_mm]
        ),
        z_edges=np.asarray(
            [
                config.quartz_center_z_mm - config.quartz_half_z_mm,
                config.quartz_center_z_mm + config.quartz_half_z_mm,
            ]
        ),
        surface=QUARTZ_INTERFACE,
        minus=QUARTZ,
        plus=LXE,
    )

    add_cathode_grid(mesh, config)

    centers = sipm_centers(config)
    for channel, (x, y) in enumerate(centers):
        add_sipm(mesh, config, float(x), float(y), channel)
    return mesh, centers


def generator_identity(config: GeometryConfig, include_anode_wires: bool) -> dict:
    source = Path(__file__).read_bytes()
    return {
        "schema": "oos.pet-4x4.geometry.v1",
        "config": asdict(config),
        "include_anode_wires": include_anode_wires,
        "generator_sha256": hashlib.sha256(source).hexdigest(),
        "boundary_regularization": {
            "photocathode_silicon_window_mm":
                config.photocathode_silicon_window_mm,
            "anode_wire_quartz_gap_mm":
                config.anode_wire_quartz_gap_mm,
            "reason": [
                "separate the LXe-silicon and silicon-photocathode boundaries",
                *(
                    ["separate the analytic lower-wire boxes from the quartz plane"]
                    if include_anode_wires
                    else []
                ),
            ],
        },
        "approximations": [
            "no_lxe_rayleigh_scattering",
            "upper_anode_wires_omitted",
            *([] if include_anode_wires else ["lower_anode_wires_omitted"]),
        ],
    }


def write_analytic_anode_wires(
    handle: h5py.File, config: GeometryConfig
) -> None:
    count = config.anode_wire_count
    centers = np.zeros((count, 3), dtype=np.float64)
    centers[:, 1] = (
        config.anode_wire_y_start_mm
        + config.anode_wire_pitch_mm * np.arange(count)
    )
    centers[:, 2] = config.anode_wire_center_z_mm
    axes = np.eye(3, dtype=np.float64)
    analytic = handle.create_group("analytic")
    analytic.create_dataset(
        "kind", data=np.full(count, 4, dtype=np.uint8)
    )
    analytic.create_dataset("center_mm", data=centers)
    analytic.create_dataset(
        "axis_x", data=np.repeat(axes[[0]], count, axis=0)
    )
    analytic.create_dataset(
        "axis_y", data=np.repeat(axes[[1]], count, axis=0)
    )
    analytic.create_dataset(
        "axis_z", data=np.repeat(axes[[2]], count, axis=0)
    )
    parameters = np.zeros((count, 4), dtype=np.float64)
    parameters[:, 0] = config.anode_wire_half_length_mm
    parameters[:, 1] = 0.5 * config.anode_wire_width_mm
    parameters[:, 2] = 0.5 * config.anode_wire_thickness_mm
    analytic.create_dataset("parameters", data=parameters)
    analytic.create_dataset(
        "normal_sign", data=np.ones(count, dtype=np.float64)
    )
    analytic.create_dataset(
        "surface_id", data=np.full(count, ANODE_WIRE, dtype=np.uint32)
    )
    analytic.create_dataset(
        "surface_basis_id",
        data=np.full(count, NO_BASIS, dtype=np.uint32),
    )
    analytic.create_dataset(
        "minus_domain_id",
        data=np.full(count, OUTSIDE, dtype=np.int32),
    )
    analytic.create_dataset(
        "plus_domain_id", data=np.full(count, LXE, dtype=np.int32)
    )
    analytic.create_dataset(
        "channel_id", data=np.full(count, -1, dtype=np.int32)
    )
    analytic.create_dataset(
        "surface_element", data=np.arange(count, dtype=np.uint64)
    )
    analytic.create_dataset(
        "hole_offset", data=np.zeros(count + 1, dtype=np.uint64)
    )
    analytic.create_dataset(
        "hole_center_uv_mm", data=np.empty((0, 2), dtype=np.float64)
    )
    analytic.create_dataset(
        "hole_radius_mm", data=np.empty(0, dtype=np.float64)
    )


def write_geometry(
    output: Path, config: GeometryConfig, *, include_anode_wires: bool = True
) -> dict:
    mesh, centers = build_mesh(config)
    identity = generator_identity(config, include_anode_wires)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with h5py.File(temporary, "w") as handle:
            geometry = handle.create_group("geometry")
            geometry.create_dataset(
                "vertices", data=np.asarray(mesh.vertices, dtype=np.float64)
            )
            geometry.create_dataset(
                "triangles", data=np.asarray(mesh.triangles, dtype=np.uint32)
            )
            geometry.create_dataset(
                "surface_id", data=np.asarray(mesh.surface_id, dtype=np.uint32)
            )
            geometry.create_dataset(
                "surface_basis_id",
                data=np.asarray(mesh.surface_basis_id, dtype=np.uint32),
            )
            geometry.create_dataset(
                "minus_domain_id",
                data=np.asarray(mesh.minus_domain_id, dtype=np.int32),
            )
            geometry.create_dataset(
                "plus_domain_id",
                data=np.asarray(mesh.plus_domain_id, dtype=np.int32),
            )
            geometry.create_dataset(
                "channel_id",
                data=np.asarray(mesh.channel_id, dtype=np.int32),
            )
            metadata = handle.create_group("metadata")
            metadata.create_dataset(
                "generator_json",
                data=np.frombuffer(
                    json.dumps(identity, sort_keys=True).encode(),
                    dtype=np.uint8,
                ),
            )
            metadata.create_dataset("sipm_xy_mm", data=centers)
            metadata.create_dataset(
                "sipm_channel_id",
                data=np.arange(len(centers), dtype=np.int32),
            )
            metadata.create_dataset(
                "source_plane_z_mm",
                data=np.asarray([config.source_z_mm], dtype=np.float64),
            )
            metadata.create_dataset(
                "anode_wire_y_mm",
                data=(
                    config.anode_wire_y_start_mm
                    + config.anode_wire_pitch_mm
                    * np.arange(config.anode_wire_count)
                    if include_anode_wires
                    else np.empty(0, dtype=np.float64)
                ),
            )
            metadata.create_dataset(
                "anode_wire_bottom_z_mm",
                data=np.asarray(
                    [config.anode_wire_bottom_z_mm], dtype=np.float64
                ),
            )
            metadata.create_dataset(
                "source_distance_max_mm",
                data=np.asarray(
                    [config.source_distance_max_mm], dtype=np.float64
                ),
            )
            if include_anode_wires:
                write_analytic_anode_wires(handle, config)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    channels = np.asarray(mesh.channel_id)
    return {
        "path": str(output.resolve()),
        "sha256": digest,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "ptfe_states": mesh._next_basis.get(PTFE, 0),
        "cathode_states": mesh._next_basis.get(CATHODE, 0),
        "sensitive_triangles": int(np.count_nonzero(channels >= 0)),
        "analytic_anode_wire_boxes": (
            config.anode_wire_count if include_anode_wires else 0
        ),
        "channels": int(len(np.unique(channels[channels >= 0]))),
        "source_plane_z_mm": config.source_z_mm,
        "anode_wire_bottom_z_mm": config.anode_wire_bottom_z_mm,
        "source_distance_max_mm": config.source_distance_max_mm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="smoke"
    )
    parser.add_argument("--ptfe-patch-size-mm", type=float)
    parser.add_argument(
        "--omit-anode-wires",
        action="store_true",
        help="omit all analytic lower-anode-wire boxes",
    )
    arguments = parser.parse_args()
    patch = (
        arguments.ptfe_patch_size_mm
        if arguments.ptfe_patch_size_mm is not None
        else PROFILES[arguments.profile]
    )
    if not math.isfinite(patch) or patch <= 0.0:
        parser.error("--ptfe-patch-size-mm must be finite and positive")
    config = replace(GeometryConfig(), ptfe_patch_size_mm=patch)
    print(
        json.dumps(
            write_geometry(
                arguments.output,
                config,
                include_anode_wires=not arguments.omit_anode_wires,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
