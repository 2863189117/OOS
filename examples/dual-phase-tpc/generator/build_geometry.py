#!/usr/bin/env python3
"""Generate a synthetic, closed dual-phase TPC example geometry.

The triangulation is an example-owned offline operation.  It uses constrained
planar Delaunay meshes for the perforated horizontal surfaces and extrudes the
same boundary vertices into the adjoining cylindrical surfaces.  Geometry
triangles are deliberately independent from the optical surface basis that
the native operator builder constructs later.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

try:
    import triangle
except ImportError as exception:  # pragma: no cover - operational message
    raise SystemExit(
        "build_geometry.py requires the example dependencies: "
        "pip install -r requirements.txt"
    ) from exception


TOP_PTFE = 0
GXE_LXE = 1
QUARTZ_WINDOW = 2
PHOTOCATHODE = 3
ABSORBING_BOUNDARY = 4
FIELD_CAGE_PTFE = 5
OUTER_WALL_PTFE = 6
QUARTZ_SIDE_ABSORBER = 7
LXE_ENCLOSURE_ABSORBER = 8
AL_FLASHING = 9

GXE = 0
QUARTZ = 1
LXE = 2
OUTSIDE = -1


@dataclass(frozen=True)
class GeometryConfig:
    # The public example is deliberately synthetic: a 2 m active diameter,
    # 2 m liquid depth, and a shallow gas region.  These values are not copied
    # from a detector design.
    height_mm: float = 60.0
    active_radius_mm: float = 1000.0
    field_cage_thickness_mm: float = 20.0
    wall_radius_mm: float = 1080.0
    aperture_radius_mm: float = 25.4
    hole_depth_mm: float = 6.0
    quartz_thickness_mm: float = 4.0
    photocathode_thickness_mm: float = 0.10
    al_flashing_thickness_mm: float = 0.50
    lxe_depth_mm: float = 2000.0
    pmt_pitch_mm: float = 64.0
    pmt_array_radius_mm: float = 900.0
    boundary_segments: int = 640
    aperture_segments: int = 24
    wall_z_segments: int = 6
    lxe_wall_z_segments: int = 72
    maximum_triangle_area_mm2: float = 100.0

    @property
    def field_cage_outer_radius_mm(self) -> float:
        return min(
            self.active_radius_mm + self.field_cage_thickness_mm,
            self.wall_radius_mm,
        )

    @property
    def quartz_bottom_z_mm(self) -> float:
        return self.height_mm + self.hole_depth_mm

    @property
    def photocathode_z_mm(self) -> float:
        return (
            self.quartz_bottom_z_mm
            + self.quartz_thickness_mm
            - self.photocathode_thickness_mm
        )

    @property
    def al_flashing_z_range_mm(self) -> tuple[float, float]:
        half_photocathode = 0.5 * self.photocathode_thickness_mm
        lower_depth = (
            0.5 * self.quartz_thickness_mm
            - half_photocathode
            - self.al_flashing_thickness_mm
        )
        upper_depth = (
            0.5 * self.quartz_thickness_mm - half_photocathode
        )
        return (
            self.quartz_bottom_z_mm + lower_depth,
            self.quartz_bottom_z_mm + upper_depth,
        )


PROFILES = {
    "smoke": {
        "boundary_segments": 80,
        "aperture_segments": 10,
        "wall_z_segments": 2,
        "lxe_wall_z_segments": 10,
        "maximum_triangle_area_mm2": 8000.0,
    },
    "coarse": {
        "boundary_segments": 240,
        "aperture_segments": 16,
        "wall_z_segments": 4,
        "lxe_wall_z_segments": 36,
        "maximum_triangle_area_mm2": 600.0,
    },
    "production": {},
    "fine": {
        "boundary_segments": 960,
        "aperture_segments": 40,
        "wall_z_segments": 10,
        "lxe_wall_z_segments": 120,
        "maximum_triangle_area_mm2": 40.0,
    },
}


def pmt_layout(config: GeometryConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return row-major channels on a regular hexagonal lattice.

    The 50.8 mm circular apertures represent generic 2-inch PMTs.  The
    lattice is generated from first principles and has no imported channel
    map, survey coordinates, or experiment-specific exceptions.
    """

    if config.pmt_pitch_mm < 2.0 * config.aperture_radius_mm:
        raise ValueError("PMT pitch must be at least the 2-inch diameter")
    usable_radius = min(
        config.pmt_array_radius_mm,
        config.active_radius_mm - config.aperture_radius_mm,
    )
    row_pitch = math.sqrt(3.0) * 0.5 * config.pmt_pitch_mm
    maximum_row = math.floor(usable_radius / row_pitch)
    xy: list[tuple[float, float]] = []
    for row in range(-maximum_row, maximum_row + 1):
        y = row * row_pitch
        x_offset = 0.5 * config.pmt_pitch_mm if row % 2 else 0.0
        maximum_column = math.ceil(usable_radius / config.pmt_pitch_mm) + 1
        for column in range(-maximum_column, maximum_column + 1):
            x = column * config.pmt_pitch_mm + x_offset
            if math.hypot(x, y) <= usable_radius + 1.0e-9:
                xy.append((x, y))
    result_xy = np.asarray(xy, dtype=np.float64)
    result_channel = np.arange(len(result_xy), dtype=np.int32)
    if len(result_xy) < 7:
        raise RuntimeError("hexagonal PMT layout contains too few positions")
    return result_xy, result_channel


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generator_identity(config: GeometryConfig) -> dict:
    return {
        "schema": "oos.dual-phase-tpc.geometry.v1",
        "config": asdict(config),
        "layout": "regular_hexagonal_lattice",
        "pmt_diameter_mm": 2.0 * config.aperture_radius_mm,
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "triangle_version": getattr(triangle, "__version__", "unknown"),
    }


def cached_identity(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with h5py.File(path, "r") as handle:
            raw = bytes(handle["/metadata/generator_json"][:])
        return json.loads(raw.decode())
    except Exception:
        return None


class PlanarPSLG:
    def __init__(self) -> None:
        self.vertices: list[tuple[float, float]] = []
        self.segments: list[tuple[int, int]] = []
        self.holes: list[tuple[float, float]] = []
        self.regions: list[tuple[float, float, int, float]] = []
        self._lookup: dict[tuple[int, int], int] = {}

    def vertex(self, x: float, y: float) -> int:
        key = (round(x * 1.0e9), round(y * 1.0e9))
        found = self._lookup.get(key)
        if found is not None:
            return found
        index = len(self.vertices)
        self.vertices.append((x, y))
        self._lookup[key] = index
        return index

    def loop(self, radius: float, angles: np.ndarray) -> list[int]:
        indices = [
            self.vertex(radius * math.cos(phi), radius * math.sin(phi))
            for phi in angles
        ]
        for first, second in zip(indices, indices[1:] + indices[:1]):
            if first != second:
                self.segments.append((first, second))
        return indices

    def translated_loop(
        self, center: np.ndarray, radius: float, angles: np.ndarray
    ) -> list[int]:
        indices = [
            self.vertex(
                float(center[0]) + radius * math.cos(phi),
                float(center[1]) + radius * math.sin(phi),
            )
            for phi in angles
        ]
        for first, second in zip(indices, indices[1:] + indices[:1]):
            if first != second:
                self.segments.append((first, second))
        return indices

    def triangulate(self) -> dict[str, np.ndarray]:
        source = {
            "vertices": np.asarray(self.vertices, dtype=np.float64),
            "segments": np.asarray(self.segments, dtype=np.int32),
            "regions": np.asarray(self.regions, dtype=np.float64),
        }
        if self.holes:
            source["holes"] = np.asarray(self.holes, dtype=np.float64)
        result = triangle.triangulate(source, "pAq28aYQ")
        if "triangles" not in result or "triangle_attributes" not in result:
            raise RuntimeError("constrained triangulation did not return regions")
        return result


class Mesh:
    def __init__(self) -> None:
        self.vertices: list[tuple[float, float, float]] = []
        self.triangles: list[tuple[int, int, int]] = []
        self.surface_id: list[int] = []
        self.minus_domain_id: list[int] = []
        self.plus_domain_id: list[int] = []
        self.channel_id: list[int] = []
        self._lookup: dict[tuple[int, int, int], int] = {}

    def vertex(self, point: tuple[float, float, float]) -> int:
        key = tuple(round(value * 1.0e9) for value in point)
        found = self._lookup.get(key)
        if found is not None:
            return found
        index = len(self.vertices)
        self.vertices.append(point)
        self._lookup[key] = index
        return index

    def triangle(
        self,
        points: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        surface: int,
        minus: int,
        plus: int,
        channel: int = -1,
    ) -> None:
        indices = tuple(self.vertex(point) for point in points)
        if len(set(indices)) != 3:
            raise RuntimeError("geometry generator produced a degenerate face")
        self.triangles.append(indices)
        self.surface_id.append(surface)
        self.minus_domain_id.append(minus)
        self.plus_domain_id.append(plus)
        self.channel_id.append(channel)

    def planar(
        self,
        result: dict[str, np.ndarray],
        z_mm: float,
        attributes: dict[int, tuple[int, int, int]],
        *,
        normal_positive_z: bool,
    ) -> None:
        vertices = np.asarray(result["vertices"], dtype=np.float64)
        triangles = np.asarray(result["triangles"], dtype=np.int64)
        markers = np.rint(
            np.asarray(result["triangle_attributes"]).reshape(-1)
        ).astype(int)
        for triangle_indices, marker in zip(triangles, markers):
            surface, minus, plus = attributes[int(marker)]
            order = (
                triangle_indices
                if normal_positive_z
                else triangle_indices[[0, 2, 1]]
            )
            points = tuple(
                (float(vertices[i, 0]), float(vertices[i, 1]), z_mm)
                for i in order
            )
            self.triangle(points, surface, minus, plus)

    def cylinder(
        self,
        center: np.ndarray,
        radius: float,
        angles: np.ndarray,
        z_values: np.ndarray,
        surface: int,
        minus: int,
        plus: int,
    ) -> None:
        rings = [
            [
                (
                    float(center[0]) + radius * math.cos(phi),
                    float(center[1]) + radius * math.sin(phi),
                    float(z),
                )
                for phi in angles
            ]
            for z in z_values
        ]
        count = len(angles)
        for lower, upper in zip(rings[:-1], rings[1:]):
            for index in range(count):
                following = (index + 1) % count
                self.triangle(
                    (lower[index], lower[following], upper[following]),
                    surface,
                    minus,
                    plus,
                )
                self.triangle(
                    (lower[index], upper[following], upper[index]),
                    surface,
                    minus,
                    plus,
                )

    def extrude_ring(
        self,
        ring_xy: np.ndarray,
        z_values: np.ndarray,
        surface: int,
        minus: int,
        plus: int,
    ) -> None:
        rings = [
            [(float(x), float(y), float(z)) for x, y in ring_xy]
            for z in z_values
        ]
        count = len(ring_xy)
        for lower, upper in zip(rings[:-1], rings[1:]):
            for index in range(count):
                following = (index + 1) % count
                self.triangle(
                    (lower[index], lower[following], upper[following]),
                    surface,
                    minus,
                    plus,
                )
                self.triangle(
                    (lower[index], upper[following], upper[index]),
                    surface,
                    minus,
                    plus,
                )

    def disk_fan(
        self,
        center: np.ndarray,
        radius: float,
        angles: np.ndarray,
        z_mm: float,
        surface: int,
        minus: int,
        plus: int,
        *,
        positive_z: bool,
        channel: int = -1,
    ) -> None:
        origin = (float(center[0]), float(center[1]), z_mm)
        ring = [
            (
                float(center[0]) + radius * math.cos(phi),
                float(center[1]) + radius * math.sin(phi),
                z_mm,
            )
            for phi in angles
        ]
        for index in range(len(ring)):
            following = (index + 1) % len(ring)
            points = (
                (origin, ring[index], ring[following])
                if positive_z
                else (origin, ring[following], ring[index])
            )
            self.triangle(points, surface, minus, plus, channel)

    def polygon_fan(
        self,
        ring_xy: np.ndarray,
        z_mm: float,
        surface: int,
        minus: int,
        plus: int,
        *,
        positive_z: bool,
    ) -> None:
        origin = (0.0, 0.0, z_mm)
        ring = [(float(x), float(y), z_mm) for x, y in ring_xy]
        for index in range(len(ring)):
            following = (index + 1) % len(ring)
            points = (
                (origin, ring[index], ring[following])
                if positive_z
                else (origin, ring[following], ring[index])
            )
            self.triangle(points, surface, minus, plus)


def circle_angles(count: int, additions: np.ndarray | None = None) -> np.ndarray:
    values = list(2.0 * math.pi * np.arange(count, dtype=float) / count)
    if additions is not None:
        values.extend(float(value % (2.0 * math.pi)) for value in additions)
    values.sort()
    unique: list[float] = []
    for value in values:
        if not unique or abs(value - unique[-1]) > 1.0e-12:
            unique.append(value)
    if len(unique) > 1 and 2.0 * math.pi - unique[-1] + unique[0] <= 1.0e-12:
        unique.pop()
    return np.asarray(unique, dtype=np.float64)


def planar_top(
    config: GeometryConfig, pmt_xy: np.ndarray
) -> tuple[dict[str, np.ndarray], list[np.ndarray], np.ndarray]:
    pslg = PlanarPSLG()
    outer_angles = circle_angles(config.boundary_segments)
    outer_pmt = np.flatnonzero(
        np.isclose(
            np.linalg.norm(pmt_xy, axis=1),
            config.active_radius_mm - config.aperture_radius_mm,
            atol=1.0e-9,
        )
    )
    tangent_angles = np.arctan2(
        pmt_xy[outer_pmt, 1], pmt_xy[outer_pmt, 0]
    )
    active_angles = circle_angles(
        config.boundary_segments, tangent_angles
    )
    pslg.loop(config.wall_radius_mm, outer_angles)
    pslg.loop(config.active_radius_mm, active_angles)
    aperture_angles: list[np.ndarray] = []
    for center in pmt_xy:
        outward = math.atan2(float(center[1]), float(center[0]))
        if np.linalg.norm(center) < 1.0e-12:
            outward = 0.0
        angles = circle_angles(config.aperture_segments) + outward
        aperture_angles.append(angles)
        pslg.translated_loop(center, config.aperture_radius_mm, angles)
        pslg.holes.append((float(center[0]), float(center[1])))
    pslg.regions.extend(
        [
            (
                40.0,
                0.0,
                TOP_PTFE,
                config.maximum_triangle_area_mm2,
            ),
            (
                0.5 * (config.active_radius_mm + config.wall_radius_mm),
                0.0,
                ABSORBING_BOUNDARY,
                config.maximum_triangle_area_mm2,
            ),
        ]
    )
    return pslg.triangulate(), aperture_angles, outer_angles


def planar_bottom(
    config: GeometryConfig, tangent_angles: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    pslg = PlanarPSLG()
    rings = {
        config.active_radius_mm: circle_angles(
            config.boundary_segments, tangent_angles
        ),
        config.field_cage_outer_radius_mm: circle_angles(
            config.boundary_segments
        ),
        config.wall_radius_mm: circle_angles(config.boundary_segments),
    }
    for radius, angles in sorted(rings.items(), reverse=True):
        pslg.loop(radius, angles)
    pslg.regions.extend(
        [
            (0.0, 0.0, GXE_LXE, config.maximum_triangle_area_mm2),
            (
                0.5
                * (
                    config.active_radius_mm
                    + config.field_cage_outer_radius_mm
                ),
                0.0,
                FIELD_CAGE_PTFE,
                config.maximum_triangle_area_mm2,
            ),
            (
                0.5
                * (
                    config.field_cage_outer_radius_mm
                    + config.wall_radius_mm
                ),
                0.0,
                ABSORBING_BOUNDARY,
                config.maximum_triangle_area_mm2,
            ),
        ]
    )
    result = pslg.triangulate()
    vertices = np.asarray(result["vertices"], dtype=np.float64)
    triangles = np.asarray(result["triangles"], dtype=np.int64)
    markers = np.rint(
        np.asarray(result["triangle_attributes"]).reshape(-1)
    ).astype(int)
    edge_count: dict[tuple[int, int], int] = {}
    for face in triangles[markers == GXE_LXE]:
        for first, second in zip(face, np.roll(face, -1)):
            edge = tuple(sorted((int(first), int(second))))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_vertices = sorted(
        {
            vertex
            for edge, count in edge_count.items()
            if count == 1
            for vertex in edge
        },
        key=lambda index: math.atan2(
            float(vertices[index, 1]), float(vertices[index, 0])
        ),
    )
    active_ring = vertices[np.asarray(boundary_vertices, dtype=int)]
    if len(active_ring) < 3:
        raise RuntimeError("cannot recover the triangulated LXe boundary")
    return result, active_ring


def write_geometry(
    output: Path,
    config: GeometryConfig,
    *,
    force: bool = False,
) -> dict:
    identity = generator_identity(config)
    if not force and cached_identity(output) == identity:
        with h5py.File(output, "r") as handle:
            vertices = int(handle["/geometry/vertices"].shape[0])
            triangles = int(handle["/geometry/triangles"].shape[0])
            pmt_count = int(handle["/metadata/pmt_xy_mm"].shape[0])
        return {
            "path": str(output.resolve()),
            "sha256": file_sha256(output),
            "vertices": vertices,
            "triangles": triangles,
            "pmt_count": pmt_count,
            "cache": "hit",
        }
    pmt_xy, channel_id = pmt_layout(config)

    top, aperture_angles, outer_angles = planar_top(config, pmt_xy)
    outer_pmt = np.flatnonzero(
        np.isclose(
            np.linalg.norm(pmt_xy, axis=1),
            config.active_radius_mm - config.aperture_radius_mm,
            atol=1.0e-9,
        )
    )
    tangent_angles = np.arctan2(
        pmt_xy[outer_pmt, 1], pmt_xy[outer_pmt, 0]
    )
    bottom, active_ring = planar_bottom(config, tangent_angles)

    mesh = Mesh()
    mesh.planar(
        top,
        config.height_mm,
        {
            TOP_PTFE: (TOP_PTFE, GXE, OUTSIDE),
            ABSORBING_BOUNDARY: (
                ABSORBING_BOUNDARY,
                GXE,
                OUTSIDE,
            ),
        },
        normal_positive_z=True,
    )
    mesh.planar(
        bottom,
        0.0,
        {
            GXE_LXE: (GXE_LXE, GXE, LXE),
            FIELD_CAGE_PTFE: (FIELD_CAGE_PTFE, GXE, OUTSIDE),
            ABSORBING_BOUNDARY: (
                ABSORBING_BOUNDARY,
                GXE,
                OUTSIDE,
            ),
        },
        normal_positive_z=False,
    )
    mesh.cylinder(
        np.zeros(2),
        config.wall_radius_mm,
        outer_angles,
        np.linspace(0.0, config.height_mm, config.wall_z_segments + 1),
        OUTER_WALL_PTFE,
        GXE,
        OUTSIDE,
    )

    for center, channel, angles in zip(
        pmt_xy, channel_id, aperture_angles
    ):
        mesh.cylinder(
            center,
            config.aperture_radius_mm,
            angles,
            np.asarray([config.height_mm, config.quartz_bottom_z_mm]),
            TOP_PTFE,
            GXE,
            OUTSIDE,
        )
        mesh.disk_fan(
            center,
            config.aperture_radius_mm,
            angles,
            config.quartz_bottom_z_mm,
            QUARTZ_WINDOW,
            GXE,
            QUARTZ,
            positive_z=True,
        )
        flashing_min_z, flashing_max_z = config.al_flashing_z_range_mm
        for lower_z, upper_z, surface in (
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
            mesh.cylinder(
                center,
                config.aperture_radius_mm,
                angles,
                np.asarray([lower_z, upper_z]),
                surface,
                QUARTZ,
                OUTSIDE,
            )
        mesh.disk_fan(
            center,
            config.aperture_radius_mm,
            angles,
            config.photocathode_z_mm,
            PHOTOCATHODE,
            QUARTZ,
            OUTSIDE,
            positive_z=True,
            channel=int(channel),
        )

    mesh.extrude_ring(
        active_ring,
        np.linspace(
            -config.lxe_depth_mm,
            0.0,
            config.lxe_wall_z_segments + 1,
        ),
        LXE_ENCLOSURE_ABSORBER,
        LXE,
        OUTSIDE,
    )
    mesh.polygon_fan(
        active_ring,
        -config.lxe_depth_mm,
        LXE_ENCLOSURE_ABSORBER,
        LXE,
        OUTSIDE,
        positive_z=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            group = handle.create_group("geometry")
            group.create_dataset(
                "vertices", data=np.asarray(mesh.vertices, dtype=np.float64)
            )
            group.create_dataset(
                "triangles", data=np.asarray(mesh.triangles, dtype=np.uint32)
            )
            group.create_dataset(
                "surface_id", data=np.asarray(mesh.surface_id, dtype=np.uint32)
            )
            # The canonical geometry starts with the exact leaf basis: one
            # independent radiance function per triangle.  Coarser and
            # hierarchical bases replace this dataset without changing the
            # triangulation used by Embree.
            group.create_dataset(
                "surface_basis_id",
                data=np.arange(len(mesh.triangles), dtype=np.uint32),
            )
            group.create_dataset(
                "minus_domain_id",
                data=np.asarray(mesh.minus_domain_id, dtype=np.int32),
            )
            group.create_dataset(
                "plus_domain_id",
                data=np.asarray(mesh.plus_domain_id, dtype=np.int32),
            )
            group.create_dataset(
                "channel_id",
                data=np.asarray(mesh.channel_id, dtype=np.int32),
            )
            triangle_count = len(mesh.triangles)
            group.create_dataset(
                "triangle_transport",
                data=np.ones(triangle_count, dtype=np.uint8),
            )
            group.create_dataset(
                "triangle_source_quadrature",
                data=np.ones(triangle_count, dtype=np.uint8),
            )
            group.create_dataset(
                "triangle_source_analytic_primitive",
                data=np.full(
                    triangle_count,
                    np.iinfo(np.uint32).max,
                    dtype=np.uint32,
                ),
            )
            metadata = handle.create_group("metadata")
            metadata.create_dataset(
                "generator_json",
                data=np.frombuffer(
                    json.dumps(identity, sort_keys=True).encode(),
                    dtype=np.uint8,
                ),
            )
            metadata.create_dataset("pmt_xy_mm", data=pmt_xy)
            metadata.create_dataset("pmt_channel_id", data=channel_id)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    digest = file_sha256(output)
    return {
        "path": str(output.resolve()),
        "sha256": digest,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "pmt_count": len(pmt_xy),
        "cache": "miss",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="production"
    )
    parser.add_argument("--height-mm", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    values = {
        **asdict(GeometryConfig()),
        **PROFILES[arguments.profile],
        "height_mm": arguments.height_mm,
    }
    config = GeometryConfig(**values)
    print(
        json.dumps(
            write_geometry(
                arguments.output,
                config,
                force=arguments.force,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
