#!/usr/bin/env python3
"""Attach a nested, conservative radiance basis to a fixed geometry mesh.

Geometry triangles remain the Embree intersection primitives.  A basis level
only assigns reflective triangles to radiance states.  Restriction sums child
power into a parent; prolongation redistributes parent power to children in
proportion to their physical area.  Consequently all transfer weights are
non-negative, columns of prolongation sum to one, and R @ P is the identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np


INVALID_BASIS = np.iinfo(np.uint32).max


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def triangle_properties(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = vertices[triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if np.any(twice_area <= 0.0):
        raise RuntimeError("surface basis cannot include degenerate triangles")
    return points.mean(axis=1), cross / twice_area[:, None], 0.5 * twice_area


def normal_class(normals: np.ndarray) -> np.ndarray:
    """Return a stable octant/dominant-axis class for crease preservation."""

    dominant = np.argmax(np.abs(normals), axis=1)
    sign = normals[np.arange(len(normals)), dominant] >= 0.0
    return 2 * dominant + sign.astype(np.int64)


def geometric_edge_triangles(
    triangles: np.ndarray,
    surface_id: np.ndarray,
    normals: np.ndarray,
    reflective: np.ndarray,
    *,
    crease_cosine: float = 0.95,
) -> np.ndarray:
    """Mark reflective triangles adjacent to a boundary or a sharp crease."""

    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    )
    edges.sort(axis=1)
    owner = np.tile(np.arange(len(triangles), dtype=np.uint32), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owner = owner[order]
    starts = np.r_[
        0,
        np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1,
        len(edges),
    ]
    marked = np.zeros(len(triangles), dtype=bool)
    for start, stop in zip(starts[:-1], starts[1:]):
        adjacent = owner[start:stop]
        if len(adjacent) != 2:
            marked[adjacent] = True
            continue
        first, second = map(int, adjacent)
        if (
            surface_id[first] != surface_id[second]
            or float(np.dot(normals[first], normals[second])) < crease_cosine
        ):
            marked[first] = True
            marked[second] = True
    return marked & reflective


def assign_level(
    centroids: np.ndarray,
    normals: np.ndarray,
    surface_id: np.ndarray,
    reflective: np.ndarray,
    patch_size_mm: float,
) -> tuple[np.ndarray, int]:
    assignment = np.full(len(surface_id), INVALID_BASIS, dtype=np.uint32)
    selected = np.flatnonzero(reflective)
    if patch_size_mm == 0.0:
        assignment[selected] = np.arange(len(selected), dtype=np.uint32)
        return assignment, len(selected)
    if not math.isfinite(patch_size_mm) or patch_size_mm <= 0.0:
        raise ValueError("patch sizes must be finite and positive, or zero")
    cell = np.floor(centroids[selected] / patch_size_mm).astype(np.int64)
    orientation = normal_class(normals[selected])
    keys = np.column_stack([surface_id[selected], orientation, cell])
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    assignment[selected] = inverse.astype(np.uint32)
    return assignment, int(inverse.max(initial=-1) + 1)


def state_areas(
    assignment: np.ndarray, triangle_area: np.ndarray, state_count: int
) -> np.ndarray:
    valid = assignment != INVALID_BASIS
    return np.bincount(
        assignment[valid], weights=triangle_area[valid], minlength=state_count
    ).astype(np.float64)


def parent_of_child(
    parent: np.ndarray, child: np.ndarray, child_count: int
) -> np.ndarray:
    result = np.full(child_count, INVALID_BASIS, dtype=np.uint32)
    valid = child != INVALID_BASIS
    for child_id, parent_id in zip(child[valid], parent[valid]):
        previous = result[child_id]
        if previous != INVALID_BASIS and previous != parent_id:
            raise RuntimeError(
                "basis patch sizes are not nested; use integer refinements "
                "with a common coordinate origin"
            )
        result[child_id] = parent_id
    if np.any(result == INVALID_BASIS):
        raise RuntimeError("a child basis state has no parent")
    return result


def write_csr(
    group: h5py.Group,
    rows: int,
    cols: int,
    row_index: np.ndarray,
    column_index: np.ndarray,
    data: np.ndarray,
) -> None:
    order = np.lexsort((column_index, row_index))
    row_index = row_index[order]
    column_index = column_index[order]
    data = data[order]
    indptr = np.zeros(rows + 1, dtype=np.uint64)
    np.add.at(indptr, row_index + 1, 1)
    np.cumsum(indptr, out=indptr)
    group.create_dataset("shape", data=np.asarray([rows, cols], np.uint64))
    group.create_dataset("indptr", data=indptr)
    group.create_dataset("indices", data=column_index.astype(np.uint32))
    group.create_dataset("data", data=data.astype(np.float64))


def build_hierarchy(
    input_path: Path,
    output_path: Path,
    patch_sizes_mm: list[float],
    reflective_surface_ids: list[int],
    active_level: int,
    *,
    force: bool = False,
) -> dict:
    if sorted(patch_sizes_mm, reverse=True) != patch_sizes_mm:
        raise ValueError("patch sizes must be ordered coarse to fine")
    if patch_sizes_mm[-1] != 0.0:
        raise ValueError("the final exact-leaf level must have patch size zero")
    if active_level < 0:
        active_level += len(patch_sizes_mm)
    if not 0 <= active_level < len(patch_sizes_mm):
        raise ValueError("active level is outside the hierarchy")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("surface-basis output must differ from its input")
    identity = {
        "schema": "oos.surface_basis.v1",
        "input_geometry_sha256": file_sha256(input_path),
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "patch_sizes_mm": patch_sizes_mm,
        "reflective_surface_ids": reflective_surface_ids,
        "active_level": active_level,
    }
    if output_path.is_file() and not force:
        try:
            with h5py.File(output_path, "r") as cached:
                stored = json.loads(
                    bytes(
                        cached["/surface_basis/generator_json"][:]
                    ).decode()
                )
                if all(stored.get(key) == value for key, value in identity.items()):
                    return {
                        "path": str(output_path.resolve()),
                        "sha256": file_sha256(output_path),
                        "triangle_count": int(
                            cached["/geometry/triangles"].shape[0]
                        ),
                        "state_counts": cached[
                            "/surface_basis/state_count"
                        ][:].tolist(),
                        "active_level": active_level,
                        "cache": "hit",
                    }
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass

    with h5py.File(input_path, "r") as source:
        vertices = np.asarray(source["/geometry/vertices"], dtype=np.float64)
        triangles = np.asarray(source["/geometry/triangles"], dtype=np.uint32)
        surface_id = np.asarray(
            source["/geometry/surface_id"], dtype=np.uint32
        )
    centroids, normals, area = triangle_properties(vertices, triangles)
    reflective = np.isin(
        surface_id, np.asarray(reflective_surface_ids, dtype=np.uint32)
    )
    edge_triangle = geometric_edge_triangles(
        triangles, surface_id, normals, reflective
    )
    levels: list[np.ndarray] = []
    counts: list[int] = []
    areas: list[np.ndarray] = []
    for size in patch_sizes_mm:
        assignment, count = assign_level(
            centroids, normals, surface_id, reflective, size
        )
        levels.append(assignment)
        counts.append(count)
        areas.append(state_areas(assignment, area, count))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(input_path, temporary)
        with h5py.File(temporary, "r+") as target:
            geometry = target["/geometry"]
            if "surface_basis_id" in geometry:
                del geometry["surface_basis_id"]
            geometry.create_dataset(
                "surface_basis_id", data=levels[active_level]
            )
            if "surface_basis" in target:
                del target["surface_basis"]
            hierarchy = target.create_group("surface_basis")
            hierarchy.attrs["schema"] = "oos.surface_basis.v1"
            hierarchy.attrs["active_level"] = active_level
            hierarchy.create_dataset(
                "patch_size_mm",
                data=np.asarray(patch_sizes_mm, dtype=np.float64),
            )
            hierarchy.create_dataset(
                "state_count", data=np.asarray(counts, dtype=np.uint64)
            )
            hierarchy.create_dataset(
                "reflective_surface_id",
                data=np.asarray(reflective_surface_ids, dtype=np.uint32),
            )
            hierarchy.create_dataset(
                "triangle_centroid_mm",
                data=centroids,
                compression="gzip",
                shuffle=True,
            )
            hierarchy.create_dataset(
                "edge_triangle",
                data=edge_triangle.astype(np.uint8),
                compression="gzip",
                shuffle=True,
            )
            level_group = hierarchy.create_group("levels")
            for index, (assignment, state_area) in enumerate(
                zip(levels, areas)
            ):
                group = level_group.create_group(str(index))
                group.create_dataset(
                    "triangle_basis_id",
                    data=assignment,
                    compression="gzip",
                    shuffle=True,
                )
                group.create_dataset("area_mm2", data=state_area)
            transfers = hierarchy.create_group("transfers")
            for parent_level in range(len(levels) - 1):
                child_level = parent_level + 1
                parent = parent_of_child(
                    levels[parent_level],
                    levels[child_level],
                    counts[child_level],
                )
                group = transfers.create_group(
                    f"{parent_level}_to_{child_level}"
                )
                restriction = group.create_group("restriction")
                write_csr(
                    restriction,
                    counts[parent_level],
                    counts[child_level],
                    parent,
                    np.arange(counts[child_level], dtype=np.uint32),
                    np.ones(counts[child_level], dtype=np.float64),
                )
                prolongation = group.create_group("prolongation")
                weights = areas[child_level] / areas[parent_level][parent]
                write_csr(
                    prolongation,
                    counts[child_level],
                    counts[parent_level],
                    np.arange(counts[child_level], dtype=np.uint32),
                    parent,
                    weights,
                )
                parent_sum = np.bincount(
                    parent, weights=weights, minlength=counts[parent_level]
                )
                if not np.allclose(parent_sum, 1.0, rtol=0.0, atol=1e-12):
                    raise RuntimeError(
                        "prolongation does not conserve parent power"
                    )
            metadata = dict(identity, state_counts=counts)
            raw = json.dumps(metadata, sort_keys=True).encode()
            hierarchy.create_dataset(
                "generator_json",
                data=np.frombuffer(raw, dtype=np.uint8),
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output_path.resolve()),
        "sha256": file_sha256(output_path),
        "triangle_count": len(triangles),
        "state_counts": counts,
        "active_level": active_level,
        "cache": "miss",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--patch-sizes-mm",
        type=float,
        nargs="+",
        default=[40.0, 20.0, 10.0, 0.0],
    )
    parser.add_argument(
        "--reflective-surface-ids",
        type=int,
        nargs="+",
        default=[0, 5, 6],
    )
    parser.add_argument("--active-level", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    result = build_hierarchy(
        arguments.input,
        arguments.output,
        arguments.patch_sizes_mm,
        arguments.reflective_surface_ids,
        arguments.active_level,
        force=arguments.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
