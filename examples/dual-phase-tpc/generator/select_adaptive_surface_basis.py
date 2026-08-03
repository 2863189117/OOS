#!/usr/bin/env python3
"""Freeze a hierarchical surface basis for one spatial source tile.

The output remains a complete canonical geometry file.  Only
``/geometry/surface_basis_id`` changes.  All candidate source positions in the
same tile therefore use an identical operator and a continuous likelihood.
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


def tile_center(value: float, tile_size_mm: float) -> float:
    return tile_size_mm * math.floor(value / tile_size_mm + 0.5)


def select_basis(
    levels: list[np.ndarray],
    centroids: np.ndarray,
    edge_triangle: np.ndarray,
    reflective: np.ndarray,
    center_xy_mm: np.ndarray,
    *,
    near_radius_mm: float,
    edge_radius_mm: float,
    near_level: int,
    edge_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    level_count = len(levels)
    if not 0 <= near_level < level_count:
        raise ValueError("near_level is outside the hierarchy")
    if not 0 <= edge_level < level_count:
        raise ValueError("edge_level is outside the hierarchy")
    distance = np.linalg.norm(
        centroids[:, :2] - center_xy_mm[np.newaxis, :], axis=1
    )
    target = np.zeros(len(centroids), dtype=np.uint8)
    target[(distance <= near_radius_mm) & reflective] = near_level
    target[
        (distance <= edge_radius_mm) & edge_triangle & reflective
    ] = np.maximum(
        target[(distance <= edge_radius_mm) & edge_triangle & reflective],
        edge_level,
    )

    active_level = np.zeros(len(centroids), dtype=np.uint8)
    for level in range(level_count - 1):
        requests = reflective & (target > level)
        refine_nodes = np.unique(levels[level][requests])
        refine_nodes = refine_nodes[refine_nodes != INVALID_BASIS]
        refine = (
            reflective
            & (active_level == level)
            & np.isin(levels[level], refine_nodes)
        )
        active_level[refine] = level + 1

    assignment = np.full(len(centroids), INVALID_BASIS, dtype=np.uint32)
    selected = np.flatnonzero(reflective)
    keys = np.column_stack(
        [
            active_level[selected],
            np.asarray(
                [
                    levels[int(level)][triangle]
                    for triangle, level in zip(
                        selected, active_level[selected]
                    )
                ],
                dtype=np.uint32,
            ),
        ]
    )
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    assignment[selected] = inverse.astype(np.uint32)
    return assignment, active_level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-x-mm", type=float, required=True)
    parser.add_argument("--source-y-mm", type=float, required=True)
    parser.add_argument("--tile-size-mm", type=float, default=100.0)
    parser.add_argument("--near-radius-mm", type=float, default=300.0)
    parser.add_argument("--edge-radius-mm", type=float, default=400.0)
    parser.add_argument("--near-level", type=int, default=2)
    parser.add_argument("--edge-level", type=int, default=3)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="write only the compact triangle-to-basis overlay",
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.tile_size_mm <= 0.0:
        parser.error("--tile-size-mm must be positive")
    center = np.asarray(
        [
            tile_center(arguments.source_x_mm, arguments.tile_size_mm),
            tile_center(arguments.source_y_mm, arguments.tile_size_mm),
        ],
        dtype=np.float64,
    )
    identity = {
        "schema": "oos.surface_basis_selection.v1",
        "geometry_sha256": file_sha256(arguments.geometry),
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "tile_center_xy_mm": center.tolist(),
        "tile_size_mm": arguments.tile_size_mm,
        "near_radius_mm": arguments.near_radius_mm,
        "edge_radius_mm": arguments.edge_radius_mm,
        "near_level": arguments.near_level,
        "edge_level": arguments.edge_level,
        "selection_only": arguments.selection_only,
    }
    if arguments.output.is_file() and not arguments.force:
        try:
            with h5py.File(arguments.output, "r") as cached:
                stored = json.loads(
                    bytes(
                        cached[
                            "/surface_basis_selection/generator_json"
                        ][:]
                    ).decode()
                )
            if stored == identity:
                print(
                    json.dumps(
                        {
                            "cache": "hit",
                            "path": str(arguments.output.resolve()),
                            **identity,
                        },
                        indent=2,
                    )
                )
                return
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass

    with h5py.File(arguments.geometry, "r") as source:
        hierarchy = source["/surface_basis"]
        level_names = sorted(
            hierarchy["levels"], key=lambda value: int(value)
        )
        levels = [
            np.asarray(
                hierarchy[f"levels/{name}/triangle_basis_id"],
                dtype=np.uint32,
            )
            for name in level_names
        ]
        centroids = np.asarray(
            hierarchy["triangle_centroid_mm"], dtype=np.float64
        )
        edge_triangle = np.asarray(
            hierarchy["edge_triangle"], dtype=bool
        )
        reflective = levels[-1] != INVALID_BASIS
    assignment, active_level = select_basis(
        levels,
        centroids,
        edge_triangle,
        reflective,
        center,
        near_radius_mm=arguments.near_radius_mm,
        edge_radius_mm=arguments.edge_radius_mm,
        near_level=arguments.near_level,
        edge_level=arguments.edge_level,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=arguments.output.name + ".", dir=arguments.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if arguments.selection_only:
            with h5py.File(temporary, "w") as target:
                group = target.create_group("surface_basis_selection")
                group.create_dataset(
                    "triangle_basis_id",
                    data=assignment,
                    compression="gzip",
                    shuffle=True,
                )
                raw = json.dumps(identity, sort_keys=True).encode()
                group.create_dataset(
                    "generator_json",
                    data=np.frombuffer(raw, dtype=np.uint8),
                )
                group.create_dataset(
                    "triangle_active_level",
                    data=active_level,
                    compression="gzip",
                    shuffle=True,
                )
        else:
            shutil.copyfile(arguments.geometry, temporary)
            with h5py.File(temporary, "r+") as target:
                del target["/geometry/surface_basis_id"]
                target["/geometry"].create_dataset(
                    "surface_basis_id", data=assignment
                )
                if "surface_basis_selection" in target:
                    del target["surface_basis_selection"]
                group = target.create_group("surface_basis_selection")
                raw = json.dumps(identity, sort_keys=True).encode()
                group.create_dataset(
                    "generator_json",
                    data=np.frombuffer(raw, dtype=np.uint8),
                )
                group.create_dataset(
                    "triangle_active_level",
                    data=active_level,
                    compression="gzip",
                    shuffle=True,
                )
        os.replace(temporary, arguments.output)
    finally:
        temporary.unlink(missing_ok=True)
    valid = assignment != INVALID_BASIS
    if not np.any(valid):
        raise RuntimeError("adaptive selection contains no reflective states")
    result = {
        "cache": "miss",
        "path": str(arguments.output.resolve()),
        "sha256": file_sha256(arguments.output),
        "state_count": int(assignment[valid].astype(np.uint64).max() + 1),
        "triangle_count_by_level": np.bincount(
            active_level[valid], minlength=len(levels)
        ).tolist(),
        **identity,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
