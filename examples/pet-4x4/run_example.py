#!/usr/bin/env python3
"""Build and solve the PET-4x4 deterministic response example."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
WIRE_CENTER_Z_MM = 17.499
WIRE_HALF_WIDTH_MM = 0.0025
WIRE_HALF_THICKNESS_MM = 0.0005
SOURCE_DISTANCE_MAX_MM = 0.006
SOURCE_LXE_Z_MAX_MM = 17.5
WIRE_Y_START_MM = -33.5
WIRE_PITCH_MM = 1.0
WIRE_COUNT = 68
SOURCE_BACKEND = "generic_bvh"
PRECOMPUTE_BATCH_SIZE = 64
GRID_BATCH_SIZE = 16


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--geometry-profile",
        choices=("smoke", "coarse", "production", "fine"),
        default="production",
    )
    parser.add_argument("--cycles", type=int, default=112)
    parser.add_argument("--grid-spacing-mm", type=float, default=2.0)
    parser.add_argument("--source-mu-order", type=int, default=32)
    parser.add_argument("--source-phi-count", type=int, default=128)
    parser.add_argument("--source-transverse-count", type=int, default=32)
    arguments = parser.parse_args()
    if arguments.cycles <= 0:
        parser.error("--cycles must be positive")
    if arguments.source_transverse_count <= 0:
        parser.error("--source-transverse-count must be positive")

    arguments.output.mkdir(parents=True, exist_ok=False)
    scene = arguments.output / "scene.yaml"
    sources = arguments.output / "sources.yaml"
    geometry = arguments.output / "geometry.h5"
    operators = arguments.output / "operators.h5"
    direct = arguments.output / "direct-response.h5"
    effective = arguments.output / f"effective-{arguments.cycles}.h5"
    bounded = arguments.output / f"response-{arguments.cycles}.h5"
    grid = arguments.output / f"grid-{arguments.grid_spacing_mm:g}mm.h5"
    shutil.copyfile(ROOT / "scene.yaml", scene)
    shutil.copyfile(ROOT / "sources.yaml", sources)

    run(
        [
            sys.executable,
            str(ROOT / "generator" / "build_geometry.py"),
            "--output",
            str(geometry),
            "--profile",
            arguments.geometry_profile,
        ]
    )
    oos = str(arguments.binary_dir / "oos")
    efficiency = str(arguments.binary_dir / "oos-efficiency")
    regress = str(arguments.binary_dir / "oos-regress")
    run([oos, "validate", str(scene)])
    run([oos, "build", str(scene), "--cache", str(operators)])
    run(
        [
            oos,
            "solve",
            str(operators),
            "--scene",
            str(scene),
            "--sources",
            str(sources),
            "--output",
            str(direct),
        ]
    )
    run(
        [
            efficiency,
            "precompute",
            str(operators),
            "--output",
            str(effective),
            "--cycles",
            str(arguments.cycles),
            "--batch-size",
            str(PRECOMPUTE_BATCH_SIZE),
            "--device",
            "cpu",
        ]
    )
    run(
        [
            efficiency,
            "calculate",
            str(operators),
            "--precomputed",
            str(effective),
            "--scene",
            str(scene),
            "--sources",
            str(sources),
            "--output",
            str(bounded),
            "--device",
            "cpu",
        ]
    )
    run(
        [
            regress,
            "grid",
            str(operators),
            "--precomputed",
            str(effective),
            "--scene",
            str(scene),
            "--output",
            str(grid),
            "--grid-shape",
            "parallel_lines",
            "--half-x-mm",
            "34",
            "--line-y-start-mm",
            str(WIRE_Y_START_MM),
            "--line-pitch-mm",
            str(WIRE_PITCH_MM),
            "--line-count",
            str(WIRE_COUNT),
            "--spacing-mm",
            str(arguments.grid_spacing_mm),
            "--source-z-mm",
            str(WIRE_CENTER_Z_MM),
            "--source-thickness-mm",
            str(SOURCE_DISTANCE_MAX_MM),
            "--source-angular-mode",
            "rectangular_line_neighborhood_isotropic_product",
            "--source-backend",
            SOURCE_BACKEND,
            "--source-transverse-count",
            str(arguments.source_transverse_count),
            "--obstacle-half-width-mm",
            str(WIRE_HALF_WIDTH_MM),
            "--obstacle-half-thickness-mm",
            str(WIRE_HALF_THICKNESS_MM),
            "--source-medium-z-max-mm",
            str(SOURCE_LXE_Z_MAX_MM),
            "--source-mu-order",
            str(arguments.source_mu_order),
            "--source-phi-count",
            str(arguments.source_phi_count),
            "--batch-size",
            str(GRID_BATCH_SIZE),
            "--device",
            "cpu",
        ]
    )
    print(f"PET example artifacts written to {arguments.output}")


if __name__ == "__main__":
    main()
