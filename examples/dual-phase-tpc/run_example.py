#!/usr/bin/env python3
"""Build and solve the synthetic dual-phase TPC example."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> dict:
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    output = log.read_text(encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=output,
        )
    return {
        "command": command,
        "wall_seconds": time.perf_counter() - started,
        "stdout": output,
    }


def link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.symlink_to(source.resolve())


def analytic_geometry_command(
    python: str,
    root: Path,
    output: Path,
    profile: str,
) -> list[str]:
    analytic_profile = "debug" if profile == "smoke" else profile
    return [
        python,
        str(root / "generator" / "build_analytic_geometry.py"),
        "--output",
        str(output),
        "--analytic-profile",
        analytic_profile,
        "--validation-profile",
        profile,
        "--height-mm",
        "60",
    ]


def lxe_generator_profile(profile: str) -> list[str]:
    return ["--test"] if profile == "smoke" else []


def read_json_dataset(handle: h5py.File, path: str) -> dict:
    return json.loads(bytes(handle[path][...]).decode())


def verify_lxe_block_geometry(geometry: Path, block: Path) -> dict:
    with h5py.File(geometry, "r") as handle:
        config = read_json_dataset(
            handle, "/metadata/generator_json"
        )["config"]
        radius_mm = float(config["active_radius_mm"])
        depth_mm = float(config["lxe_depth_mm"])
        if "/analytic/kind" in handle:
            kind = handle["/analytic/kind"][:]
            surface = handle["/analytic/surface_id"][:]
            parameters = handle["/analytic/parameters"][:]
            selected = np.flatnonzero((kind == 1) & (surface == 1))
            if len(selected) != 1:
                raise RuntimeError(
                    "geometry must declare exactly one analytic LXe disk"
                )
            analytic_radius = float(parameters[int(selected[0]), 0])
            if abs(analytic_radius - radius_mm) > 1.0e-9:
                raise RuntimeError(
                    "analytic LXe disk radius disagrees with geometry metadata"
                )
    with h5py.File(block, "r") as handle:
        generator = read_json_dataset(handle, "/metadata/generator_json")
        phase_grid = read_json_dataset(handle, "/metadata/phase_grid_json")
    liquid = generator["lxe_config"]
    if (
        abs(float(liquid["radius_mm"]) - radius_mm) > 1.0e-9
        or abs(float(liquid["depth_mm"]) - depth_mm) > 1.0e-9
        or abs(float(phase_grid["radius_mm"]) - radius_mm) > 1.0e-9
    ):
        raise RuntimeError(
            "LXe block dimensions do not match the selected geometry"
        )
    return {
        "radius_mm": radius_mm,
        "depth_mm": depth_mm,
        "coefficient_layout": generator.get("coefficient_layout"),
        "generator_sha256": generator.get("generator_sha256"),
        "compute_backend": generator.get("compute_backend"),
        "collision_sample_power": generator.get("collision_sample_power"),
        "geometry_contract_sha256": generator.get(
            "geometry_contract_sha256"
        ),
        "test_profile": arguments_is_test_profile(phase_grid),
    }


def arguments_is_test_profile(phase_grid: dict) -> bool:
    return (
        int(phase_grid["position_radial_bins"]) == 4
        and int(phase_grid["position_phi_bins"]) == 8
        and int(phase_grid["direction_mu_bins"]) == 2
        and int(phase_grid["direction_phi_bins"]) == 4
    )


def render_scene(template: Path, destination: Path, lxe_block: Path) -> dict:
    with h5py.File(lxe_block, "r") as handle:
        generator = read_json_dataset(handle, "/metadata/generator_json")
        phase_grid = read_json_dataset(handle, "/metadata/phase_grid_json")
    liquid = generator["lxe_config"]
    plugin_config = {
        "geometry": "finite_cylinder",
        "radius_mm": liquid["radius_mm"],
        "depth_mm": liquid["depth_mm"],
        "explicit_collision_order": generator["explicit_collision_order"],
        "nonlocal_domain_id": 2,
        "side_reflectivity": liquid["side_reflectivity"],
        "phase_grid": {
            name: phase_grid[name]
            for name in (
                "position_radial_bins",
                "position_phi_bins",
                "direction_mu_bins",
                "direction_phi_bins",
                "direction_mu_minimum",
            )
        },
        "inward_normal": [0.0, 0.0, -1.0],
        "factorized_block_hdf5": "lxe-factorized-block.h5",
    }
    lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("plugin_config_json:")
    )
    indentation = len(lines[start]) - len(lines[start].lstrip())
    stop = start + 1
    while stop < len(lines):
        stripped = lines[stop].lstrip()
        current_indentation = len(lines[stop]) - len(stripped)
        if stripped.strip() and current_indentation <= indentation:
            break
        stop += 1
    encoded = json.dumps(plugin_config, separators=(",", ":"), sort_keys=True)
    lines[start:stop] = [
        f"{' ' * indentation}plugin_config_json: '{encoded}'\n"
    ]
    destination.write_text("".join(lines), encoding="utf-8")
    return plugin_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oos", type=Path, required=True)
    parser.add_argument(
        "--oos-efficiency",
        type=Path,
        help="native production efficiency entry point; defaults beside --oos",
    )
    parser.add_argument("--lxe-plugin", type=Path, required=True)
    parser.add_argument(
        "--geometry",
        type=Path,
        help=(
            "use an existing canonical geometry; when omitted the example "
            "generates and caches geometry.h5"
        ),
    )
    parser.add_argument(
        "--geometry-profile",
        choices=("smoke", "coarse", "production", "fine"),
        default="coarse",
    )
    parser.add_argument(
        "--surface-basis",
        type=Path,
        help=(
            "use an existing analytic geometry/basis file; defaults to "
            "--geometry"
        ),
    )
    parser.add_argument(
        "--surface-basis-selection",
        type=Path,
        help=(
            "compact tile-frozen triangle-to-basis overlay passed to the "
            "native builder and source tracer"
        ),
    )
    parser.add_argument(
        "--lxe-block",
        type=Path,
        help=(
            "use an existing canonical factorized block; when omitted the "
            "example generator builds and caches one from --geometry"
        ),
    )
    parser.add_argument(
        "--python", default=os.environ.get("PYTHON", sys.executable)
    )
    parser.add_argument("--generator-processes", type=int, default=24)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--direct-solve",
        action="store_true",
        help="bypass the default materialized seven-cycle response",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    arguments = parser.parse_args()
    if arguments.oos_efficiency is None:
        arguments.oos_efficiency = arguments.oos.with_name("oos-efficiency")

    root = Path(__file__).resolve().parent
    arguments.output.mkdir(parents=True, exist_ok=False)
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)
    scene = arguments.output / "scene.yaml"
    sources = arguments.output / "sources.yaml"
    shutil.copyfile(root / "sources.yaml", sources)
    stages = []
    geometry = arguments.geometry
    if geometry is None:
        geometry = arguments.cache_dir / (
            f"geometry-gas-height-60-analytic-"
            f"{arguments.geometry_profile}.h5"
        )
        geometry_command = analytic_geometry_command(
            arguments.python,
            root,
            geometry,
            arguments.geometry_profile,
        )
        if arguments.force_rebuild:
            geometry_command.append("--force")
        stages.append(
            run(
                geometry_command,
                arguments.output / "build-geometry.log",
            )
        )
    surface_basis = arguments.surface_basis
    if surface_basis is None:
        surface_basis = geometry
    link(surface_basis, arguments.output / "geometry.h5")
    lxe_block = arguments.lxe_block
    if lxe_block is None:
        suffix = "-test" if arguments.geometry_profile == "smoke" else ""
        lxe_block = arguments.cache_dir / f"lxe-factorized-block{suffix}.h5"
        generator = [
            arguments.python,
            str(root / "generator" / "build_lxe_function_block.py"),
            "--geometry",
            str(surface_basis),
            "--output",
            str(lxe_block),
            "--processes",
            str(arguments.generator_processes),
            *lxe_generator_profile(arguments.geometry_profile),
        ]
        if arguments.force_rebuild:
            generator.append("--force")
        stages.append(
            run(generator, arguments.output / "build-lxe-function.log")
        )
    lxe_contract = verify_lxe_block_geometry(surface_basis, lxe_block)
    link(lxe_block, arguments.output / "lxe-factorized-block.h5")
    link(arguments.lxe_plugin, arguments.output / "liboos_lxe_plugin.so")
    lxe_plugin_config = render_scene(root / "scene.yaml", scene, lxe_block)

    operators = (
        arguments.cache_dir
        / f"operators-analytic-{arguments.geometry_profile}.h5"
    )
    effective = (
        arguments.cache_dir
        / (
            f"effective-adjoint-seven-cycle-v3-analytic-"
            f"{arguments.geometry_profile}.h5"
        )
    )
    response = arguments.output / "response.h5"
    validate = [str(arguments.oos), "validate", str(scene)]
    if arguments.surface_basis_selection is not None:
        validate.extend(
            [
                "--surface-basis",
                str(arguments.surface_basis_selection),
            ]
        )
    stages.append(run(validate, arguments.output / "validate.log"))
    build = [
        str(arguments.oos),
        "build",
        str(scene),
        "--energy-eV",
        "7",
        "--cache",
        str(operators),
    ]
    if arguments.force_rebuild:
        build.append("--force")
    if arguments.surface_basis_selection is not None:
        build.extend(
            [
                "--surface-basis",
                str(arguments.surface_basis_selection),
            ]
        )
    stages.append(run(build, arguments.output / "build.log"))
    if not arguments.direct_solve and (
        arguments.force_rebuild or not effective.is_file()
    ):
        stages.append(
            run(
                [
                    str(arguments.oos_efficiency),
                    "precompute",
                    str(operators),
                    "--output",
                    str(effective),
                    "--cycles",
                    "7",
                    "--device",
                    arguments.device,
                ],
                arguments.output / "precompute.log",
            )
        )
    solve = [
        str(arguments.oos_efficiency),
        "calculate",
        str(operators),
        "--scene",
        str(scene),
        "--sources",
        str(sources),
        "--output",
        str(response),
        "--device",
        arguments.device,
    ]
    if arguments.direct_solve:
        solve.append("--direct")
    else:
        solve.extend(["--precomputed", str(effective)])
    if arguments.surface_basis_selection is not None:
        solve.extend(
            [
                "--surface-basis",
                str(arguments.surface_basis_selection),
            ]
        )
    stages.append(run(solve, arguments.output / "solve.log"))
    manifest = {
        "schema": "oos.dual-phase-tpc.example-run.v2",
        "inputs": {
            "geometry": {
                "path": str(surface_basis.resolve()),
                "sha256": sha256(surface_basis),
                "generated_by_example": arguments.geometry is None,
            },
            "canonical_geometry": {
                "path": str(geometry.resolve()),
                "sha256": sha256(geometry),
            },
            "surface_basis_is_geometry": arguments.surface_basis is None,
            "surface_basis_selection": (
                {
                    "path": str(
                        arguments.surface_basis_selection.resolve()
                    ),
                    "sha256": sha256(arguments.surface_basis_selection),
                }
                if arguments.surface_basis_selection is not None
                else None
            ),
            "lxe_block": {
                "path": str(lxe_block.resolve()),
                "sha256": sha256(lxe_block),
                "generated_by_example": arguments.lxe_block is None,
                "geometry_contract": lxe_contract,
            },
            "lxe_plugin": {
                "path": str(arguments.lxe_plugin.resolve()),
                "sha256": sha256(arguments.lxe_plugin),
            },
            "lxe_plugin_config": lxe_plugin_config,
        },
        "operators": {
            "path": str(operators.resolve()),
            "sha256": sha256(operators),
        },
        "effective_response": (
            None
            if arguments.direct_solve
            else {
                "path": str(effective.resolve()),
                "sha256": sha256(effective),
                "cycles": 7,
            }
        ),
        "response": {
            "path": str(response.resolve()),
            "sha256": sha256(response),
        },
        "device": arguments.device,
        "stages": stages,
    }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
