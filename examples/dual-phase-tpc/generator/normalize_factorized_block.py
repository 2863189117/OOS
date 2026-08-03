#!/usr/bin/env python3
"""Normalize metadata in an already computed factorized LXe block."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(handle: h5py.File, path: str) -> dict:
    return json.loads(
        bytes(np.asarray(handle[path], dtype=np.uint8)).decode()
    )


def replace_json(handle: h5py.File, path: str, value: dict) -> None:
    del handle[path]
    handle.create_dataset(
        path,
        data=np.frombuffer(
            json.dumps(value, sort_keys=True).encode(), dtype=np.uint8
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() and not arguments.force:
        parser.error("--output exists; pass --force to replace it")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=arguments.output.name + ".",
        dir=arguments.output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(arguments.input, temporary)
        with h5py.File(temporary, "r+") as handle:
            coefficient_shape = handle[
                "/function/coefficients_real"
            ].shape
            if len(coefficient_shape) != 5:
                raise ValueError("factorized coefficients must be rank five")
            phase = read_json(handle, "/metadata/phase_grid_json")
            metadata = read_json(handle, "/metadata/generator_json")
            nd, nr, nm, _, surface_radial = coefficient_shape
            np_ = int(phase["position_phi_bins"])
            state_count = int(nr * np_ * nm * nd)
            egress_count = int(
                handle["/nonlocal/egress/surface_element"].shape[0]
            )
            angular_count = int(
                handle["/function/angular_weight"].shape[0]
            )
            denominator = surface_radial * angular_count
            if denominator == 0 or egress_count % denominator:
                raise ValueError("egress basis does not match function shape")
            previous_state_count = metadata.get("state_count")
            if (
                previous_state_count is not None
                and int(previous_state_count) != state_count
            ):
                metadata["coefficient_row_count"] = int(
                    previous_state_count
                )
            metadata.update(
                {
                    "schema": "oos.nonlocal.function.v1",
                    "surface_relative": True,
                    "execution": "function",
                    "state_count": state_count,
                    "egress_count": egress_count,
                    "surface_phi_bins": int(
                        egress_count // denominator
                    ),
                    "angular_count": angular_count,
                }
            )
            replace_json(
                handle, "/metadata/generator_json", metadata
            )
        os.replace(temporary, arguments.output)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "input": str(arguments.input.resolve()),
                "input_sha256": sha256(arguments.input),
                "output": str(arguments.output.resolve()),
                "output_sha256": sha256(arguments.output),
                "state_count": state_count,
                "coefficient_row_count": metadata.get(
                    "coefficient_row_count"
                ),
                "egress_count": egress_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
