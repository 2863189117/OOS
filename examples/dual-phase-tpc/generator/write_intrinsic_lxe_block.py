#!/usr/bin/env python3
"""Write an LXe block that is intrinsic to the LXe surface.

The physics generator should call :func:`write_block` directly.  The CLI form
accepts an NPZ with the same named arrays and is useful for format-conversion tests.
No PMT channel, external primitive, or PTFE target is accepted here; those
couplings are constructed later by the C++ Embree preprocessor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse


def _validated_csr(value: sparse.spmatrix, name: str) -> sparse.csr_matrix:
    result = value.tocsr().astype(np.float64)
    result.sum_duplicates()
    result.sort_indices()
    if result.data.size and (
        not np.all(np.isfinite(result.data)) or np.any(result.data < 0.0)
    ):
        raise ValueError(f"{name} contains an invalid weight")
    return result


def _write_csr(parent: h5py.Group, name: str, value: sparse.csr_matrix) -> None:
    group = parent.create_group(name)
    group.create_dataset("shape", data=np.asarray(value.shape, dtype=np.uint64))
    group.create_dataset("indptr", data=value.indptr.astype(np.uint64))
    group.create_dataset("indices", data=value.indices.astype(np.uint32))
    group.create_dataset("data", data=value.data.astype(np.float64))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(input_path: Path, loss_names: list[str]) -> str:
    material = {
        "schema": "oos.nonlocal.intrinsic.v1",
        "input_sha256": _sha256(input_path),
        "loss_names": loss_names,
        "writer_sha256": _sha256(Path(__file__)),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _stored_cache_key(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with h5py.File(path, "r") as handle:
            raw = bytes(np.asarray(
                handle["/metadata/generator_json"], dtype=np.uint8
            ))
        return json.loads(raw.decode()).get("cache_key_sha256")
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def write_block(
    output: Path,
    *,
    internal_transition: sparse.spmatrix,
    emission: sparse.spmatrix,
    internal_losses: sparse.spmatrix,
    surface_element: np.ndarray,
    barycentric: np.ndarray,
    side: np.ndarray,
    direction_local: np.ndarray,
    loss_names: list[str],
    metadata: dict[str, Any] | None = None,
    tolerance: float = 1.0e-10,
) -> None:
    """Write the canonical geometry-independent nonlocal surface block."""

    transition = _validated_csr(internal_transition, "internal_transition")
    emitted = _validated_csr(emission, "emission")
    losses = _validated_csr(internal_losses, "internal_losses")
    element = np.asarray(surface_element, dtype=np.uint64)
    bary = np.asarray(barycentric, dtype=np.float64)
    output_side = np.asarray(side, dtype=np.uint64)
    direction = np.asarray(direction_local, dtype=np.float64)
    states = transition.shape[0]
    egress_count = element.size
    if transition.shape != (states, states):
        raise ValueError("internal_transition must be square")
    if emitted.shape != (states, egress_count):
        raise ValueError("emission columns must match the egress basis")
    if losses.shape != (states, len(loss_names)):
        raise ValueError("internal_losses columns must match loss_names")
    if bary.shape != (egress_count, 3) or direction.shape != (egress_count, 3):
        raise ValueError("egress barycentric and direction arrays must be [Ne,3]")
    if output_side.shape != (egress_count,) or np.any(output_side > 1):
        raise ValueError("egress side must be a length-Ne zero/one array")
    if np.any(bary < 0.0) or not np.allclose(
        bary.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("egress barycentric coordinates are invalid")
    if (
        not np.all(np.isfinite(direction))
        or np.any(direction[:, 2] <= 0.0)
        or not np.allclose(
            np.linalg.norm(direction, axis=1), 1.0, rtol=0.0, atol=1.0e-12
        )
    ):
        raise ValueError("egress directions must be unit vectors with local z>0")
    row_total = (
        np.asarray(transition.sum(axis=1)).ravel()
        + np.asarray(emitted.sum(axis=1)).ravel()
        + np.asarray(losses.sum(axis=1)).ravel()
    )
    if not np.allclose(row_total, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("intrinsic state rows do not conserve probability")

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        root = handle.create_group("nonlocal")
        _write_csr(root, "internal_transition", transition)
        _write_csr(root, "emission", emitted)
        _write_csr(root, "internal_losses", losses)
        egress = root.create_group("egress")
        egress.create_dataset("surface_element", data=element)
        egress.create_dataset("barycentric", data=bary)
        egress.create_dataset("side", data=output_side)
        egress.create_dataset("direction_local", data=direction)
        meta = handle.create_group("metadata")
        meta.create_dataset(
            "loss_names_json",
            data=np.frombuffer(json.dumps(loss_names).encode(), dtype=np.uint8),
        )
        payload = {
            "schema": "oos.nonlocal.intrinsic.v1",
            "surface_relative": True,
            **(metadata or {}),
        }
        meta.create_dataset(
            "generator_json",
            data=np.frombuffer(json.dumps(payload, sort_keys=True).encode(),
                               dtype=np.uint8),
        )


def write_factorized_block(
    output: Path,
    *,
    coefficients: np.ndarray,
    expected_return: np.ndarray,
    audit_values: np.ndarray,
    surface_radius_mm: np.ndarray,
    surface_ring_area_mm2: np.ndarray,
    angular_weight: np.ndarray,
    surface_element: np.ndarray,
    barycentric: np.ndarray,
    side: np.ndarray,
    direction_local: np.ndarray,
    stokes: np.ndarray,
    reference_axis_local: np.ndarray,
    phase_grid: dict[str, Any],
    audit_names: list[str],
    surface_ring_offsets: np.ndarray | None = None,
    surface_phi_rad: np.ndarray | None = None,
    surface_area_mm2: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    tolerance: float = 1.0e-10,
) -> None:
    """Write a canonical matrix-free LXe function block.

    ``coefficients`` retain the Fourier--Bessel factorization used by the
    explicit-seven-collision/diffusion generator.  Rank-five coefficients use
    the legacy position-only surface marginal; rank-six coefficients add the
    egress-angle axis and therefore preserve the joint exit position and gas
    direction.  Geometry-dependent egress destinations are deliberately
    absent: the payload contains only points and polarized directions relative
    to its own surface group.
    """

    modes = np.asarray(coefficients, dtype=np.complex128)
    expected = np.asarray(expected_return, dtype=np.float64)
    audit = np.asarray(audit_values, dtype=np.float64)
    radius = np.asarray(surface_radius_mm, dtype=np.float64)
    ring_area = np.asarray(surface_ring_area_mm2, dtype=np.float64)
    angular = np.asarray(angular_weight, dtype=np.float64)
    element = np.asarray(surface_element, dtype=np.uint64)
    bary = np.asarray(barycentric, dtype=np.float64)
    output_side = np.asarray(side, dtype=np.uint64)
    direction = np.asarray(direction_local, dtype=np.float64)
    polarization = np.asarray(stokes, dtype=np.float64)
    reference = np.asarray(reference_axis_local, dtype=np.float64)
    ragged_values = (
        surface_ring_offsets,
        surface_phi_rad,
        surface_area_mm2,
    )
    ragged = any(value is not None for value in ragged_values)
    if ragged and not all(value is not None for value in ragged_values):
        raise ValueError(
            "ragged surface layout requires offsets, phi, and area together"
        )

    required_grid = (
        "position_radial_bins",
        "position_phi_bins",
        "direction_mu_bins",
        "direction_phi_bins",
    )
    if any(int(phase_grid.get(name, 0)) <= 0 for name in required_grid):
        raise ValueError("phase_grid is incomplete")
    nr, np_, nm, nd = (int(phase_grid[name]) for name in required_grid)
    if modes.ndim not in (5, 6) or modes.shape[:3] != (nd, nr, nm):
        raise ValueError("factorized coefficient array has wrong phase shape")
    if expected.shape != (nd, nr, nm):
        raise ValueError("expected_return has wrong shape")
    if audit.shape[:3] != (nd, nr, nm) or audit.ndim != 4:
        raise ValueError("audit_values has wrong shape")
    if audit.shape[3] != len(audit_names):
        raise ValueError("audit_names do not match audit_values")
    if modes.shape[4] != radius.size or ring_area.shape != radius.shape:
        raise ValueError("surface radial quadrature does not match coefficients")
    joint_angular = modes.ndim == 6
    if joint_angular and modes.shape[5] != angular.size:
        raise ValueError("joint coefficient angle axis does not match angular_weight")
    if (
        np.any(~np.isfinite(modes))
        or np.any(~np.isfinite(radius))
        or np.any(radius < 0.0)
        or np.any(~np.isfinite(ring_area))
        or np.any(ring_area <= 0.0)
    ):
        raise ValueError("modal coefficients or radial quadrature are invalid")
    if (
        np.any(~np.isfinite(expected))
        or np.any(expected < -tolerance)
        or np.any(expected >= 1.0)
    ):
        raise ValueError("expected_return must lie in [0,1)")
    if (
        angular.ndim != 1
        or angular.size == 0
        or np.any(~np.isfinite(angular))
        or np.any(angular < 0.0)
        or not np.isclose(angular.sum(), 1.0, rtol=0.0, atol=tolerance)
    ):
        raise ValueError("angular weights must be nonnegative and sum to one")
    egress_count = element.size
    if (
        egress_count % angular.size
        or bary.shape != (egress_count, 3)
        or output_side.shape != (egress_count,)
        or direction.shape != (egress_count, 3)
        or polarization.shape != (egress_count, 4)
        or reference.shape != (egress_count, 3)
    ):
        raise ValueError("egress arrays have inconsistent shapes")
    surface_points = egress_count // angular.size
    if ragged:
        ring_offsets = np.asarray(surface_ring_offsets, dtype=np.uint64)
        surface_phi = np.asarray(surface_phi_rad, dtype=np.float64)
        surface_area = np.asarray(surface_area_mm2, dtype=np.float64)
        if (
            ring_offsets.shape != (radius.size + 1,)
            or ring_offsets[0] != 0
            or ring_offsets[-1] != surface_points
            or np.any(np.diff(ring_offsets.astype(np.int64)) <= 0)
        ):
            raise ValueError("surface_ring_offsets do not partition the rings")
        if surface_phi.shape != (surface_points,) or surface_area.shape != (
            surface_points,
        ):
            raise ValueError("ragged surface arrays have inconsistent shapes")
        if (
            np.any(~np.isfinite(surface_phi))
            or np.any(surface_phi < 0.0)
            or np.any(surface_phi >= 2.0 * np.pi)
            or np.any(~np.isfinite(surface_area))
            or np.any(surface_area <= 0.0)
        ):
            raise ValueError("ragged surface phi/area values are invalid")
        area_by_ring = np.add.reduceat(
            surface_area, ring_offsets[:-1].astype(np.int64)
        )
        if not np.allclose(
            area_by_ring,
            ring_area,
            rtol=2.0e-12,
            atol=max(tolerance, 5.0e-8),
        ):
            raise ValueError(
                "ragged point areas do not reproduce the modal ring areas"
            )
        surface_phi_bins = None
    else:
        surface_phi_bins = surface_points // radius.size
        if surface_phi_bins * radius.size != surface_points:
            raise ValueError("egress basis does not match the radial quadrature")
    if np.any(output_side > 1):
        raise ValueError("egress side must be zero or one")
    if np.any(bary < 0.0) or not np.allclose(
        bary.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("egress barycentric coordinates are invalid")
    if (
        np.any(~np.isfinite(direction))
        or np.any(direction[:, 2] <= 0.0)
        or not np.allclose(
            np.linalg.norm(direction, axis=1), 1.0,
            rtol=0.0, atol=1.0e-12
        )
    ):
        raise ValueError("egress directions are invalid")
    if (
        np.any(~np.isfinite(polarization))
        or not np.allclose(
            polarization[:, 0], 1.0, rtol=0.0, atol=1.0e-12
        )
        or np.any(np.linalg.norm(polarization[:, 1:], axis=1) > 1.0 + 1e-12)
    ):
        raise ValueError("egress Stokes vectors are invalid")

    state_count = nr * np_ * nm * nd
    producer_metadata = dict(metadata or {})
    coefficient_row_count = producer_metadata.pop("state_count", None)
    if coefficient_row_count is not None:
        producer_metadata["coefficient_row_count"] = int(
            coefficient_row_count
        )
    payload: dict[str, Any] = {
        **producer_metadata,
        "schema": (
            "oos.nonlocal.function.v2"
            if ragged
            else "oos.nonlocal.function.v1"
        ),
        "surface_relative": True,
        "execution": "function",
        "state_count": state_count,
        "egress_count": egress_count,
        "angular_count": int(angular.size),
        "contraction_bound": float(np.max(expected)),
    }
    if joint_angular:
        payload["coefficient_layout"] = "joint_surface_angle_v1"
    if ragged:
        payload.update(
            {
                "surface_layout": "ragged_ring_v1",
                "surface_point_count": surface_points,
            }
        )
        payload.pop("surface_phi_bins", None)
    else:
        payload["surface_phi_bins"] = surface_phi_bins
        payload.pop("surface_layout", None)
        payload.pop("surface_point_count", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        root = handle.create_group("nonlocal")
        egress = root.create_group("egress")
        egress.create_dataset("surface_element", data=element)
        egress.create_dataset("barycentric", data=bary)
        egress.create_dataset("side", data=output_side)
        egress.create_dataset("direction_local", data=direction)
        egress.create_dataset("stokes", data=polarization)
        egress.create_dataset("reference_axis_local", data=reference)

        function = handle.create_group("function")
        function.create_dataset(
            "coefficients_real", data=modes.real, compression="gzip",
            shuffle=True
        )
        function.create_dataset(
            "coefficients_imag", data=modes.imag, compression="gzip",
            shuffle=True
        )
        function.create_dataset("expected_return", data=expected)
        function.create_dataset("audit_values", data=audit)
        function.create_dataset("surface_radius_mm", data=radius)
        function.create_dataset("surface_ring_area_mm2", data=ring_area)
        if ragged:
            function.create_dataset(
                "surface_ring_offsets", data=ring_offsets
            )
            function.create_dataset("surface_phi_rad", data=surface_phi)
            function.create_dataset("surface_area_mm2", data=surface_area)
        function.create_dataset("angular_weight", data=angular)

        meta = handle.create_group("metadata")
        meta.create_dataset(
            "loss_names_json",
            data=np.frombuffer(
                json.dumps(["lxe_nonreturn"]).encode(), dtype=np.uint8
            ),
        )
        meta.create_dataset(
            "audit_names_json",
            data=np.frombuffer(json.dumps(audit_names).encode(), dtype=np.uint8),
        )
        meta.create_dataset(
            "phase_grid_json",
            data=np.frombuffer(
                json.dumps(phase_grid, sort_keys=True).encode(), dtype=np.uint8
            ),
        )
        meta.create_dataset(
            "generator_json",
            data=np.frombuffer(
                json.dumps(payload, sort_keys=True).encode(), dtype=np.uint8
            ),
        )


def _csr(payload: np.lib.npyio.NpzFile, prefix: str) -> sparse.csr_matrix:
    shape = tuple(np.asarray(payload[f"{prefix}_shape"], dtype=np.uint64))
    return sparse.csr_matrix(
        (
            payload[f"{prefix}_data"],
            payload[f"{prefix}_indices"],
            payload[f"{prefix}_indptr"],
        ),
        shape=shape,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss-names-json", required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    loss_names = json.loads(arguments.loss_names_json)
    cache_key = _cache_key(arguments.input, loss_names)
    if not arguments.force and _stored_cache_key(arguments.output) == cache_key:
        print(f"cache hit: {arguments.output}")
        print(f"cache key: {cache_key}")
        return
    with np.load(arguments.input, allow_pickle=False) as payload:
        temporary = arguments.output.with_suffix(
            arguments.output.suffix + ".tmp"
        )
        write_block(
            temporary,
            internal_transition=_csr(payload, "internal_transition"),
            emission=_csr(payload, "emission"),
            internal_losses=_csr(payload, "internal_losses"),
            surface_element=payload["egress_surface_element"],
            barycentric=payload["egress_barycentric"],
            side=payload["egress_side"],
            direction_local=payload["egress_direction_local"],
            loss_names=loss_names,
            metadata={
                "input_npz": str(arguments.input.resolve()),
                "input_sha256": _sha256(arguments.input),
                "cache_key_sha256": cache_key,
            },
        )
    temporary.replace(arguments.output)
    print(f"built: {arguments.output}")
    print(f"cache key: {cache_key}")


if __name__ == "__main__":
    main()
