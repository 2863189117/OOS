#!/usr/bin/env python3
"""Deterministic LXe diffusion/first-return calculation in a finite cylinder.

The calculation is deliberately separate from the GXe radiosity solver.  It
quantifies the part that the first boundary-optics implementation classified
as permanent ``lxe_transmission`` loss:

* exact Fresnel/Snell transmission into LXe;
* an exponentially distributed first Rayleigh-scattering flight;
* diffusion with bulk absorption after that first scatter;
* Fresnel-limited escape through the liquid surface;
* partially absorbing Lambertian PTFE side and bottom boundaries.

The diffusion equation is discretised with an axisymmetric finite-volume
scheme.  Consequently there is no Monte Carlo or Poisson noise in the result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .numerics import fresnel_power, gauss_interval


@dataclass(frozen=True)
class LXeDiffusionConfig:
    """Physical and numerical settings, with all lengths in millimetres."""

    # Radius and depth of the synthetic cylindrical liquid domain.
    radius_mm: float = 1000.0
    depth_mm: float = 2000.0
    # Linear interpolation of the Geant4 LXe RAYLEIGH table at 7.0 eV.
    rayleigh_length_mm: float = 341.51442280354416
    absorption_length_mm: float = 70_000.0
    n_lxe: float = 1.6829
    n_gxe: float = 1.000702
    side_reflectivity: float = 0.95
    bottom_reflectivity: float = 0.0
    # The production default derives the top Robin length from the P1 Fresnel
    # moments. A named override is reserved for controlled tail-only A/B
    # studies, such as a separately converged Fresnel--Milne solve.
    top_boundary_model: str = "p1_moments"
    top_boundary_length_mm: float | None = None
    radial_cells: int = 120
    depth_cells: int = 200
    incident_mu_order: int = 64
    first_flight_order: int = 64

    def validate(self) -> None:
        for name in (
            "radius_mm",
            "depth_mm",
            "rayleigh_length_mm",
            "absorption_length_mm",
            "n_lxe",
            "n_gxe",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("side_reflectivity", "bottom_reflectivity"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.top_boundary_model == "p1_moments":
            if self.top_boundary_length_mm is not None:
                raise ValueError(
                    "top_boundary_length_mm requires a non-default "
                    "top_boundary_model"
                )
        elif (
            not self.top_boundary_model
            or self.top_boundary_length_mm is None
            or not math.isfinite(self.top_boundary_length_mm)
            or self.top_boundary_length_mm <= 0.0
        ):
            raise ValueError(
                "a non-default top boundary model requires a finite positive "
                "top_boundary_length_mm"
            )
        if self.radial_cells < 2 or self.depth_cells < 2:
            raise ValueError("the diffusion grid must have at least two cells")
        if self.incident_mu_order < 2 or self.first_flight_order < 2:
            raise ValueError("quadrature orders must be at least two")

    @property
    def scattering_coefficient_per_mm(self) -> float:
        return 1.0 / self.rayleigh_length_mm

    @property
    def field_cage_radius_mm(self) -> float:
        """Physical name for the cylindrical LXe diffusion boundary."""

        return self.radius_mm

    @property
    def absorption_coefficient_per_mm(self) -> float:
        return 1.0 / self.absorption_length_mm

    @property
    def extinction_coefficient_per_mm(self) -> float:
        return (
            self.scattering_coefficient_per_mm
            + self.absorption_coefficient_per_mm
        )

    @property
    def transport_diffusion_mm(self) -> float:
        # Rayleigh scattering has <cos(theta)>=0, hence mu_s' = mu_s.
        return 1.0 / (3.0 * self.extinction_coefficient_per_mm)

    @property
    def diffusion_absorption_length_mm(self) -> float:
        return math.sqrt(
            self.transport_diffusion_mm
            / self.absorption_coefficient_per_mm
        )


def diffuse_fresnel_escape_fraction(
    n_inside: float, n_outside: float, order: int = 256
) -> float:
    """Cosine-current averaged power transmission out of a dielectric."""

    first_moment, _ = diffuse_fresnel_reflection_moments(
        n_inside, n_outside, order
    )
    return 1.0 - first_moment


def diffuse_fresnel_reflection_moments(
    n_inside: float, n_outside: float, order: int = 256
) -> tuple[float, float]:
    """Return the P1 Fresnel moments ``C1`` and ``C2``.

    For unpolarized radiance incident on the boundary from inside the
    dielectric,

    ``C1 = 2 integral R(mu) mu dmu`` and
    ``C2 = 3 integral R(mu) mu^2 dmu``.

    The total-internal-reflection interval is integrated analytically and the
    smooth remainder is evaluated with Gauss--Legendre quadrature.  Keeping
    both moments is required by the P1/Marshak boundary condition; replacing
    ``C2`` by ``C1`` is only valid for angle-independent reflectivity.
    """

    if n_inside <= 0.0 or n_outside <= 0.0:
        raise ValueError("refractive indices must be positive")
    if order < 2:
        raise ValueError("quadrature order must be at least two")

    critical_mu = 0.0
    if n_inside > n_outside:
        ratio = n_outside / n_inside
        critical_mu = math.sqrt(max(0.0, 1.0 - ratio * ratio))

    # R(mu)=1 throughout the total-internal-reflection interval.
    first_moment = critical_mu**2
    second_moment = critical_mu**3
    if critical_mu < 1.0:
        mu, weights = gauss_interval(order, critical_mu, 1.0)
        for cosine, weight in zip(mu, weights):
            reflection, _, _ = fresnel_power(
                n_inside, n_outside, float(cosine)
            )
            unpolarized = 0.5 * float(np.sum(reflection))
            first_moment += (
                2.0 * float(cosine) * float(weight) * unpolarized
            )
            second_moment += (
                3.0
                * float(cosine) ** 2
                * float(weight)
                * unpolarized
            )
    return first_moment, second_moment


def fresnel_extrapolation_length_mm(
    diffusion_mm: float,
    n_inside: float,
    n_outside: float,
    order: int = 256,
) -> float:
    """P1/Marshak extrapolation length for a Fresnel dielectric boundary."""

    if diffusion_mm <= 0.0:
        raise ValueError("diffusion coefficient must be positive")
    first_moment, second_moment = diffuse_fresnel_reflection_moments(
        n_inside, n_outside, order
    )
    escape_current = 1.0 - first_moment
    if escape_current <= 0.0:
        return math.inf
    boundary_factor = (1.0 + second_moment) / escape_current
    return 2.0 * diffusion_mm * boundary_factor


def configured_top_extrapolation_length_mm(config: LXeDiffusionConfig) -> float:
    """Return the explicitly selected top-boundary extrapolation length."""

    config.validate()
    if config.top_boundary_length_mm is not None:
        return config.top_boundary_length_mm
    return fresnel_extrapolation_length_mm(
        config.transport_diffusion_mm, config.n_lxe, config.n_gxe
    )


def extrapolation_length_mm(diffusion_mm: float, reflectivity: float) -> float:
    """P1/Marshak extrapolation length for a diffuse reflecting boundary."""

    if reflectivity >= 1.0:
        return math.inf
    return 2.0 * diffusion_mm * (1.0 + reflectivity) / (1.0 - reflectivity)


def first_scatter_source(
    config: LXeDiffusionConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Axisymmetric first-scatter source per photon transmitted into LXe."""

    config.validate()
    nr = config.radial_cells
    nz = config.depth_cells
    dr = config.radius_mm / nr
    dz = config.depth_mm / nz
    source = np.zeros((nz, nr), dtype=float)

    incident_mu, incident_weights = gauss_interval(
        config.incident_mu_order, 0.0, 1.0
    )
    # A photon emitted isotropically immediately above a planar boundary has
    # uniform mu over the downward hemisphere.  Condition on transmission.
    transmitted_weights = np.zeros_like(incident_weights)
    liquid_mu = np.zeros_like(incident_mu)
    for index, cosine in enumerate(incident_mu):
        _, transmission, cos_t = fresnel_power(
            config.n_gxe, config.n_lxe, float(cosine)
        )
        transmitted_weights[index] = (
            incident_weights[index] * 0.5 * float(np.sum(transmission))
        )
        if cos_t is None:
            raise RuntimeError("GXe to LXe refraction unexpectedly has TIR")
        liquid_mu[index] = cos_t
    transmitted_weights /= float(transmitted_weights.sum())

    # Gauss-Laguerre integrates exp(-x) f(x), with physical path
    # s=x/mu_t.  At the first interaction, scattering wins over absorption
    # with probability mu_s/mu_t.
    laguerre_x, laguerre_weights = np.polynomial.laguerre.laggauss(
        config.first_flight_order
    )
    scatter_fraction = (
        config.scattering_coefficient_per_mm
        / config.extinction_coefficient_per_mm
    )
    ballistic_absorption = 1.0 - scatter_fraction
    escaped_geometry = 0.0

    for mu_lxe, angle_weight in zip(liquid_mu, transmitted_weights):
        transverse = math.sqrt(max(0.0, 1.0 - float(mu_lxe) ** 2))
        for path_x, path_weight in zip(laguerre_x, laguerre_weights):
            path_mm = float(path_x) / config.extinction_coefficient_per_mm
            radius = path_mm * transverse
            depth = path_mm * float(mu_lxe)
            weight = (
                float(angle_weight)
                * float(path_weight)
                * scatter_fraction
            )
            if radius >= config.radius_mm or depth >= config.depth_mm:
                # This is negligible for a source near the symmetry axis
                # dimensions, but it is explicitly audited.
                escaped_geometry += weight
                continue
            ir = min(int(radius / dr), nr - 1)
            iz = min(int(depth / dz), nz - 1)
            source[iz, ir] += weight

    audit = {
        "source_scatter_weight": float(source.sum()),
        "ballistic_absorption": ballistic_absorption,
        "ballistic_boundary_reach": escaped_geometry,
        "accounted": float(
            source.sum() + ballistic_absorption + escaped_geometry
        ),
    }
    return source, audit


def _cell_geometry(
    config: LXeDiffusionConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    dr = config.radius_mm / config.radial_cells
    dz = config.depth_mm / config.depth_cells
    radial_edges = np.arange(config.radial_cells + 1, dtype=float) * dr
    annulus_area = math.pi * (
        radial_edges[1:] ** 2 - radial_edges[:-1] ** 2
    )
    volumes = dz * annulus_area
    return radial_edges, annulus_area, dr, dz


def build_diffusion_operator(
    config: LXeDiffusionConfig,
) -> tuple[sparse.csr_matrix, dict[str, np.ndarray]]:
    """Build the conservative finite-volume diffusion operator."""

    config.validate()
    nr = config.radial_cells
    nz = config.depth_cells
    count = nr * nz
    radial_edges, annulus_area, dr, dz = _cell_geometry(config)
    diffusion = config.transport_diffusion_mm
    absorption = config.absorption_coefficient_per_mm

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diagonal = np.zeros(count, dtype=float)

    def linear(iz: int, ir: int) -> int:
        return iz * nr + ir

    # Radial internal faces.
    for ir in range(nr - 1):
        face_area = 2.0 * math.pi * radial_edges[ir + 1] * dz
        conductance = diffusion * face_area / dr
        for iz in range(nz):
            left = linear(iz, ir)
            right = linear(iz, ir + 1)
            diagonal[left] += conductance
            diagonal[right] += conductance
            rows.extend((left, right))
            cols.extend((right, left))
            data.extend((-conductance, -conductance))

    # Depth internal faces.
    for iz in range(nz - 1):
        for ir in range(nr):
            conductance = diffusion * annulus_area[ir] / dz
            upper = linear(iz, ir)
            lower = linear(iz + 1, ir)
            diagonal[upper] += conductance
            diagonal[lower] += conductance
            rows.extend((upper, lower))
            cols.extend((lower, upper))
            data.extend((-conductance, -conductance))

    volumes = np.tile(annulus_area * dz, nz)
    volume_sink = absorption * volumes
    diagonal += volume_sink

    first_moment, second_moment = diffuse_fresnel_reflection_moments(
        config.n_lxe, config.n_gxe
    )
    escape_fraction = 1.0 - first_moment
    top_reflectivity = first_moment
    top_extrapolation = configured_top_extrapolation_length_mm(config)
    top_sink = np.zeros(count, dtype=float)
    for ir in range(nr):
        index = linear(0, ir)
        conductance = (
            diffusion
            * annulus_area[ir]
            / (0.5 * dz + top_extrapolation)
        )
        diagonal[index] += conductance
        top_sink[index] = conductance

    side_extrapolation = extrapolation_length_mm(
        diffusion, config.side_reflectivity
    )
    side_sink = np.zeros(count, dtype=float)
    if math.isfinite(side_extrapolation):
        face_area = 2.0 * math.pi * config.radius_mm * dz
        conductance = diffusion * face_area / (
            0.5 * dr + side_extrapolation
        )
        for iz in range(nz):
            index = linear(iz, nr - 1)
            diagonal[index] += conductance
            side_sink[index] = conductance

    bottom_extrapolation = extrapolation_length_mm(
        diffusion, config.bottom_reflectivity
    )
    bottom_sink = np.zeros(count, dtype=float)
    if math.isfinite(bottom_extrapolation):
        for ir in range(nr):
            index = linear(nz - 1, ir)
            conductance = (
                diffusion
                * annulus_area[ir]
                / (0.5 * dz + bottom_extrapolation)
            )
            diagonal[index] += conductance
            bottom_sink[index] = conductance

    rows.extend(range(count))
    cols.extend(range(count))
    data.extend(diagonal.tolist())
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(count, count)
    ).tocsr()
    sinks = {
        "top_escape": top_sink,
        "side_absorption": side_sink,
        "bottom_absorption": bottom_sink,
        "volume_absorption": volume_sink,
        "top_effective_reflectivity": np.asarray([top_reflectivity]),
        "top_diffuse_escape_fraction": np.asarray([escape_fraction]),
        "top_fresnel_first_moment": np.asarray([first_moment]),
        "top_fresnel_second_moment": np.asarray([second_moment]),
        "top_extrapolation_length_mm": np.asarray([top_extrapolation]),
    }
    return matrix, sinks


def solve_lxe_return(config: LXeDiffusionConfig) -> dict:
    """Solve for the return probability and surface radial distribution."""

    source_grid, first_flight = first_scatter_source(config)
    operator, sinks = build_diffusion_operator(config)
    source = source_grid.ravel()
    fluence = sparse_linalg.spsolve(operator, source)
    if np.any(~np.isfinite(fluence)) or np.any(fluence < -1.0e-12):
        raise RuntimeError("diffusion solve produced an invalid fluence")
    fluence = np.maximum(fluence, 0.0)

    top_by_cell = sinks["top_escape"] * fluence
    side_absorption = float(np.dot(sinks["side_absorption"], fluence))
    bottom_absorption = float(np.dot(sinks["bottom_absorption"], fluence))
    volume_absorption = float(np.dot(sinks["volume_absorption"], fluence))
    diffuse_accounted = float(
        top_by_cell.sum()
        + side_absorption
        + bottom_absorption
        + volume_absorption
    )
    radial_edges, _, _, _ = _cell_geometry(config)
    top_radial = top_by_cell[: config.radial_cells]
    cumulative = np.cumsum(top_radial)
    total_return = float(top_radial.sum())

    def quantile_radius(probability: float) -> float:
        if total_return <= 0:
            return math.nan
        target = probability * total_return
        index = int(np.searchsorted(cumulative, target, side="left"))
        index = min(index, config.radial_cells - 1)
        return float(radial_edges[index + 1])

    total_accounted = (
        total_return
        + side_absorption
        + bottom_absorption
        + volume_absorption
        + first_flight["ballistic_absorption"]
        + first_flight["ballistic_boundary_reach"]
    )
    return {
        "config": asdict(config),
        "derived": {
            "diffusion_coefficient_mm": config.transport_diffusion_mm,
            "diffusion_absorption_length_mm": (
                config.diffusion_absorption_length_mm
            ),
            "top_diffuse_escape_fraction_per_incidence": float(
                sinks["top_diffuse_escape_fraction"][0]
            ),
            "top_effective_reflectivity": float(
                sinks["top_effective_reflectivity"][0]
            ),
            "top_fresnel_first_moment": float(
                sinks["top_fresnel_first_moment"][0]
            ),
            "top_fresnel_second_moment": float(
                sinks["top_fresnel_second_moment"][0]
            ),
            "top_extrapolation_length_mm": float(
                sinks["top_extrapolation_length_mm"][0]
            ),
        },
        "first_flight": first_flight,
        "probabilities_per_lxe_entry": {
            "return_to_gxe": total_return,
            "lxe_volume_absorption": (
                volume_absorption + first_flight["ballistic_absorption"]
            ),
            "lxe_ptfe_side_absorption": side_absorption,
            "bottom_absorption": bottom_absorption,
            "ballistic_boundary_reach": first_flight[
                "ballistic_boundary_reach"
            ],
            "accounted": total_accounted,
            "diffuse_accounted": diffuse_accounted,
        },
        "return_radius_mm": {
            "median": quantile_radius(0.5),
            "q90": quantile_radius(0.9),
            "q95": quantile_radius(0.95),
            "q99": quantile_radius(0.99),
        },
        "radial_edges_mm": radial_edges.tolist(),
        "return_probability_by_annulus": top_radial.tolist(),
    }


def write_result(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "lxe_diffusion_return.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    radial_edges = np.asarray(result["radial_edges_mm"])
    probability = np.asarray(result["return_probability_by_annulus"])
    np.savez_compressed(
        output / "lxe_diffusion_return.npz",
        radial_edges_mm=radial_edges,
        return_probability_by_annulus=probability,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--radius-mm", type=float, default=1000.0)
    result.add_argument("--depth-mm", type=float, default=2000.0)
    result.add_argument(
        "--rayleigh-mm", type=float, default=341.51442280354416
    )
    result.add_argument("--absorption-mm", type=float, default=70_000.0)
    result.add_argument("--side-reflectivity", type=float, default=0.95)
    result.add_argument("--bottom-reflectivity", type=float, default=0.0)
    result.add_argument("--radial-cells", type=int, default=120)
    result.add_argument("--depth-cells", type=int, default=200)
    return result


def main() -> None:
    arguments = parser().parse_args()
    config = LXeDiffusionConfig(
        radius_mm=arguments.radius_mm,
        depth_mm=arguments.depth_mm,
        rayleigh_length_mm=arguments.rayleigh_mm,
        absorption_length_mm=arguments.absorption_mm,
        side_reflectivity=arguments.side_reflectivity,
        bottom_reflectivity=arguments.bottom_reflectivity,
        radial_cells=arguments.radial_cells,
        depth_cells=arguments.depth_cells,
    )
    solved = solve_lxe_return(config)
    write_result(solved, arguments.output)
    print(
        json.dumps(
            {
                "probabilities_per_lxe_entry": solved[
                    "probabilities_per_lxe_entry"
                ],
                "return_radius_mm": solved["return_radius_mm"],
                "derived": solved["derived"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
