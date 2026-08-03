"""Conservative phase-space grid for rays entering the liquid domain."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class LXePhaseSpaceGrid:
    """Finite grid in entry position and refracted liquid direction."""

    radius_mm: float
    position_radial_bins: int
    position_phi_bins: int
    direction_mu_bins: int
    direction_phi_bins: int
    direction_phi_relative_to_position: bool = False
    deposition: str = "nearest"
    radial_node_spacing: str = "chebyshev"
    direction_mu_minimum: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.radius_mm) or self.radius_mm <= 0.0:
            raise ValueError("phase-space radius must be finite and positive")
        for name in (
            "position_radial_bins",
            "position_phi_bins",
            "direction_mu_bins",
            "direction_phi_bins",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.deposition not in {"nearest", "multilinear"}:
            raise ValueError("phase-space deposition must be nearest or multilinear")
        if self.radial_node_spacing not in {"linear", "chebyshev"}:
            raise ValueError(
                "phase-space radial-node spacing must be linear or chebyshev"
            )
        if (
            not math.isfinite(self.direction_mu_minimum)
            or self.direction_mu_minimum < 0.0
            or self.direction_mu_minimum >= 1.0
        ):
            raise ValueError("direction-mu minimum must lie in [0,1)")
        if self.deposition == "multilinear":
            if self.position_radial_bins < 2:
                raise ValueError(
                    "multilinear deposition needs at least two radial nodes"
                )
            if self.direction_mu_bins < 2:
                raise ValueError(
                    "multilinear deposition needs at least two direction-mu nodes"
                )

    @property
    def size(self) -> int:
        return (
            self.position_radial_bins
            * self.position_phi_bins
            * self.direction_mu_bins
            * self.direction_phi_bins
        )

    @property
    def position_radius_values(self) -> np.ndarray:
        if self.deposition == "nearest":
            return self.radius_mm * np.sqrt(
                (np.arange(self.position_radial_bins, dtype=float) + 0.5)
                / self.position_radial_bins
            )
        coordinate = np.linspace(0.0, 1.0, self.position_radial_bins)
        if self.radial_node_spacing == "chebyshev":
            # Chebyshev--Radau-like placement: retain the symmetry-axis node
            # but exclude the cylindrical boundary itself.  An entry exactly
            # on the wall with an outward transverse direction is a singular
            # immediate-boundary-hit state and must not be interpolated with
            # smooth interior responses.
            coordinate = 0.5 * (
                1.0
                - np.cos(
                    math.pi
                    * np.arange(self.position_radial_bins, dtype=float)
                    / self.position_radial_bins
                )
            )
        values = self.radius_mm * coordinate
        if self.radial_node_spacing == "linear":
            # The linear option is retained for controlled comparisons.  Keep
            # its last nominal endpoint safely inside the diffusion domain.
            values[-1] = self.radius_mm * (1.0 - 1.0e-12)
        return values

    @property
    def position_phi_values(self) -> np.ndarray:
        offset = 0.5 if self.deposition == "nearest" else 0.0
        return (
            2.0
            * math.pi
            * (np.arange(self.position_phi_bins, dtype=float) + offset)
            / self.position_phi_bins
        )

    @property
    def direction_mu_values(self) -> np.ndarray:
        if self.deposition == "nearest":
            return (
                np.arange(self.direction_mu_bins, dtype=float) + 0.5
            ) / self.direction_mu_bins
        # A exactly tangential ray is not a valid LXeEntryRay.  The response
        # has a regular one-sided limit there, represented by a tiny positive
        # endpoint; physical GXe-to-LXe rays are far above this endpoint.
        values = np.linspace(
            self.direction_mu_minimum,
            1.0,
            self.direction_mu_bins,
        )
        if values[0] == 0.0:
            values[0] = 1.0e-12
        return values

    @property
    def direction_phi_values(self) -> np.ndarray:
        offset = 0.5 if self.deposition == "nearest" else 0.0
        return (
            2.0
            * math.pi
            * (np.arange(self.direction_phi_bins, dtype=float) + offset)
            / self.direction_phi_bins
        )

    def flat_index(
        self,
        position_radial: int,
        position_angular: int,
        direction_mu: int,
        direction_angular: int,
    ) -> int:
        return (
            (
                (
                    position_radial * self.position_phi_bins
                    + position_angular
                )
                * self.direction_mu_bins
                + direction_mu
            )
            * self.direction_phi_bins
            + direction_angular
        )

    @staticmethod
    def _bounded_stencil(
        value: float,
        nodes: np.ndarray,
    ) -> tuple[tuple[int, float], ...]:
        if value <= float(nodes[0]):
            return ((0, 1.0),)
        if value >= float(nodes[-1]):
            return ((len(nodes) - 1, 1.0),)
        upper = int(np.searchsorted(nodes, value, side="right"))
        lower = upper - 1
        fraction = (value - float(nodes[lower])) / float(
            nodes[upper] - nodes[lower]
        )
        if fraction <= 1.0e-15:
            return ((lower, 1.0),)
        if fraction >= 1.0 - 1.0e-15:
            return ((upper, 1.0),)
        return ((lower, 1.0 - fraction), (upper, fraction))

    @staticmethod
    def _periodic_stencil(
        angle: float,
        count: int,
    ) -> tuple[tuple[int, float], ...]:
        if count == 1:
            return ((0, 1.0),)
        scaled = (angle % (2.0 * math.pi)) * count / (2.0 * math.pi)
        lower_unwrapped = math.floor(scaled)
        fraction = scaled - lower_unwrapped
        lower = int(lower_unwrapped) % count
        if fraction <= 1.0e-15:
            return ((lower, 1.0),)
        upper = (lower + 1) % count
        return ((lower, 1.0 - fraction), (upper, fraction))

    def stencil(
        self,
        position_xy_mm: np.ndarray,
        liquid_direction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative phase-node indices and interpolation weights."""

        self.validate()
        position = np.asarray(position_xy_mm, dtype=float)
        direction = np.asarray(liquid_direction, dtype=float)
        if position.shape != (2,) or direction.shape != (3,):
            raise ValueError("phase-space position/direction has invalid shape")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(direction)):
            raise ValueError("phase-space position/direction must be finite")
        radius = float(np.linalg.norm(position))
        if radius > self.radius_mm * (1.0 + 1.0e-10):
            raise ValueError("LXe entry position lies outside phase-space disk")
        radius = min(radius, self.radius_mm)
        position_phi = math.atan2(float(position[1]), float(position[0]))
        position_phi %= 2.0 * math.pi
        mu = min(1.0, max(0.0, float(direction[2])))
        direction_phi = math.atan2(float(direction[1]), float(direction[0]))
        if self.direction_phi_relative_to_position:
            direction_phi -= position_phi
        direction_phi %= 2.0 * math.pi

        if self.deposition == "nearest":
            u = min(1.0 - np.finfo(float).eps, (radius / self.radius_mm) ** 2)
            radial = min(
                int(u * self.position_radial_bins),
                self.position_radial_bins - 1,
            )
            position_angular = min(
                int(
                    position_phi
                    / (2.0 * math.pi)
                    * self.position_phi_bins
                ),
                self.position_phi_bins - 1,
            )
            mu_index = min(
                int(
                    min(1.0 - np.finfo(float).eps, mu)
                    * self.direction_mu_bins
                ),
                self.direction_mu_bins - 1,
            )
            direction_angular = min(
                int(
                    direction_phi
                    / (2.0 * math.pi)
                    * self.direction_phi_bins
                ),
                self.direction_phi_bins - 1,
            )
            return (
                np.asarray(
                    [
                        self.flat_index(
                            radial,
                            position_angular,
                            mu_index,
                            direction_angular,
                        )
                    ],
                    dtype=np.int64,
                ),
                np.asarray([1.0]),
            )

        radial_stencil = self._bounded_stencil(
            radius, self.position_radius_values
        )
        # At the origin the position azimuth is not a physical degree of
        # freedom.  Pin it to zero and keep the absolute ray azimuth in the
        # relative-direction coordinate.
        if radius <= 1.0e-12 * self.radius_mm:
            position_stencil = ((0, 1.0),)
            if self.direction_phi_relative_to_position:
                direction_phi = math.atan2(
                    float(direction[1]), float(direction[0])
                )
                direction_phi %= 2.0 * math.pi
        else:
            position_stencil = self._periodic_stencil(
                position_phi, self.position_phi_bins
            )
        mu_stencil = self._bounded_stencil(mu, self.direction_mu_values)
        direction_stencil = self._periodic_stencil(
            direction_phi, self.direction_phi_bins
        )

        accumulated: dict[int, float] = {}
        for radial, position_angular, mu_index, direction_angular in (
            itertools.product(
                radial_stencil,
                position_stencil,
                mu_stencil,
                direction_stencil,
            )
        ):
            indices = (
                radial[0],
                position_angular[0],
                mu_index[0],
                direction_angular[0],
            )
            weight = (
                radial[1]
                * position_angular[1]
                * mu_index[1]
                * direction_angular[1]
            )
            if weight <= 0.0:
                continue
            column = self.flat_index(*indices)
            accumulated[column] = accumulated.get(column, 0.0) + weight
        columns = np.asarray(sorted(accumulated), dtype=np.int64)
        weights = np.asarray([accumulated[index] for index in columns])
        weights /= float(weights.sum())
        return columns, weights

    def encode(
        self, position_xy_mm: np.ndarray, liquid_direction: np.ndarray
    ) -> int:
        """Return the nearest representative index.

        ``stencil`` must be used when conservative multilinear deposition is
        desired.  This method remains for diagnostics and existing callers.
        """

        indices, weights = self.stencil(position_xy_mm, liquid_direction)
        return int(indices[int(np.argmax(weights))])

    def decode(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(indices, dtype=np.int64)
        direction_angular = values % self.direction_phi_bins
        values = values // self.direction_phi_bins
        direction_mu = values % self.direction_mu_bins
        values = values // self.direction_mu_bins
        position_angular = values % self.position_phi_bins
        position_radial = values // self.position_phi_bins

        radius = self.position_radius_values[position_radial]
        position_phi = self.position_phi_values[position_angular]
        position = np.column_stack(
            [radius * np.cos(position_phi), radius * np.sin(position_phi)]
        )
        mu = self.direction_mu_values[direction_mu]
        direction_phi = self.direction_phi_values[direction_angular]
        if self.direction_phi_relative_to_position:
            direction_phi += position_phi
        transverse = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
        direction = np.column_stack(
            [
                transverse * np.cos(direction_phi),
                transverse * np.sin(direction_phi),
                mu,
            ]
        )
        return position, direction

    def bin_entries(self, entries: list[LXeBoundaryEntry]) -> np.ndarray:
        result = np.zeros(self.size, dtype=float)
        for entry in entries:
            indices, weights = self.stencil(
                entry.position_xy_mm, entry.liquid_direction
            )
            result[indices] += float(entry.weight) * weights
        return result

    def entries_from_weights(
        self, weights: np.ndarray, cutoff: float = 0.0
    ) -> list[LXeBoundaryEntry]:
        values = np.asarray(weights, dtype=float)
        if values.shape != (self.size,):
            raise ValueError("phase-space weight vector has wrong shape")
        indices = np.flatnonzero(values > cutoff)
        position, direction = self.decode(indices)
        return [
            LXeBoundaryEntry(position[index], direction[index], float(values[column]))
            for index, column in enumerate(indices)
        ]
