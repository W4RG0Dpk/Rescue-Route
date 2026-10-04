"""
RescueRoute Dynamic Flood Model
================================

Controlled stochastic flood evolution for the RescueRoute simulator.

Design goals
------------
The flood model is NOT intended to be a scientifically exact
hydrological simulator.

It is a reproducible stochastic environment component whose purpose
is to create changing disaster conditions for UAV navigation.

Flood evolution depends on:

    1. Current flood state
    2. Number of neighboring flooded cells
    3. Rain / flood intensity
    4. Wind speed
    5. Wind direction
    6. A small spontaneous-flood probability

Important:
    The simulator is seeded so that the same scenario can be
    reproduced exactly.

Grid convention
---------------
    x -> East
    y -> South

Flood grid:
    0 -> dry / not flooded
    1 -> flooded
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..weather.wind import WindState


class DynamicFloodModel:
    """
    Stochastic flood expansion model.

    Flooded cells remain flooded during an episode.

    Expansion is influenced by:
        - local flood pressure
        - rain intensity
        - wind direction
        - wind strength
        - small spontaneous formation probability

    All values are simulator parameters rather than claims about
    real-world hydrology.
    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        initial_flood_ratio: float = 0.015,
        expansion_rate: float = 0.03,
        rain_intensity: float = 0.50,
        rain_variability: float = 0.15,
        rain_correlation: float = 0.85,
        wind_influence: float = 0.05,
        spontaneous_flood_probability: float = 0.001,
        max_expansion_probability: float = 0.60,
        min_rain_intensity: float = 0.0,
        max_rain_intensity: float = 1.0,
        seed: Optional[int] = None,
    ):
        # -------------------------------------------------------------
        # Validate dimensions
        # -------------------------------------------------------------

        if grid_width <= 0:
            raise ValueError(
                "grid_width must be positive."
            )

        if grid_height <= 0:
            raise ValueError(
                "grid_height must be positive."
            )

        # -------------------------------------------------------------
        # Validate parameters
        # -------------------------------------------------------------

        if not 0.0 <= initial_flood_ratio <= 1.0:
            raise ValueError(
                "initial_flood_ratio must be between 0 and 1."
            )

        if expansion_rate < 0.0:
            raise ValueError(
                "expansion_rate must be >= 0."
            )

        if not 0.0 <= rain_intensity <= 1.0:
            raise ValueError(
                "rain_intensity must be between 0 and 1."
            )

        if rain_variability < 0.0:
            raise ValueError(
                "rain_variability must be >= 0."
            )

        if not 0.0 <= rain_correlation < 1.0:
            raise ValueError(
                "rain_correlation must satisfy 0 <= rho < 1."
            )

        if wind_influence < 0.0:
            raise ValueError(
                "wind_influence must be >= 0."
            )

        if not 0.0 <= spontaneous_flood_probability <= 1.0:
            raise ValueError(
                "spontaneous_flood_probability must be between "
                "0 and 1."
            )

        if not 0.0 < max_expansion_probability <= 1.0:
            raise ValueError(
                "max_expansion_probability must be in (0, 1]."
            )

        if min_rain_intensity < 0.0:
            raise ValueError(
                "min_rain_intensity must be >= 0."
            )

        if max_rain_intensity <= min_rain_intensity:
            raise ValueError(
                "max_rain_intensity must be greater than "
                "min_rain_intensity."
            )

        # -------------------------------------------------------------
        # Static parameters
        # -------------------------------------------------------------

        self.width = int(grid_width)
        self.height = int(grid_height)

        self.initial_ratio = float(
            initial_flood_ratio
        )

        self.expansion_rate = float(
            expansion_rate
        )

        self.rain_intensity_mean = float(
            rain_intensity
        )

        self.rain_variability = float(
            rain_variability
        )

        self.rain_rho = float(
            rain_correlation
        )

        self.wind_influence = float(
            wind_influence
        )

        self.spontaneous_probability = float(
            spontaneous_flood_probability
        )

        self.max_expansion_probability = float(
            max_expansion_probability
        )

        self.min_rain_intensity = float(
            min_rain_intensity
        )

        self.max_rain_intensity = float(
            max_rain_intensity
        )

        # -------------------------------------------------------------
        # Random generator
        # -------------------------------------------------------------

        self.rng = np.random.default_rng(
            seed
        )

        # -------------------------------------------------------------
        # Dynamic state
        # -------------------------------------------------------------

        self.flood_grid = np.zeros(
            (
                self.height,
                self.width,
            ),
            dtype=np.int32,
        )

        self.rain_intensity = (
            self.rain_intensity_mean
        )

        self.step_count = 0

        # Coordinates protected from initial flood.
        self.protected_coords: set[
            Tuple[int, int]
        ] = set()

    # =================================================================
    # RESET
    # =================================================================

    def reset(
        self,
        building_grid: np.ndarray,
        safe_coords: Optional[
            List[Tuple[int, int]]
        ] = None,
        seed: Optional[int] = None,
        *,
        rain_intensity: Optional[float] = None,
        rain_variability: Optional[float] = None,
    ) -> np.ndarray:
        """
        Reset the flood episode.

        Parameters
        ----------
        building_grid:
            2D map where:
                0 = traversable
                1 = statically blocked

        safe_coords:
            Coordinates that must not be initial flood seeds.

        seed:
            Random seed.

        rain_intensity:
            Optional scenario-specific mean rain/flood intensity.

        rain_variability:
            Optional scenario-specific rain variability.
        """

        # -------------------------------------------------------------
        # Validate map
        # -------------------------------------------------------------

        building_grid = np.asarray(
            building_grid
        )

        expected_shape = (
            self.height,
            self.width,
        )

        if building_grid.shape != expected_shape:
            raise ValueError(
                "building_grid shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {building_grid.shape}."
            )

        # -------------------------------------------------------------
        # Reset RNG
        # -------------------------------------------------------------

        if seed is not None:
            self.rng = np.random.default_rng(
                seed
            )

        # -------------------------------------------------------------
        # Optional scenario-specific overrides
        # -------------------------------------------------------------

        if rain_intensity is not None:

            if not 0.0 <= rain_intensity <= 1.0:
                raise ValueError(
                    "rain_intensity must be between 0 and 1."
                )

            self.rain_intensity_mean = float(
                rain_intensity
            )

        if rain_variability is not None:

            if rain_variability < 0.0:
                raise ValueError(
                    "rain_variability must be >= 0."
                )

            self.rain_variability = float(
                rain_variability
            )

        # -------------------------------------------------------------
        # Reset state
        # -------------------------------------------------------------

        self.flood_grid = np.zeros(
            (
                self.height,
                self.width,
            ),
            dtype=np.int32,
        )

        self.rain_intensity = float(
            np.clip(
                self.rain_intensity_mean
                + self.rng.normal(
                    0.0,
                    self.rain_variability,
                ),
                self.min_rain_intensity,
                self.max_rain_intensity,
            )
        )

        self.step_count = 0

        self.protected_coords = set(
            safe_coords
            if safe_coords is not None
            else []
        )

        # -------------------------------------------------------------
        # Find valid initial flood cells
        # -------------------------------------------------------------

        free_coords = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (
                building_grid[y, x] == 0
                and (x, y)
                not in self.protected_coords
            )
        ]

        if not free_coords:
            return self.flood_grid.copy()

        # -------------------------------------------------------------
        # Number of initial flood seed sources
        # -------------------------------------------------------------

        num_seeds = max(
            1,
            int(
                round(
                    self.width
                    * self.height
                    * self.initial_ratio
                )
            ),
        )

        num_seeds = min(
            num_seeds,
            len(free_coords),
        )

        # -------------------------------------------------------------
        # Select initial seed locations
        #
        # We use independent sources rather than making the entire
        # flood start as one giant block.
        # -------------------------------------------------------------

        selected_indices = self.rng.choice(
            len(free_coords),
            size=num_seeds,
            replace=False,
        )

        for idx in selected_indices:

            x, y = free_coords[int(idx)]

            self.flood_grid[
                y,
                x
            ] = 1

        return self.flood_grid.copy()

    # =================================================================
    # RAIN UPDATE
    # =================================================================

    def _update_rain_intensity(self) -> float:
        """
        Evolve rain/flood intensity using a correlated stochastic
        process.

            R[t+1] =
                mean
                + rho * (R[t] - mean)
                + noise

        Values are clipped to [0, 1].
        """

        innovation_std = (
            self.rain_variability
            * np.sqrt(
                max(
                    0.0,
                    1.0 - self.rain_rho ** 2,
                )
            )
        )

        noise = self.rng.normal(
            0.0,
            innovation_std,
        )

        self.rain_intensity = float(
            np.clip(
                self.rain_intensity_mean
                + self.rain_rho
                * (
                    self.rain_intensity
                    - self.rain_intensity_mean
                )
                + noise,
                self.min_rain_intensity,
                self.max_rain_intensity,
            )
        )

        return self.rain_intensity

    # =================================================================
    # WIND HELPERS
    # =================================================================

    @staticmethod
    def _wind_unit_vector(
        wind_state: Optional[WindState],
    ) -> Tuple[float, float]:
        """
        Return normalized wind vector.

        Wind vector convention inherited from WindModel:

            vector points in the direction the air moves.

        Returns:
            (0, 0) if there is effectively no wind.
        """

        if wind_state is None:
            return 0.0, 0.0

        vx, vy = wind_state.wind_vector

        magnitude = float(
            np.sqrt(
                vx * vx
                + vy * vy
            )
        )

        if magnitude < 1e-8:
            return 0.0, 0.0

        return (
            float(vx / magnitude),
            float(vy / magnitude),
        )

    # =================================================================
    # SINGLE-CELL EXPANSION PROBABILITY
    # =================================================================

    def _expansion_probability(
        self,
        flooded_neighbors: int,
        wind_alignment: float,
        wind_strength: float,
    ) -> float:
        """
        Calculate flood probability for one candidate cell.

        Parameters
        ----------
        flooded_neighbors:
            Number of adjacent flooded cells, 1 to 4.

        wind_alignment:
            [-1, 1]

            +1 = wind strongly pushes water toward the candidate
            -1 = wind pushes away from the candidate

        wind_strength:
            Normalized wind strength [0, 1].

        Probability components
        ----------------------
        Local flood pressure
        +
        rain amplification
        +
        wind directional effect
        """

        if flooded_neighbors <= 0:
            return 0.0

        # -------------------------------------------------------------
        # Local flood pressure.
        #
        # More flooded neighbors -> greater chance of expansion.
        # -------------------------------------------------------------

        local_pressure = (
            self.expansion_rate
            * float(flooded_neighbors)
        )

        # -------------------------------------------------------------
        # Rain multiplier.
        #
        # Minimum multiplier is 0.5 so that low rain does not turn
        # the flood completely static.
        # -------------------------------------------------------------

        rain_multiplier = (
            0.50
            + 0.75
            * self.rain_intensity
        )

        rain_component = (
            local_pressure
            * rain_multiplier
        )

        # -------------------------------------------------------------
        # Wind component.
        #
        # Only wind pushing TOWARD the candidate contributes
        # positively.
        # -------------------------------------------------------------

        positive_alignment = max(
            0.0,
            wind_alignment,
        )

        wind_component = (
            self.wind_influence
            * positive_alignment
            * wind_strength
        )

        probability = (
            rain_component
            + wind_component
        )

        return float(
            np.clip(
                probability,
                0.0,
                self.max_expansion_probability,
            )
        )

    # =================================================================
    # STEP
    # =================================================================

    def step(
        self,
        building_grid: np.ndarray,
        wind_state: Optional[WindState] = None,
    ) -> np.ndarray:
        """
        Advance flood dynamics by one simulation timestep.

        Sequence:

            current flood
                 +
            current wind
                 +
            updated rain intensity
                 ↓
            stochastic expansion
                 ↓
            next flood grid

        Existing flood cells remain flooded.
        """

        building_grid = np.asarray(
            building_grid
        )

        expected_shape = (
            self.height,
            self.width,
        )

        if building_grid.shape != expected_shape:
            raise ValueError(
                "building_grid shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {building_grid.shape}."
            )

        self.step_count += 1

        # -------------------------------------------------------------
        # Update rain before expansion
        # -------------------------------------------------------------

        self._update_rain_intensity()

        # -------------------------------------------------------------
        # Start with current flood state.
        #
        # Flooded cells persist.
        # -------------------------------------------------------------

        new_flood = self.flood_grid.copy()

        # -------------------------------------------------------------
        # Wind information
        # -------------------------------------------------------------

        wind_x, wind_y = self._wind_unit_vector(
            wind_state
        )

        if wind_state is not None:

            total_wind_speed = max(
                0.0,
                float(
                    wind_state.total_speed
                ),
            )

        else:

            total_wind_speed = 0.0

        # Normalize wind strength.
        wind_strength = float(
            np.clip(
                total_wind_speed / 15.0,
                0.0,
                1.0,
            )
        )

        # -------------------------------------------------------------
        # 4-neighbor expansion
        #
        # N, S, E, W
        # -------------------------------------------------------------

        neighbors = [
            (0, -1),
            (0, 1),
            (1, 0),
            (-1, 0),
        ]

        for y in range(self.height):

            for x in range(self.width):

                # Already flooded.
                if self.flood_grid[y, x] == 1:
                    continue

                # Static blocked/NFZ cell.
                #
                # Flood model operates on navigable cells only.
                if building_grid[y, x] != 0:
                    continue

                # -----------------------------------------------------
                # Count flooded neighbors
                # -----------------------------------------------------

                flooded_neighbors = 0

                alignment_sum = 0.0

                for dx, dy in neighbors:

                    nx = x + dx
                    ny = y + dy

                    if not (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                    ):
                        continue

                    if self.flood_grid[
                        ny,
                        nx
                    ] != 1:
                        continue

                    flooded_neighbors += 1

                    # -------------------------------------------------
                    # Direction from flooded neighbor -> candidate.
                    #
                    # Candidate:
                    #       (x, y)
                    #
                    # Flooded neighbor:
                    #       (x+dx, y+dy)
                    #
                    # Direction water must move:
                    #       (-dx, -dy)
                    #
                    # Positive dot with wind means wind pushes water
                    # from the flooded cell toward this candidate.
                    # -------------------------------------------------

                    toward_candidate_x = -dx
                    toward_candidate_y = -dy

                    norm = np.sqrt(
                        toward_candidate_x ** 2
                        + toward_candidate_y ** 2
                    )

                    direction_x = (
                        toward_candidate_x
                        / norm
                    )

                    direction_y = (
                        toward_candidate_y
                        / norm
                    )

                    alignment = (
                        direction_x * wind_x
                        + direction_y * wind_y
                    )

                    alignment_sum += alignment

                # -----------------------------------------------------
                # Local flood pressure exists only when a flooded
                # neighbor is present.
                # -----------------------------------------------------

                if flooded_neighbors > 0:

                    average_alignment = (
                        alignment_sum
                        / flooded_neighbors
                    )

                    probability = (
                        self._expansion_probability(
                            flooded_neighbors,
                            average_alignment,
                            wind_strength,
                        )
                    )

                    if (
                        self.rng.random()
                        < probability
                    ):

                        new_flood[
                            y,
                            x
                        ] = 1

                else:

                    # -------------------------------------------------
                    # Very small chance for spontaneous flood creation.
                    #
                    # This prevents the flood model from requiring
                    # every future flooded region to remain adjacent
                    # to the original flood forever.
                    # -------------------------------------------------

                    if (
                        self.rng.random()
                        < (
                            self.spontaneous_probability
                            * self.rain_intensity
                        )
                    ):

                        new_flood[
                            y,
                            x
                        ] = 1

        # -------------------------------------------------------------
        # Never flood protected coordinates.
        #
        # Safe start/target positions are protected from the initial
        # seeds, but once the episode evolves they should still be
        # allowed to become unsafe if the environment genuinely
        # reaches them.
        #
        # Therefore we do NOT permanently protect them here.
        # -------------------------------------------------------------

        self.flood_grid = new_flood

        return self.flood_grid.copy()

    # =================================================================
    # ACCESSORS
    # =================================================================

    def is_flooded(
        self,
        x: int,
        y: int,
    ) -> bool:
        """
        Check whether a cell is currently flooded.
        """

        if not (
            0 <= x < self.width
            and 0 <= y < self.height
        ):
            return False

        return bool(
            self.flood_grid[y, x] == 1
        )

    def get_rain_intensity(self) -> float:
        """
        Return current normalized rain/flood intensity [0, 1].
        """

        return float(
            self.rain_intensity
        )

    def get_flood_ratio(self) -> float:
        """
        Return fraction of all map cells currently flooded.
        """

        return float(
            np.mean(
                self.flood_grid
            )
        )