"""
RescueRoute Probabilistic Forecast Model
========================================

Forecasts future environmental conditions from the CURRENT state.

CRITICAL RESEARCH PROPERTY
--------------------------
This model does NOT simulate the actual future and then corrupt it.

Instead it directly propagates probabilities using a statistical
transition model.

The actual simulator independently samples its future using the
DynamicFloodModel and WindModel.

Therefore:

    forecast != actual future

and the agent never receives the sampled future trajectory.

Forecast outputs:
    - flood probability map at each future horizon step
    - final flood probability map
    - forecast wind speed
    - forecast wind direction
    - forecast wind speed sequence
    - forecast wind direction sequence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..weather.wind import WindState
from .reliability import ForecastReliabilityEngine


# =====================================================================
# Forecast state
# =====================================================================

@dataclass
class ForecastState:
    """
    Agent-visible probabilistic forecast.
    """

    horizon_steps: int

    # Probability map at final forecast horizon.
    flood_prob_map: np.ndarray

    # Probability map for every future timestep:
    #
    # shape = (H, height, width)
    flood_prob_sequence: np.ndarray

    # Mean forecast wind over the horizon.
    forecast_wind_speed: float
    forecast_wind_dir_deg: float

    # Forecast wind for each future timestep.
    forecast_wind_speed_sequence: np.ndarray
    forecast_wind_dir_sequence: np.ndarray

    # Reliability.
    reliability_alpha: float


# =====================================================================
# Forecast model
# =====================================================================

class ForecastModel:
    """
    Direct probabilistic forecast model.

    The model is deliberately lightweight at this stage.

    Flood forecast:
        probabilistic propagation of flood occupancy.

    Wind forecast:
        deterministic expected mean trajectory based on persistence
        toward a base/reference state.

    No future simulator rollout is used.
    """

    def __init__(
        self,
        horizon_steps: int = 10,
        alpha_fc: float = 0.80,

        # Flood forecast parameters.
        flood_expansion_rate: float = 0.03,
        wind_influence: float = 0.05,
        spontaneous_flood_probability: float = 0.001,

        # Default rain intensity used until the environment passes
        # the actual current rain state explicitly.
        rain_intensity: float = 0.50,

        # Forecast background prior.
        background_flood_prior: float = 0.05,

        # Expected wind persistence.
        wind_rho: float = 0.85,
        wind_base_speed: float = 5.0,

        seed: Optional[int] = None,
    ):
        if horizon_steps <= 0:
            raise ValueError(
                "horizon_steps must be > 0."
            )

        if not 0.0 <= alpha_fc <= 1.0:
            raise ValueError(
                "alpha_fc must be between 0 and 1."
            )

        if flood_expansion_rate < 0.0:
            raise ValueError(
                "flood_expansion_rate must be >= 0."
            )

        if wind_influence < 0.0:
            raise ValueError(
                "wind_influence must be >= 0."
            )

        if not 0.0 <= rain_intensity <= 1.0:
            raise ValueError(
                "rain_intensity must be between 0 and 1."
            )

        if not 0.0 <= spontaneous_flood_probability <= 1.0:
            raise ValueError(
                "spontaneous_flood_probability must be "
                "between 0 and 1."
            )

        if not 0.0 <= wind_rho < 1.0:
            raise ValueError(
                "wind_rho must satisfy 0 <= rho < 1."
            )

        if wind_base_speed < 0.0:
            raise ValueError(
                "wind_base_speed must be >= 0."
            )

        self.horizon = int(
            horizon_steps
        )

        self.flood_expansion_rate = float(
            flood_expansion_rate
        )

        self.wind_influence = float(
            wind_influence
        )

        self.spontaneous_probability = float(
            spontaneous_flood_probability
        )

        self.rain_intensity = float(
            rain_intensity
        )

        self.wind_rho = float(
            wind_rho
        )

        self.wind_base_speed = float(
            wind_base_speed
        )

        self.reliability_engine = (
            ForecastReliabilityEngine(
                alpha_fc=alpha_fc,
                background_flood_prior=background_flood_prior,
                seed=seed,
            )
        )

        self.seed = seed

        self.rng = np.random.default_rng(
            seed
        )

    # =================================================================
    # RESET
    # =================================================================

    def reset(
        self,
        alpha_fc: Optional[float] = None,
        seed: Optional[int] = None,
        *,
        rain_intensity: Optional[float] = None,
    ) -> None:
        """
        Reset forecast state.

        No future trajectory is generated here.
        """

        if seed is not None:

            self.seed = int(seed)

            self.rng = np.random.default_rng(
                seed
            )

            self.reliability_engine.reset(
                seed=seed
            )

        if alpha_fc is not None:

            self.reliability_engine.set_reliability(
                alpha_fc
            )

        if rain_intensity is not None:

            if not 0.0 <= rain_intensity <= 1.0:
                raise ValueError(
                    "rain_intensity must be between 0 and 1."
                )

            self.rain_intensity = float(
                rain_intensity
            )

    # =================================================================
    # WIND FORECAST
    # =================================================================

    def _forecast_wind(
        self,
        current_wind: WindState,
    ):
        """
        Generate expected future wind trajectory.

        We do NOT sample the actual future.

        The expected speed follows mean reversion:

            E[W(t+k)] =
                mu + rho^k (W_t - mu)

        Direction is assumed to remain centered around the current
        direction at this stage.

        Later this can be replaced by a learned weather predictor.
        """

        speed_sequence = []

        direction_sequence = []

        current_speed = float(
            current_wind.speed
        )

        current_direction = float(
            current_wind.direction_deg
        )

        for k in range(
            1,
            self.horizon + 1,
        ):

            expected_speed = (
                self.wind_base_speed
                + (
                    self.wind_rho ** k
                )
                * (
                    current_speed
                    - self.wind_base_speed
                )
            )

            expected_speed = max(
                0.0,
                expected_speed,
            )

            speed_sequence.append(
                expected_speed
            )

            # Expected direction remains centered at the current
            # direction because our direction process is a symmetric
            # random walk around its current value.
            direction_sequence.append(
                current_direction
            )

        return (
            np.asarray(
                speed_sequence,
                dtype=np.float32,
            ),
            np.asarray(
                direction_sequence,
                dtype=np.float32,
            ),
        )

    # =================================================================
    # WIND VECTOR
    # =================================================================

    @staticmethod
    def _wind_unit_vector(
        speed: float,
        direction_deg: float,
    ):
        """
        Convert wind direction to a unit vector.

        Convention:
            0°   = North
            90°  = East
            180° = South
            270° = West
        """

        if speed <= 1e-8:
            return 0.0, 0.0

        theta = np.radians(
            direction_deg
        )

        vx = np.sin(theta)
        vy = -np.cos(theta)

        return (
            float(vx),
            float(vy),
        )

    # =================================================================
    # FLOOD FORECAST
    # =================================================================

    def _forecast_flood_probabilities(
        self,
        current_flood: np.ndarray,
        building_grid: np.ndarray,
        wind_speed_sequence: np.ndarray,
        wind_dir_sequence: np.ndarray,
        rain_intensity: float,
    ) -> np.ndarray:
        """
        Propagate flood OCCUPANCY PROBABILITIES forward.

        This is NOT a sampled flood trajectory.

        Input:
            current_flood
                observed current flood state

        Output:
            probability sequence

            shape:
                (H, height, width)

        Each future timestep contains the probability that the cell
        is flooded by that horizon.

        Existing flooded cells persist probabilistically as certainty.
        """

        current_flood = np.asarray(
            current_flood,
            dtype=np.float32,
        )

        building_grid = np.asarray(
            building_grid
        )

        if current_flood.shape != building_grid.shape:
            raise ValueError(
                "current_flood and building_grid "
                "must have identical shapes."
            )

        height, width = (
            current_flood.shape
        )

        # Initial probability state.
        probability = np.clip(
            current_flood,
            0.0,
            1.0,
        ).astype(np.float32)

        sequence = []

        neighbors = [
            (0, -1),   # N
            (0, 1),    # S
            (1, 0),    # E
            (-1, 0),   # W
        ]

        for h in range(
            self.horizon
        ):

            next_probability = (
                probability.copy()
            )

            wind_speed = float(
                wind_speed_sequence[h]
            )

            wind_direction = float(
                wind_dir_sequence[h]
            )

            wind_x, wind_y = (
                self._wind_unit_vector(
                    wind_speed,
                    wind_direction,
                )
            )

            wind_strength = float(
                np.clip(
                    wind_speed / 15.0,
                    0.0,
                    1.0,
                )
            )

            for y in range(height):

                for x in range(width):

                    # Static blocked cells are not forecast as
                    # flight-area flood.
                    if building_grid[y, x] != 0:
                        next_probability[
                            y, x
                        ] = 0.0

                        continue

                    # Already certain flooded.
                    if probability[y, x] >= 0.999999:
                        next_probability[
                            y, x
                        ] = 1.0

                        continue

                    neighbor_pressure = 0.0

                    wind_alignment_sum = 0.0

                    for dx, dy in neighbors:

                        nx = x + dx
                        ny = y + dy

                        if not (
                            0 <= nx < width
                            and 0 <= ny < height
                        ):
                            continue

                        neighbor_probability = float(
                            probability[ny, nx]
                        )

                        if neighbor_probability <= 0.0:
                            continue

                        neighbor_pressure += (
                            neighbor_probability
                        )

                        # Direction from flooded neighbor toward
                        # candidate cell.
                        toward_x = -dx
                        toward_y = -dy

                        norm = np.sqrt(
                            toward_x ** 2
                            + toward_y ** 2
                        )

                        toward_x /= norm
                        toward_y /= norm

                        alignment = (
                            toward_x * wind_x
                            + toward_y * wind_y
                        )

                        wind_alignment_sum += (
                            neighbor_probability
                            * alignment
                        )

                    # Normalize expected number of flooded
                    # neighbors to [0, 1].
                    neighbor_pressure = np.clip(
                        neighbor_pressure / 4.0,
                        0.0,
                        1.0,
                    )

                    if neighbor_pressure > 0.0:

                        weighted_alignment = (
                            wind_alignment_sum
                            / max(
                                1e-8,
                                neighbor_pressure * 4.0,
                            )
                        )

                        weighted_alignment = np.clip(
                            weighted_alignment,
                            -1.0,
                            1.0,
                        )

                        positive_alignment = max(
                            0.0,
                            float(
                                weighted_alignment
                            ),
                        )

                    else:

                        positive_alignment = 0.0

                    # -------------------------------------------------
                    # Rain effect
                    # -------------------------------------------------

                    rain_multiplier = (
                        0.50
                        + 0.75
                        * rain_intensity
                    )

                    local_expansion = (
                        self.flood_expansion_rate
                        * 4.0
                        * neighbor_pressure
                        * rain_multiplier
                    )

                    # -------------------------------------------------
                    # Wind effect
                    # -------------------------------------------------

                    wind_component = (
                        self.wind_influence
                        * positive_alignment
                        * wind_strength
                    )

                    # -------------------------------------------------
                    # Spontaneous formation
                    # -------------------------------------------------

                    spontaneous = (
                        self.spontaneous_probability
                        * rain_intensity
                    )

                    # -------------------------------------------------
                    # Convert pressure to transition probability.
                    #
                    # We use:
                    #
                    #     1 - exp(-lambda)
                    #
                    # so the result remains bounded.
                    # -------------------------------------------------

                    hazard_rate = (
                        local_expansion
                        + wind_component
                        + spontaneous
                    )

                    transition_probability = (
                        1.0
                        - np.exp(
                            -max(
                                0.0,
                                hazard_rate,
                            )
                        )
                    )

                    transition_probability = float(
                        np.clip(
                            transition_probability,
                            0.0,
                            0.95,
                        )
                    )

                    # -------------------------------------------------
                    # Probability that this cell remains dry:
                    #
                    # current probability already flooded
                    # OR
                    # previously dry and expands this step
                    # -------------------------------------------------

                    probability_flooded = (
                        probability[y, x]
                        + (
                            1.0
                            - probability[y, x]
                        )
                        * transition_probability
                    )

                    next_probability[
                        y, x
                    ] = np.clip(
                        probability_flooded,
                        0.0,
                        1.0,
                    )

            probability = next_probability

            sequence.append(
                probability.copy()
            )

        return np.stack(
            sequence,
            axis=0,
        ).astype(np.float32)

    # =================================================================
    # PUBLIC FORECAST
    # =================================================================

    def generate_forecast(
        self,
        current_flood: np.ndarray,
        building_grid: np.ndarray,
        current_wind: WindState,
        current_rain_intensity: Optional[float] = None,
    ) -> ForecastState:
        """
        Generate an agent-visible probabilistic forecast.

        Parameters
        ----------
        current_flood:
            Current observed flood state.

        building_grid:
            Static blocked map.

        current_wind:
            Current wind state.

        current_rain_intensity:
            Current rain intensity in [0, 1].

        IMPORTANT:
            No future simulator state is accessed.
            No future DynamicFloodModel is instantiated.
            No realized future is sampled.
        """

        current_flood = np.asarray(
            current_flood
        )

        building_grid = np.asarray(
            building_grid
        )

        if (
            current_flood.ndim != 2
            or building_grid.ndim != 2
        ):
            raise ValueError(
                "current_flood and building_grid "
                "must be 2D arrays."
            )

        if (
            current_flood.shape
            != building_grid.shape
        ):
            raise ValueError(
                "current_flood and building_grid "
                "must have the same shape."
            )

        # -------------------------------------------------------------
        # Rain intensity
        # -------------------------------------------------------------

        if current_rain_intensity is None:

            rain_intensity = (
                self.rain_intensity
            )

        else:

            if not 0.0 <= current_rain_intensity <= 1.0:
                raise ValueError(
                    "current_rain_intensity must be "
                    "between 0 and 1."
                )

            rain_intensity = float(
                current_rain_intensity
            )

            self.rain_intensity = (
                rain_intensity
            )

        # -------------------------------------------------------------
        # Forecast wind
        # -------------------------------------------------------------

        (
            wind_speed_sequence,
            wind_dir_sequence,
        ) = self._forecast_wind(
            current_wind
        )

        # -------------------------------------------------------------
        # Direct probabilistic flood prediction
        # -------------------------------------------------------------

        raw_probability_sequence = (
            self._forecast_flood_probabilities(
                current_flood=current_flood,
                building_grid=building_grid,
                wind_speed_sequence=wind_speed_sequence,
                wind_dir_sequence=wind_dir_sequence,
                rain_intensity=rain_intensity,
            )
        )

        # -------------------------------------------------------------
        # Forecast reliability transformation
        # -------------------------------------------------------------

        visible_final_probability = (
            self.reliability_engine.calibrate_probability(
                raw_probability_sequence[-1],
                current_flood=current_flood,
                blocked_mask=building_grid != 0,
            )
        )

        # Apply same calibration to every forecast horizon.
        visible_sequence = []

        for h in range(
            self.horizon
        ):

            visible_map = (
                self.reliability_engine.calibrate_probability(
                    raw_probability_sequence[h],
                    current_flood=current_flood,
                    blocked_mask=building_grid != 0,
                )
            )

            visible_sequence.append(
                visible_map
            )

        visible_sequence = np.stack(
            visible_sequence,
            axis=0,
        ).astype(np.float32)

        # -------------------------------------------------------------
        # Keep the final horizon map consistent with the sequence.
        # -------------------------------------------------------------

        visible_final_probability = (
            visible_sequence[-1].copy()
        )

        # -------------------------------------------------------------
        # Wind summary
        # -------------------------------------------------------------

        mean_speed = float(
            np.mean(
                wind_speed_sequence
            )
        )

        # Circular mean for wind direction.
        angles = np.radians(
            wind_dir_sequence
        )

        mean_sin = np.mean(
            np.sin(angles)
        )

        mean_cos = np.mean(
            np.cos(angles)
        )

        mean_direction = float(
            np.degrees(
                np.arctan2(
                    mean_sin,
                    mean_cos,
                )
            )
            % 360.0
        )

        return ForecastState(
            horizon_steps=self.horizon,

            flood_prob_map=(
                visible_final_probability
            ),

            flood_prob_sequence=(
                visible_sequence
            ),

            forecast_wind_speed=(
                mean_speed
            ),

            forecast_wind_dir_deg=(
                mean_direction
            ),

            forecast_wind_speed_sequence=(
                wind_speed_sequence
            ),

            forecast_wind_dir_sequence=(
                wind_dir_sequence
            ),

            reliability_alpha=(
                self.reliability_engine.alpha_fc
            ),
        )