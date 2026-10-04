"""
RescueRoute stochastic wind model.

The model provides:
    - temporally correlated wind speed
    - slowly changing wind direction
    - stochastic gusts
    - wind vector
    - along-wind component
    - crosswind component
    - movement energy multiplier

Coordinate convention
---------------------
RescueRoute grid coordinates:

    x -> East  (+)
    y -> South (+)

Wind direction convention:

    0   degrees -> North
    90  degrees -> East
    180 degrees -> South
    270 degrees -> West

The wind vector represents the direction TOWARD which the air is
moving.

Therefore:

    movement aligned with wind
        -> tailwind
        -> lower energy

    movement opposite wind
        -> headwind
        -> higher energy

This is a controlled stochastic simulation model. The numerical
parameters are simulation parameters, not claims about real UAV
aerodynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# =====================================================================
# Wind state
# =====================================================================

@dataclass
class WindState:
    """
    Instantaneous wind state.

    Attributes
    ----------
    speed:
        Base/current sustained wind speed in m/s.

    direction_deg:
        Wind direction in degrees.
        0 = North, 90 = East, 180 = South, 270 = West.

    gust_strength:
        Additional transient gust speed in m/s.

    wind_vector:
        Total wind vector (vx, vy), including gust contribution.

    total_speed:
        Sustained speed + gust strength.
    """

    speed: float
    direction_deg: float
    gust_strength: float
    wind_vector: Tuple[float, float]
    total_speed: float


# =====================================================================
# Wind model
# =====================================================================

class WindModel:
    """
    Controlled stochastic wind process.

    Wind speed:
        AR(1)-style temporally correlated process.

    Wind direction:
        Slowly varying random walk.

    Gust:
        Stochastic event with exponential-like decay.

    All random behavior is controlled by a local NumPy RNG so that
    reset(seed=...) is reproducible.
    """

    def __init__(
        self,
        base_speed: float = 5.0,
        speed_std: float = 2.0,
        correlation_rho: float = 0.85,
        gust_probability: float = 0.15,
        max_gust: float = 8.0,
        min_speed: float = 0.5,
        direction_drift_deg: float = 15.0,
        gust_decay: float = 0.5,
        min_gust: float = 2.0,
        seed: Optional[int] = None,
    ):
        # -------------------------------------------------------------
        # Validate parameters
        # -------------------------------------------------------------

        if base_speed < 0.0:
            raise ValueError(
                "base_speed must be >= 0."
            )

        if speed_std < 0.0:
            raise ValueError(
                "speed_std must be >= 0."
            )

        if not 0.0 <= correlation_rho < 1.0:
            raise ValueError(
                "correlation_rho must satisfy 0 <= rho < 1."
            )

        if not 0.0 <= gust_probability <= 1.0:
            raise ValueError(
                "gust_probability must be between 0 and 1."
            )

        if max_gust < 0.0:
            raise ValueError(
                "max_gust must be >= 0."
            )

        if min_speed < 0.0:
            raise ValueError(
                "min_speed must be >= 0."
            )

        if direction_drift_deg < 0.0:
            raise ValueError(
                "direction_drift_deg must be >= 0."
            )

        if not 0.0 < gust_decay <= 1.0:
            raise ValueError(
                "gust_decay must satisfy 0 < gust_decay <= 1."
            )

        if min_gust < 0.0:
            raise ValueError(
                "min_gust must be >= 0."
            )

        if min_gust > max_gust and max_gust > 0:
            raise ValueError(
                "min_gust cannot exceed max_gust."
            )

        # -------------------------------------------------------------
        # Parameters
        # -------------------------------------------------------------

        self.base_speed = float(base_speed)
        self.speed_std = float(speed_std)
        self.rho = float(correlation_rho)

        self.gust_prob = float(gust_probability)
        self.max_gust = float(max_gust)
        self.min_gust = float(min_gust)
        self.gust_decay = float(gust_decay)

        self.min_speed = float(min_speed)

        self.direction_drift_deg = float(
            direction_drift_deg
        )

        # -------------------------------------------------------------
        # RNG
        # -------------------------------------------------------------

        self.rng = np.random.default_rng(seed)

        # -------------------------------------------------------------
        # Dynamic state
        # -------------------------------------------------------------

        self.current_speed = self.base_speed

        self.current_direction = float(
            self.rng.uniform(0.0, 360.0)
        )

        self.current_gust = 0.0

    # =================================================================
    # Reset
    # =================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        *,
        base_speed: Optional[float] = None,
        speed_std: Optional[float] = None,
        gust_probability: Optional[float] = None,
    ) -> WindState:
        """
        Reset the wind state.

        Optional condition parameters allow future mission-level
        scenario control without changing the public API.

        Example later:

            wind_model.reset(
                seed=123,
                base_speed=10.0,
                speed_std=3.0,
                gust_probability=0.30,
            )
        """

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # -------------------------------------------------------------
        # Optional scenario overrides
        # -------------------------------------------------------------

        if base_speed is not None:
            if base_speed < 0.0:
                raise ValueError(
                    "base_speed must be >= 0."
                )

            self.base_speed = float(
                base_speed
            )

        if speed_std is not None:
            if speed_std < 0.0:
                raise ValueError(
                    "speed_std must be >= 0."
                )

            self.speed_std = float(
                speed_std
            )

        if gust_probability is not None:
            if not 0.0 <= gust_probability <= 1.0:
                raise ValueError(
                    "gust_probability must be between 0 and 1."
                )

            self.gust_prob = float(
                gust_probability
            )

        # -------------------------------------------------------------
        # Initial stochastic state
        # -------------------------------------------------------------

        self.current_speed = max(
            self.min_speed,
            self.base_speed
            + self.rng.normal(
                0.0,
                self.speed_std,
            ),
        )

        self.current_direction = float(
            self.rng.uniform(
                0.0,
                360.0,
            )
        )

        self.current_gust = 0.0

        return self.get_state()

    # =================================================================
    # Step
    # =================================================================

    def step(self) -> WindState:
        """
        Advance wind by one simulation timestep.

        Wind speed follows a correlated stochastic process:

            W[t+1] =
                mu
                + rho * (W[t] - mu)
                + noise

        The noise variance is scaled so the long-run variability
        remains approximately controlled by speed_std.
        """

        # -------------------------------------------------------------
        # AR(1) wind speed update
        # -------------------------------------------------------------

        innovation_std = (
            self.speed_std
            * np.sqrt(
                max(
                    0.0,
                    1.0 - self.rho ** 2,
                )
            )
        )

        noise = self.rng.normal(
            0.0,
            innovation_std,
        )

        self.current_speed = max(
            self.min_speed,
            self.base_speed
            + self.rho
            * (
                self.current_speed
                - self.base_speed
            )
            + noise,
        )

        # -------------------------------------------------------------
        # Direction random walk
        # -------------------------------------------------------------

        direction_change = self.rng.uniform(
            -self.direction_drift_deg,
            self.direction_drift_deg,
        )

        self.current_direction = (
            self.current_direction
            + direction_change
        ) % 360.0

        # -------------------------------------------------------------
        # Gust process
        # -------------------------------------------------------------

        if (
            self.max_gust > 0.0
            and self.rng.random() < self.gust_prob
        ):

            self.current_gust = float(
                self.rng.uniform(
                    self.min_gust,
                    self.max_gust,
                )
            )

        else:

            # Exponential-like decay.
            self.current_gust *= self.gust_decay

            # Remove tiny residual numerical values.
            if self.current_gust < 1e-6:
                self.current_gust = 0.0

        return self.get_state()

    # =================================================================
    # State
    # =================================================================

    def get_state(self) -> WindState:
        """
        Return the current wind state.

        Vector convention:

            vx = speed * sin(theta)
            vy = -speed * cos(theta)

        which gives:

            0°   -> North
            90°  -> East
            180° -> South
            270° -> West
        """

        total_speed = (
            self.current_speed
            + self.current_gust
        )

        radians = np.radians(
            self.current_direction
        )

        vx = total_speed * np.sin(
            radians
        )

        vy = -total_speed * np.cos(
            radians
        )

        return WindState(
            speed=float(
                self.current_speed
            ),

            direction_deg=float(
                self.current_direction
            ),

            gust_strength=float(
                self.current_gust
            ),

            wind_vector=(
                float(vx),
                float(vy),
            ),

            total_speed=float(
                total_speed
            ),
        )

    # =================================================================
    # Vector helpers
    # =================================================================

    @staticmethod
    def _normalize_vector(
        vector: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Return normalized 2D vector.

        Returns None for zero vector, used by Hover.
        """

        dx, dy = vector

        norm = np.sqrt(
            float(dx * dx + dy * dy)
        )

        if norm < 1e-8:
            return None

        return np.array(
            [
                dx / norm,
                dy / norm,
            ],
            dtype=np.float64,
        )

    # =================================================================
    # Along-wind component
    # =================================================================

    def get_along_wind_component(
        self,
        move_dir: Tuple[int, int],
    ) -> float:
        """
        Calculate wind component along movement direction.

        Positive:
            wind is moving in the same direction as the UAV.

            -> tailwind

        Negative:
            wind opposes UAV movement.

            -> headwind

        Units:
            approximately m/s.
        """

        movement = self._normalize_vector(
            move_dir
        )

        if movement is None:
            return 0.0

        vx, vy = self.get_state().wind_vector

        wind_vector = np.array(
            [vx, vy],
            dtype=np.float64,
        )

        return float(
            np.dot(
                movement,
                wind_vector,
            )
        )

    # =================================================================
    # Crosswind component
    # =================================================================

    def get_crosswind_component(
        self,
        move_dir: Tuple[int, int],
    ) -> float:
        """
        Calculate absolute crosswind component.

        Large values represent strong sideways wind loading, which
        will later be useful for payload stability.

        Units:
            approximately m/s.
        """

        movement = self._normalize_vector(
            move_dir
        )

        if movement is None:
            return 0.0

        vx, vy = self.get_state().wind_vector

        wind_vector = np.array(
            [vx, vy],
            dtype=np.float64,
        )

        # In 2D, magnitude of the perpendicular component:
        along = np.dot(
            movement,
            wind_vector,
        )

        wind_sq = np.dot(
            wind_vector,
            wind_vector,
        )

        cross_sq = max(
            0.0,
            wind_sq - along ** 2,
        )

        return float(
            np.sqrt(cross_sq)
        )

    # =================================================================
    # Energy multiplier
    # =================================================================

    def get_energy_multiplier(
        self,
        move_dir: Tuple[int, int],
    ) -> float:
        """
        Calculate wind-dependent energy multiplier.

        Hover:
            1.0

        Tailwind:
            multiplier < 1.0

        No wind alignment:
            approximately 1.0

        Headwind:
            multiplier > 1.0

        The effect is intentionally capped.

        This is a simulation-level energy approximation, not a
        detailed aerodynamic model.
        """

        movement = self._normalize_vector(
            move_dir
        )

        # Hover has its normal base energy cost.
        if movement is None:
            return 1.0

        state = self.get_state()

        total_speed = max(
            state.total_speed,
            1e-6,
        )

        vx, vy = state.wind_vector

        wind_unit = np.array(
            [
                vx / total_speed,
                vy / total_speed,
            ],
            dtype=np.float64,
        )

        alignment = float(
            np.dot(
                movement,
                wind_unit,
            )
        )

        # -------------------------------------------------------------
        # Strength of wind relative to a reference range.
        #
        # This prevents a 1 m/s wind from having the same effect as
        # a 15 m/s wind.
        # -------------------------------------------------------------

        wind_strength = np.clip(
            total_speed / 15.0,
            0.0,
            1.0,
        )

        # -------------------------------------------------------------
        # Correct physical direction:
        #
        # alignment > 0
        #     tailwind -> lower energy
        #
        # alignment < 0
        #     headwind -> higher energy
        # -------------------------------------------------------------

        multiplier = (
            1.0
            - 0.20
            * alignment
            * wind_strength
        )

        return float(
            np.clip(
                multiplier,
                0.75,
                1.40,
            )
        )