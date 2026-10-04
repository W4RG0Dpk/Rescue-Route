# Stochastic Wind and Gust Model for RescueRoute Environment.
# Simulates temporally correlated wind speed, dynamic wind direction, and stochastic gusts.
# Used to calculate headwind energy penalties, crosswind payload instabilities, and flood biases.

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# Wind State Data Container.
# Holds instantaneous wind speed, direction angle, gust strength, and vector components.
@dataclass
class WindState:
    speed: float           # Current wind speed in m/s (e.g. 0.0 to 15.0 m/s)
    direction_deg: float   # Current wind direction angle in degrees (0 = North, 90 = East, 180 = South, 270 = West)
    gust_strength: float   # Additional transient gust speed in m/s
    wind_vector: Tuple[float, float] # (vx, vy) 2D wind vector components


# Stochastic Wind Simulation Engine.
# Uses first-order autoregressive AR(1) noise process for smooth temporal correlation.
class WindModel:

    def __init__(
        self,
        base_speed: float = 6.0,
        speed_std: float = 2.0,
        correlation_rho: float = 0.85,
        gust_probability: float = 0.15,
        max_gust: float = 8.0,
        seed: Optional[int] = None
    ):
        self.base_speed = base_speed
        self.speed_std = speed_std
        self.rho = correlation_rho
        self.gust_prob = gust_probability
        self.max_gust = max_gust

        self.rng = np.random.default_rng(seed)

        # State variables
        self.current_speed = base_speed
        self.current_direction = self.rng.uniform(0.0, 360.0)
        self.current_gust = 0.0

    # Reset wind state with optional seed.
    def reset(self, seed: Optional[int] = None) -> WindState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_speed = max(0.0, self.base_speed + self.rng.normal(0, 1.0))
        self.current_direction = self.rng.uniform(0.0, 360.0)
        self.current_gust = 0.0
        return self.get_state()

    # Advance wind simulation by 1 time step.
    # Returns updated WindState object.
    def step(self) -> WindState:
        # AR(1) autoregressive speed update for temporal smoothness
        noise = self.rng.normal(0.0, self.speed_std * np.sqrt(1 - self.rho**2))
        self.current_speed = max(0.5, self.base_speed + self.rho * (self.current_speed - self.base_speed) + noise)

        # Small random drift in wind direction angle (-15 deg to +15 deg)
        dir_drift = self.rng.uniform(-15.0, 15.0)
        self.current_direction = (self.current_direction + dir_drift) % 360.0

        # Stochastic gust trigger check
        if self.rng.random() < self.gust_prob:
            self.current_gust = self.rng.uniform(2.0, self.max_gust)
        else:
            self.current_gust = max(0.0, self.current_gust * 0.5) # Decay previous gust

        return self.get_state()

    # Get current WindState data object.
    def get_state(self) -> WindState:
        total_speed = self.current_speed + self.current_gust
        rad = np.radians(self.current_direction)

        # Convert polar (speed, direction) to 2D Cartesian vector (vx, vy)
        vx = total_speed * np.sin(rad)
        vy = -total_speed * np.cos(rad)

        return WindState(
            speed=self.current_speed,
            direction_deg=self.current_direction,
            gust_strength=self.current_gust,
            wind_vector=(vx, vy)
        )

    # Calculate headwind component affecting drone movement in direction (dx, dy).
    # Returns relative energy multiplier factor (headwind > 1.0, tailwind < 1.0).
    def get_energy_multiplier(self, move_dir: Tuple[int, int]) -> float:
        dx, dy = move_dir
        if dx == 0 and dy == 0:
            return 1.0 # Hovering

        # Normalize movement direction vector
        move_len = np.sqrt(dx**2 + dy**2)
        u_dx, u_dy = dx / move_len, dy / move_len

        # Calculate dot product of move vector and wind vector
        vx, vy = self.get_state().wind_vector
        wind_len = np.sqrt(vx**2 + vy**2) + 1e-6

        # Dot product measures headwind vs tailwind alignment
        dot = u_dx * (vx / wind_len) + u_dy * (vy / wind_len)

        # Headwind (opposite to movement) increases energy consumption by up to +40%
        # Tailwind (same direction) reduces energy consumption by up to -15%
        multiplier = 1.0 + 0.3 * dot
        return max(0.7, min(1.5, multiplier))
