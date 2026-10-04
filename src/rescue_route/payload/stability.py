# Payload Stability Model for RescueRoute Environment.
# Models physical payload integrity and stability degradation ($S_t \in [0.0, 1.0]$).
# Fragile payloads (e.g. vaccines/plasma) degrade under crosswinds, severe gusts, and sharp turning,
# while Hover stabilization actions provide recovery contribution ($R_t$).

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from ..weather.wind import WindState


# Payload State Data Container.
# Holds current stability ratio, payload type, and decay parameters.
@dataclass
class PayloadState:
    stability: float       # Current payload stability ratio (1.0 = PERFECT, 0.0 = DAMAGED/DESTROYED)
    payload_type: str      # 'STANDARD' or 'FRAGILE'
    required_threshold: float # Minimum stability required at delivery destination (e.g. 0.80)
    is_damaged: bool       # True if stability falls below required threshold


# Payload Stability Simulation Engine.
class PayloadStabilityModel:

    def __init__(
        self,
        payload_type: str = "FRAGILE",
        required_threshold: float = 0.80,
        alpha_wind: float = 0.02,
        beta_gust: float = 0.05,
        gamma_turn: float = 0.03,
        hover_recovery: float = 0.04
    ):
        self.payload_type = payload_type
        self.threshold = required_threshold
        self.alpha_wind = alpha_wind
        self.beta_gust = beta_gust
        self.gamma_turn = gamma_turn
        self.hover_recovery = hover_recovery

        # Current stability level (1.0 = 100%)
        self.stability = 1.0

    # Reset payload stability to 1.0 (100% intact).
    def reset(self, payload_type: Optional[str] = None, threshold: Optional[float] = None) -> PayloadState:
        if payload_type is not None:
            self.payload_type = payload_type
        if threshold is not None:
            self.threshold = threshold

        self.stability = 1.0
        return self.get_state()

    # Update payload stability for 1 flight step given drone move vector and wind state.
    # Parameters:
    #   action: Action integer ID (0 to 8)
    #   move_vector: (dx, dy) direction vector
    #   prev_move_vector: Previous step (dx, dy) direction vector for turning calculation
    #   wind_state: Current WindState object
    # Returns:
    #   Updated PayloadState object.
    def step(
        self,
        action: int,
        move_vector: Tuple[int, int],
        prev_move_vector: Tuple[int, int],
        wind_state: WindState
    ) -> PayloadState:
        # Standard payloads do not experience stability degradation
        if self.payload_type == "STANDARD":
            return self.get_state()

        dx, dy = move_vector
        p_dx, p_dy = prev_move_vector

        # Action 8 is Hover: Hovering allows drone gimbal/flight controller to stabilize payload
        if action == 8 or (dx == 0 and dy == 0):
            # Recovery contribution R_t
            self.stability = min(1.0, self.stability + self.hover_recovery)
            return self.get_state()

        # 1. Calculate crosswind component perpendicular to movement direction
        vx, vy = wind_state.wind_vector
        move_len = np.sqrt(dx**2 + dy**2) + 1e-6
        u_dx, u_dy = dx / move_len, dy / move_len
        
        # Perpendicular unit vector (-u_dy, u_dx)
        crosswind = abs(-u_dy * vx + u_dx * vy)

        # 2. Calculate turning sharpness (angle difference from previous step)
        turn_severity = 0.0
        if p_dx != 0 or p_dy != 0:
            prev_len = np.sqrt(p_dx**2 + p_dy**2)
            u_pdx, u_pdy = p_dx / prev_len, p_dy / prev_len
            dot_turn = u_dx * u_pdx + u_dy * u_pdy
            # 90 degree turn -> dot = 0, 180 turn -> dot = -1
            turn_severity = max(0.0, 1.0 - dot_turn)

        # 3. Calculate total decay D_t
        decay = (
            self.alpha_wind * (crosswind / 10.0) +
            self.beta_gust * (wind_state.gust_strength / 8.0) +
            self.gamma_turn * turn_severity
        )

        # Update stability state: S_{t+1} = clip(S_t - D_t, 0.0, 1.0)
        self.stability = max(0.0, min(1.0, self.stability - decay))
        return self.get_state()

    # Get current PayloadState data object.
    def get_state(self) -> PayloadState:
        is_damaged = self.stability < self.threshold
        return PayloadState(
            stability=self.stability,
            payload_type=self.payload_type,
            required_threshold=self.threshold,
            is_damaged=is_damaged
        )
