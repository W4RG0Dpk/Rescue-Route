"""
RescueRoute Payload Stability Model
===================================

Models the stability / integrity state of the medical payload during
UAV flight.

The model is intentionally a controlled simulation model rather than
a high-fidelity physical model.

Payload types
-------------
STANDARD
    Lower sensitivity to environmental disturbance.

FRAGILE
    Higher sensitivity to crosswind, gusts and sharp turns.

Main effects
------------
1. Crosswind
2. Gust strength
3. Direction changes / turning
4. Hover-based stabilization

Important distinction
---------------------
We distinguish between:

    stability
        Current continuous payload stability score [0, 1].

    is_damaged
        Stability is below the required delivery threshold.
        This means the payload is currently NOT VALID for delivery.

    catastrophic_damage
        Stability has fallen below a much lower irreversible
        threshold. Hover can no longer recover the payload.

Thus, a payload can temporarily fall below the delivery threshold
without being permanently destroyed.

This distinction will be useful later for:
    - payload-stability metrics
    - delivery success
    - recoverability
    - safety filtering
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..weather.wind import WindState


# =====================================================================
# Payload state
# =====================================================================

@dataclass
class PayloadState:
    """
    Current payload state.

    Attributes
    ----------
    stability:
        Continuous payload stability score.

        1.0 -> fully stable
        0.0 -> completely failed

    payload_type:
        "STANDARD" or "FRAGILE"

    required_threshold:
        Minimum stability required for successful delivery.

    is_damaged:
        True when current stability is below the delivery threshold.

        NOTE:
        This is intentionally compatible with the existing environment.
        It means "delivery-invalid", not necessarily permanently
        physically destroyed.

    catastrophic_damage:
        True when stability has fallen below the irreversible damage
        threshold.

    instability:
        Current instability level = 1 - stability.
    """

    stability: float
    payload_type: str
    required_threshold: float

    is_damaged: bool
    catastrophic_damage: bool
    instability: float


# =====================================================================
# Payload stability model
# =====================================================================

class PayloadStabilityModel:
    """
    Simulates payload stability under UAV movement and wind.

    The model uses three main disturbance components:

        crosswind_load
        gust_load
        turning_load

    and one recovery component:

        hover_recovery

    All parameters are simulation parameters, not real-world physical
    constants.
    """

    VALID_PAYLOAD_TYPES = {
        "STANDARD",
        "FRAGILE",
    }

    # -------------------------------------------------------------
    # Payload sensitivity multipliers
    #
    # STANDARD is affected, but much less strongly than FRAGILE.
    # -------------------------------------------------------------

    PAYLOAD_SENSITIVITY = {
        "STANDARD": {
            "wind": 0.35,
            "gust": 0.30,
            "turn": 0.35,
            "hover_recovery": 0.75,
        },

        "FRAGILE": {
            "wind": 1.00,
            "gust": 1.00,
            "turn": 1.00,
            "hover_recovery": 1.00,
        },
    }

    def __init__(
        self,
        payload_type: str = "FRAGILE",
        required_threshold: float = 0.80,

        # Base disturbance coefficients.
        alpha_wind: float = 0.025,
        beta_gust: float = 0.050,
        gamma_turn: float = 0.035,

        # Recovery when hovering.
        hover_recovery: float = 0.040,

        # Once stability reaches this level, the payload is considered
        # catastrophically damaged and cannot be restored through hover.
        catastrophic_threshold: float = 0.15,

        # Normalization scales.
        crosswind_reference: float = 10.0,
        gust_reference: float = 8.0,
    ):
        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        payload_type = payload_type.upper()

        if payload_type not in self.VALID_PAYLOAD_TYPES:
            raise ValueError(
                f"Unknown payload type '{payload_type}'. "
                f"Expected one of {sorted(self.VALID_PAYLOAD_TYPES)}."
            )

        if not 0.0 <= required_threshold <= 1.0:
            raise ValueError(
                "required_threshold must be between 0 and 1."
            )

        if alpha_wind < 0.0:
            raise ValueError(
                "alpha_wind must be >= 0."
            )

        if beta_gust < 0.0:
            raise ValueError(
                "beta_gust must be >= 0."
            )

        if gamma_turn < 0.0:
            raise ValueError(
                "gamma_turn must be >= 0."
            )

        if hover_recovery < 0.0:
            raise ValueError(
                "hover_recovery must be >= 0."
            )

        if not 0.0 < catastrophic_threshold < 1.0:
            raise ValueError(
                "catastrophic_threshold must be between 0 and 1."
            )

        if crosswind_reference <= 0.0:
            raise ValueError(
                "crosswind_reference must be > 0."
            )

        if gust_reference <= 0.0:
            raise ValueError(
                "gust_reference must be > 0."
            )

        if catastrophic_threshold >= required_threshold:
            raise ValueError(
                "catastrophic_threshold should be below "
                "required_threshold."
            )

        # -------------------------------------------------------------
        # Parameters
        # -------------------------------------------------------------

        self.payload_type = payload_type

        self.threshold = float(
            required_threshold
        )

        self.alpha_wind = float(
            alpha_wind
        )

        self.beta_gust = float(
            beta_gust
        )

        self.gamma_turn = float(
            gamma_turn
        )

        self.hover_recovery = float(
            hover_recovery
        )

        self.catastrophic_threshold = float(
            catastrophic_threshold
        )

        self.crosswind_reference = float(
            crosswind_reference
        )

        self.gust_reference = float(
            gust_reference
        )

        # -------------------------------------------------------------
        # Dynamic state
        # -------------------------------------------------------------

        self.stability = 1.0

    # =================================================================
    # RESET
    # =================================================================

    def reset(
        self,
        payload_type: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> PayloadState:
        """
        Reset payload stability to 1.0.
        """

        if payload_type is not None:

            payload_type = payload_type.upper()

            if (
                payload_type
                not in self.VALID_PAYLOAD_TYPES
            ):
                raise ValueError(
                    f"Unknown payload type '{payload_type}'. "
                    f"Expected one of "
                    f"{sorted(self.VALID_PAYLOAD_TYPES)}."
                )

            self.payload_type = payload_type

        if threshold is not None:

            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "threshold must be between 0 and 1."
                )

            if (
                self.catastrophic_threshold
                >= threshold
            ):
                raise ValueError(
                    "threshold must be above "
                    "catastrophic_threshold."
                )

            self.threshold = float(
                threshold
            )

        self.stability = 1.0

        return self.get_state()

    # =================================================================
    # TURN SEVERITY
    # =================================================================

    @staticmethod
    def _calculate_turn_severity(
        move_vector: Tuple[int, int],
        previous_vector: Tuple[int, int],
    ) -> float:
        """
        Calculate normalized turn severity in [0, 1].

        Approximate interpretation:

            0.0 -> same direction
            0.5 -> 90 degree turn
            1.0 -> 180 degree reversal

        A missing previous direction means no turn penalty.
        """

        dx, dy = move_vector
        p_dx, p_dy = previous_vector

        # No movement.
        if dx == 0 and dy == 0:
            return 0.0

        # No previous movement / initial heading.
        if p_dx == 0 and p_dy == 0:
            return 0.0

        current_norm = np.sqrt(
            dx ** 2 + dy ** 2
        )

        previous_norm = np.sqrt(
            p_dx ** 2 + p_dy ** 2
        )

        if (
            current_norm < 1e-8
            or previous_norm < 1e-8
        ):
            return 0.0

        current_unit = np.array(
            [
                dx / current_norm,
                dy / current_norm,
            ],
            dtype=np.float64,
        )

        previous_unit = np.array(
            [
                p_dx / previous_norm,
                p_dy / previous_norm,
            ],
            dtype=np.float64,
        )

        dot_product = float(
            np.dot(
                current_unit,
                previous_unit,
            )
        )

        dot_product = float(
            np.clip(
                dot_product,
                -1.0,
                1.0,
            )
        )

        # Map [-1, 1] -> [0, 1].
        #
        # same direction:
        #     dot = 1  -> 0
        #
        # 90 degrees:
        #     dot = 0  -> 0.5
        #
        # opposite:
        #     dot = -1 -> 1
        #
        return float(
            np.clip(
                (1.0 - dot_product) / 2.0,
                0.0,
                1.0,
            )
        )

    # =================================================================
    # CROSSWIND
    # =================================================================

    @staticmethod
    def _calculate_crosswind(
        move_vector: Tuple[int, int],
        wind_state: WindState,
    ) -> float:
        """
        Calculate absolute wind component perpendicular to movement.

        This uses the instantaneous wind vector, including gusts.
        """

        dx, dy = move_vector

        move_length = np.sqrt(
            dx ** 2 + dy ** 2
        )

        if move_length < 1e-8:
            return 0.0

        unit_dx = (
            dx / move_length
        )

        unit_dy = (
            dy / move_length
        )

        vx, vy = wind_state.wind_vector

        # Perpendicular component:
        #
        # |cross(unit_movement, wind_vector)|
        #
        crosswind = abs(
            -unit_dy * vx
            + unit_dx * vy
        )

        return float(
            crosswind
        )

    # =================================================================
    # STEP
    # =================================================================

    def step(
        self,
        action: int,
        move_vector: Tuple[int, int],
        prev_move_vector: Tuple[int, int],
        wind_state: WindState,
    ) -> PayloadState:
        """
        Update payload stability for one simulation timestep.

        Movement:
            stability decreases according to:
                crosswind
                gust
                turn severity

        Hover:
            stability partially recovers.

        Catastrophic damage:
            once stability reaches the catastrophic threshold,
            normal hover recovery no longer restores it.
        """

        # -------------------------------------------------------------
        # Current sensitivity
        # -------------------------------------------------------------

        payload_settings = (
            self.PAYLOAD_SENSITIVITY[
                self.payload_type
            ]
        )

        # -------------------------------------------------------------
        # Hover
        # -------------------------------------------------------------

        if (
            action == 8
            or (
                move_vector[0] == 0
                and move_vector[1] == 0
            )
        ):

            # Do not recover catastrophic damage.
            if (
                self.stability
                > self.catastrophic_threshold
            ):

                recovery = (
                    self.hover_recovery
                    * payload_settings[
                        "hover_recovery"
                    ]
                )

                self.stability = min(
                    1.0,
                    self.stability
                    + recovery,
                )

            return self.get_state()

        # -------------------------------------------------------------
        # Environmental disturbance
        # -------------------------------------------------------------

        crosswind = (
            self._calculate_crosswind(
                move_vector,
                wind_state,
            )
        )

        gust_strength = max(
            0.0,
            float(
                wind_state.gust_strength
            ),
        )

        # -------------------------------------------------------------
        # Normalize loads
        # -------------------------------------------------------------

        normalized_crosswind = np.clip(
            crosswind
            / self.crosswind_reference,
            0.0,
            2.0,
        )

        normalized_gust = np.clip(
            gust_strength
            / self.gust_reference,
            0.0,
            2.0,
        )

        turn_severity = (
            self._calculate_turn_severity(
                move_vector,
                prev_move_vector,
            )
        )

        # -------------------------------------------------------------
        # Payload-specific decay
        # -------------------------------------------------------------

        wind_decay = (
            self.alpha_wind
            * normalized_crosswind
            * payload_settings["wind"]
        )

        gust_decay = (
            self.beta_gust
            * normalized_gust
            * payload_settings["gust"]
        )

        turn_decay = (
            self.gamma_turn
            * turn_severity
            * payload_settings["turn"]
        )

        total_decay = (
            wind_decay
            + gust_decay
            + turn_decay
        )

        # -------------------------------------------------------------
        # Update stability
        # -------------------------------------------------------------

        self.stability = float(
            np.clip(
                self.stability
                - total_decay,
                0.0,
                1.0,
            )
        )

        return self.get_state()

    # =================================================================
    # STATE
    # =================================================================

    def get_state(self) -> PayloadState:
        """
        Return the current payload state.
        """

        stability = float(
            np.clip(
                self.stability,
                0.0,
                1.0,
            )
        )

        is_damaged = (
            stability
            < self.threshold
        )

        catastrophic_damage = (
            stability
            <= self.catastrophic_threshold
        )

        instability = (
            1.0 - stability
        )

        return PayloadState(
            stability=stability,
            payload_type=self.payload_type,
            required_threshold=self.threshold,
            is_damaged=is_damaged,
            catastrophic_damage=catastrophic_damage,
            instability=float(
                np.clip(
                    instability,
                    0.0,
                    1.0,
                )
            ),
        )