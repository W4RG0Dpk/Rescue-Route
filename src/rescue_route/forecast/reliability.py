"""
RescueRoute Forecast Reliability
================================

Transforms a probabilistic forecast according to the assumed forecast
reliability.

IMPORTANT
---------
This module does NOT receive or modify a ground-truth future.

The forecast is already a probability distribution produced from the
current state.

Reliability controls how strongly the forecast trusts that predictive
distribution.

alpha_fc:
    1.0 -> trust the predictive model strongly
    0.0 -> fall back toward a weak background prior

This is intentionally different from:

    true future -> corrupt true future -> call it forecast

which would leak information from a realized future trajectory into
the forecast-generation process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class ForecastReliabilityConfig:
    """
    Configuration for forecast reliability.

    alpha_fc:
        Forecast confidence in [0, 1].

    background_flood_prior:
        Weak prior used when forecast confidence is low.

    probability_floor:
        Prevents exact zero probability for non-blocked cells when
        uncertainty is high.

    probability_ceiling:
        Prevents exact certainty unless the current state itself is
        known with certainty.
    """

    alpha_fc: float = 0.80

    background_flood_prior: float = 0.05

    probability_floor: float = 0.0

    probability_ceiling: float = 1.0


# =====================================================================
# Reliability engine
# =====================================================================

class ForecastReliabilityEngine:
    """
    Applies uncertainty to an already probabilistic forecast.

    Input:
        predictive flood probabilities

    Output:
        agent-visible flood probabilities

    No ground-truth future is used.
    """

    def __init__(
        self,
        alpha_fc: float = 0.80,
        background_flood_prior: float = 0.05,
        seed: Optional[int] = None,
    ):
        if not 0.0 <= alpha_fc <= 1.0:
            raise ValueError(
                "alpha_fc must be between 0 and 1."
            )

        if not 0.0 <= background_flood_prior <= 1.0:
            raise ValueError(
                "background_flood_prior must be between 0 and 1."
            )

        self.alpha_fc = float(alpha_fc)

        self.background_flood_prior = float(
            background_flood_prior
        )

        # Retained for reproducible future extensions.
        # The current reliability transformation is deterministic.
        self.rng = np.random.default_rng(seed)

    # =================================================================
    # SET RELIABILITY
    # =================================================================

    def set_reliability(
        self,
        alpha_fc: float,
    ) -> None:
        """
        Update forecast reliability.
        """

        if not 0.0 <= alpha_fc <= 1.0:
            raise ValueError(
                "alpha_fc must be between 0 and 1."
            )

        self.alpha_fc = float(
            alpha_fc
        )

    # =================================================================
    # RESET
    # =================================================================

    def reset(
        self,
        alpha_fc: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Reset reliability configuration and RNG.
        """

        if alpha_fc is not None:
            self.set_reliability(
                alpha_fc
            )

        if seed is not None:
            self.rng = np.random.default_rng(
                seed
            )

    # =================================================================
    # CALIBRATE FLOOD FORECAST
    # =================================================================

    def calibrate_probability(
        self,
        probability_map: np.ndarray,
        current_flood: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Transform predictive probabilities according to alpha_fc.

        Formula:

            p_visible =
                alpha * p_model
                + (1-alpha) * p_prior

        where:

            p_prior = background_flood_prior

        Thus:

            alpha = 1
                -> model forecast is preserved

            alpha = 0
                -> weak background prior

        Current known flooded cells can remain certain because their
        current state is observed, not predicted.
        """

        probability_map = np.asarray(
            probability_map,
            dtype=np.float32,
        )

        if probability_map.ndim != 2:
            raise ValueError(
                "probability_map must be a 2D array."
            )

        p_model = np.clip(
            probability_map,
            0.0,
            1.0,
        )

        p_prior = (
            self.background_flood_prior
        )

        p_visible = (
            self.alpha_fc * p_model
            + (
                1.0 - self.alpha_fc
            ) * p_prior
        )

        # -------------------------------------------------------------
        # Known current flood cells
        #
        # These are observed current-state facts, not future truth.
        # -------------------------------------------------------------

        if current_flood is not None:

            current_flood = np.asarray(
                current_flood
            )

            if (
                current_flood.shape
                != probability_map.shape
            ):
                raise ValueError(
                    "current_flood shape must match "
                    "probability_map."
                )

            current_mask = (
                current_flood.astype(bool)
            )

            p_visible[
                current_mask
            ] = 1.0

        # -------------------------------------------------------------
        # Static blocked cells
        #
        # There is no future flood probability for an impossible
        # flight cell.
        # -------------------------------------------------------------

        if blocked_mask is not None:

            blocked_mask = np.asarray(
                blocked_mask
            )

            if (
                blocked_mask.shape
                != probability_map.shape
            ):
                raise ValueError(
                    "blocked_mask shape must match "
                    "probability_map."
                )

            p_visible[
                blocked_mask.astype(bool)
            ] = 0.0

        return np.clip(
            p_visible,
            0.0,
            1.0,
        ).astype(np.float32)