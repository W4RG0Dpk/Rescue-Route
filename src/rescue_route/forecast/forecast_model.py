# Forecast Model Generator for RescueRoute Environment.
# Implements Dual-World Architecture:
# - Actual Environment Future: Hidden sampled ground-truth weather & flood progression.
# - Forecast Generator: Probabilistic future forecast available to the UAV agent over lookahead horizon H=10.

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from ..weather.wind import WindModel, WindState
from ..hazards.flood import DynamicFloodModel
from .reliability import ForecastReliabilityEngine


# Forecast State Data Container.
# Holds lookahead horizon predictions, flood probability grid, and wind forecast vector.
@dataclass
class ForecastState:
    horizon_steps: int                  # Forecast lookahead horizon H (e.g. 10 steps)
    flood_prob_map: np.ndarray          # 2D float array in [0.0, 1.0] of predicted flood probabilities
    forecast_wind_speed: float         # Predicted average wind speed over horizon
    forecast_wind_dir_deg: float       # Predicted average wind direction over horizon
    reliability_alpha: float            # Forecast reliability factor alpha_fc


# Dual-World Forecast Generator Engine.
class ForecastModel:

    def __init__(
        self,
        horizon_steps: int = 10,
        alpha_fc: float = 0.80,
        seed: Optional[int] = None
    ):
        self.horizon = horizon_steps
        self.reliability_engine = ForecastReliabilityEngine(alpha_fc=alpha_fc, seed=seed)
        self.seed = seed

    # Reset forecast model state.
    def reset(self, alpha_fc: Optional[float] = None, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        if alpha_fc is not None:
            self.reliability_engine.set_reliability(alpha_fc)

    # Generate probabilistic forecast over lookahead horizon H.
    # Simulates future trajectory rollout using current environment state,
    # then applies reliability noise (alpha_fc) to generate agent-visible forecast.
    # Parameters:
    #   current_flood: 2D array of current flood state
    #   building_grid: 2D array of static building obstacles
    #   current_wind: Current WindState object
    # Returns:
    #   ForecastState object.
    def generate_forecast(
        self,
        current_flood: np.ndarray,
        building_grid: np.ndarray,
        current_wind: WindState
    ) -> ForecastState:
        height, width = current_flood.shape

        # Simulate rollout of true future flood grid H steps ahead
        sim_flood_model = DynamicFloodModel(width, height, seed=self.seed)
        sim_flood_model.flood_grid = current_flood.copy()

        # Step simulated flood model forward over lookahead horizon H
        for _ in range(self.horizon):
            sim_flood_model.step(building_grid, current_wind)

        true_future_flood = sim_flood_model.flood_grid.copy()

        # Apply forecast reliability noise (alpha_fc) to ground-truth future flood
        prob_flood_map = self.reliability_engine.apply_reliability_noise(true_future_flood, building_grid)

        # Wind forecast over horizon (adds small forecast prediction variation)
        fc_wind_speed = current_wind.speed + self.reliability_engine.rng.normal(0, (1.0 - self.reliability_engine.alpha_fc) * 2.0)
        fc_wind_speed = max(0.0, fc_wind_speed)
        fc_wind_dir = (current_wind.direction_deg + self.reliability_engine.rng.normal(0, (1.0 - self.reliability_engine.alpha_fc) * 15.0)) % 360.0

        return ForecastState(
            horizon_steps=self.horizon,
            flood_prob_map=prob_flood_map,
            forecast_wind_speed=fc_wind_speed,
            forecast_wind_dir_deg=fc_wind_dir,
            reliability_alpha=self.reliability_engine.alpha_fc
        )
