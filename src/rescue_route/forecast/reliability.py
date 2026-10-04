# Forecast Reliability Engine for RescueRoute Environment.
# Models forecast prediction accuracy, degradation, and noise via reliability parameter alpha_fc in [0.0, 1.0].
# Used to simulate real-world meteorological/hydrological forecast uncertainty.

import numpy as np
from dataclasses import dataclass
from typing import Optional


# Forecast Reliability Config Data Container.
@dataclass
class ForecastReliabilityConfig:
    alpha_fc: float       # Reliability factor in [0.0, 1.0] (1.0 = PERFECT FORECAST, 0.0 = NOISY/UNRELIABLE)
    false_positive_rate: float # Probability of forecasting flood where none occurs
    false_negative_rate: float # Probability of missing an actual flood expansion


# Forecast Reliability Engine.
# Corrupts ground-truth future hazard maps based on alpha_fc reliability factor.
class ForecastReliabilityEngine:

    def __init__(self, alpha_fc: float = 0.80, seed: Optional[int] = None):
        self.alpha_fc = max(0.0, min(1.0, alpha_fc))
        self.rng = np.random.default_rng(seed)

    # Set or reset reliability factor alpha_fc.
    def set_reliability(self, alpha_fc: float):
        self.alpha_fc = max(0.0, min(1.0, alpha_fc))

    # Apply forecast reliability noise to ground-truth future flood grid.
    # Parameters:
    #   true_future_flood: 2D numpy array of actual future flood states
    #   building_grid: 2D array of static building obstacles
    # Returns:
    #   Probabilistic float 2D array representing agent-visible flood forecast probabilities in [0.0, 1.0].
    def apply_reliability_noise(self, true_future_flood: np.ndarray, building_grid: np.ndarray) -> np.ndarray:
        height, width = true_future_flood.shape
        forecast_grid = true_future_flood.astype(np.float32)

        if self.alpha_fc == 1.0:
            # Perfect forecast matching true future
            forecast_grid[building_grid == 1] = 0.0
            return forecast_grid

        # Noise level inversely proportional to alpha_fc
        noise_level = 1.0 - self.alpha_fc

        # Add Gaussian prediction uncertainty to forecast probabilities
        noise = self.rng.normal(0.0, noise_level * 0.35, size=(height, width))
        forecast_grid = np.clip(forecast_grid + noise, 0.0, 1.0)

        # Zero out building obstacle cells in forecast
        forecast_grid[building_grid == 1] = 0.0
        return forecast_grid
