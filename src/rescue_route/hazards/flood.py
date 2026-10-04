# Dynamic Flood Expansion Model for RescueRoute Environment.
# Simulates stochastic flood spreading across the grid map over time.
# Uses controlled cellular automaton expansion influenced by wind direction and rain intensity.
# Ensures start base and target locations are protected from initial flood epicenters.

import numpy as np
from typing import Tuple, List, Optional
from ..weather.wind import WindState


# Dynamic Flood Model Engine.
# Maintains a 2D boolean/float grid matrix tracking flooded airspace cells.
class DynamicFloodModel:

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        initial_flood_ratio: float = 0.015,
        expansion_rate: float = 0.03,
        seed: Optional[int] = None
    ):
        self.width = grid_width
        self.height = grid_height
        self.initial_ratio = initial_flood_ratio
        self.expansion_rate = expansion_rate
        self.rng = np.random.default_rng(seed)

        # 2D Grid tracking flooded cells (1 = FLOODED/HAZARD, 0 = DRY/SAFE)
        self.flood_grid = np.zeros((self.height, self.width), dtype=np.int32)

    # Initialize flood map with initial seed clusters, excluding protected start/target coords.
    # Parameters:
    #   building_grid: 2D array of static building obstacles
    #   safe_coords: List of protected coordinates (start base, target) to exclude from initial flood
    #   seed: Random seed for simulation
    def reset(self, building_grid: np.ndarray, safe_coords: Optional[List[Tuple[int, int]]] = None, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.flood_grid = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Build set of protected coordinates
        safe_set = set(safe_coords) if safe_coords else set()

        # Seed initial flood epicenters in non-building, non-safe cells
        num_seeds = max(2, int(self.width * self.height * self.initial_ratio))
        free_coords = [
            (x, y) for y in range(self.height) for x in range(self.width) 
            if building_grid[y, x] == 0 and (x, y) not in safe_set
        ]

        if free_coords:
            seed_idx = self.rng.choice(len(free_coords), size=min(num_seeds, len(free_coords)), replace=False)
            for idx in seed_idx:
                sx, sy = free_coords[idx]
                self.flood_grid[sy, sx] = 1

        return self.flood_grid.copy()

    # Advance flood expansion by 1 simulation step.
    # Parameters:
    #   building_grid: 2D array of static building obstacles
    #   wind_state: Optional WindState object biasing expansion direction
    # Returns:
    #   Updated 2D binary numpy array of flooded cells.
    def step(self, building_grid: np.ndarray, wind_state: Optional[WindState] = None) -> np.ndarray:
        new_flood = self.flood_grid.copy()
        
        # Compute wind bias vector if wind state provided
        bias_x, bias_y = 0.0, 0.0
        if wind_state is not None:
            vx, vy = wind_state.wind_vector
            norm = np.sqrt(vx**2 + vy**2) + 1e-6
            bias_x, bias_y = vx / norm, vy / norm

        # 4-neighbor directional offsets (N, S, E, W)
        neighbors = [(0, -1), (0, 1), (1, 0), (-1, 0)]

        for y in range(self.height):
            for x in range(self.width):
                # Only dry airspace cells can flood
                if self.flood_grid[y, x] == 0 and building_grid[y, x] == 0:
                    # Count adjacent flooded neighbors
                    flooded_neighbors = 0
                    wind_advantage = 0.0

                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.flood_grid[ny, nx] == 1:
                                flooded_neighbors += 1
                                # Wind blowing toward (x,y) increases flood probability
                                dot = dx * bias_x + dy * bias_y
                                if dot > 0:
                                    wind_advantage += dot * 0.05

                    if flooded_neighbors > 0:
                        # Expansion probability proportional to flooded neighbors + wind direction
                        p_expand = self.expansion_rate * flooded_neighbors + wind_advantage
                        if self.rng.random() < p_expand:
                            new_flood[y, x] = 1

        self.flood_grid = new_flood
        return self.flood_grid.copy()

    # Check if a specific grid cell (x, y) is currently flooded.
    def is_flooded(self, x: int, y: int) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.flood_grid[y, x] == 1
        return False
