# Mission Generator Module for RescueRoute Environment.
# Defines mission parameters (start location, emergency target, initial battery, time deadline)
# and ensures seeded reproducibility and maximum long-distance target placement across the map grid.

import random
from collections import deque
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from ..map.map_loader import MapData


# Mission Configuration Data Container.
# Holds episode-specific mission goals and physical constraints.
@dataclass
class MissionConfig:
    mission_id: int                     # Unique identifier integer for this mission
    seed: int                           # Random seed for reproducible generation
    map_name: str                       # Map identifier ('manhattan32' or 'urban50')
    start_pos: Tuple[int, int]          # Initial drone deployment coordinate (x, y)
    target_pos: Tuple[int, int]         # Emergency medical delivery destination (x, y)
    initial_battery: float              # Initial drone battery level (e.g. 300.0 energy units)
    deadline_steps: int                 # Maximum allowed flight steps before mission failure
    payload_type: str                   # Type of payload ('STANDARD' or 'FRAGILE')
    required_stability: float          # Required minimum payload stability threshold (e.g. 0.80)


# Extract all connected components of free airspace cells (0) in the grid map using BFS.
def get_connected_components(grid_map: np.ndarray) -> List[List[Tuple[int, int]]]:
    height, width = grid_map.shape
    visited = set()
    components = []

    # 8-directional movement offsets
    directions = [(0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)]

    for y in range(height):
        for x in range(width):
            if grid_map[y, x] == 0 and (x, y) not in visited:
                comp = []
                queue = deque([(x, y)])
                visited.add((x, y))

                while queue:
                    cx, cy = queue.popleft()
                    comp.append((cx, cy))
                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if grid_map[ny, nx] == 0 and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny))

                components.append(comp)

    # Sort components descending by size
    components.sort(key=len, reverse=True)
    return components


# Generate a valid, connected, long-distance mission across the map.
# Maximizes distance between start base and target to challenge the UAV navigation.
# Parameters:
#   map_data: Loaded MapData object
#   seed: Random seed for deterministic scenario creation
# Returns:
#   MissionConfig object ready for environment initialization.
def generate_mission(map_data: MapData, seed: int = 42) -> MissionConfig:
    # Set Python and Numpy random seeds to guarantee exact scenario reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Get connected components of free airspace cells
    components = get_connected_components(map_data.grid_map)

    if not components:
        raise ValueError("Map does not contain any free airspace cells.")

    # Select the primary large connected component (road network)
    primary_comp = components[0]

    # Select start base coordinate from component
    start_pos = random.choice(primary_comp)

    # Calculate Manhattan distance from start_pos to all cells in primary component
    cell_distances = [
        (c, abs(c[0] - start_pos[0]) + abs(c[1] - start_pos[1])) 
        for c in primary_comp
    ]
    # Sort descending by distance
    cell_distances.sort(key=lambda item: item[1], reverse=True)

    # Pick from the top 15% furthest coordinates in the component
    top_furthest_count = max(1, len(cell_distances) // 7)
    top_candidates = [c for c, dist in cell_distances[:top_furthest_count]]
    target_pos = random.choice(top_candidates)

    # Calculate generous battery and deadline budgets for long-distance flight
    dist = abs(target_pos[0] - start_pos[0]) + abs(target_pos[1] - start_pos[1])
    deadline = max(100, int(dist * 4.5))
    initial_battery = max(300.0, float(dist * 8.0))

    return MissionConfig(
        mission_id=seed,
        seed=seed,
        map_name=map_data.name,
        start_pos=start_pos,
        target_pos=target_pos,
        initial_battery=initial_battery,
        deadline_steps=deadline,
        payload_type="FRAGILE",
        required_stability=0.80
    )


# Alias function for random mission creation
def generate_random_mission(map_data: MapData, seed: Optional[int] = None) -> MissionConfig:
    if seed is None:
        seed = random.randint(1, 999999)
    return generate_mission(map_data, seed)
