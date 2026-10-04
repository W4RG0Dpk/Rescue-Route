"""
Mission generation for RescueRoute.

Responsibilities:
    - Select a valid benchmark base as the UAV start position.
    - Select a reachable emergency destination.
    - Use an 8-connected / octile distance model.
    - Generate reproducible battery and deadline conditions.
    - Generate standard or fragile payload missions.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..map.map_loader import MapData


# ---------------------------------------------------------------------
# Mission configuration
# ---------------------------------------------------------------------

@dataclass
class MissionConfig:
    """Complete episode configuration for RescueRoute."""

    mission_id: int
    seed: int
    map_name: str

    # -------------------------------------------------------------
    # Mission geometry
    # -------------------------------------------------------------

    start_pos: Tuple[int, int]
    target_pos: Tuple[int, int]

    # -------------------------------------------------------------
    # Mission constraints
    # -------------------------------------------------------------

    initial_battery: float
    deadline_steps: int

    # -------------------------------------------------------------
    # Payload
    # -------------------------------------------------------------

    payload_type: str
    required_stability: float

    # -------------------------------------------------------------
    # Wind scenario
    # -------------------------------------------------------------

    wind_base_speed: float = 5.0
    wind_speed_std: float = 1.5
    wind_correlation: float = 0.85

    wind_gust_probability: float = 0.15
    wind_max_gust: float = 8.0
    wind_min_gust: float = 2.0

    wind_direction_drift_deg: float = 15.0

    # -------------------------------------------------------------
    # Flood / rain scenario
    # -------------------------------------------------------------

    flood_initial_ratio: float = 0.015
    flood_expansion_rate: float = 0.03

    rain_intensity: float = 0.50
    rain_variability: float = 0.15
    rain_correlation: float = 0.85

    flood_wind_influence: float = 0.05
    flood_spontaneous_probability: float = 0.001

    # -------------------------------------------------------------
    # Forecast
    # -------------------------------------------------------------

    forecast_horizon: int = 10
    forecast_reliability: float = 0.80

# ---------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------

def get_connected_components(
    grid_map: np.ndarray,
) -> List[List[Tuple[int, int]]]:
    """
    Find connected traversable regions using 8-connected movement.

    grid_map:
        0 -> traversable
        1 -> blocked

    Returns:
        Components sorted from largest to smallest.
    """

    height, width = grid_map.shape

    visited = set()
    components: List[List[Tuple[int, int]]] = []

    # 8-connected movement because RescueRoute allows diagonal actions.
    directions = [
        (0, -1),   # N
        (0, 1),    # S
        (1, 0),    # E
        (-1, 0),   # W
        (1, -1),   # NE
        (-1, -1),  # NW
        (1, 1),    # SE
        (-1, 1),   # SW
    ]

    for y in range(height):
        for x in range(width):

            if grid_map[y, x] != 0:
                continue

            if (x, y) in visited:
                continue

            component = []
            queue = deque([(x, y)])
            visited.add((x, y))

            while queue:

                cx, cy = queue.popleft()
                component.append((cx, cy))

                for dx, dy in directions:
                    nx = cx + dx
                    ny = cy + dy

                    if not (0 <= nx < width and 0 <= ny < height):
                        continue

                    if grid_map[ny, nx] != 0:
                        continue

                    if (nx, ny) in visited:
                        continue

                    visited.add((nx, ny))
                    queue.append((nx, ny))

            components.append(component)

    components.sort(key=len, reverse=True)

    return components


# ---------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------

def octile_distance(
    a: Tuple[int, int],
    b: Tuple[int, int],
) -> float:
    """
    Octile distance for an 8-connected grid.

    Cardinal movement cost:
        1

    Diagonal movement cost:
        sqrt(2)
    """

    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])

    diagonal = min(dx, dy)
    straight = max(dx, dy) - diagonal

    return straight + math.sqrt(2.0) * diagonal


# ---------------------------------------------------------------------
# Mission generation
# ---------------------------------------------------------------------

def generate_mission(
    map_data: MapData,
    seed: int = 42,
    battery_factor_range: Tuple[float, float] = (1.35, 2.20),
    deadline_factor_range: Tuple[float, float] = (1.35, 2.00),
    fragile_probability: float = 0.50,
) -> MissionConfig:
    """
    Generate one reproducible RescueRoute mission.

    Start:
        Selected from actual blue benchmark base/landing cells.

    Target:
        Selected from traversable cells in the same connected
        component as the chosen base, preferring cells that are
        far from the base.

    Battery/deadline:
        Generated from the minimum geometric distance, with
        controlled variation.

    Payload:
        STANDARD or FRAGILE.
    """

    # -------------------------------------------------------------
    # Validate arguments
    # -------------------------------------------------------------

    if battery_factor_range[0] <= 0:
        raise ValueError("Battery factor must be positive.")

    if deadline_factor_range[0] <= 0:
        raise ValueError("Deadline factor must be positive.")

    if not 0.0 <= fragile_probability <= 1.0:
        raise ValueError(
            "fragile_probability must be between 0 and 1."
        )

    # -------------------------------------------------------------
    # Local RNGs
    #
    # Do NOT modify global random state.
    # -------------------------------------------------------------

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # -------------------------------------------------------------
    # Find connected traversable regions
    # -------------------------------------------------------------

    components = get_connected_components(map_data.grid_map)

    if not components:
        raise ValueError(
            "Map does not contain any traversable cells."
        )

    # -------------------------------------------------------------
    # Find bases that belong to traversable components
    # -------------------------------------------------------------

    valid_bases = []

    for base in map_data.base_locations:
        x, y = base

        if map_data.is_inside(x, y) and not map_data.is_blocked(x, y):
            valid_bases.append(base)

    if not valid_bases:
        raise ValueError(
            "Map contains no valid traversable base/landing cells."
        )

    # -------------------------------------------------------------
    # Select a base
    #
    # We only select bases that belong to a component large enough
    # to support a meaningful delivery mission.
    # -------------------------------------------------------------

    base_to_component = {}

    for component_id, component in enumerate(components):
        component_set = set(component)

        for base in valid_bases:
            if base in component_set:
                base_to_component[base] = component_id

    if not base_to_component:
        raise ValueError(
            "No benchmark base belongs to a traversable "
            "connected component."
        )

    # Prefer a base in the largest component(s).
    largest_component_ids = {
        component_id
        for component_id, component in enumerate(components)
        if len(component) == len(components[0])
    }

    preferred_bases = [
        base
        for base, component_id in base_to_component.items()
        if component_id in largest_component_ids
    ]

    if not preferred_bases:
        preferred_bases = list(base_to_component.keys())

    start_pos = py_rng.choice(preferred_bases)

    start_component_id = base_to_component[start_pos]
    start_component = components[start_component_id]

    # -------------------------------------------------------------
    # Select a destination
    #
    # Prefer distant cells to create meaningful navigation missions.
    # The destination itself must be traversable and cannot be the
    # same cell as the base.
    # -------------------------------------------------------------

    candidate_distances = []

    for cell in start_component:

        if cell == start_pos:
            continue

        distance = octile_distance(start_pos, cell)

        candidate_distances.append((cell, distance))

    if not candidate_distances:
        raise ValueError(
            "The selected base does not have another traversable "
            "cell available for a destination."
        )

    # Sort from farthest to nearest.
    candidate_distances.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    # Select from the farthest ~20% rather than always picking
    # exactly the farthest cell. This preserves mission variety.
    top_count = max(
        1,
        int(math.ceil(len(candidate_distances) * 0.20)),
    )

    top_candidates = [
        cell
        for cell, _ in candidate_distances[:top_count]
    ]

    target_pos = py_rng.choice(top_candidates)

    # -------------------------------------------------------------
    # Geometric mission distance
    # -------------------------------------------------------------

    geometric_distance = octile_distance(
        start_pos,
        target_pos,
    )

    minimum_steps = max(
        1,
        int(math.ceil(geometric_distance)),
    )

    # -------------------------------------------------------------
    # Battery
    #
    # This is a simulation resource, not a physical UAV battery
    # measurement yet.
    #
    # Later the energy model will incorporate:
    #   - cell size
    #   - UAV speed
    #   - wind
    #   - payload
    #   - hover
    #   - recovery
    # -------------------------------------------------------------

    battery_factor = float(
        np_rng.uniform(
            battery_factor_range[0],
            battery_factor_range[1],
        )
    )

    initial_battery = max(
        20.0,
        geometric_distance * 8.0 * battery_factor,
    )

    # -------------------------------------------------------------
    # Deadline
    #
    # Currently based on navigation steps only.
    # Wind-dependent travel time will be incorporated later.
    # -------------------------------------------------------------

    deadline_factor = float(
        np_rng.uniform(
            deadline_factor_range[0],
            deadline_factor_range[1],
        )
    )

    deadline_steps = max(
        minimum_steps + 5,
        int(math.ceil(minimum_steps * deadline_factor)),
    )

    # -------------------------------------------------------------
    # Payload
    # -------------------------------------------------------------

    if py_rng.random() < fragile_probability:
        payload_type = "FRAGILE"
        required_stability = 0.80
    else:
        payload_type = "STANDARD"
        required_stability = 0.40

    # -------------------------------------------------------------
    # Return mission
    # -------------------------------------------------------------

        # -------------------------------------------------------------
    # Generate environment difficulty
    #
    # These are controlled simulation parameters.
    # They are not real-world physical constants.
    # -------------------------------------------------------------

    # Wind severity:
    # approximately 0 = calm, 1 = strong.
    wind_severity = float(
        np_rng.uniform(0.20, 1.00)
    )

    wind_base_speed = float(
        3.0 + 9.0 * wind_severity
    )

    wind_speed_std = float(
        0.8 + 2.5 * wind_severity
    )

    wind_gust_probability = float(
        0.05 + 0.25 * wind_severity
    )

    wind_max_gust = float(
        3.0 + 8.0 * wind_severity
    )

    # -------------------------------------------------------------
    # Flood / rain severity
    # -------------------------------------------------------------

    rain_intensity = float(
        np_rng.uniform(0.15, 0.90)
    )

    rain_variability = float(
        0.05 + 0.20 * rain_intensity
    )

    flood_initial_ratio = float(
        0.005 + 0.025 * rain_intensity
    )

    flood_expansion_rate = float(
        0.015 + 0.045 * rain_intensity
    )

    # -------------------------------------------------------------
    # Forecast reliability
    #
    # We intentionally vary this because forecast uncertainty is
    # part of the research problem.
    # -------------------------------------------------------------

    forecast_reliability = float(
        np_rng.uniform(0.50, 0.95)
    )

    # -------------------------------------------------------------
    # Return complete mission configuration
    # -------------------------------------------------------------

    return MissionConfig(
        mission_id=seed,
        seed=seed,
        map_name=map_data.name,
        start_pos=start_pos,
        target_pos=target_pos,
        initial_battery=float(initial_battery),
        deadline_steps=int(deadline_steps),
        payload_type=payload_type,
        required_stability=float(required_stability),

        # Wind
        wind_base_speed=wind_base_speed,
        wind_speed_std=wind_speed_std,
        wind_correlation=0.85,
        wind_gust_probability=wind_gust_probability,
        wind_max_gust=wind_max_gust,
        wind_min_gust=2.0,
        wind_direction_drift_deg=15.0,

        # Flood
        flood_initial_ratio=flood_initial_ratio,
        flood_expansion_rate=flood_expansion_rate,
        rain_intensity=rain_intensity,
        rain_variability=rain_variability,
        rain_correlation=0.85,
        flood_wind_influence=0.05,
        flood_spontaneous_probability=0.001,

        # Forecast
        forecast_horizon=10,
        forecast_reliability=forecast_reliability,
    )


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------

def generate_random_mission(
    map_data: MapData,
    seed: Optional[int] = None,
) -> MissionConfig:
    """
    Generate a mission using a random seed when one is not supplied.
    """

    if seed is None:
        seed = random.randint(1, 999999)

    return generate_mission(
        map_data,
        seed=seed,
    )