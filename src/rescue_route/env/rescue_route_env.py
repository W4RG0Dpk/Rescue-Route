# Gymnasium Environment implementation for RescueRoute UAV Emergency Navigation.
# Integrates Phase 3 Dual-World Forecast & Partial Observability Architecture:
# - Hidden sampled ground-truth future vs Probabilistic Forecast Grid
# - Forecast Reliability Parameter (alpha_fc in [0.0, 1.0])
# - Multi-Channel Grid Tensor Observation (5, Height, Width) ready for CNN/RL Policy networks

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from ..map.map_loader import MapData, load_map
from .mission import MissionConfig, generate_mission
from ..weather.wind import WindModel, WindState
from ..hazards.flood import DynamicFloodModel
from ..payload.stability import PayloadStabilityModel, PayloadState
from ..forecast.forecast_model import ForecastModel, ForecastState


# Main RescueRoute Gymnasium Environment class.
class RescueRouteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    # Action constants mapping integer IDs to directional movement vectors (dx, dy)
    ACTIONS = {
        0: (0, -1),   # North (Up)
        1: (0, 1),    # South (Down)
        2: (1, 0),    # East (Right)
        3: (-1, 0),   # West (Left)
        4: (1, -1),   # North-East
        5: (-1, -1),  # North-West
        6: (1, 1),    # South-East
        7: (-1, 1),   # South-West
        8: (0, 0),    # Hover (Stay in place)
    }

    # Base energy drain costs per action type
    BASE_ENERGY_COSTS = {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0,
        4: 1.414, 5: 1.414, 6: 1.414, 7: 1.414,
        8: 0.5,
    }

    # Initialize environment instance with map, mission, and physical/forecast models.
    def __init__(
        self,
        map_name: str = "manhattan32",
        mission: Optional[MissionConfig] = None,
        alpha_fc: float = 0.80,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        # Load grid map dataset (buildings, dimensions, base locations)
        self.map_data: MapData = load_map(map_name)
        self.render_mode = render_mode
        self.alpha_fc = alpha_fc

        # Set or generate mission parameters
        if mission is None:
            self.mission = generate_mission(self.map_data, seed=42)
        else:
            self.mission = mission

        # Initialize Phase 2 & Phase 3 Models
        self.wind_model = WindModel(base_speed=5.0, seed=self.mission.seed)
        self.flood_model = DynamicFloodModel(self.map_data.width, self.map_data.height, seed=self.mission.seed)
        self.payload_model = PayloadStabilityModel(
            payload_type=self.mission.payload_type,
            required_threshold=self.mission.required_stability
        )
        self.forecast_model = ForecastModel(horizon_steps=10, alpha_fc=self.alpha_fc, seed=self.mission.seed)

        # Define Gymnasium Discrete Action Space (9 discrete choices)
        self.action_space = spaces.Discrete(9)

        # Define Gymnasium Multi-Channel Tensor Observation Space (5, Height, Width)
        self.observation_space = spaces.Dict({
            "spatial_tensor": spaces.Box(
                low=0.0, high=1.0, 
                shape=(5, self.map_data.height, self.map_data.width), 
                dtype=np.float32
            ),
            "vector_state": spaces.Box(low=-50.0, high=50.0, shape=(7,), dtype=np.float32),
        })

        # Dynamic state variables
        self.drone_pos: Tuple[int, int] = self.mission.start_pos
        self.current_battery: float = self.mission.initial_battery
        self.current_step: int = 0
        self.trajectory: list = []
        self.last_move_vector: Tuple[int, int] = (0, 0)
        self.current_forecast: Optional[ForecastState] = None

    # Reset environment state.
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        if options and "mission" in options:
            self.mission = options["mission"]
        elif seed is not None:
            self.mission = generate_mission(self.map_data, seed=seed)

        # Reset state variables
        self.drone_pos = self.mission.start_pos
        self.current_battery = self.mission.initial_battery
        self.current_step = 0
        self.trajectory = [self.drone_pos]
        self.last_move_vector = (0, 0)

        # Reset models (protect start base and target from initial flood epicenters)
        eff_seed = seed if seed is not None else self.mission.seed
        self.wind_model.reset(seed=eff_seed)
        self.flood_model.reset(
            self.map_data.grid_map, 
            safe_coords=[self.mission.start_pos, self.mission.target_pos], 
            seed=eff_seed
        )
        self.payload_model.reset(payload_type=self.mission.payload_type, threshold=self.mission.required_stability)
        self.forecast_model.reset(alpha_fc=self.alpha_fc, seed=eff_seed)

        # Generate initial forecast
        current_wind = self.wind_model.get_state()
        self.current_forecast = self.forecast_model.generate_forecast(
            self.flood_model.flood_grid,
            self.map_data.grid_map,
            current_wind
        )

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # Execute simulation step.
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        dx, dy = self.ACTIONS[action]

        # 1. Step Wind simulation
        wind_state = self.wind_model.step()

        # 2. Step Flood expansion simulation
        flood_grid = self.flood_model.step(self.map_data.grid_map, wind_state)

        # 3. Step Forecast generation
        self.current_forecast = self.forecast_model.generate_forecast(
            flood_grid,
            self.map_data.grid_map,
            wind_state
        )

        # 4. Calculate wind-adjusted energy drain cost
        base_cost = self.BASE_ENERGY_COSTS[action]
        wind_mult = self.wind_model.get_energy_multiplier((dx, dy))
        energy_used = base_cost * wind_mult

        # Drain battery
        self.current_battery = max(0.0, self.current_battery - energy_used)

        # 5. Step Payload stability simulation
        payload_state = self.payload_model.step(
            action=action,
            move_vector=(dx, dy),
            prev_move_vector=self.last_move_vector,
            wind_state=wind_state
        )

        # Candidate position
        new_x = self.drone_pos[0] + dx
        new_y = self.drone_pos[1] + dy

        terminated = False
        truncated = False
        reward = -0.1

        # Check map boundary collision
        out_of_bounds = (
            new_x < 0 or new_x >= self.map_data.width or 
            new_y < 0 or new_y >= self.map_data.height
        )

        if out_of_bounds:
            reward = -50.0
            terminated = True
            collision_type = "BOUNDARY_CRASH"
        else:
            # Check building obstacle collision & flood hazard collision
            is_building = (self.map_data.grid_map[new_y, new_x] == 1)
            is_flooded = self.flood_model.is_flooded(new_x, new_y)

            if is_building:
                reward = -50.0
                terminated = True
                collision_type = "BUILDING_CRASH"
            elif is_flooded:
                reward = -30.0
                terminated = True
                collision_type = "FLOOD_HAZARD"
            else:
                # Valid move into free airspace
                old_dist = abs(self.drone_pos[0] - self.mission.target_pos[0]) + abs(self.drone_pos[1] - self.mission.target_pos[1])
                new_dist = abs(new_x - self.mission.target_pos[0]) + abs(new_y - self.mission.target_pos[1])

                # Progress reward
                reward += (old_dist - new_dist) * 1.5

                # Payload degradation penalty
                if payload_state.stability < 1.0:
                    reward -= (1.0 - payload_state.stability) * 0.5

                self.drone_pos = (new_x, new_y)
                self.trajectory.append(self.drone_pos)
                self.last_move_vector = (dx, dy)
                collision_type = "NONE"

                # Check target delivery arrival
                if self.drone_pos == self.mission.target_pos:
                    if payload_state.is_damaged:
                        reward += 30.0
                    else:
                        reward += 100.0
                    terminated = True

        # Check truncation conditions
        if not terminated:
            if self.current_battery <= 0.0:
                reward -= 50.0
                truncated = True
            elif self.current_step >= self.mission.deadline_steps:
                reward -= 50.0
                truncated = True

        obs = self._get_obs()
        info = self._get_info()
        info["collision_type"] = collision_type
        info["energy_used"] = energy_used
        info["wind_speed"] = wind_state.speed
        info["payload_stability"] = payload_state.stability
        info["is_payload_damaged"] = payload_state.is_damaged
        info["alpha_fc"] = self.alpha_fc

        return obs, reward, terminated, truncated, info

    # Format multi-channel tensor observation (5, Height, Width) & vector state for RL policy.
    def _get_obs(self) -> Dict[str, Any]:
        h, w = self.map_data.height, self.map_data.width
        spatial_tensor = np.zeros((5, h, w), dtype=np.float32)

        # Channel 0: Building obstacle map (1 = BUILDING, 0 = FREE)
        spatial_tensor[0] = self.map_data.grid_map.astype(np.float32)

        # Channel 1: Current drone position map
        dx, dy = self.drone_pos
        if 0 <= dx < w and 0 <= dy < h:
            spatial_tensor[1, dy, dx] = 1.0

        # Channel 2: Emergency target position map
        tx, ty = self.mission.target_pos
        if 0 <= tx < w and 0 <= ty < h:
            spatial_tensor[2, ty, tx] = 1.0

        # Channel 3: Current ground-truth flood mask (1 = FLOODED)
        spatial_tensor[3] = self.flood_model.flood_grid.astype(np.float32)

        # Channel 4: Probabilistic forecast flood map ([0.0, 1.0])
        if self.current_forecast is not None:
            spatial_tensor[4] = self.current_forecast.flood_prob_map.astype(np.float32)

        # Vector state (7 parameters)
        norm_battery = np.float32(self.current_battery / max(1.0, self.mission.initial_battery))
        rem_steps = max(0, self.mission.deadline_steps - self.current_step)
        norm_deadline = np.float32(rem_steps / max(1, self.mission.deadline_steps))

        wind_state = self.wind_model.get_state()
        vx, vy = wind_state.wind_vector
        payload_state = self.payload_model.get_state()

        vector_state = np.array([
            norm_battery,
            norm_deadline,
            vx, vy,
            np.float32(payload_state.stability),
            np.float32(self.alpha_fc),
            np.float32(self.current_step / max(1, self.mission.deadline_steps))
        ], dtype=np.float32)

        return {
            "spatial_tensor": spatial_tensor,
            "vector_state": vector_state
        }

    # Metadata summary.
    def _get_info(self) -> Dict[str, Any]:
        dist_to_target = abs(self.drone_pos[0] - self.mission.target_pos[0]) + abs(self.drone_pos[1] - self.mission.target_pos[1])
        wind_state = self.wind_model.get_state()
        payload_state = self.payload_model.get_state()

        return {
            "step": self.current_step,
            "drone_pos": self.drone_pos,
            "target_pos": self.mission.target_pos,
            "distance_to_target": dist_to_target,
            "battery_remaining": self.current_battery,
            "wind_speed": wind_state.speed,
            "wind_direction": wind_state.direction_deg,
            "payload_stability": payload_state.stability,
            "alpha_fc": self.alpha_fc,
            "flooded_cells_count": int(np.sum(self.flood_model.flood_grid == 1)),
            "trajectory_length": len(self.trajectory)
        }
