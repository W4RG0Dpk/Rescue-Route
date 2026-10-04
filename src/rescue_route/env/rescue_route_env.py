"""
RescueRoute Gymnasium Environment
=================================

Single-UAV emergency medical navigation environment.

Current environment responsibilities:
    - Static benchmark map
    - Emergency mission
    - Battery and deadline
    - Stochastic wind
    - Dynamic flooding
    - Payload stability
    - Probabilistic forecast observation
    - Local partial observation
    - Reproducible seeded episodes

Important architecture rule:
    The agent receives the CURRENT local environment + forecast.

    The agent does NOT receive:
        - the full map state
        - the full current flood map
        - future wind trajectory
        - future flood trajectory

Forecast/truth separation will be strengthened further when the
forecast module is refactored in the next environment stage.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..map.map_loader import MapData, load_map
from .mission import (
    MissionConfig,
    generate_mission,
    octile_distance,
)
from ..weather.wind import WindModel, WindState
from ..hazards.flood import DynamicFloodModel
from ..payload.stability import (
    PayloadStabilityModel,
    PayloadState,
)
from ..forecast.forecast_model import (
    ForecastModel,
    ForecastState,
)


class RescueRouteEnv(gym.Env):
    """
    RescueRoute emergency UAV navigation environment.

    The UAV must:
        1. Reach the emergency destination.
        2. Reach it before the delivery deadline.
        3. Preserve payload validity.
        4. Avoid static and dynamic hazards.
        5. Manage battery.

    Action space:
        0 = North
        1 = South
        2 = East
        3 = West
        4 = North-East
        5 = North-West
        6 = South-East
        7 = South-West
        8 = Hover

    PPO / RL policy later receives:
        - local spatial map
        - normalized scalar state
        - forecast information
    """

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 10,
    }

    # ------------------------------------------------------------------
    # Action definitions
    # ------------------------------------------------------------------

    ACTIONS = {
        0: (0, -1),    # North
        1: (0, 1),     # South
        2: (1, 0),     # East
        3: (-1, 0),    # West
        4: (1, -1),    # North-East
        5: (-1, -1),   # North-West
        6: (1, 1),     # South-East
        7: (-1, 1),    # South-West
        8: (0, 0),     # Hover
    }

    # ------------------------------------------------------------------
    # Base energy costs
    #
    # These are simulator units for now.
    # They will later be tied more explicitly to physical distance,
    # UAV speed and the calibrated energy model.
    # ------------------------------------------------------------------

    BASE_ENERGY_COSTS = {
        0: 1.0,
        1: 1.0,
        2: 1.0,
        3: 1.0,
        4: np.sqrt(2.0),
        5: np.sqrt(2.0),
        6: np.sqrt(2.0),
        7: np.sqrt(2.0),
        8: 0.5,
    }

    # ------------------------------------------------------------------
    # Reward parameters
    # ------------------------------------------------------------------

    STEP_PENALTY = -0.05
    PROGRESS_REWARD_SCALE = 1.5

    ENERGY_PENALTY_SCALE = 0.02
    STABILITY_PENALTY_SCALE = 0.50

    DELIVERY_REWARD = 100.0

    COLLISION_PENALTY = -50.0
    FLOOD_PENALTY = -30.0
    BATTERY_FAILURE_PENALTY = -50.0
    DEADLINE_FAILURE_PENALTY = -50.0
    PAYLOAD_FAILURE_PENALTY = -40.0

    # ------------------------------------------------------------------
    # Observation configuration
    # ------------------------------------------------------------------

    # Local observation around UAV.
    #
    # 17x17 gives a useful local neighborhood while still preserving
    # partial observability.
    LOCAL_VIEW_SIZE = 17

    # Half-width for the centered crop.
    LOCAL_VIEW_RADIUS = LOCAL_VIEW_SIZE // 2

    # Number of spatial channels.
    #
    # 0 = blocked cells
    # 1 = NFZ
    # 2 = fly-over buildings
    # 3 = UAV
    # 4 = destination
    # 5 = current local flood
    # 6 = forecast flood probability
    NUM_SPATIAL_CHANNELS = 7

    # Vector observation:
    #
    # 0  battery normalized
    # 1  remaining deadline normalized
    # 2  goal dx normalized
    # 3  goal dy normalized
    # 4  goal distance normalized
    # 5  heading dx
    # 6  heading dy
    # 7  wind speed normalized
    # 8  wind direction x
    # 9  wind direction y
    # 10 payload stability
    # 11 required stability
    # 12 payload type (0 standard, 1 fragile)
    # 13 forecast reliability
    # 14 normalized current timestep
    VECTOR_STATE_SIZE = 15

    # Wind normalization is a simulator scaling parameter.
    # It is not a claimed real-world maximum wind speed.
    WIND_SPEED_NORMALIZATION = 15.0

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        map_name: str = "manhattan32",
        mission: Optional[MissionConfig] = None,
        alpha_fc: Optional[float] = None,
        render_mode: Optional[str] = None,
        local_view_size: int = LOCAL_VIEW_SIZE,
    ):
        super().__init__()

        # --------------------------------------------------------------
        # Validate local-view size
        # --------------------------------------------------------------

        if local_view_size <= 0:
            raise ValueError("local_view_size must be positive.")

        if local_view_size % 2 == 0:
            raise ValueError(
                "local_view_size must be odd, e.g. 11, 15 or 17."
            )

        self.map_data: MapData = load_map(map_name)

        self.render_mode = render_mode

        # Optional experiment override.
        #
        # None:
        #     use mission.forecast_reliability
        #
        # Numeric value:
        #     explicitly override the mission reliability for controlled
        #     experiments.
        self._alpha_fc_override = (
            None
            if alpha_fc is None
            else float(alpha_fc)
        )

        if (
            self._alpha_fc_override is not None
            and not 0.0 <= self._alpha_fc_override <= 1.0
        ):
            raise ValueError(
                "alpha_fc must be between 0.0 and 1.0."
            )

        if (
            self._alpha_fc_override is not None
            and not 0.0 <= self._alpha_fc_override <= 1.0
        ):
            raise ValueError(
                "alpha_fc must be between 0.0 and 1.0."
            )

        self.local_view_size = int(local_view_size)
        self.local_view_radius = self.local_view_size // 2

        # --------------------------------------------------------------
        # Mission handling
        #
        # If mission is supplied explicitly, keep using that mission.
        #
        # If mission is not supplied, every reset() without an explicit
        # seed generates a new reproducible episode from Gymnasium's RNG.
        # --------------------------------------------------------------

        self.fixed_mission = mission is not None

        if mission is not None:
            self.mission = mission
        else:
            self.mission = generate_mission(
                self.map_data,
                seed=42,
            )

        # --------------------------------------------------------------
        # Physical/environment models
        #
        # Wind configuration will later be passed from MissionConfig.
        # For now preserve the existing model API.
        # --------------------------------------------------------------

        # --------------------------------------------------------------
# Initialize models using the complete mission configuration.
# --------------------------------------------------------------

        self._initialize_episode_models()

        # --------------------------------------------------------------
        # Gym action space
        # --------------------------------------------------------------

        self.action_space = spaces.Discrete(9)

        # --------------------------------------------------------------
        # Gym observation space
        # --------------------------------------------------------------

        self.observation_space = spaces.Dict(
            {
                "spatial_tensor": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        self.NUM_SPATIAL_CHANNELS,
                        self.local_view_size,
                        self.local_view_size,
                    ),
                    dtype=np.float32,
                ),

                # We normalize each scalar into [-1, 1].
                "vector_state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.VECTOR_STATE_SIZE,),
                    dtype=np.float32,
                ),
            }
        )

        # --------------------------------------------------------------
        # Dynamic episode state
        # --------------------------------------------------------------

        self.drone_pos: Tuple[int, int] = self.mission.start_pos

        self.current_battery: float = (
            float(self.mission.initial_battery)
        )

        self.current_step: int = 0

        self.trajectory = []

        # Heading is maintained separately from movement.
        #
        # Hover does NOT erase the UAV's previous heading.
        self.heading: Tuple[int, int] = (0, 0)

        self.current_forecast: Optional[ForecastState] = None

        self.episode_seed: Optional[int] = None

        self.last_action: Optional[int] = None

        self.last_collision_type: str = "NONE"

        self.terminated_reason: Optional[str] = None

    # ==================================================================
    # RESET
    # ==================================================================
    
    
    def _initialize_episode_models(self) -> None:
        """
        Create all stochastic/environment models from the current mission.

        This ensures that the mission completely defines the episode's
        environmental conditions.

        Mission controls:
            - wind severity/variability/gusts
            - flood severity
            - rain intensity
            - forecast horizon/reliability

        The models are recreated at episode reset so a new mission cannot
        accidentally inherit parameters from the previous episode.
        """

        # --------------------------------------------------------------
        # Forecast reliability
        # --------------------------------------------------------------

        if self._alpha_fc_override is None:
            self.alpha_fc = float(
                self.mission.forecast_reliability
            )
        else:
            self.alpha_fc = float(
                self._alpha_fc_override
            )

        # --------------------------------------------------------------
        # Wind
        # --------------------------------------------------------------

        self.wind_model = WindModel(
            base_speed=self.mission.wind_base_speed,
            speed_std=self.mission.wind_speed_std,
            correlation_rho=self.mission.wind_correlation,
            gust_probability=self.mission.wind_gust_probability,
            max_gust=self.mission.wind_max_gust,
            min_gust=self.mission.wind_min_gust,
            direction_drift_deg=(
                self.mission.wind_direction_drift_deg
            ),
            seed=self.mission.seed,
        )

        # --------------------------------------------------------------
        # Flood
        # --------------------------------------------------------------

        self.flood_model = DynamicFloodModel(
            self.map_data.width,
            self.map_data.height,

            initial_flood_ratio=(
                self.mission.flood_initial_ratio
            ),

            expansion_rate=(
                self.mission.flood_expansion_rate
            ),

            rain_intensity=(
                self.mission.rain_intensity
            ),

            rain_variability=(
                self.mission.rain_variability
            ),

            rain_correlation=(
                self.mission.rain_correlation
            ),

            wind_influence=(
                self.mission.flood_wind_influence
            ),

            spontaneous_flood_probability=(
                self.mission.flood_spontaneous_probability
            ),

            seed=self.mission.seed,
        )

        # --------------------------------------------------------------
        # Payload
        # --------------------------------------------------------------

        self.payload_model = PayloadStabilityModel(
            payload_type=self.mission.payload_type,
            required_threshold=(
                self.mission.required_stability
            ),
        )

        # --------------------------------------------------------------
        # Forecast
        # --------------------------------------------------------------

        self.forecast_model = ForecastModel(
            horizon_steps=(
                self.mission.forecast_horizon
            ),

            alpha_fc=self.alpha_fc,

            flood_expansion_rate=(
                self.mission.flood_expansion_rate
            ),

            wind_influence=(
                self.mission.flood_wind_influence
            ),

            spontaneous_flood_probability=(
                self.mission.flood_spontaneous_probability
            ),

            rain_intensity=(
                self.mission.rain_intensity
            ),

            wind_rho=self.mission.wind_correlation,

            wind_base_speed=(
                self.mission.wind_base_speed
            ),

            seed=self.mission.seed,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset one RescueRoute episode.

        Seed behavior:
            reset(seed=X)
                -> deterministic mission/environment using X.

            reset()
                -> when no fixed mission is supplied, generate a new
                   episode seed from Gymnasium's internal RNG.

            options={"mission": mission}
                -> directly use the supplied mission.
        """

        super().reset(seed=seed)

        # --------------------------------------------------------------
        # Determine episode mission
        # --------------------------------------------------------------

        if options is not None and "mission" in options:
            self.mission = options["mission"]

        elif not self.fixed_mission:
            if seed is not None:
                episode_seed = int(seed)
            else:
                # Generate a fresh deterministic seed from Gymnasium's
                # internal RNG.
                episode_seed = int(
                    self.np_random.integers(
                        0,
                        np.iinfo(np.int32).max,
                    )
                )

            self.mission = generate_mission(
                self.map_data,
                seed=episode_seed,
            )
        # --------------------------------------------------------------
        # Validate forecast reliability from mission.
        # --------------------------------------------------------------

        if not (
            0.0
            <= self.mission.forecast_reliability
            <= 1.0
        ):
            raise ValueError(
                "Mission forecast_reliability must be "
                "between 0 and 1."
            )

        # If fixed_mission == True and no new mission/seed is supplied,
        # continue using the explicitly provided mission.

        self.episode_seed = int(self.mission.seed)

        # --------------------------------------------------------------
        # Reset dynamic state
        # --------------------------------------------------------------

        self.drone_pos = self.mission.start_pos

        self.current_battery = float(
            self.mission.initial_battery
        )

        self.current_step = 0

        self.trajectory = [
            self.drone_pos
        ]

        self.heading = (0, 0)

        self.last_action = None

        self.last_collision_type = "NONE"

        self.terminated_reason = None

        # --------------------------------------------------------------
        # Reset all environment models
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # Rebuild all episode models from the current mission.
        #
        # This guarantees that the scenario parameters belong to this
        # episode and are not inherited from the previous episode.
        # --------------------------------------------------------------

        self._initialize_episode_models()

        eff_seed = self.episode_seed

        # --------------------------------------------------------------
        # Reset wind
        # --------------------------------------------------------------

        self.wind_model.reset(
            seed=eff_seed,
        )

        # --------------------------------------------------------------
        # Reset flood
        # --------------------------------------------------------------

        self.flood_model.reset(
            self.map_data.grid_map,

            safe_coords=[
                self.mission.start_pos,
                self.mission.target_pos,
            ],

            seed=eff_seed,

            rain_intensity=(
                self.mission.rain_intensity
            ),

            rain_variability=(
                self.mission.rain_variability
            ),
        )

        # --------------------------------------------------------------
        # Reset payload
        # --------------------------------------------------------------

        self.payload_model.reset(
            payload_type=(
                self.mission.payload_type
            ),

            threshold=(
                self.mission.required_stability
            ),
        )

        # --------------------------------------------------------------
        # Reset forecast
        # --------------------------------------------------------------

        self.forecast_model.reset(
            alpha_fc=self.alpha_fc,
            seed=eff_seed,
            rain_intensity=(
                self.flood_model.get_rain_intensity()
            ),
        )
        # --------------------------------------------------------------
        # Generate initial forecast.
        #
        # The agent only receives the forecast representation in the
        # observation. The sampled future remains hidden.
        # --------------------------------------------------------------

        current_wind = self.wind_model.get_state()

        self.current_forecast = (
            self.forecast_model.generate_forecast(
                self.flood_model.flood_grid,
                self.map_data.grid_map,
                current_wind,
                current_rain_intensity=(
                    self.flood_model.get_rain_intensity()
                ),
            )
        )

        # --------------------------------------------------------------
        # Build first observation
        # --------------------------------------------------------------

        obs = self._get_obs()
        info = self._get_info()

        info["episode_seed"] = self.episode_seed

        return obs, info

    # ==================================================================
    # STEP
    # ==================================================================

    def step(
        self,
        action: int,
    ) -> Tuple[
        Dict[str, Any],
        float,
        bool,
        bool,
        Dict[str, Any],
    ]:
        """
        Execute one environment action.

        Timing convention:

            observation at time t
                    ↓
                action a_t
                    ↓
            action consequences
                    ↓
            energy / time / payload
                    ↓
            movement + immediate hazard checks
                    ↓
            environment advances to t+1
                    ↓
            next forecast
                    ↓
            next observation
        """

        # --------------------------------------------------------------
        # Validate action
        # --------------------------------------------------------------

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. "
                f"Expected integer in [0, 8]."
            )

        action = int(action)

        self.last_action = action

        dx, dy = self.ACTIONS[action]

        # --------------------------------------------------------------
        # Current state before action
        # --------------------------------------------------------------

        old_position = self.drone_pos

        old_distance = octile_distance(
            old_position,
            self.mission.target_pos,
        )

        wind_state: WindState = (
            self.wind_model.get_state()
        )

        # --------------------------------------------------------------
        # 1. Energy calculation
        #
        # Energy uses CURRENT wind because action a_t is selected using
        # the current observation at time t.
        # --------------------------------------------------------------

        base_cost = self.BASE_ENERGY_COSTS[action]

        wind_multiplier = self.wind_model.get_energy_multiplier(
            (dx, dy)
        )

        energy_used = float(
            base_cost * wind_multiplier
        )

        self.current_battery = max(
            0.0,
            self.current_battery - energy_used,
        )

        # --------------------------------------------------------------
        # 2. Time advances by one simulation step
        # --------------------------------------------------------------

        self.current_step += 1

        # --------------------------------------------------------------
        # 3. Payload stability update
        #
        # IMPORTANT:
        # Hover does not erase the previous heading.
        # --------------------------------------------------------------

        payload_state: PayloadState = (
            self.payload_model.step(
                action=action,
                move_vector=(dx, dy),
                prev_move_vector=self.heading,
                wind_state=wind_state,
            )
        )

        # --------------------------------------------------------------
        # Reward starts with small time/step cost
        # --------------------------------------------------------------

        reward = float(self.STEP_PENALTY)

        # --------------------------------------------------------------
        # 4. Calculate candidate position
        # --------------------------------------------------------------

        new_x = old_position[0] + dx
        new_y = old_position[1] + dy

        terminated = False
        truncated = False

        collision_type = "NONE"

        # --------------------------------------------------------------
        # 5. Boundary check
        # --------------------------------------------------------------

        out_of_bounds = (
            new_x < 0
            or new_x >= self.map_data.width
            or new_y < 0
            or new_y >= self.map_data.height
        )

        if out_of_bounds:

            reward += self.COLLISION_PENALTY

            terminated = True

            collision_type = "BOUNDARY_CRASH"

            self.terminated_reason = collision_type

        else:

            # ----------------------------------------------------------
            # 6. Static map check
            # ----------------------------------------------------------

            is_blocked = bool(
                self.map_data.blocked_mask[
                    new_y,
                    new_x,
                ]
            )

            is_nfz = bool(
                self.map_data.nfz_mask[
                    new_y,
                    new_x,
                ]
            )

            # ----------------------------------------------------------
            # 7. Current flood check
            #
            # This is the environmental state BEFORE it advances to
            # the next timestep.
            # ----------------------------------------------------------

            is_flooded = bool(
                self.flood_model.is_flooded(
                    new_x,
                    new_y,
                )
            )

            # ----------------------------------------------------------
            # 8. Diagonal corner-cutting prevention
            #
            # Example:
            #
            #     X .
            #     . U
            #
            # NE movement must not pass diagonally between two blocked
            # cells.
            # ----------------------------------------------------------

            diagonal_corner_blocked = False
            diagonal_corner_flooded = False

            if dx != 0 and dy != 0:

                side_a_x = old_position[0] + dx
                side_a_y = old_position[1]

                side_b_x = old_position[0]
                side_b_y = old_position[1] + dy

                side_a_blocked = (
                    not self.map_data.is_inside(
                        side_a_x,
                        side_a_y,
                    )
                    or self.map_data.is_blocked(
                        side_a_x,
                        side_a_y,
                    )
                )

                side_b_blocked = (
                    not self.map_data.is_inside(
                        side_b_x,
                        side_b_y,
                    )
                    or self.map_data.is_blocked(
                        side_b_x,
                        side_b_y,
                    )
                )

                diagonal_corner_blocked = (
                    side_a_blocked or side_b_blocked
                )

                if not side_a_blocked:
                    diagonal_corner_flooded |= (
                        self.flood_model.is_flooded(
                            side_a_x,
                            side_a_y,
                        )
                    )

                if not side_b_blocked:
                    diagonal_corner_flooded |= (
                        self.flood_model.is_flooded(
                            side_b_x,
                            side_b_y,
                        )
                    )

            # ----------------------------------------------------------
            # 9. Collision / immediate hazard handling
            # ----------------------------------------------------------

            if is_blocked:

                reward += self.COLLISION_PENALTY

                terminated = True

                collision_type = (
                    "NFZ_VIOLATION"
                    if is_nfz
                    else "STATIC_BLOCKED"
                )

                self.terminated_reason = collision_type

            elif diagonal_corner_blocked:

                reward += self.COLLISION_PENALTY

                terminated = True

                collision_type = (
                    "DIAGONAL_CORNER_COLLISION"
                )

                self.terminated_reason = collision_type

            elif is_flooded:

                reward += self.FLOOD_PENALTY

                terminated = True

                collision_type = "FLOOD_HAZARD"

                self.terminated_reason = collision_type

            elif diagonal_corner_flooded:

                reward += self.FLOOD_PENALTY

                terminated = True

                collision_type = (
                    "DIAGONAL_FLOOD_HAZARD"
                )

                self.terminated_reason = collision_type

            else:

                # ------------------------------------------------------
                # 10. Valid movement
                # ------------------------------------------------------

                self.drone_pos = (
                    new_x,
                    new_y,
                )

                self.trajectory.append(
                    self.drone_pos
                )

                # ------------------------------------------------------
                # Update heading ONLY for movement.
                #
                # Hover keeps the previous heading.
                # ------------------------------------------------------

                if dx != 0 or dy != 0:
                    self.heading = (
                        int(np.sign(dx)),
                        int(np.sign(dy)),
                    )

                collision_type = "NONE"

                # ------------------------------------------------------
                # 11. Progress reward
                # ------------------------------------------------------

                new_distance = octile_distance(
                    self.drone_pos,
                    self.mission.target_pos,
                )

                progress = old_distance - new_distance

                reward += (
                    progress
                    * self.PROGRESS_REWARD_SCALE
                )

                # ------------------------------------------------------
                # 12. Stability penalty
                # ------------------------------------------------------

                stability_loss = max(
                    0.0,
                    1.0 - float(
                        payload_state.stability
                    ),
                )

                reward -= (
                    stability_loss
                    * self.STABILITY_PENALTY_SCALE
                )

                # ------------------------------------------------------
                # 13. Energy penalty
                # ------------------------------------------------------

                reward -= (
                    energy_used
                    * self.ENERGY_PENALTY_SCALE
                )

        # =================================================================
        # 14. Delivery check
        # =================================================================

        if not terminated:

            if self.drone_pos == self.mission.target_pos:

                # ------------------------------------------------------
                # Delivery is successful ONLY when:
                #
                #   - destination reached
                #   - deadline not exceeded
                #   - payload is still valid
                #
                # Recovery is intentionally NOT checked here.
                # Recovery will be added in the later recoverability
                # stage.
                # ------------------------------------------------------

                on_time = (
                    self.current_step
                    <= self.mission.deadline_steps
                )

                payload_valid = (
                    not payload_state.is_damaged
                )

                if on_time and payload_valid:

                    reward += self.DELIVERY_REWARD

                    terminated = True

                    self.terminated_reason = (
                        "SUCCESSFUL_DELIVERY"
                    )

                elif not on_time:

                    reward += (
                        self.DEADLINE_FAILURE_PENALTY
                    )

                    terminated = True

                    collision_type = (
                        "DEADLINE_EXCEEDED"
                    )

                    self.terminated_reason = (
                        "DEADLINE_EXCEEDED"
                    )

                else:

                    reward += (
                        self.PAYLOAD_FAILURE_PENALTY
                    )

                    terminated = True

                    collision_type = (
                        "PAYLOAD_FAILURE"
                    )

                    self.terminated_reason = (
                        "PAYLOAD_FAILURE"
                    )

        # =================================================================
        # 15. Battery failure
        # =================================================================

        if not terminated:

            if self.current_battery <= 0.0:

                reward += (
                    self.BATTERY_FAILURE_PENALTY
                )

                terminated = True

                self.terminated_reason = (
                    "BATTERY_DEPLETED"
                )

        # =================================================================
        # 16. Deadline failure
        # =================================================================

        if not terminated:

            if (
                self.current_step
                >= self.mission.deadline_steps
            ):

                reward += (
                    self.DEADLINE_FAILURE_PENALTY
                )

                terminated = True

                self.terminated_reason = (
                    "DEADLINE_EXCEEDED"
                )

        # =================================================================
        # 17. Advance dynamic environment
        #
        # We do this ONLY if the mission did not already end.
        # =================================================================

        next_wind_state = wind_state

        if not terminated:

            # ----------------------------------------------------------
            # Wind advances to t+1
            # ----------------------------------------------------------

            next_wind_state = (
                self.wind_model.step()
            )

            # ----------------------------------------------------------
            # Flood advances to t+1
            # ----------------------------------------------------------

            next_flood_grid = (
                self.flood_model.step(
                    self.map_data.grid_map,
                    next_wind_state,
                )
            )

            # ----------------------------------------------------------
            # If flood evolution makes the UAV's current cell unsafe,
            # terminate the mission.
            # ----------------------------------------------------------

            current_x, current_y = self.drone_pos

            if self.flood_model.is_flooded(
                current_x,
                current_y,
            ):

                reward += self.FLOOD_PENALTY

                terminated = True

                collision_type = (
                    "FLOOD_ENGULFMENT"
                )

                self.terminated_reason = (
                    "FLOOD_ENGULFMENT"
                )

        # =================================================================
        # 18. Generate next forecast
        # =================================================================

        if not terminated:

            self.current_forecast = (
                self.forecast_model.generate_forecast(
                    self.flood_model.flood_grid,
                    self.map_data.grid_map,
                    next_wind_state,
                    current_rain_intensity=(
                        self.flood_model.get_rain_intensity()
                    ),
                )
            )

        # =================================================================
        # 19. Final observation + info
        # =================================================================

        self.last_collision_type = collision_type

        obs = self._get_obs()

        info = self._get_info()

        info["collision_type"] = collision_type

        info["energy_used"] = float(
            energy_used
        )

        info["wind_speed"] = float(
            next_wind_state.speed
        )

        info["wind_direction"] = float(
            next_wind_state.direction_deg
        )

        info["payload_stability"] = float(
            payload_state.stability
        )

        info["is_payload_damaged"] = bool(
            payload_state.is_damaged
        )

        info["alpha_fc"] = float(
            self.alpha_fc
        )

        info["terminated_reason"] = (
            self.terminated_reason
        )

        info["is_success"] = (
            self.terminated_reason
            == "SUCCESSFUL_DELIVERY"
        )

        info["remaining_deadline"] = max(
            0,
            self.mission.deadline_steps
            - self.current_step,
        )

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ==================================================================
    # LOCAL OBSERVATION HELPERS
    # ==================================================================

    def _extract_local(
        self,
        full_grid: np.ndarray,
        center_x: int,
        center_y: int,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        """
        Extract an odd-sized local window centered at (center_x, center_y).

        Outside-map cells are padded with pad_value.

        This prevents the policy from receiving the complete map.
        """

        radius = self.local_view_radius
        size = self.local_view_size

        output = np.full(
            (size, size),
            pad_value,
            dtype=np.float32,
        )

        min_x = center_x - radius
        max_x = center_x + radius

        min_y = center_y - radius
        max_y = center_y + radius

        src_x0 = max(0, min_x)
        src_x1 = min(
            self.map_data.width - 1,
            max_x,
        )

        src_y0 = max(0, min_y)
        src_y1 = min(
            self.map_data.height - 1,
            max_y,
        )

        if src_x0 > src_x1 or src_y0 > src_y1:
            return output

        dst_x0 = src_x0 - min_x
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        dst_y0 = src_y0 - min_y
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        output[
            dst_y0 : dst_y1 + 1,
            dst_x0 : dst_x1 + 1,
        ] = full_grid[
            src_y0 : src_y1 + 1,
            src_x0 : src_x1 + 1,
        ].astype(np.float32)

        return output

    def _get_local_spatial_tensor(
        self,
    ) -> np.ndarray:
        """
        Construct the local multi-channel spatial observation.

        Channels:
            0 -> blocked
            1 -> NFZ
            2 -> fly-over building
            3 -> UAV
            4 -> destination
            5 -> current flood
            6 -> forecast flood probability
        """

        h = self.map_data.height
        w = self.map_data.width

        x, y = self.drone_pos

        spatial = np.zeros(
            (
                self.NUM_SPATIAL_CHANNELS,
                self.local_view_size,
                self.local_view_size,
            ),
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Channel 0: hard blocked cells
        #
        # Outside map is also treated as blocked.
        # --------------------------------------------------------------

        spatial[0] = self._extract_local(
            self.map_data.blocked_mask,
            x,
            y,
            pad_value=1.0,
        )

        # --------------------------------------------------------------
        # Channel 1: NFZ
        # --------------------------------------------------------------

        spatial[1] = self._extract_local(
            self.map_data.nfz_mask,
            x,
            y,
            pad_value=0.0,
        )

        # --------------------------------------------------------------
        # Channel 2: fly-over buildings
        # --------------------------------------------------------------

        spatial[2] = self._extract_local(
            self.map_data.fly_over_building_mask,
            x,
            y,
            pad_value=0.0,
        )

        # --------------------------------------------------------------
        # Channel 3: local UAV position
        #
        # Since the window is centered on UAV, this is always the
        # center cell.
        # --------------------------------------------------------------

        center = self.local_view_radius

        spatial[3, center, center] = 1.0

        # --------------------------------------------------------------
        # Channel 4: destination
        #
        # It only appears when the destination lies inside the
        # observable local window.
        # --------------------------------------------------------------

        tx, ty = self.mission.target_pos

        local_tx = tx - x + center
        local_ty = ty - y + center

        if (
            0 <= local_tx < self.local_view_size
            and 0 <= local_ty < self.local_view_size
        ):
            spatial[
                4,
                local_ty,
                local_tx,
            ] = 1.0

        # --------------------------------------------------------------
        # Channel 5: current local flood state
        # --------------------------------------------------------------

        spatial[5] = self._extract_local(
            self.flood_model.flood_grid,
            x,
            y,
            pad_value=0.0,
        )

        # --------------------------------------------------------------
        # Channel 6: forecast flood probability
        #
        # The forecast map is visible.
        # The sampled future is NOT.
        # --------------------------------------------------------------

        if self.current_forecast is not None:

            forecast_map = (
                self.current_forecast.flood_prob_map
            )

            spatial[6] = self._extract_local(
                forecast_map,
                x,
                y,
                pad_value=0.0,
            )

        return np.clip(
            spatial,
            0.0,
            1.0,
        ).astype(np.float32)

    # ==================================================================
    # VECTOR OBSERVATION
    # ==================================================================

    def _get_vector_state(self) -> np.ndarray:
        """
        Construct normalized scalar state.

        All values are constrained to approximately [-1, 1].
        """

        # --------------------------------------------------------------
        # Battery
        # --------------------------------------------------------------

        battery_norm = (
            self.current_battery
            / max(
                1.0,
                self.mission.initial_battery,
            )
        )

        battery_norm = np.clip(
            battery_norm,
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Remaining deadline
        # --------------------------------------------------------------

        remaining_steps = max(
            0,
            self.mission.deadline_steps
            - self.current_step,
        )

        deadline_norm = (
            remaining_steps
            / max(
                1,
                self.mission.deadline_steps,
            )
        )

        deadline_norm = np.clip(
            deadline_norm,
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Goal direction
        # --------------------------------------------------------------

        gx = (
            self.mission.target_pos[0]
            - self.drone_pos[0]
        )

        gy = (
            self.mission.target_pos[1]
            - self.drone_pos[1]
        )

        goal_dx_norm = gx / max(
            1,
            self.map_data.width - 1,
        )

        goal_dy_norm = gy / max(
            1,
            self.map_data.height - 1,
        )

        goal_dx_norm = np.clip(
            goal_dx_norm,
            -1.0,
            1.0,
        )

        goal_dy_norm = np.clip(
            goal_dy_norm,
            -1.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Goal distance
        # --------------------------------------------------------------

        goal_distance = octile_distance(
            self.drone_pos,
            self.mission.target_pos,
        )

        max_distance = octile_distance(
            (0, 0),
            (
                self.map_data.width - 1,
                self.map_data.height - 1,
            ),
        )

        goal_distance_norm = (
            goal_distance
            / max(
                1.0,
                max_distance,
            )
        )

        goal_distance_norm = np.clip(
            goal_distance_norm,
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Heading
        # --------------------------------------------------------------

        heading_dx = float(
            np.clip(
                self.heading[0],
                -1,
                1,
            )
        )

        heading_dy = float(
            np.clip(
                self.heading[1],
                -1,
                1,
            )
        )

        # --------------------------------------------------------------
        # Wind
        # --------------------------------------------------------------

        wind_state = self.wind_model.get_state()

        wind_speed_norm = (
            float(wind_state.speed)
            / self.WIND_SPEED_NORMALIZATION
        )

        wind_speed_norm = np.clip(
            wind_speed_norm,
            0.0,
            1.0,
        )

        wind_x, wind_y = wind_state.wind_vector

        wind_direction_x = np.clip(
            float(wind_x),
            -1.0,
            1.0,
        )

        wind_direction_y = np.clip(
            float(wind_y),
            -1.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Payload
        # --------------------------------------------------------------

        payload_state = (
            self.payload_model.get_state()
        )

        stability = np.clip(
            float(payload_state.stability),
            0.0,
            1.0,
        )

        required_stability = np.clip(
            float(self.mission.required_stability),
            0.0,
            1.0,
        )

        payload_is_fragile = (
            1.0
            if self.mission.payload_type.upper()
            == "FRAGILE"
            else 0.0
        )

        # --------------------------------------------------------------
        # Time
        # --------------------------------------------------------------

        time_norm = (
            self.current_step
            / max(
                1,
                self.mission.deadline_steps,
            )
        )

        time_norm = np.clip(
            time_norm,
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Forecast reliability
        # --------------------------------------------------------------

        alpha_fc = np.clip(
            float(self.alpha_fc),
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Final vector
        # --------------------------------------------------------------

        vector_state = np.array(
            [
                # 0
                battery_norm * 2.0 - 1.0,

                # 1
                deadline_norm * 2.0 - 1.0,

                # 2
                goal_dx_norm,

                # 3
                goal_dy_norm,

                # 4
                goal_distance_norm * 2.0 - 1.0,

                # 5
                heading_dx,

                # 6
                heading_dy,

                # 7
                wind_speed_norm * 2.0 - 1.0,

                # 8
                wind_direction_x,

                # 9
                wind_direction_y,

                # 10
                stability * 2.0 - 1.0,

                # 11
                required_stability * 2.0 - 1.0,

                # 12
                payload_is_fragile * 2.0 - 1.0,

                # 13
                alpha_fc * 2.0 - 1.0,

                # 14
                time_norm * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )

        return np.clip(
            vector_state,
            -1.0,
            1.0,
        )

    # ==================================================================
    # FULL OBSERVATION
    # ==================================================================

    def _get_obs(self) -> Dict[str, Any]:
        """
        Return the agent observation.

        IMPORTANT:
            This function exposes only a local spatial window.

            The current flood is locally observable.

            The future sampled flood is NOT observable.

            Forecast information is exposed only through the forecast
            representation.
        """

        return {
            "spatial_tensor": (
                self._get_local_spatial_tensor()
            ),
            "vector_state": (
                self._get_vector_state()
            ),
        }

    # ==================================================================
    # INFO
    # ==================================================================

    def _get_info(self) -> Dict[str, Any]:
        """
        Return debugging/evaluation metadata.

        Info is NOT part of the agent observation.

        Therefore evaluation information can contain quantities that
        are intentionally hidden from the policy.
        """

        wind_state = (
            self.wind_model.get_state()
        )

        payload_state = (
            self.payload_model.get_state()
        )

        goal_distance = octile_distance(
            self.drone_pos,
            self.mission.target_pos,
        )

        return {
            "episode_seed": self.episode_seed,

            "step": int(
                self.current_step
            ),

            "drone_pos": tuple(
                self.drone_pos
            ),

            "target_pos": tuple(
                self.mission.target_pos
            ),

            "start_pos": tuple(
                self.mission.start_pos
            ),

            "distance_to_target": float(
                goal_distance
            ),

            "battery_remaining": float(
                self.current_battery
            ),

            "initial_battery": float(
                self.mission.initial_battery
            ),

            "deadline_steps": int(
                self.mission.deadline_steps
            ),

            "remaining_deadline": int(
                max(
                    0,
                    self.mission.deadline_steps
                    - self.current_step,
                )
            ),

            "wind_speed": float(
                wind_state.speed
            ),

            "wind_direction": float(
                wind_state.direction_deg
            ),

            "payload_type": (
                self.mission.payload_type
            ),

            "required_stability": float(
                self.mission.required_stability
            ),

            "payload_stability": float(
                payload_state.stability
            ),

            "is_payload_damaged": bool(
                payload_state.is_damaged
            ),

            "alpha_fc": float(
                self.alpha_fc
            ),

            "flooded_cells_count": int(
                np.sum(
                    self.flood_model.flood_grid
                    == 1
                )
            ),

            "trajectory_length": int(
                len(self.trajectory)
            ),

            "heading": tuple(
                self.heading
            ),

            "last_action": self.last_action,

            "terminated_reason": (
                self.terminated_reason
            ),
            "mission_id": int(
                self.mission.mission_id
            ),

            "scenario_seed": int(
                self.mission.seed
            ),

            "wind_base_speed": float(
                self.mission.wind_base_speed
            ),

            "wind_speed_std": float(
                self.mission.wind_speed_std
            ),

            "wind_gust_probability": float(
                self.mission.wind_gust_probability
            ),

            "wind_max_gust": float(
                self.mission.wind_max_gust
            ),

            "rain_intensity": float(
                self.flood_model.get_rain_intensity()
            ),

            "flood_expansion_rate": float(
                self.mission.flood_expansion_rate
            ),

            "forecast_reliability": float(
                self.alpha_fc
            ),
        }

    # ==================================================================
    # RENDER
    # ==================================================================

    def render(self):
        """
        Lightweight environment renderer.

        The existing project visualization module can later be used
        for richer rendering and trajectory plots.

        For now this returns the raw RGB benchmark map with the UAV
        and target overlaid.
        """

        image = (
            self.map_data.raw_image.copy()
        )

        # --------------------------------------------------------------
        # Draw UAV as cyan
        # --------------------------------------------------------------

        x, y = self.drone_pos

        if 0 <= x < self.map_data.width and 0 <= y < self.map_data.height:
            image[y, x] = np.array(
                [0, 255, 255],
                dtype=np.uint8,
            )

        # --------------------------------------------------------------
        # Draw destination as magenta
        # --------------------------------------------------------------

        tx, ty = self.mission.target_pos

        if 0 <= tx < self.map_data.width and 0 <= ty < self.map_data.height:
            image[ty, tx] = np.array(
                [255, 0, 255],
                dtype=np.uint8,
            )

        image = image.astype(np.uint8)

        if self.render_mode == "rgb_array":
            return image

        if self.render_mode == "human":
            try:
                import matplotlib.pyplot as plt

                plt.clf()
                plt.imshow(image)
                plt.axis("off")
                plt.pause(1.0 / self.metadata["render_fps"])

            except ImportError:
                pass

        return None

    # ==================================================================
    # CLOSE
    # ==================================================================

    def close(self):
        """
        Close any resources.

        No persistent resources are currently held.
        """
        pass