# Main Execution Script for RescueRoute Phase 3 — Dual-World Forecast & Partial Observability Demo.
# Demonstrates probabilistic flood forecast generation, forecast reliability degradation (alpha_fc),
# multi-channel 2D spatial tensor observation generation, and forecast-aware flight trajectory simulation.

import os
import sys
import heapq
import numpy as np

# Add src folder to python import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from rescue_route.map.map_loader import load_map
from rescue_route.env.mission import generate_mission
from rescue_route.env.rescue_route_env import RescueRouteEnv
from rescue_route.utils.visualization import DroneVisualizer


# Forecast-Aware A* Pathfinder considering buildings, current floods, AND predicted forecast risk cells.
def plan_forecast_aware_astar(grid_map: np.ndarray, flood_grid: np.ndarray, prob_flood_map: np.ndarray, start: tuple, target: tuple):
    height, width = grid_map.shape
    directions = [(0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)]
    action_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    action_taken = {}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            actions = []
            curr = target
            while curr in action_taken:
                act, prev = action_taken[curr]
                actions.append(act)
                curr = prev
            actions.reverse()
            return actions

        for act_id, (dx, dy) in zip(action_ids, directions):
            neighbor = (current[0] + dx, current[1] + dy)
            nx, ny = neighbor

            if 0 <= nx < width and 0 <= ny < height:
                # Avoid building obstacles and currently flooded cells
                if grid_map[ny, nx] == 1 or flood_grid[ny, nx] == 1:
                    continue

                # Add risk penalty for predicted high-probability flood forecast cells
                risk_penalty = prob_flood_map[ny, nx] * 3.0
                move_cost = (1.414 if (dx != 0 and dy != 0) else 1.0) + risk_penalty

                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    action_taken[neighbor] = (act_id, current)
                    f_score = tentative_g + heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score, neighbor))

    return []


# Run Phase 3 Forecast Architecture Simulation Demo.
def main():
    print("=" * 80)
    print(" RescueRoute Phase 3 — Dual-World Forecast & Partial Observability Architecture Demo ")
    print("=" * 80)

    # Step 1: Load Manhattan32 Map Dataset
    map_name = "manhattan32"
    print(f"\n[1/5] Loading benchmark map dataset: '{map_name}'...")
    map_data = load_map(map_name)

    # Step 2: Generate Seeded Mission Scenario
    print("\n[2/5] Generating seeded emergency mission scenario...")
    mission = generate_mission(map_data, seed=105)
    print(f"      - Start Base Pos:      {mission.start_pos}")
    print(f"      - Emergency Target:    {mission.target_pos}")
    print(f"      - Flight Deadline:      {mission.deadline_steps} steps")
    print(f"      - Payload Type:         {mission.payload_type}")

    # Step 3: Initialize RescueRoute Gymnasium Environment with Phase 3 Forecast Architecture
    alpha_fc = 0.80 # 80% forecast reliability index
    print(f"\n[3/5] Initializing RescueRoute Environment with Forecast Architecture (alpha_fc = {alpha_fc})...")
    env = RescueRouteEnv(map_name=map_name, mission=mission, alpha_fc=alpha_fc)
    obs, info = env.reset(seed=105)
    
    spatial_tensor = obs["spatial_tensor"]
    vector_state = obs["vector_state"]
    print("      - Environment reset successfully.")
    print(f"      - Multi-Channel Spatial Tensor Shape: {spatial_tensor.shape} (5 channels, {spatial_tensor.shape[1]}x{spatial_tensor.shape[2]})")
    print("        * Channel 0: Building Obstacles")
    print("        * Channel 1: Drone Position Map")
    print("        * Channel 2: Emergency Target Position Map")
    print("        * Channel 3: Ground-Truth Dynamic Flood Mask")
    print("        * Channel 4: Probabilistic Flood Forecast Grid (H=10)")
    print(f"      - Vector Observation State Length:    {len(vector_state)} parameters")
    print(f"      - Initial Forecast Reliability alpha_fc: {info['alpha_fc']*100:.0f}%")

    # Step 4: Execute Simulation Steps with Forecast-Aware Pathfinder
    print("\n[4/5] Simulating episode steps using forecast-aware path planning & multi-channel state tracking...")

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        step_count += 1

        # Replan trajectory considering current flood map AND forecast flood probabilities
        flood_grid = env.flood_model.flood_grid
        prob_map = env.current_forecast.flood_prob_map if env.current_forecast else np.zeros_like(flood_grid)
        
        path = plan_forecast_aware_astar(map_data.grid_map, flood_grid, prob_map, env.drone_pos, mission.target_pos)

        if path:
            action = path[0]
        else:
            action = 8 # Hover stabilization fallback

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"        Step {step_count:02d} | Act: {action} | Pos: {info['drone_pos']} | Dist: {info['distance_to_target']:02d} | "
              f"Battery: {info['battery_remaining']:.1f} | Wind: {info['wind_speed']:.1f}m/s | Flooded: {info['flooded_cells_count']} | "
              f"Fc Alpha: {info['alpha_fc']*100:.0f}% | Rwd: {reward:+.2f}")

        if step_count >= 50:
            break

    print(f"\nPhase 3 Simulation Completed in {step_count} steps!")
    print(f"  - Final Drone Position:   {info['drone_pos']}")
    print(f"  - Total Episode Reward:   {total_reward:+.2f}")
    print(f"  - Remaining Battery:      {info['battery_remaining']:.1f}")
    print(f"  - Final Payload Stability:{info['payload_stability']*100:.1f}%")
    print(f"  - Forecast Reliability alpha_fc: {info['alpha_fc']*100:.0f}%")

    # Step 5: Render and Save Phase 3 Trajectory Animation GIF & PNG
    print("\n[5/5] Generating Phase 3 trajectory animation with forecast heatmap, flood overlay & HUD...")
    visualizer = DroneVisualizer(env)
    gif_path = os.path.abspath("outputs/animations/rescue_route_phase3_demo.gif")
    gif_file, png_file = visualizer.create_episode_animation(output_gif_path=gif_path, fps=4)

    print("\n" + "=" * 80)
    print(" PHASE 3 COMPLETED SUCCESSFULLY! ")
    print("=" * 80)
    print(f"\nVisualization Outputs Generated:")
    print(f"  1. Animated GIF: {gif_file}")
    print(f"  2. Final Frame:  {png_file}")
    print("\nHow to view the Phase 3 visualization:")
    print("  - Open 'outputs/animations/rescue_route_phase3_demo.gif' in any browser or IDE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
