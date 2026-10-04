# Main Execution Script for RescueRoute Phase 2 — Physical Dynamics & Hazards Demo.
# Demonstrates stochastic wind & gust simulation, dynamic flood expansion,
# fragile payload stability tracking, hazard collision avoidance, and trajectory animation generation.

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


# Dynamic A* Pathfinder considering both static buildings AND dynamic flooded cells.
def plan_dynamic_astar(grid_map: np.ndarray, flood_grid: np.ndarray, start: tuple, target: tuple):
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
                # Avoid both building obstacles AND flooded hazard cells
                if grid_map[ny, nx] == 1 or flood_grid[ny, nx] == 1:
                    continue

                tentative_g = g_score[current] + (1.414 if (dx != 0 and dy != 0) else 1.0)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    action_taken[neighbor] = (act_id, current)
                    f_score = tentative_g + heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score, neighbor))

    return []


# Run Phase 2 Physical Dynamics & Hazard Simulation Demo.
def main():
    print("=" * 75)
    print(" RescueRoute Phase 2 — Physical Dynamics & Dynamic Hazard Simulation Demo ")
    print("=" * 75)

    # Step 1: Load Manhattan32 Map Dataset
    map_name = "manhattan32"
    print(f"\n[1/5] Loading benchmark map dataset: '{map_name}'...")
    map_data = load_map(map_name)

    # Step 2: Generate Seeded Mission Scenario
    print("\n[2/5] Generating seeded emergency mission scenario...")
    mission = generate_mission(map_data, seed=105)
    print(f"      - Start Base Pos:      {mission.start_pos}")
    print(f"      - Emergency Target:    {mission.target_pos}")
    print(f"      - Initial Battery:      {mission.initial_battery:.1f} units")
    print(f"      - Flight Deadline:      {mission.deadline_steps} steps")
    print(f"      - Payload Type:         {mission.payload_type} (Required Stability: {mission.required_stability*100:.0f}%)")

    # Step 3: Initialize RescueRoute Gymnasium Environment with Phase 2 Dynamics
    print("\n[3/5] Initializing RescueRoute Environment with Wind, Flood & Payload Dynamics...")
    env = RescueRouteEnv(map_name=map_name, mission=mission)
    obs, info = env.reset(seed=105)
    print("      - Environment reset successfully.")
    print(f"      - Initial Wind Speed:   {info['wind_speed']:.1f} m/s (Direction: {info['wind_direction']:.1f}°)")
    print(f"      - Initial Flooded Cells:{info['flooded_cells_count']} cells")
    print(f"      - Initial Stability:    {info['payload_stability']*100:.0f}%")

    # Step 4: Execute Simulation Steps with Dynamic Pathfinder
    print("\n[4/5] Simulating dynamic episode steps with wind, flood expansion & fragile payload tracking...")
    
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        step_count += 1

        # Replan path dynamically considering updated flood expansion grid
        flood_grid = env.flood_model.flood_grid
        path = plan_dynamic_astar(map_data.grid_map, flood_grid, env.drone_pos, mission.target_pos)

        if path:
            action = path[0] # Pick next step action towards target
        else:
            action = 8 # Hover stabilization fallback if path temporarily blocked

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"        Step {step_count:02d} | Act: {action} | Pos: {info['drone_pos']} | Dist: {info['distance_to_target']:02d} | "
              f"Battery: {info['battery_remaining']:.1f} | Wind: {info['wind_speed']:.1f}m/s | Flooded: {info['flooded_cells_count']} | "
              f"Stability: {info['payload_stability']*100:.1f}% | Rwd: {reward:+.2f}")

        if step_count >= 50:
            break

    print(f"\nPhase 2 Simulation Completed in {step_count} steps!")
    print(f"  - Final Drone Position:  {info['drone_pos']}")
    print(f"  - Total Episode Reward:  {total_reward:+.2f}")
    print(f"  - Remaining Battery:     {info['battery_remaining']:.1f}")
    print(f"  - Final Payload Integrity:{info['payload_stability']*100:.1f}% (Damaged: {info['is_payload_damaged']})")
    print(f"  - Total Flooded Cells:   {info['flooded_cells_count']}")

    # Step 5: Render and Save Phase 2 Trajectory Animation GIF & PNG
    print("\n[5/5] Generating Phase 2 trajectory animation with flood overlay, wind HUD & drone icon...")
    visualizer = DroneVisualizer(env)
    gif_path = os.path.abspath("outputs/animations/rescue_route_phase2_demo.gif")
    gif_file, png_file = visualizer.create_episode_animation(output_gif_path=gif_path, fps=4)

    print("\n" + "=" * 75)
    print(" PHASE 2 COMPLETED SUCCESSFULLY! ")
    print("=" * 75)
    print(f"\nVisualization Outputs Generated:")
    print(f"  1. Animated GIF: {gif_file}")
    print(f"  2. Final Frame:  {png_file}")
    print("\nHow to view the Phase 2 visualization:")
    print("  - Open 'outputs/animations/rescue_route_phase2_demo.gif' in any browser or IDE.")
    print("=" * 75)


if __name__ == "__main__":
    main()
