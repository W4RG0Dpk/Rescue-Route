# Main Execution Script for RescueRoute Phase 1 — Environment Core & Visualization.
# Demonstrates loading benchmark map datasets, initializing the Gymnasium environment,
# simulating a complete drone flight mission, and generating an animated GIF visualization.

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


# Simple A* Grid Pathfinder to generate a collision-free flight trajectory for Phase 1 demo simulation.
# Used to demonstrate the environment step dynamics and trajectory animation before RL training.
def plan_astar_path(grid_map: np.ndarray, start: tuple, target: tuple):
    height, width = grid_map.shape

    # 8-directional movement offsets + hover
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
            # Reconstruct sequence of action IDs from start to target
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

            # Check bounds and building obstacles
            if 0 <= nx < width and 0 <= ny < height:
                if grid_map[ny, nx] == 1:
                    continue # Obstacle/building collision

                tentative_g = g_score[current] + (1.414 if (dx != 0 and dy != 0) else 1.0)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    action_taken[neighbor] = (act_id, current)
                    f_score = tentative_g + heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score, neighbor))

    return [] # Return empty list if no valid path found


# Run Phase 1 Environment & Animation Demo.
def main():
    print("=" * 70)
    print(" RescueRoute Phase 1 — Environment Core & Visualization Demo ")
    print("=" * 70)

    # Step 1: Load Manhattan32 Map Dataset
    map_name = "manhattan32"
    print(f"\n[1/4] Loading benchmark map dataset: '{map_name}'...")
    map_data = load_map(map_name)
    print(f"      - Map Dimensions: {map_data.width}x{map_data.height} cells")
    print(f"      - Number of Base Stations: {len(map_data.base_locations)}")
    print(f"      - Obstacle Cells Count: {np.sum(map_data.grid_map == 1)}")

    # Step 2: Generate Seeded Emergency Mission Scenario
    print("\n[2/4] Generating seeded emergency mission scenario...")
    mission = generate_mission(map_data, seed=42)
    print(f"      - Start Base Pos:  {mission.start_pos}")
    print(f"      - Emergency Target: {mission.target_pos}")
    print(f"      - Initial Battery:  {mission.initial_battery:.1f} units")
    print(f"      - Deadline Steps:   {mission.deadline_steps} steps")

    # Step 3: Initialize RescueRoute Gymnasium Environment
    print("\n[3/4] Initializing Gymnasium RescueRoute Environment...")
    env = RescueRouteEnv(map_name=map_name, mission=mission)
    obs, info = env.reset(seed=42)
    print("      - Environment reset successfully.")
    print(f"      - Observation keys: {list(obs.keys())}")
    print(f"      - Initial distance to target: {info['distance_to_target']} cells")

    # Step 4: Plan Navigable Path using A* Pathfinder and Execute Episode Steps
    print("\n[4/4] Planning collision-free flight trajectory and simulating episode steps...")
    action_plan = plan_astar_path(map_data.grid_map, mission.start_pos, mission.target_pos)

    if not action_plan:
        print("      [WARNING] No direct path found! Using default hover/step actions.")
        action_plan = [0, 2, 0, 2]

    print(f"      - Planned trajectory action sequence length: {len(action_plan)} steps")

    total_reward = 0.0
    step_count = 0
    
    # Step through environment episode using planned actions
    for action in action_plan:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        print(f"        Step {step_count:02d} | Action: {action} | Pos: {info['drone_pos']} | "
              f"Target Dist: {info['distance_to_target']} | Battery: {info['battery_remaining']:.1f} | Reward: {reward:+.2f}")

        if terminated or truncated:
            break

    print(f"\nEpisode Simulation Completed in {step_count} steps!")
    print(f"  - Final Drone Position: {info['drone_pos']}")
    print(f"  - Total Episode Reward: {total_reward:+.2f}")
    print(f"  - Remaining Battery:    {info['battery_remaining']:.1f}")

    # Step 5: Render and Save Animation GIF / Static Screenshot
    print("\n[5/5] Generating trajectory animation with drone icon...")
    visualizer = DroneVisualizer(env)
    gif_path = os.path.abspath("outputs/animations/rescue_route_demo.gif")
    gif_file, png_file = visualizer.create_episode_animation(output_gif_path=gif_path, fps=4)

    print("\n" + "=" * 70)
    print(" PHASE 1 COMPLETED SUCCESSFULLY! ")
    print("=" * 70)
    print(f"\nVisualization Outputs Generated:")
    print(f"  1. Animated GIF: {gif_file}")
    print(f"  2. Final Frame:  {png_file}")
    print("\nHow to view the visualization:")
    print("  - Open 'outputs/animations/rescue_route_demo.gif' in any image viewer, web browser, or VS Code / IDE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
