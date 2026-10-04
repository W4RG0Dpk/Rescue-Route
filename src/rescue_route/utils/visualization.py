# Visualization Engine for RescueRoute UAV Environment.
# Renders 2D urban maps, custom quadcopter drone icons, start/target markers,
# flight trajectory paths, dynamic flood expansion overlays, probabilistic forecast heatmaps,
# wind HUD compass, payload stability gauge, and forecast reliability index (alpha_fc).

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import imageio
from typing import List, Tuple, Optional
from ..env.rescue_route_env import RescueRouteEnv


# Drone Visualizer Class for Phase 3 Environment.
class DroneVisualizer:

    def __init__(self, env: RescueRouteEnv):
        self.env = env
        self.map_data = env.map_data
        self.frames: List[np.ndarray] = []

    # Draw detailed quadcopter drone icon at grid position (x, y).
    def draw_drone_icon(self, ax, x: float, y: float, heading_angle: float = 0.0):
        cx = x + 0.5
        cy = y + 0.5
        radius = 0.35

        # Quadcopter center body chassis (cyan circle)
        body = patches.Circle((cx, cy), radius * 0.4, color='#00E5FF', zorder=10)
        ax.add_patch(body)

        # Draw 4 rotor arms extending diagonally from body
        arm_len = radius * 0.85
        offsets = [(-arm_len, -arm_len), (arm_len, -arm_len), (-arm_len, arm_len), (arm_len, arm_len)]
        for dx, dy in offsets:
            ax.plot([cx, cx + dx], [cy, cy + dy], color='white', linewidth=2, zorder=9)
            rotor = patches.Circle((cx + dx, cy + dy), radius * 0.3, color='#FF9100', alpha=0.9, zorder=11)
            ax.add_patch(rotor)

        # Directional arrow indicating drone flight heading
        rad = np.radians(heading_angle)
        arrow_dx = 0.4 * np.sin(rad)
        arrow_dy = -0.4 * np.cos(rad)
        ax.arrow(cx, cy, arrow_dx, arrow_dy, head_width=0.15, head_length=0.15, fc='#FFEA00', ec='#FFD600', zorder=12)

    # Render a single frame of the current Phase 3 simulation state.
    def render_frame(self, step_num: int, status_text: str = "NAVIGATING") -> np.ndarray:
        fig, ax = plt.subplots(figsize=(9.5, 9.5), dpi=100)

        # 1. Render background map image
        map_img = self.map_data.raw_image.copy()
        ax.imshow(map_img)

        # 2. Render probabilistic forecast flood heatmap overlay (yellow/orange cells)
        if self.env.current_forecast is not None:
            prob_map = self.env.current_forecast.flood_prob_map
            for fy in range(self.map_data.height):
                for fx in range(self.map_data.width):
                    prob = prob_map[fy, fx]
                    if prob > 0.3 and self.env.flood_model.flood_grid[fy, fx] == 0:
                        # Render forecast risk prediction in orange/yellow
                        fc_box = patches.Rectangle(
                            (fx, fy), 1.0, 1.0,
                            linewidth=0.5, edgecolor='#FF6D00', facecolor='#FFAB00', alpha=min(0.4, prob * 0.5), zorder=2
                        )
                        ax.add_patch(fc_box)

        # 3. Render ground-truth dynamic flood overlay (water blue/cyan cells)
        flood_grid = self.env.flood_model.flood_grid
        for fy in range(self.map_data.height):
            for fx in range(self.map_data.width):
                if flood_grid[fy, fx] == 1:
                    flood_box = patches.Rectangle(
                        (fx, fy), 1.0, 1.0,
                        linewidth=0.5, edgecolor='#00B0FF', facecolor='#00B0FF', alpha=0.5, zorder=3
                    )
                    ax.add_patch(flood_box)

        # Get state metadata
        wind_state = self.env.wind_model.get_state()
        payload_state = self.env.payload_model.get_state()

        # Set HUD titles
        ax.set_title(
            f"RescueRoute Emergency UAV Simulation — {self.map_data.name.upper()}\n"
            f"Step: {step_num:02d} | Battery: {self.env.current_battery:.1f} | Wind: {wind_state.speed:.1f}m/s | "
            f"Payload: {payload_state.stability*100:.0f}% | Forecast Reliability α_fc: {self.env.alpha_fc*100:.0f}%",
            fontsize=11, fontweight='bold', pad=10
        )
        ax.set_xlim(-0.5, self.map_data.width - 0.5)
        ax.set_ylim(self.map_data.height - 0.5, -0.5)
        ax.set_xticks(range(0, self.map_data.width, 5))
        ax.set_yticks(range(0, self.map_data.height, 5))
        ax.grid(True, color='gray', linestyle=':', alpha=0.4)

        # 4. Draw Start Base location
        start_x, start_y = self.env.mission.start_pos
        start_box = patches.Rectangle(
            (start_x, start_y), 1.0, 1.0, 
            linewidth=2, edgecolor='#00E676', facecolor='#00E676', alpha=0.4, zorder=4, label='Start Base'
        )
        ax.add_patch(start_box)
        ax.text(start_x + 0.5, start_y + 0.5, "START", color='white', weight='bold', fontsize=8, ha='center', va='center', zorder=5)

        # 5. Draw Target Emergency Delivery Location
        target_x, target_y = self.env.mission.target_pos
        target_box = patches.Rectangle(
            (target_x, target_y), 1.0, 1.0, 
            linewidth=2, edgecolor='#FF1744', facecolor='#FF1744', alpha=0.5, zorder=4, label='Emergency Target'
        )
        ax.add_patch(target_box)
        ax.text(target_x + 0.5, target_y + 0.5, "TARGET", color='white', weight='bold', fontsize=8, ha='center', va='center', zorder=5)

        # Dummy patches for legend
        dummy_flood = patches.Rectangle((0,0), 0, 0, facecolor='#00B0FF', alpha=0.5, label='Actual Flood')
        dummy_fc = patches.Rectangle((0,0), 0, 0, facecolor='#FFAB00', alpha=0.4, label='Forecast Risk (H=10)')
        ax.add_patch(dummy_flood)
        ax.add_patch(dummy_fc)

        # 6. Draw flight trajectory line
        if len(self.env.trajectory) > 1:
            traj_x = [p[0] + 0.5 for p in self.env.trajectory]
            traj_y = [p[1] + 0.5 for p in self.env.trajectory]
            ax.plot(traj_x, traj_y, color='#FFEA00', linewidth=2.5, linestyle='--', marker='o', markersize=3, zorder=6, label='Flight Path')

        # 7. Draw Quadcopter Drone Icon
        drone_x, drone_y = self.env.drone_pos
        heading = 0.0
        if len(self.env.trajectory) >= 2:
            prev_x, prev_y = self.env.trajectory[-2]
            dx = drone_x - prev_x
            dy = drone_y - prev_y
            if dx != 0 or dy != 0:
                heading = np.degrees(np.arctan2(dx, -dy))

        self.draw_drone_icon(ax, drone_x, drone_y, heading)

        # 8. Render Wind Compass Vector HUD
        w_rad = np.radians(wind_state.direction_deg)
        w_dx = 1.5 * np.sin(w_rad)
        w_dy = -1.5 * np.cos(w_rad)
        ax.arrow(2.5, 2.5, w_dx, w_dy, head_width=0.6, head_length=0.6, fc='#D500F9', ec='#AA00FF', linewidth=2, zorder=15)
        ax.text(2.5, 4.5, f"WIND {wind_state.speed:.1f} m/s", color='#D500F9', weight='bold', fontsize=9, ha='center', zorder=16)

        # HUD Legend
        ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

        # Convert matplotlib canvas to numpy array
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frame = rgba[:, :, :3].copy()
        plt.close(fig)

        return frame

    # Record and save full episode simulation animation to GIF file.
    def create_episode_animation(
        self,
        output_gif_path: str = "outputs/animations/rescue_route_phase3_demo.gif",
        fps: int = 4
    ):
        os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
        print(f"Generating Phase 3 trajectory animation frames ({len(self.env.trajectory)} steps)...")

        frames = []
        saved_trajectory = list(self.env.trajectory)

        # Replay trajectory step by step to capture frame sequence
        for step_idx in range(len(saved_trajectory)):
            self.env.trajectory = saved_trajectory[:step_idx + 1]
            self.env.drone_pos = saved_trajectory[step_idx]

            status = "NAVIGATING"
            if step_idx == len(saved_trajectory) - 1:
                if self.env.drone_pos == self.env.mission.target_pos:
                    status = "TARGET REACHED!"
                else:
                    status = "COMPLETED"

            frame = self.render_frame(step_idx, status_text=status)
            frames.append(frame)

        # Save frames as animated GIF using imageio
        print(f"Saving animated GIF to '{output_gif_path}'...")
        imageio.mimsave(output_gif_path, frames, fps=fps)
        print("Phase 3 Animation GIF generated successfully!")

        # Also save static screenshot of final frame
        png_path = output_gif_path.replace(".gif", "_final.png")
        Image.fromarray(frames[-1]).save(png_path)
        print(f"Final Phase 3 trajectory screenshot saved to '{png_path}'.")

        return output_gif_path, png_path
