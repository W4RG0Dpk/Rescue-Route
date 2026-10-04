# RescueRoute: Forecast-Aware Recoverability Safety Layer for Autonomous Emergency UAV Path Planning in Stochastic Urban Flood Environments

## Overview
**RescueRoute** is an autonomous emergency UAV navigation and delivery platform for urban disaster zones. This repository implements Phase 1 of the architecture: Environment Core, Benchmark Map Engine, Kinematic Simulation, and Animation Renderer.

---

## Phase 1 Modules
- `src/rescue_route/map/map_loader.py`: Benchmark grid map dataset loader (`manhattan32` and `urban50`) parsing obstacle structures and base stations.
- `src/rescue_route/env/mission.py`: Seeded scenario generator for start base, emergency target, battery capacity, and flight deadlines.
- `src/rescue_route/env/rescue_route_env.py`: Gymnasium reinforcement learning environment implementing 9 discrete flight actions, boundary/building collision checks, energy drain, and distance reward shaping.
- `src/rescue_route/utils/visualization.py`: Trajectory rendering engine with a custom quadcopter drone icon, flight path overlay, and animated GIF/MP4 generator.

---

## Quick Start — How to Run Phase 1

### 1. Run Demo Execution & Animation Generator
```bash
python run_phase1_demo.py
```

### 2. View Generated Animation
- Open `outputs/animations/rescue_route_demo.gif` in any web browser, image viewer, or IDE to view the quadcopter drone flying to the emergency target.
