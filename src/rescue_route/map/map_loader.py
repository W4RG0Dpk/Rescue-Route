# Map Loader Module for RescueRoute Environment.
# Used to load benchmark map images (PNG) and metadata configs (JSON), 
# parse static building obstacles vs free airspace, and extract base locations.

import os
import json
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, List, Optional


# Data container class holding parsed grid map data and metadata.
# Used across the environment, pathfinder, and visualization engine to query map bounds and obstacles.
@dataclass
class MapData:
    name: str                           # Map identifier name (e.g. 'manhattan32' or 'urban50')
    width: int                          # Grid map width in cells (e.g. 32 or 50)
    height: int                         # Grid map height in cells (e.g. 32 or 50)
    grid_map: np.ndarray                # 2D integer array (0 = FREE AIRSPACE, 1 = BUILDING OBSTACLE)
    raw_image: np.ndarray               # Original RGB image array loaded from PNG
    base_locations: List[Tuple[int, int]]# List of available drone base/landing coordinates (x, y)
    config: dict                        # Full raw configuration dictionary from JSON config


# Load PNG image and JSON config for a specified map dataset.
# Parameters:
#   map_name: Name of map folder inside data/maps/ ('manhattan32' or 'urban50')
#   base_dir: Root project directory path
# Returns:
#   MapData object containing obstacle masks and map dimensions.
def load_map(map_name: str = "manhattan32", base_dir: Optional[str] = None) -> MapData:
    # If base_dir is not provided, resolve path relative to repository root directory
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    map_folder = os.path.join(base_dir, "data", "maps", map_name)
    png_path = os.path.join(map_folder, "map.png")
    json_path = os.path.join(map_folder, "config.json")

    # Ensure map files exist before trying to read them
    if not os.path.exists(png_path) or not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Map files not found for '{map_name}' at '{map_folder}'. "
            "Please run data download script first."
        )

    # Load PNG image using PIL and convert to RGB numpy matrix
    img = Image.open(png_path).convert("RGB")
    raw_img = np.array(img)
    height, width, _ = raw_img.shape

    # Read JSON configuration metadata file
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Create 2D grid matrix where 1 represents building/obstacle and 0 represents navigable airspace.
    # In benchmark map PNGs:
    # - Black pixels [0, 0, 0] are BUILDING OBSTACLES (grid = 1)
    # - Non-black colored pixels (blue bases, yellow roads, green/red zones) are FREE AIRSPACE (grid = 0)
    grid = np.zeros((height, width), dtype=np.int32)
    base_locs = []

    for y in range(height):
        for x in range(width):
            r, g, b = raw_img[y, x]
            if r == 0 and g == 0 and b == 0:
                grid[y, x] = 1 # Mark cell as BUILDING OBSTACLE
            elif r < 50 and g < 50 and b > 200:
                # Blue pixels mark base stations
                base_locs.append((x, y))

    # If no blue base pixels found in image, check JSON config or use corners
    if not base_locs:
        if "bases" in config and isinstance(config["bases"], list):
            for b in config["bases"]:
                base_locs.append((int(b[0]), int(b[1])))
        else:
            base_locs = [(2, 2), (width - 3, height - 3), (2, height - 3), (width - 3, 2)]

    return MapData(
        name=map_name,
        width=width,
        height=height,
        grid_map=grid,
        raw_image=raw_img,
        base_locations=base_locs,
        config=config
    )
