"""
Map loading and parsing for RescueRoute.

The benchmark maps use one PNG pixel per grid cell.

Benchmark color semantics:
    RED    #FF0000 -> No-Fly Zone (NFZ)
    GREEN  #00FF00 -> Building, but UAV can fly over
    BLUE   #0000FF -> Start / Landing Zone
    YELLOW #FFFF00 -> Building + NFZ, UAV cannot fly over

RescueRoute keeps these semantics as separate masks so that the
environment can later distinguish:
    - hard flight obstacles
    - no-fly zones
    - fly-over structures
    - valid base / landing cells
    - free airspace
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# RGB color definitions from the benchmark
# ---------------------------------------------------------------------

RED = np.array([255, 0, 0], dtype=np.uint8)
GREEN = np.array([0, 255, 0], dtype=np.uint8)
BLUE = np.array([0, 0, 255], dtype=np.uint8)
YELLOW = np.array([255, 255, 0], dtype=np.uint8)


# ---------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------

@dataclass
class MapData:
    """
    Parsed RescueRoute map.

    Coordinate convention:
        (x, y)
        x -> column
        y -> row

    Array convention:
        array[y, x]
    """

    name: str

    width: int
    height: int

    # Original PNG image
    raw_image: np.ndarray

    # Semantic masks
    nfz_mask: np.ndarray
    fly_over_building_mask: np.ndarray
    blocked_mask: np.ndarray
    base_mask: np.ndarray
    free_mask: np.ndarray

    # Backward-compatible main navigation grid:
    #   0 -> traversable
    #   1 -> blocked
    grid_map: np.ndarray

    # Coordinates of valid start / landing cells
    base_locations: List[Tuple[int, int]]

    # Original JSON configuration
    config: Dict

    @property
    def shape(self) -> Tuple[int, int]:
        """Return map shape as (height, width)."""
        return self.height, self.width

    def is_inside(self, x: int, y: int) -> bool:
        """Return True when (x, y) lies inside the map."""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, x: int, y: int) -> bool:
        """Return True if the UAV cannot fly through this cell."""
        if not self.is_inside(x, y):
            return True

        return bool(self.blocked_mask[y, x])

    def is_nfz(self, x: int, y: int) -> bool:
        """Return True if the cell belongs to an NFZ."""
        if not self.is_inside(x, y):
            return False

        return bool(self.nfz_mask[y, x])

    def is_base(self, x: int, y: int) -> bool:
        """Return True if the cell is a valid base / landing cell."""
        if not self.is_inside(x, y):
            return False

        return bool(self.base_mask[y, x])

    def traversable_cells(self) -> np.ndarray:
        """
        Return coordinates of all traversable cells as an N x 2 array.

        Each row is [x, y].
        """
        ys, xs = np.where(~self.blocked_mask)
        return np.column_stack((xs, ys))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _project_root() -> Path:
    """
    Resolve the repository root.

    map_loader.py is expected at:
        <root>/src/rescue_route/map/map_loader.py
    """
    return Path(__file__).resolve().parents[3]


def _color_mask(image: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Return a boolean mask for exact RGB color matches."""
    return np.all(image == color, axis=-1)


def _load_config(json_path: Path) -> Dict:
    """Load benchmark JSON configuration."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------

def load_map(
    map_name: str = "manhattan32",
    base_dir: Optional[str] = None,
) -> MapData:
    """
    Load and parse a RescueRoute benchmark map.

    Expected directory:

        data/
        └── maps/
            └── <map_name>/
                ├── map.png
                └── config.json

    Parameters
    ----------
    map_name:
        Map directory name, for example:
        "manhattan32" or "urban50"

    base_dir:
        Optional repository root override.

    Returns
    -------
    MapData
        Parsed map representation.
    """

    # -------------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------------

    if base_dir is None:
        root = _project_root()
    else:
        root = Path(base_dir).resolve()

    map_folder = root / "data" / "maps" / map_name

    png_path = map_folder / "map.png"
    json_path = map_folder / "config.json"

    if not png_path.exists():
        raise FileNotFoundError(
            f"Map PNG not found:\n{png_path}"
        )

    if not json_path.exists():
        raise FileNotFoundError(
            f"Map JSON/config not found:\n{json_path}"
        )

    # -------------------------------------------------------------
    # Load image
    # -------------------------------------------------------------

    img = Image.open(png_path).convert("RGB")
    raw_image = np.asarray(img, dtype=np.uint8)

    if raw_image.ndim != 3 or raw_image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), "
            f"got {raw_image.shape}"
        )

    height, width, _ = raw_image.shape

    # -------------------------------------------------------------
    # Load JSON metadata
    # -------------------------------------------------------------

    config = _load_config(json_path)

    # -------------------------------------------------------------
    # Decode benchmark colors
    # -------------------------------------------------------------

    red_mask = _color_mask(raw_image, RED)
    green_mask = _color_mask(raw_image, GREEN)
    blue_mask = _color_mask(raw_image, BLUE)
    yellow_mask = _color_mask(raw_image, YELLOW)

    # -------------------------------------------------------------
    # Semantic layers
    # -------------------------------------------------------------

    # Red = NFZ
    nfz_mask = red_mask.copy()

    # Green = building that UAV can fly over
    fly_over_building_mask = green_mask.copy()

    # Red = NFZ
    # Yellow = building + NFZ
    # Neither is flyable.
    blocked_mask = nfz_mask | yellow_mask

    # Blue = valid starting / landing area
    base_mask = blue_mask.copy()

    # All cells that are not hard blocked are traversable.
    free_mask = ~blocked_mask

    # Keep the old grid_map interface for existing code:
    #
    #   0 = traversable
    #   1 = blocked
    #
    grid_map = blocked_mask.astype(np.int8)

    # -------------------------------------------------------------
    # Extract base coordinates
    # -------------------------------------------------------------

    ys, xs = np.where(base_mask)
    base_locations: List[Tuple[int, int]] = list(
        zip(xs.tolist(), ys.tolist())
    )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid map dimensions: width={width}, height={height}"
        )

    if not base_locations:
        raise ValueError(
            f"No blue start/landing cells were found in map "
            f"'{map_name}'. Check map.png color semantics."
        )

    # A base should never be blocked.
    if np.any(base_mask & blocked_mask):
        raise ValueError(
            f"Map '{map_name}' contains base cells marked as blocked."
        )

    # -------------------------------------------------------------
    # Build object
    # -------------------------------------------------------------

    return MapData(
        name=map_name,
        width=width,
        height=height,
        raw_image=raw_image,
        nfz_mask=nfz_mask,
        fly_over_building_mask=fly_over_building_mask,
        blocked_mask=blocked_mask,
        base_mask=base_mask,
        free_mask=free_mask,
        grid_map=grid_map,
        base_locations=base_locations,
        config=config,
    )