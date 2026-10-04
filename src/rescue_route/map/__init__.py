# Map processing package initialization file.
# Contains utilities for parsing grid maps and building obstacle representations.

from .map_loader import MapData, load_map

__all__ = ["MapData", "load_map"]
