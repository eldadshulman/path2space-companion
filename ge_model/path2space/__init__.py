"""
path2space: H&E -> spatial gene-expression prediction.

Top-level usage:

    from path2space import Path2SpacePipeline
    pipe = Path2SpacePipeline()
    out = pipe.predict_grid("slide.svs", tile_size_px=224)
    out["pred"]    # rows = tiles, cols = genes (plus slide_name/x/y/grid_x/grid_y)
    out["smooth"]  # KDTree-smoothed version

For predefined Visium spots:

    spots_df = pd.read_csv("spots.csv")  # cols: pixel_x, pixel_y, spot_id, select
    out = pipe.predict_spots("slide.tif", spots=spots_df)
"""

# Force-import cv2 BEFORE anything that pulls in libtiff via timm/skimage.
# In some environments (notably the vision_ml conda env), importing
# skimage/timm first loads an incompatible libtiff that breaks cv2.
import cv2 as _cv2  # noqa: F401

from .pipeline import (
    Path2SpacePipeline,
    WeightsLayout,
    DEFAULT_EDGE_MAG_THRESHOLD,
    DEFAULT_EDGE_FRACTION_THRESHOLD_GRID,
    DEFAULT_EDGE_FRACTION_THRESHOLD_SPOTS,
)

__version__ = "0.1.0"
__all__ = [
    "Path2SpacePipeline",
    "WeightsLayout",
    "DEFAULT_EDGE_MAG_THRESHOLD",
    "DEFAULT_EDGE_FRACTION_THRESHOLD_GRID",
    "DEFAULT_EDGE_FRACTION_THRESHOLD_SPOTS",
]
