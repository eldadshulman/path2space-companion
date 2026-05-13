"""
Tile-coordinate generators.

Two strategies share the same iterator protocol — each yields
(tile_name, np.ndarray) pairs in HxWx3 uint8 RGB:

  - GridTiler: lays an axis-aligned, non-overlapping grid over the slide.
    Tile size defaults to 224 px; pass `tile_size_px` to override (e.g.
    for MPP-standardized tiling derived from `tile_size_px_100_micron`).
    Matches the behavior of Cell_revisions/prediction_pipline/1main_tcga.py.

  - SpotsTiler: reads a spots table (CSV or pickle of a DataFrame with
    columns pixel_x, pixel_y, spot_id, select/selected) and yields one
    tile per selected spot, centered on (pixel_x, pixel_y).

Both rely on a SlideReader for pixel access — backend (openslide vs
skimage) is chosen by SlideReader at open time.

Coordinate convention for SpotsTiler: by default we match the paper's
spots-mode script (`st_validation/feature_extraction/1main_MultiFMExtract.py`),
which extracts tiles via skimage as `slide_img[pixel_x-h:pixel_x+h, pixel_y-h:pixel_y+h, :]`.
That treats `pixel_x` as the ROW index (numpy axis 0) and `pixel_y` as the COL
index — the opposite of standard Visium / openslide convention, but it's what
generates the paper's saved features and predictions. We swap accordingly when
reading via openslide so output matches paper bit-for-bit. Pass
`paper_compatible=False` to use the standard pixel_x=col, pixel_y=row convention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from .slide import SlideReader

log = logging.getLogger(__name__)


class GridTiler:
    """
    Axis-aligned, non-overlapping tile grid over an entire slide.

    Parameters
    ----------
    reader : SlideReader
    tile_size_px : int
        Side length of each tile in slide-pixel units. Use the slide's
        `tile_size_px_100_micron` value for MPP-standardized tiling.
    """

    def __init__(self, reader: SlideReader, tile_size_px: int, shuffle: bool = False, seed: int = 0):
        """
        Parameters
        ----------
        shuffle : if True, iterate tiles in random order (seeded by `seed`).
            Useful when combined with `iter_tiles(limit=N)` for a
            representative subsample of the slide. Default False (raster
            order, matches the reference scripts).
        """
        self.reader = reader
        self.tile_size = int(tile_size_px)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    def _coords(self) -> list[tuple[int, int]]:
        width, height = self.reader.dimensions
        xs = np.arange(0, width - self.tile_size + 1, self.tile_size)
        ys = np.arange(0, height - self.tile_size + 1, self.tile_size)
        coords = [(int(x), int(y)) for x in xs for y in ys]
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(coords)
        return coords

    def iter_tiles(self, slide_name: str, limit: int | None = None) -> Iterator[Tuple[str, np.ndarray, int, int]]:
        """
        Yield (tile_name, tile_rgb_uint8, x, y) for every tile in the grid.

        The tile name is f"{slide_name}_{x}x{y}" — matching the format used by
        the reference scripts so downstream code can split coords back out.
        """
        coords = self._coords()
        if limit is not None:
            coords = coords[:limit]
        for x, y in coords:
            tile = self.reader.read_region((x, y), (self.tile_size, self.tile_size))
            if tile.shape[0] == 0 or tile.shape[1] == 0:
                continue
            yield f"{slide_name}_{x}x{y}", tile, x, y


class SpotsTiler:
    """
    Visium-style tiling: one tile per pre-defined spot.

    Parameters
    ----------
    reader : SlideReader
    spots : pd.DataFrame
        Must contain columns: pixel_x, pixel_y, spot_id, and either
        'select' or 'selected' (rows with value 1 are kept).
    tile_size_px : int
        Side length of each tile, centered on (pixel_x, pixel_y).
        Defaults to 224 to match the reference spots-mode script.
    """

    REQUIRED_COLS = ("pixel_x", "pixel_y", "spot_id")

    def __init__(self, reader: SlideReader, spots: pd.DataFrame, tile_size_px: int = 224,
                 paper_compatible: bool = True):
        self.reader = reader
        self.tile_size = int(tile_size_px)
        self.paper_compatible = bool(paper_compatible)
        self.spots = self._normalize(spots)

    @staticmethod
    def _normalize(spots: pd.DataFrame) -> pd.DataFrame:
        df = spots.copy()
        if "selected" in df.columns and "select" not in df.columns:
            df = df.rename(columns={"selected": "select"})
        for c in SpotsTiler.REQUIRED_COLS:
            if c not in df.columns:
                raise ValueError(f"spots table missing required column: {c}")
        if "select" in df.columns:
            df = df[df["select"] == 1]
        return df.sort_values(["pixel_x", "pixel_y"]).reset_index(drop=True)

    @classmethod
    def from_path(cls, reader: SlideReader, spots_path: str | Path, tile_size_px: int = 224,
                  paper_compatible: bool = True) -> "SpotsTiler":
        p = str(spots_path)
        if p.endswith(".csv"):
            spots = pd.read_csv(p)
        elif p.endswith(".pkl") or p.endswith(".pickle"):
            spots = pd.read_pickle(p)
        else:
            raise ValueError(f"Unsupported spots file: {spots_path}")
        return cls(reader, spots, tile_size_px=tile_size_px, paper_compatible=paper_compatible)

    def iter_tiles(self, limit: int | None = None) -> Iterator[Tuple[str, np.ndarray, int, int]]:
        """
        Yield (spot_id, tile_rgb_uint8, pixel_x, pixel_y).

        Tiles are centered on (pixel_x, pixel_y) using openslide / numpy
        standard convention (pixel_x = column, pixel_y = row).
        Spots whose tile falls partially outside the slide are still yielded
        (the SlideReader clamps), but the SlideReader returns an HxWx3 array
        possibly smaller than (tile_size, tile_size) — evaluate_tile will
        reject such tiles downstream.
        """
        half = self.tile_size // 2
        rows = self.spots.itertuples(index=False)
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                break
            x = int(round(row.pixel_x))
            y = int(round(row.pixel_y))
            # Paper-compatible mode swaps the axes when reading: the reference
            # spots-mode script does `slide_img[pixel_x:pixel_x+T, pixel_y:pixel_y+T, :]`,
            # which is `arr[row, col] = arr[pixel_x, pixel_y]` — treating pixel_x
            # as row. Equivalent openslide call is read_region((pixel_y, pixel_x), T),
            # i.e. swap the args.
            if self.paper_compatible:
                top_col, top_row = y - half, x - half  # openslide expects (col, row)
            else:
                top_col, top_row = x - half, y - half
            tile = self.reader.read_region((top_col, top_row), (self.tile_size, self.tile_size))
            if tile.size == 0 or min(tile.shape[:2]) == 0:
                log.debug("Skipping empty tile for spot %s", row.spot_id)
                continue
            yield str(row.spot_id), tile, x, y
