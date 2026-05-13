"""
Slide reader.

Single interface for two backends:
  - OpenSlide (default; handles pyramidal .svs/.ndpi/.tiff/.mrxs cheaply)
  - skimage.io.imread fallback (whole image into RAM; used when OpenSlide
    can't open the file — e.g. plain PNG/JPEG/TIFF without the right tile
    structure).

Both backends expose the same `read_region((x, y), size=(w, h))` API so the
tiler doesn't need to care which one is active.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class SlideReader:
    """
    Lazy slide reader. Tries OpenSlide first; falls back to skimage on failure.

    Coordinate convention: `read_region((x, y), size=(w, h))` where (x, y) is the
    top-left pixel in slide coordinates and `w`/`h` are width/height. Returns
    an HxWx3 uint8 RGB numpy array.
    """

    def __init__(self, path: str | Path, prefer: str = "openslide"):
        """
        Parameters
        ----------
        path : slide path.
        prefer : "openslide" (default; falls back to skimage on failure) or
                 "skimage" (loads the whole image into RAM; falls back to
                 openslide on failure). The paper's spots-mode script tries
                 skimage FIRST — set `prefer="skimage"` for bit-for-bit
                 parity on Visium slides.
        """
        if prefer not in ("openslide", "skimage"):
            raise ValueError(f"prefer must be 'openslide' or 'skimage'; got {prefer!r}")
        self.path = str(path)
        self.prefer = prefer
        self._osh = None      # openslide handle
        self._array = None    # skimage array (H, W, 3) uint8
        self._backend = self._open()

    def _open(self) -> str:
        order = ("openslide", "skimage") if self.prefer == "openslide" else ("skimage", "openslide")
        last_err = None
        for backend in order:
            try:
                if backend == "openslide":
                    import openslide
                    self._osh = openslide.OpenSlide(self.path)
                    log.debug("Opened %s with openslide; dims=%s", self.path, self._osh.dimensions)
                    return "openslide"
                else:
                    import skimage.io as skio
                    img = skio.imread(self.path)
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=-1)
                    elif img.shape[-1] == 4:
                        img = img[..., :3]
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    self._array = img
                    log.debug("Opened %s with skimage.io; shape=%s", self.path, img.shape)
                    return "skimage"
            except Exception as e:
                last_err = e
                log.warning("%s failed for %s (%s); trying next backend.", backend, self.path, e)
        raise RuntimeError(f"All slide backends failed for {self.path}: {last_err}")

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dimensions(self) -> Tuple[int, int]:
        """(width, height) of the slide, matching openslide's convention."""
        if self._backend == "openslide":
            return self._osh.dimensions
        h, w = self._array.shape[:2]
        return (w, h)

    def read_region(self, location: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
        """
        Read a tile.

        Parameters
        ----------
        location : (x, y) top-left in slide coordinates (pixels at level 0).
        size     : (w, h) tile size in pixels.

        Returns
        -------
        np.ndarray (h, w, 3) uint8 RGB.
        """
        x, y = location
        w, h = size

        if self._backend == "openslide":
            tile = self._osh.read_region((x, y), level=0, size=(w, h)).convert("RGB")
            return np.asarray(tile)

        # skimage fallback. Mirrors the slicing used by the reference
        # spots-mode script: slide_img[start_x:end_x, start_y:end_y, :].
        # That script treated its first axis as "x" (rows) and second as "y"
        # (cols); we replicate that here for parity.
        arr = self._array
        H, W = arr.shape[:2]
        # Clamp to image bounds
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + w)
        y1 = min(H, y + h)
        # numpy is (row, col, c) = (y, x, c)
        return arr[y0:y1, x0:x1, :]

    def close(self):
        if self._osh is not None:
            self._osh.close()
            self._osh = None
        self._array = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
