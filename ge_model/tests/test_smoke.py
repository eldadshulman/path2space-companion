"""
Smoke tests that do not require the trained weights.

Run with:
    cd path2space-companion && python -m pytest tests/ -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skimage.io as skio
import torch

from path2space.frozen.ctrans import CTransPath
from path2space.frozen.utils_color_norm import macenko_normalizer
from path2space.frozen.utils_preprocessing import evaluate_tile, init_random_seed
from path2space.model_mlp import MLP_regression_relu_two
from path2space.slide import SlideReader
from path2space.smoothing import smooth_genes_kdtree
from path2space.tiler import GridTiler, SpotsTiler


def _fake_slide(tmp: Path, w: int = 1000, h: int = 800) -> Path:
    """Create a deterministic RGB image so SlideReader can open it via skimage."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    p = tmp / "fake.tif"
    skio.imsave(str(p), img)
    return p


def test_mlp_forward():
    m = MLP_regression_relu_two(n_inputs=768, n_hiddens=768, n_outputs=50, dropout=0.2)
    out = m(torch.randn(7, 768))
    assert out.shape == (7, 50)
    assert (out >= 0).all(), "output ReLU should be non-negative"


def test_ctranspath_arch():
    model = CTransPath(num_classes=0)
    assert isinstance(model.head, torch.nn.Identity)


def test_evaluate_tile_rejects_blank():
    blank = np.full((224, 224, 3), 240, dtype=np.uint8)
    assert evaluate_tile(blank, 15, 0.5) == 0


def test_evaluate_tile_accepts_textured():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    assert evaluate_tile(noisy, 15, 0.5) == 1


def test_slide_reader_skimage_fallback(tmp_path):
    p = _fake_slide(tmp_path)
    with SlideReader(p) as r:
        assert r.backend in ("openslide", "skimage")
        w, h = r.dimensions
        assert w == 1000 and h == 800
        tile = r.read_region((100, 50), (64, 64))
        assert tile.shape == (64, 64, 3)
        assert tile.dtype == np.uint8


def test_grid_tiler(tmp_path):
    p = _fake_slide(tmp_path, w=500, h=400)
    with SlideReader(p) as r:
        tiles = list(GridTiler(r, tile_size_px=224).iter_tiles("fake"))
    # 500//224 = 2, 400//224 = 1 -> 2 tiles
    assert len(tiles) == 2
    for name, tile, x, y in tiles:
        assert name.startswith("fake_")
        assert tile.shape == (224, 224, 3)


def test_spots_tiler(tmp_path):
    p = _fake_slide(tmp_path, w=2000, h=1500)
    spots = pd.DataFrame({
        "pixel_x": [500, 1000, 1500],
        "pixel_y": [400, 700, 1100],
        "spot_id": ["s0", "s1", "s2"],
        "select": [1, 1, 0],
    })
    with SlideReader(p) as r:
        tiles = list(SpotsTiler(r, spots, tile_size_px=224).iter_tiles())
    assert [t[0] for t in tiles] == ["s0", "s1"]  # s2 filtered out by select==0
    for name, tile, x, y in tiles:
        assert tile.shape == (224, 224, 3)


def test_smoothing_grid_coords():
    df = pd.DataFrame({
        "grid_x": [0, 1, 2, 0, 1, 2],
        "grid_y": [0, 0, 0, 1, 1, 1],
        "GENE_A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "GENE_B": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    # Use radius=1.01 so only the rook-4-neighborhood (incl. self) is captured.
    out = smooth_genes_kdtree(df, genes=["GENE_A", "GENE_B"], radius=1.01)
    assert out.shape == df.shape
    # Spot (1,0): neighbors within dist<=1.01 are (0,0)=1, (2,0)=3, (1,1)=5, (1,0)=2.
    # Mean = (1+3+5+2)/4 = 2.75.
    center = out.loc[(df["grid_x"] == 1) & (df["grid_y"] == 0), "GENE_A"].iloc[0]
    assert abs(center - 2.75) < 1e-9


def test_color_norm_runs():
    """Quick check that Macenko can transform a textured tile (just runs end-to-end)."""
    rng = np.random.default_rng(0)
    img = rng.integers(80, 220, size=(64, 64, 3), dtype=np.uint8)
    norm = macenko_normalizer()
    out = norm.transform(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_init_random_seed_is_reproducible():
    init_random_seed(42)
    a = torch.randn(3, 4)
    init_random_seed(42)
    b = torch.randn(3, 4)
    assert torch.equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
