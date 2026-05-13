"""
End-to-end path2space pipeline.

Given an H&E slide, produces a spots-x-genes DataFrame of predicted
expression. Two entry points share the same backend:

  predict_grid(slide_path, tile_size_px, ...)   # 1main_tcga.py-style
  predict_spots(slide_path, spots, ...)         # 1main_enable_medicine_brca.py-style

Stages
------
1. Load slide (openslide first, skimage fallback).
2. Generate tile coordinates (grid auto or from spots table).
3. Tile-quality filter via frozen evaluate_tile.
4. Macenko color normalization via frozen macenko_normalizer.
5. CTransPath feature extraction (768-dim per tile).
6. 22 x 7 MLP ensemble prediction.
7. (optional) KDTree spatial smoothing.

Output: DataFrame indexed by tile/spot id, with columns
['slide_name', 'x', 'y', 'grid_x', 'grid_y', <gene_1>, ..., <gene_N>].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .ensemble import MLPEnsemble, N_IK_FOLDS, N_IL_FOLDS
from .features import CTransPathExtractor, CTRANSPATH_FEATURE_DIM
from .frozen.utils_color_norm import macenko_normalizer
from .frozen.utils_preprocessing import evaluate_tile, init_random_seed
from .slide import SlideReader
from .smoothing import smooth_genes_kdtree
from .tiler import GridTiler, SpotsTiler

log = logging.getLogger(__name__)


# Default edge thresholds match the two reference scripts.
DEFAULT_EDGE_MAG_THRESHOLD = 15
DEFAULT_EDGE_FRACTION_THRESHOLD_GRID = 0.5     # 1main_tcga.py
DEFAULT_EDGE_FRACTION_THRESHOLD_SPOTS = 0.5    # 1main_MultiFMExtract.py (HEST validation; the
                                               # value used to generate the st_preds paper outputs).
                                               # 1main_enable_medicine_brca.py uses 0.2 in its
                                               # current state; pass edge_fraction_threshold=0.2
                                               # explicitly to reproduce that variant.

# Smoothing defaults — match each reference pipeline.
DEFAULT_SMOOTH_GRID = dict(radius=2.0, coord_cols=("grid_x", "grid_y"))
DEFAULT_SMOOTH_SPOTS = dict(radius=2.0, coord_cols=("grid_x", "grid_y"))


@dataclass
class WeightsLayout:
    """Locations of weight artifacts on disk, all relative to the companion root."""

    ctranspath: Path
    mlp_ensemble: Path
    genes: Path  # one gene symbol per line OR a pickle with a 'gene' column

    @classmethod
    def default(cls, package_root: str | Path | None = None) -> "WeightsLayout":
        if package_root is None:
            package_root = Path(__file__).resolve().parent.parent
        root = Path(package_root)
        return cls(
            ctranspath=root / "weights" / "ctranspath.pth",
            mlp_ensemble=root / "weights" / "mlp_ensemble",
            genes=root / "weights" / "genes.txt",
        )


def _load_genes(path: Path) -> list[str]:
    path = Path(path)
    if path.suffix == ".pkl":
        df = pd.read_pickle(path)
        return df["gene"].astype(str).tolist()
    if path.suffix == ".txt":
        return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        col = "gene" if "gene" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()
    raise ValueError(f"Unsupported gene list format: {path}")


class Path2SpacePipeline:
    """
    Stateful pipeline. Loads CTransPath + 154 MLPs once; reusable across slides.

    Parameters
    ----------
    weights : WeightsLayout or None (default).
    device : torch device or None (auto-detect).
    batch_size : CTransPath inference batch size.
    n_ik_folds, n_il_folds : ensemble shape; override only for testing.
    random_seed : seeds Python/torch/cuDNN for reproducibility (default 42).
    """

    def __init__(
        self,
        weights: WeightsLayout | None = None,
        device: str | torch.device | None = None,
        batch_size: int = 128,
        n_ik_folds: int = N_IK_FOLDS,
        n_il_folds: int = N_IL_FOLDS,
        random_seed: int = 42,
    ):
        init_random_seed(random_seed)

        self.weights = weights or WeightsLayout.default()
        for label, p in [
            ("ctranspath weights", self.weights.ctranspath),
            ("MLP ensemble dir", self.weights.mlp_ensemble),
            ("gene list", self.weights.genes),
        ]:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Missing {label}: {p}\n"
                    f"Run scripts/copy_weights.sh first to populate weights/."
                )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        log.info("Loading CTransPath from %s", self.weights.ctranspath)
        self.extractor = CTransPathExtractor(
            weights_path=self.weights.ctranspath,
            device=self.device,
            batch_size=batch_size,
        )

        self.genes = _load_genes(self.weights.genes)
        log.info("Loaded %d output genes", len(self.genes))

        log.info("Loading MLP ensemble (%d x %d) from %s",
                 n_ik_folds, n_il_folds, self.weights.mlp_ensemble)
        self.ensemble = MLPEnsemble(
            ensemble_dir=self.weights.mlp_ensemble,
            n_inputs=CTRANSPATH_FEATURE_DIM,
            n_genes=len(self.genes),
            device=self.device,
            n_ik_folds=n_ik_folds,
            n_il_folds=n_il_folds,
        )

        self.color_normalizer = macenko_normalizer()

    # --------------------------------------------------------------------- #
    # Shared internals                                                      #
    # --------------------------------------------------------------------- #

    def _tiles_to_features(
        self,
        tile_stream,
        edge_mag_threshold: int,
        edge_fraction_threshold: float,
    ) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
        """
        Iterate a tile stream, filter + color-normalize, return features.

        Parameters
        ----------
        tile_stream : iterable of (name, np.uint8 RGB tile, x, y).
        edge_mag_threshold, edge_fraction_threshold : evaluate_tile knobs.

        Returns
        -------
        features : (N, 768) float32 — only tiles passing evaluate_tile.
        names    : list of length N — tile/spot identifiers.
        coords   : list of length N — (x, y) per tile (slide-pixel coords).
        """
        normalized: list[Image.Image] = []
        names: list[str] = []
        coords: list[tuple[int, int]] = []

        for name, tile, x, y in tile_stream:
            if tile.shape[0] == 0 or tile.shape[1] == 0:
                continue
            if not evaluate_tile(tile, edge_mag_threshold, edge_fraction_threshold):
                continue
            try:
                normed = self.color_normalizer.transform(tile)
            except Exception as e:
                log.debug("Color-norm failed for %s (%s); skipping", name, e)
                continue
            normalized.append(Image.fromarray(normed))
            names.append(name)
            coords.append((x, y))

        features = self.extractor.extract(normalized)
        return features, names, coords

    def _predict_from_features(
        self,
        features: np.ndarray,
        names: list[str],
        coords: list[tuple[int, int]],
        slide_name: str,
        tile_size_px: int,
    ) -> pd.DataFrame:
        if features.shape[0] == 0:
            log.warning("No tiles passed quality filtering for %s", slide_name)
            cols = ["slide_name", "x", "y", "grid_x", "grid_y"] + list(self.genes)
            return pd.DataFrame(columns=cols)

        preds = self.ensemble.predict(features)
        xs = np.asarray([c[0] for c in coords], dtype=float)
        ys = np.asarray([c[1] for c in coords], dtype=float)

        grid_x = ((xs - xs.min()) / tile_size_px + 1).astype(int)
        grid_y = ((ys - ys.min()) / tile_size_px + 1).astype(int)

        meta = pd.DataFrame(
            {"slide_name": slide_name, "x": xs, "y": ys, "grid_x": grid_x, "grid_y": grid_y},
            index=pd.Index(names, name="spot_id"),
        )
        gene_df = pd.DataFrame(preds, index=meta.index, columns=self.genes)
        out = pd.concat([meta, gene_df], axis=1).sort_values(["slide_name", "x", "y"])
        return out

    # --------------------------------------------------------------------- #
    # Public entry points                                                   #
    # --------------------------------------------------------------------- #

    def predict_grid(
        self,
        slide_path: str | Path,
        tile_size_px: int,
        slide_name: str | None = None,
        edge_mag_threshold: int = DEFAULT_EDGE_MAG_THRESHOLD,
        edge_fraction_threshold: float = DEFAULT_EDGE_FRACTION_THRESHOLD_GRID,
        smooth: bool = True,
        tile_limit: int | None = None,
        shuffle_tiles: bool = False,
        shuffle_seed: int = 0,
    ) -> dict[str, pd.DataFrame]:
        """
        WSI-mode prediction with an auto-generated tile grid.

        Returns
        -------
        dict with keys:
            'pred'   : raw per-tile predictions (DataFrame).
            'smooth' : KDTree-smoothed predictions (DataFrame), if smooth=True.
        """
        slide_path = Path(slide_path)
        slide_name = slide_name or slide_path.stem

        with SlideReader(slide_path) as reader:
            log.info("Backend: %s; dims=%s", reader.backend, reader.dimensions)
            tiler = GridTiler(reader, tile_size_px=tile_size_px,
                              shuffle=shuffle_tiles, seed=shuffle_seed)
            features, names, coords = self._tiles_to_features(
                tiler.iter_tiles(slide_name, limit=tile_limit),
                edge_mag_threshold,
                edge_fraction_threshold,
            )

        df = self._predict_from_features(features, names, coords, slide_name, tile_size_px)
        out = {"pred": df}
        if smooth and len(df):
            out["smooth"] = smooth_genes_kdtree(df, self.genes, **DEFAULT_SMOOTH_GRID)
        return out

    def predict_spots(
        self,
        slide_path: str | Path,
        spots: pd.DataFrame | str | Path,
        slide_name: str | None = None,
        tile_size_px: int = 224,
        edge_mag_threshold: int = DEFAULT_EDGE_MAG_THRESHOLD,
        edge_fraction_threshold: float = DEFAULT_EDGE_FRACTION_THRESHOLD_SPOTS,
        smooth: bool = True,
        tile_limit: int | None = None,
        paper_compatible: bool = False,
        slide_backend: str = "openslide",
    ) -> dict[str, pd.DataFrame]:
        """
        Visium-mode prediction with predefined spot coordinates.

        Parameters
        ----------
        spots : DataFrame or path to a CSV/PKL with columns
            pixel_x, pixel_y, spot_id, select/selected.
        paper_compatible : if True (default), match the paper's spots-mode
            tile-extraction convention (treats pixel_x as the row index — see
            tiler.SpotsTiler for the rationale).
        slide_backend : "skimage" (default) or "openslide". The paper's HEST
            script tries skimage first; for bit-for-bit parity on .tif slides
            keep this on "skimage".
        """
        slide_path = Path(slide_path)
        slide_name = slide_name or slide_path.stem

        with SlideReader(slide_path, prefer=slide_backend) as reader:
            log.info("Backend: %s; dims=%s", reader.backend, reader.dimensions)
            if isinstance(spots, (str, Path)):
                tiler = SpotsTiler.from_path(reader, spots, tile_size_px=tile_size_px,
                                             paper_compatible=paper_compatible)
            else:
                tiler = SpotsTiler(reader, spots, tile_size_px=tile_size_px,
                                   paper_compatible=paper_compatible)
            features, names, coords = self._tiles_to_features(
                tiler.iter_tiles(limit=tile_limit),
                edge_mag_threshold,
                edge_fraction_threshold,
            )

        df = self._predict_from_features(features, names, coords, slide_name, tile_size_px)
        out = {"pred": df}
        if smooth and len(df):
            out["smooth"] = smooth_genes_kdtree(df, self.genes, **DEFAULT_SMOOTH_SPOTS)
        return out
