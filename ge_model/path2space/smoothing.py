"""
KDTree-based spatial gene-expression smoothing.

Copied from the reference scripts and parameterized. Two coordinate
conventions are supported, matching the two reference pipelines:

  - 'grid'  : smooth in integer grid coordinates with `radius=2`.
              Matches Cell_revisions/prediction_pipline/1main_enable_medicine_brca.py.
              Tile-size independent, MPP-independent — recommended default.

  - 'pixel' : smooth in pixel coordinates with `radius=tile_size_px`.
              Matches Cell_revisions/prediction_pipline/1main_tcga.py.
              The radius scales with the tile size used for grid-mode tiling.

Both modes produce a copy of `slide_df` with gene columns replaced by the
neighborhood-mean values.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def smooth_genes_kdtree(
    slide_df: pd.DataFrame,
    genes: Sequence[str],
    radius: float = 2.0,
    coord_cols: tuple[str, str] = ("grid_x", "grid_y"),
    weights: str | float = "uniform",
) -> pd.DataFrame:
    """
    Spatial smoothing over neighboring spots.

    Parameters
    ----------
    slide_df : DataFrame indexed by spot/tile name with at least `coord_cols`
        and one column per gene in `genes`.
    genes : list of gene-column names to smooth.
    radius : neighborhood radius in the units of `coord_cols`.
    coord_cols : ('grid_x', 'grid_y') for grid-coordinate smoothing (default,
        recommended) or ('x', 'y') for pixel-coordinate smoothing.
    weights : 'uniform' for an arithmetic mean over neighbors, or a numeric
        weight w to up-weight the central spot:
            smooth_i = ((K * mean_neighbors) + (w - 1) * value_i) / (K + w - 1)
        where K = #neighbors. Matches the reference implementation.

    Returns
    -------
    A copy of `slide_df` with the `genes` columns replaced by their smoothed
    values. Spots with no neighbors keep their original value.
    """
    coords = slide_df[list(coord_cols)].values
    gene_data = slide_df[list(genes)].values

    tree = cKDTree(coords)
    indices = [tree.query_ball_point(p, r=radius) for p in coords]

    smoothed = np.zeros_like(gene_data)
    for i, neighbors in enumerate(indices):
        if neighbors:
            m = gene_data[neighbors].mean(axis=0)
            if weights != "uniform":
                w = float(weights)
                k = len(neighbors)
                m = (k * m + (w - 1) * gene_data[i]) / (k + w - 1)
            smoothed[i] = m
        else:
            smoothed[i] = gene_data[i]

    out = slide_df.copy()
    out[list(genes)] = smoothed
    return out
