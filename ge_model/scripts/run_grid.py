#!/usr/bin/env python
"""
CLI: run path2space in grid (WSI) mode.

Example:
    python scripts/run_grid.py \\
        --slide /path/to/slide.svs \\
        --out   /path/to/out_dir \\
        --tile-size 224
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from path2space import Path2SpacePipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="path2space grid (WSI) prediction.")
    ap.add_argument("--slide", required=True, type=Path, help="Path to the H&E slide.")
    ap.add_argument("--out", required=True, type=Path, help="Output directory.")
    ap.add_argument("--tile-size", type=int, default=224, help="Tile size in pixels (default 224; "
                                                                "set to tile_size_px_100_micron for MPP-standardized tiling).")
    ap.add_argument("--slide-name", default=None, help="Override slide name (default: filename stem).")
    ap.add_argument("--edge-mag", type=int, default=15)
    ap.add_argument("--edge-fraction", type=float, default=0.5,
                    help="evaluate_tile threshold; 0.5 matches 1main_tcga.py.")
    ap.add_argument("--no-smooth", action="store_true", help="Skip KDTree smoothing.")
    ap.add_argument("--limit", type=int, default=None, help="Cap on # of tiles (debug only).")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log.upper(), format="%(levelname)s %(name)s: %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    name = args.slide_name or args.slide.stem

    pipe = Path2SpacePipeline()
    out = pipe.predict_grid(
        slide_path=args.slide,
        slide_name=name,
        tile_size_px=args.tile_size,
        edge_mag_threshold=args.edge_mag,
        edge_fraction_threshold=args.edge_fraction,
        smooth=not args.no_smooth,
        tile_limit=args.limit,
    )

    pred_path = args.out / f"{name}_pred.pkl"
    out["pred"].to_pickle(pred_path)
    print(f"wrote {pred_path}  ({len(out['pred'])} tiles, {len(out['pred'].columns) - 5} genes)")

    if "smooth" in out:
        smooth_path = args.out / f"{name}_smooth.pkl"
        out["smooth"].to_pickle(smooth_path)
        print(f"wrote {smooth_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
