# Path2Space — gene-expression model (`ge_model/`)

Self-contained, reproducible inference pipeline for the gene-expression
component of Path2Space (Shulman et al., *Cell* 2026):
**H&E image → spatial gene-expression matrix (rows = tiles/spots, columns = genes).**

This is one component of the larger
[`path2space-companion`](../README.md) repository; it can be installed and
used on its own.

```
┌────────────┐   ┌────────────┐   ┌────────────────┐   ┌──────────────┐   ┌────────────┐
│ H&E slide  │ → │ Tile grid  │ → │ Macenko color  │ → │ CTransPath   │ → │ 154-MLP    │ → spots × genes
│ (svs/tiff) │   │ or spots   │   │ normalization  │   │ feature ext. │   │ ensemble   │
└────────────┘   └────────────┘   └────────────────┘   └──────────────┘   └────────────┘
                  evaluate_tile        spams (frozen)        768-dim         22 × 7 folds
                  filter
```

## Quick start

```bash
git clone https://github.com/eldadshulman/path2space-companion.git
cd path2space-companion/ge_model

conda env create -f environment.yml
conda activate path2space-companion
pip install -e .
```

### SPAMS install troubleshooting

SPAMS (used for Macenko stain normalization) can be finicky to install. If
`conda env create` fails on the SPAMS step, install it explicitly from
conda-forge after the env is created:

```bash
conda install -n path2space-companion -c conda-forge python-spams
```

On systems without conda, build dependencies (BLAS, LAPACK, a C++ compiler)
are required and macOS in particular often needs `brew install openblas`
first. As a last resort, `pip install spams-bin` provides a prebuilt wheel
on Linux.

Verify with `python -c "import spams; print(spams.__file__)"`.

### Get the weights

The trained weights are too large for git (~6.7 GB total) and live outside
this repo. You have three options:

1. **Direct download** (recommended for end users) — download the
   `ge_model_weights.tar.gz` artifact from the
   [Zenodo deposit](https://doi.org/10.5281/zenodo.14729337) and unpack it
   inside `ge_model/` so the tree looks like
   `ge_model/weights/{ctranspath.pth, mlp_ensemble/, genes.txt}`.
2. **Copy from the source** (Ruppin Lab members only) — run
   `bash scripts/copy_weights.sh` from a host with read access to
   `/data/Ruppin_ST/`. This will copy ctranspath.pth + 154 MLP checkpoints + gene
   list into `weights/`.
3. **Custom paths** — set env vars before running the copy script:
   `PATH2SPACE_CTRANSPATH_SRC`, `PATH2SPACE_ENSEMBLE_SRC`, `PATH2SPACE_GENES_SRC`.

### Run

```bash
# WSI / grid mode:
python scripts/run_grid.py --slide /path/to/slide.svs --out /tmp/out --tile-size 224

# Visium / spots mode:
python scripts/run_spots.py --slide /path/to/slide.tif --spots /path/to/spots.csv --out /tmp/out
```

Or, as a library:

```python
from path2space import Path2SpacePipeline

pipe = Path2SpacePipeline()
out = pipe.predict_grid("slide.svs", tile_size_px=224)
out["pred"]    # raw per-tile predictions
out["smooth"]  # KDTree-smoothed version
```

## Pipeline stages

1. **Slide loading.** OpenSlide by default; falls back to `skimage.io.imread`
   if OpenSlide cannot open the file (plain PNG/JPEG/non-pyramidal TIFF).
2. **Tile coordinates.** `GridTiler` lays a non-overlapping grid; `SpotsTiler`
   reads pixel coords from a Visium-style CSV/PKL.
3. **Tile-quality filter (frozen).** `evaluate_tile` thresholds the
   Sobel-edge-magnitude histogram (see *Reproducibility* below).
4. **Color normalization (frozen).** Macenko stain normalization via SPAMS.
5. **Feature extraction.** CTransPath (SwinTransformer + ConvStem) →
   768-dim per tile.
6. **Ensemble MLP.** Two-layer ReLU MLP × 154 checkpoints (22 ik-fold × 7
   il-fold). Averaging order matches the paper: inner mean over il-folds,
   then outer mean over ik-folds.
7. **Optional smoothing.** KDTree neighborhood mean in grid coordinates
   (radius=2 by default; see `path2space.smoothing`).

## Reproducibility

### Measured paper-parity on HEST NCBI776 (spots mode)

`examples/predict_demo.ipynb` reproduces the paper's saved predictions for
the public HEST NCBI776 sample and reports:

| metric | value |
|---|---|
| paper spots                   | 4295 |
| companion spots               | 4296 |
| common spots (compared)       | 4295 |
| max abs diff (any gene)       | 0.52 |
| mean abs diff (across genes)  | 7.4e-3 |
| **per-gene Pearson r (median)** | **0.977** |
| per-gene r > 0.99             | 0.2% |

The companion accepts one extra spot relative to the paper's filter (likely a
borderline tile near the edge-magnitude threshold); the 4295 common spots
are compared on the gene matrix.

The remaining ~2% drift is residual SPAMS stain-matrix noise and float
reduction order; the predictions are quantitatively faithful to the paper at
a per-gene level. (Bit-for-bit would require pinning the exact SPAMS revision
and processing each tile individually rather than in batches.)

### Reference implementations

The reference implementations these entry points were refactored from are
maintained internally by the Ruppin Lab and are available on request.

### What is frozen (do not modify)

`path2space/frozen/` is paper-canonical and must not be modernized:

- `utils_preprocessing.py` — `tile_transform` (Resize 224 BICUBIC → ToTensor →
  Normalize with `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`),
  `evaluate_tile` (Sobel-edge histogram), `init_random_seed`.
- `utils_color_norm.py` — Macenko `macenko_normalizer` with the hard-coded
  target stain matrix and `maxC_target = [1.9705, 1.0308]`.
- `ctrans/` — CTransPath `SwinTransformer` + `ConvStem` architecture; weights
  must be the published `ctranspath.pth`.

Changes here will silently change predictions. The dead/commented code from
the originals (vit/dino branches, ResNet50, cvxpy `get_concentrations` variants,
viz helpers) has been stripped; everything else is verbatim.

### What was refactored (vs. the reference scripts)

- Two clean entry points (`predict_grid`, `predict_spots`) replacing the two
  monolithic scripts.
- All hard-coded `/data/Ruppin_AI/...` paths replaced by a `weights/` layout
  inside the companion. No external paths at inference time.
- `SlideReader` unifies OpenSlide and skimage with an OpenSlide-first /
  skimage-fallback policy.
- The reference spots-mode script's tile slicing treats `pixel_x` as a row
  index — opposite of the standard Visium convention. **The companion uses
  the standard convention** (`pixel_x = column`, `pixel_y = row`). Spots-mode
  outputs will therefore not match the reference at the pixel level. The
  parity contract is the grid-mode reference.
- A scanner-specific blue-background→white preprocessing step
  (`bg_blue = [220, 240, 255]`, threshold 70) used by one of the reference
  scripts is **not part of the canonical pipeline** and has been dropped.
- `EDGE_FRACTION_THRESHOLD` defaults: 0.5 (grid) and 0.2 (spots), with the
  spots-mode value overridable in the 0.2–0.9 range via `--edge-fraction`.
- The MLP forward pass keeps per-tile predictions (the reference has the
  per-tile `torch.mean` commented out — matches what we do).

## Outputs

Both entry points return a dict:

```python
{
  "pred":   pd.DataFrame,  # indexed by tile/spot id; cols: slide_name, x, y, grid_x, grid_y, <gene_1>...
  "smooth": pd.DataFrame,  # optional; same shape, gene cols spatially smoothed
}
```

`x`, `y` are slide-pixel coordinates of the tile center (grid mode) or spot
(spots mode). `grid_x`, `grid_y` are integer grid indices (computed from `x`/`y`
and the tile size).

## Layout

```
ge_model/
├── README.md
├── environment.yml          # pinned, matches vision_ml
├── setup.py
├── path2space/              # Python package — `import path2space`
│   ├── __init__.py          # exports Path2SpacePipeline, WeightsLayout
│   ├── frozen/              # DO NOT MODIFY
│   │   ├── utils_preprocessing.py
│   │   ├── utils_color_norm.py
│   │   └── ctrans/{ctranspath,swin_transformer}.py
│   ├── model_mlp.py         # MLP_regression_relu_two (inference-only)
│   ├── slide.py             # SlideReader (openslide + skimage fallback)
│   ├── tiler.py             # GridTiler, SpotsTiler
│   ├── features.py          # CTransPathExtractor
│   ├── ensemble.py          # MLPEnsemble (22 × 7)
│   ├── smoothing.py         # KDTree spatial smoothing
│   └── pipeline.py          # Path2SpacePipeline (orchestrator)
├── examples/
│   └── predict_demo.ipynb   # TCGA-BRCA + HEST NCBI776 demo with parity check
├── scripts/
│   ├── copy_weights.sh      # one-time: populate weights/
│   ├── run_grid.py          # CLI for grid mode
│   ├── run_spots.py         # CLI for spots mode
│   └── verify_parity.py     # companion vs the internal grid-mode reference
├── weights/                 # populated by copy_weights.sh (gitignored)
│   ├── ctranspath.pth                                (~107 MB)
│   ├── mlp_ensemble/result_{0..21}_{0..6}_0/model_trained.pth   (154 files, ~6.6 GB)
│   └── genes.txt
└── tests/
    └── test_smoke.py
```
