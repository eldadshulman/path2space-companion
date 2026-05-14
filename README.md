# Path2Space — companion code

[![CI](https://github.com/eldadshulman/path2space-companion/actions/workflows/ci.yml/badge.svg)](https://github.com/eldadshulman/path2space-companion/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20172521.svg)](https://doi.org/10.5281/zenodo.20172521)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Companion code for:

> **AI-predicted spatial transcriptomics unlocks breast cancer biomarkers from
> pathology.**
> Shulman et al., *Cell* (2026).
> [doi:10.1016/j.cell.2026.04.023](https://doi.org/10.1016/j.cell.2026.04.023)

This repository is organized as a collection of **independent components**.
Each component lives in its own top-level subfolder, has its own
`README.md`, `setup.py`, and tests, and can be installed and used on its
own.

## Components

| Folder | Description |
|---|---|
| [`ge_model/`](ge_model/) | Path2Space gene-expression prediction model. Predicts spot-level spatial transcriptomics (≈14 K genes) from an H&E histopathology slide using a CTransPath feature extractor + a 22 × 7 MLP ensemble. See [`ge_model/README.md`](ge_model/README.md) for installation and usage. |

Future components (deconvolution, cluster identity, SPAND, …) will be added
as additional sibling folders.

## Citation and archives

If you use any code or model in this repository, please cite the paper:

> Shulman, E.D., et al. (2026). AI-predicted spatial transcriptomics unlocks breast cancer biomarkers from pathology. *Cell*. [doi:10.1016/j.cell.2026.04.023](https://doi.org/10.1016/j.cell.2026.04.023)

A machine-readable citation is also available in [`CITATION.cff`](./CITATION.cff).

### Related Zenodo records

| Record | DOI | Contents |
|---|---|---|
| **This repository (versioned code releases)** | [10.5281/zenodo.20172521](https://doi.org/10.5281/zenodo.20172521) | Citable snapshot of the companion code. Updated automatically on each GitHub release. Cite this for the code. |
| **Trained model weights** | [10.5281/zenodo.20174301](https://doi.org/10.5281/zenodo.20174301) | Trained model weights (154-checkpoint MLP ensemble + CTransPath). Required to run inference on new data. |
| **HEST demo data** | [10.5281/zenodo.20183759](https://doi.org/10.5281/zenodo.20183759) | HEST NCBI776 Visium slide used as the demo notebook input. Auto-fetched by the notebook on first run. |
| **Original training scripts and bundled data (Jan 2025 snapshot)** | [10.5281/zenodo.20171390](https://doi.org/10.5281/zenodo.20171390) | The training-era codebase, tutorial input data, example outputs, and trained weights (~4 GB). |

The `ge_model/` component fetches the trained weights from the weights record above; see [`ge_model/README.md`](./ge_model/README.md) for the expected layout under `ge_model/weights/`.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
