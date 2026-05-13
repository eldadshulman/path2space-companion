# Path2Space — companion code

[![CI](https://github.com/eldadshulman/path2space-companion/actions/workflows/ci.yml/badge.svg)](https://github.com/eldadshulman/path2space-companion/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14729337.svg)](https://doi.org/10.5281/zenodo.14729337)
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

## Bundled data, weights, and example outputs

Trained weights, tutorial data, and example predictions are available on
Zenodo:

**https://doi.org/10.5281/zenodo.14729337**

Each component's README explains how to drop those artifacts into its
expected layout (typically `<component>/weights/`).

## Citation

If you use any code or model in this repository, please cite the paper —
see [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
