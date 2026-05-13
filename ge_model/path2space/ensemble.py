"""
MLP ensemble inference.

The published path2space model is a 22 x 7 = 154 ensemble of
MLP_regression_relu_two checkpoints, trained on TNBC_BC_data with CTransPath
features. The reference scripts iterate the folds in nested loops:

    preds_ik = []
    for ik in range(22):
        preds_il = []
        for il in range(7):
            load model_trained.pth from result_{ik}_{il}_0
            preds_il.append(predict(features))
        preds_ik.append(mean(preds_il, axis=0))      # inner mean over 7 il-folds
    final = mean(preds_ik, axis=0)                   # outer mean over 22 ik-folds

This module preserves that order exactly. The implementation pre-loads all
154 state_dicts upfront and runs the features through each MLP once.

Outputs an (N_tiles, n_genes) array.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .model_mlp import MLP_regression_relu_two

log = logging.getLogger(__name__)


N_IK_FOLDS = 22
N_IL_FOLDS = 7


class MLPEnsemble:
    """
    Loads and runs the 22 x 7 MLP ensemble.

    Parameters
    ----------
    ensemble_dir : path containing subdirs `result_{ik}_{il}_0/model_trained.pth`
        for ik in 0..21 and il in 0..6.
    n_inputs : feature dimensionality (768 for CTransPath).
    n_genes  : number of output genes.
    device   : torch device.
    n_ik_folds, n_il_folds : override the 22/7 default if running a smaller
                             ensemble for testing.
    """

    def __init__(
        self,
        ensemble_dir: str | Path,
        n_inputs: int,
        n_genes: int,
        device: str | torch.device | None = None,
        n_ik_folds: int = N_IK_FOLDS,
        n_il_folds: int = N_IL_FOLDS,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.ensemble_dir = Path(ensemble_dir)
        self.n_inputs = int(n_inputs)
        self.n_genes = int(n_genes)
        self.n_ik = int(n_ik_folds)
        self.n_il = int(n_il_folds)

        self.models: dict[tuple[int, int], MLP_regression_relu_two] = {}
        self._load_all()

    def _load_all(self) -> None:
        missing: list[Path] = []
        for ik in range(self.n_ik):
            for il in range(self.n_il):
                ckpt = self.ensemble_dir / f"result_{ik}_{il}_0" / "model_trained.pth"
                if not ckpt.exists():
                    missing.append(ckpt)
                    continue
                model = MLP_regression_relu_two(
                    n_inputs=self.n_inputs,
                    n_hiddens=self.n_inputs,
                    n_outputs=self.n_genes,
                    dropout=0.2,
                    bias_init=None,
                ).to(self.device)
                state = torch.load(ckpt, map_location=self.device)
                model.load_state_dict(state)
                model.eval()
                self.models[(ik, il)] = model
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} MLP checkpoint(s) missing from {self.ensemble_dir}. "
                f"First missing: {missing[0]}"
            )
        log.info("Loaded %d MLP checkpoints from %s", len(self.models), self.ensemble_dir)

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run the 22 x 7 ensemble on a feature matrix.

        Parameters
        ----------
        features : (N_tiles, n_inputs) float array.

        Returns
        -------
        (N_tiles, n_genes) float64 array — mean over ik-folds of (mean over
        il-folds of (per-tile MLP prediction)). Order preserved per spec.
        """
        if features.ndim != 2 or features.shape[1] != self.n_inputs:
            raise ValueError(f"features must be (N, {self.n_inputs}); got {features.shape}")

        n_tiles = features.shape[0]
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)

        preds_ik = np.zeros((self.n_ik, n_tiles, self.n_genes), dtype=np.float64)
        for ik in range(self.n_ik):
            preds_il = np.zeros((self.n_il, n_tiles, self.n_genes), dtype=np.float64)
            for il in range(self.n_il):
                model = self.models[(ik, il)]
                preds_il[il] = model(x).cpu().numpy()
            preds_ik[ik] = preds_il.mean(axis=0)
            log.debug("ik=%d done", ik)
        return preds_ik.mean(axis=0)
