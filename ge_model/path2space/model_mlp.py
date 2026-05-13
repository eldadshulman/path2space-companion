"""
MLP_regression_relu_two — inference-only.

Architecture must match the checkpoints in weights/mlp_ensemble/result_{ik}_{il}_0/model_trained.pth
exactly. State-dict keys: layer0.0.{weight,bias}, layer1.0.{weight,bias}.

The original model_MLP.py in the reference repo also contained training/eval
plumbing (training_epoch, fit, analyze_result, ...). None of that is needed
for inference, so it's omitted here.

Note: the per-tile aggregation (`torch.mean(x, dim=0)`) is intentionally NOT
applied — the published path2space pipeline keeps per-tile predictions and
emits one prediction per tile/spot.
"""

from __future__ import annotations

import torch
from torch import nn


class MLP_regression_relu_two(nn.Module):
    """
    Two-layer ReLU MLP for per-tile gene-expression regression.

    Shape contract (must not change — checkpoint compatibility):
      Input:  (..., n_inputs)        # n_inputs = 768 for CTransPath features
      Hidden: Linear -> ReLU -> Dropout(0.2)
      Output: Linear -> ReLU         # (..., n_outputs) = (..., n_genes)
    """

    def __init__(
        self,
        n_inputs: int,
        n_hiddens: int,
        n_outputs: int,
        dropout: float,
        bias_init: torch.Tensor | None = None,
    ):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layer1 = nn.Sequential(
            nn.Linear(n_hiddens, n_outputs),
            nn.ReLU(),
        )

        if bias_init is not None:
            with torch.no_grad():
                self.layer1[0].bias = nn.Parameter(bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        return x
