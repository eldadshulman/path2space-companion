"""
CTransPath feature extractor.

Wraps the frozen CTransPath architecture + the published `ctranspath.pth`
weights into a clean batched-inference class.

The mean/std and the (Resize -> ToTensor -> Normalize) ordering are inherited
from the frozen utils_preprocessing.tile_transform helper to guarantee parity
with the reference scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .frozen.ctrans import CTransPath


def _tile_transform_single(tile, data_mean, data_std, tile_size):
    """
    Per-tile transform matching the inline `tile_transform_single` in the
    reference scripts (1main_tcga.py, 1main_enable_medicine_brca.py,
    1main_MultiFMExtract.py).

    NOTE: uses torchvision's default Resize interpolation (BILINEAR), NOT the
    BICUBIC used by the (dead) `tile_transform` helper in
    func/utils_preprocessing.py. The reference scripts never call the helper —
    they define their own inline version with default-BILINEAR. Matching them
    here is required for numerical parity against the paper predictions.
    """
    # antialias=False forces the historical torchvision default for PIL inputs
    # (False before 0.17, True after). The paper was generated with the False
    # behavior; matching it is needed for ~bit-for-bit parity.
    data_transform = transforms.Compose([
        transforms.Resize(tile_size, antialias=False),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ])
    return data_transform(tile).unsqueeze(0)

log = logging.getLogger(__name__)


CTRANSPATH_MEAN = (0.485, 0.456, 0.406)
CTRANSPATH_STD = (0.229, 0.224, 0.225)
CTRANSPATH_FEATURE_DIM = 768
CTRANSPATH_TILE_SIZE = 224


class CTransPathExtractor:
    """
    Stateful batched feature extractor.

    Usage
    -----
    >>> extractor = CTransPathExtractor("weights/ctranspath.pth")
    >>> features = extractor.extract([tile_pil_1, tile_pil_2, ...])  # (N, 768)
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: str | torch.device | None = None,
        batch_size: int = 128,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

        self.model = CTransPath(num_classes=0).to(self.device)
        state = torch.load(str(weights_path), map_location=self.device)
        # CTransPath checkpoint stores weights under "model" key.
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()

        self.mean = CTRANSPATH_MEAN
        self.std = CTRANSPATH_STD

    @torch.no_grad()
    def _extract_batch(self, batch_tiles: Sequence[Image.Image]) -> np.ndarray:
        if not batch_tiles:
            return np.zeros((0, CTRANSPATH_FEATURE_DIM), dtype=np.float32)
        # Per-tile transform then concat, matching the reference loop body.
        per_tile = [
            _tile_transform_single(t, self.mean, self.std, CTRANSPATH_TILE_SIZE)
            for t in batch_tiles
        ]
        x = torch.cat(per_tile, dim=0).to(self.device).float()
        y = self.model(x)
        return y.cpu().numpy()

    @torch.no_grad()
    def extract(self, tiles: Iterable[Image.Image]) -> np.ndarray:
        """
        Extract features for an iterable of PIL.Image tiles.

        Returns (N, 768) float32 array, one row per input tile in the original
        iteration order.
        """
        chunks: list[np.ndarray] = []
        batch: list[Image.Image] = []
        for tile in tiles:
            batch.append(tile)
            if len(batch) == self.batch_size:
                chunks.append(self._extract_batch(batch))
                batch.clear()
        if batch:
            chunks.append(self._extract_batch(batch))
        if not chunks:
            return np.zeros((0, CTRANSPATH_FEATURE_DIM), dtype=np.float32)
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
