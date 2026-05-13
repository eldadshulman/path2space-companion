"""
Tile-quality filter (evaluate_tile) and reproducibility helper (init_random_seed).

FROZEN — do not modify.
Copied from Cell_revisions/prediction_pipline/func/utils_preprocessing.py.
The unused Feature_Extraction (ResNet50) class, the unused tiles2features
multi-encoder dispatcher, and the dead ViT/DINO branches have been removed
because they were never reached by the path2space pipeline.

What remains:
  - tile_transform: Compose(Resize -> ToTensor -> Normalize). Frozen ordering
    and arguments. Used by the feature extractor.
  - evaluate_tile: Sobel-edge based tissue-quality filter. Reads the same
    EDGE_MAG_THRESHOLD / EDGE_FRACTION_THRESHOLD knobs as the reference scripts.
  - init_random_seed: seeds Python, torch, cuDNN for determinism.
"""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def tile_transform(tiles_list, data_mean, data_std, device=None):
    """
    Apply Resize(224, BICUBIC) -> ToTensor -> Normalize(mean, std) to each tile.

    The ordering, the BICUBIC interpolation, and the CTransPath-specific
    mean/std passed in by the caller are part of the frozen pipeline.

    Returns a single (N, 3, 224, 224) tensor on `device` (CPU if device is None).
    """
    if device is None:
        device = torch.device('cpu')

    data_transform = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std)
    ])

    tiles = []
    for tile in tiles_list:
        transformed_tile = data_transform(tile).unsqueeze(0).to(device)
        tiles.append(transformed_tile)

    return torch.cat(tiles, dim=0)


def evaluate_tile(img_np, edge_mag_thrsh, edge_fraction_thrsh):
    """
    Sobel-edge tissue-quality filter.

    Returns 1 if the tile passes (enough edge content -> tissue),
    0 if the tile is rejected (mostly background / blurry).

    A tile is REJECTED when the fraction of weak-gradient pixels
    (magnitude below `edge_mag_thrsh`) exceeds `edge_fraction_thrsh`.
    """
    select = 1

    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)

    sobelx1 = cv2.convertScaleAbs(sobelx)
    sobely1 = cv2.convertScaleAbs(sobely)
    mag = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5, 0)

    unique, counts = np.unique(mag, return_counts=True)
    edge_mag = counts[np.argwhere(unique < edge_mag_thrsh)].sum() / (img_np.shape[0] * img_np.shape[1])

    if edge_mag > edge_fraction_thrsh:
        select = 0

    return select


def init_random_seed(random_seed=42):
    """Set Python/torch/cuDNN seeds for reproducibility."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Disable TF32 — the paper predictions were generated before TF32 became the
    # default for matmul (PyTorch >=1.12). With TF32 enabled, matmul drifts by
    # ~1e-4 per op, compounding through Swin's hundreds of layers.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
