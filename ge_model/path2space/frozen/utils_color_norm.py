"""
Stain normalization based on the method of:

M. Macenko et al., 'A method for normalizing histology slides for quantitative analysis',
in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107-1110.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html
(https://anaconda.org/conda-forge/python-spams
https://pypi.org/project/spams/)

FROZEN — do not modify.
Copied verbatim from Cell_revisions/prediction_pipline/func/utils_color_norm.py;
only commented-out experimental cvxpy/spams variants and unused viz helpers
have been removed.
"""

from __future__ import division

import numpy as np
import cv2 as cv
import spams


def read_image(path):
    """Read an image to RGB uint8."""
    im = cv.imread(path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im


def standardize_brightness(I):
    p = np.percentile(I, 95)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(I):
    """Remove zeros from a uint8 array, replace with 1's."""
    mask = (I == 0)
    I[mask] = 1
    return I


def RGB_to_OD(I):
    """Convert from RGB to optical density."""
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def OD_to_RGB(OD):
    """Convert from optical density to RGB."""
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def normalize_rows(A):
    return A / np.linalg.norm(A, axis=1)[:, None]


def notwhite_mask(I, thresh=0.8):
    """Binary mask where true denotes 'not white'."""
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)


def sign(x):
    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0


def get_concentrations(I, stain_matrix, lamda=0.01):
    """Get concentrations, an (npix, 2) matrix."""
    OD = RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T


def get_stain_matrix(I, beta=0.15, alpha=1):
    """Get stain matrix (2x3)."""
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[(OD > beta).any(axis=1), :]
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0:
        V[:, 0] *= -1
    if V[0, 1] < 0:
        V[:, 1] *= -1
    That = np.dot(OD, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    return normalize_rows(HE)


class macenko_normalizer(object):
    """A stain normalization object. FROZEN — do not modify target matrix/constants."""

    def __init__(self):
        self.stain_matrix_target = np.array(
            [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], dtype=np.float32).T
        self.target_concentrations = None

    def fit(self, target):
        target = standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)

    def target_stains(self):
        return OD_to_RGB(self.stain_matrix_target)

    def transform(self, I):
        I = standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        maxC_target = np.array([1.9705, 1.0308], dtype=float).reshape((1, 2))
        source_concentrations *= maxC_target / maxC_source
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(np.uint8)

    def hematoxylin(self, I):
        I = standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H
