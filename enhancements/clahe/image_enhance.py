"""CLAHE-based low-light enhancement (multi-step fusion pipeline).

Adapted from ThomasWangWeiHong/Low-Light-Image-Enhancement-CLAHE-Based
(``image_enhance.py``), a reproduction of a published algorithm.

Idea: an inverted low-light image looks like a hazy image, so enhance the
inverted brightness channel, fuse two enhanced versions with PCA-derived
weights, and invert back.

Pipeline (all on the HSV "V" channel; H and S are left untouched):
    1. invert V
    2. CLAHE on the inverted V
    3. gamma (=5) boost -> morphological top-hat (7x7 ellipse) to pull out detail
    4. PCA fusion: weights from the principal eigenvector of the 2x2
       covariance of (CLAHE output, top-hat output)
    5. invert back and write into V

Input/output follow the project contract (enhancements/base.py):
HxWx3 uint8 BGR in -> HxWx3 uint8 BGR out.
"""

from __future__ import annotations

import cv2
import numpy as np

GAMMA = 5
TOPHAT_KERNEL_SIZE = (7, 7)


def make_clahe(clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
    """Build the cv2 CLAHE object used by :func:`low_light_enhance`."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)


def low_light_enhance(image: np.ndarray, clahe) -> np.ndarray:
    """Enhance one low-light BGR uint8 image.

    Args:
        image: HxWx3 uint8 BGR image.
        clahe: object from :func:`make_clahe`.

    Returns:
        HxWx3 uint8 BGR image, same shape as the input.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Invert the intensity channel.
    inverted = 255 - hsv[:, :, 2]

    # 2. CLAHE on the inverted intensity, normalised to [0, 1].
    clahe_i = clahe.apply(inverted).astype(np.float32) / 255.0

    # 3. Gamma boost, then morphological top-hat.
    gamma_u8 = np.clip(np.power(clahe_i, GAMMA) * 255.0, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, TOPHAT_KERNEL_SIZE)
    tophat = cv2.morphologyEx(gamma_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32) / 255.0

    # 4. PCA fusion weights.
    cov = np.cov(np.vstack((clahe_i.ravel(), tophat.ravel())))
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    total = principal.sum()
    if abs(total) < 1e-8:  # degenerate (e.g. perfectly flat image): equal weights
        w1 = w2 = 0.5
    else:
        w1, w2 = principal[0] / total, principal[1] / total
    fused = np.clip(w1 * clahe_i + w2 * tophat, 0.0, 1.0)

    # 5. Invert back and replace V.
    hsv[:, :, 2] = np.rint((1.0 - fused) * 255.0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
