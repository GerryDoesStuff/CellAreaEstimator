from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import logging

from io_utils import to_uint8

logger = logging.getLogger(__name__)

try:
    from skimage.segmentation import morphological_chan_vese
    SKIMAGE_AVAILABLE = True
except Exception as exc:
    logger.warning("skimage not available; Chan–Vese segmentation disabled: %s", exc)
    SKIMAGE_AVAILABLE = False


@dataclass
class RegSegParams:
    """Container for registration and segmentation parameters."""
    numSpaSam: int = 5000       # not directly used by ECC but preserved for UI
    numHisBin: int = 64         # not used by ECC
    allPix: bool = True         # not used by ECC
    groFac: float = 1.5         # not used by ECC
    epsilon: float = 1e-6       # ECC termination epsilon
    iniRad: float = 1.0         # not used by ECC
    maxIter: int = 200          # ECC max iterations
    gausBlurDif: float = 2.0    # Gaussian sigma for difference mask
    gausBlurIn: float = 2.0     # Gaussian sigma for input image
    bwThresh: int = 50          # threshold (>=0) or Otsu (<0)
    recMaSize: int = 3          # rectangular opening kernel size
    segIter: int = 50           # Chan–Vese iterations if available


def ksize_from_sigma(sigma: float) -> int:
    """Given a Gaussian sigma, compute an odd kernel size for OpenCV."""
    if sigma <= 0:
        return 0
    k = int(6 * sigma + 1)
    return k if k % 2 == 1 else k + 1


def clahe_equalize(img: np.ndarray) -> np.ndarray:
    """Apply contrast limited adaptive histogram equalization."""
    img8 = to_uint8(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img8)


def complement(img: np.ndarray) -> np.ndarray:
    """Return the complement (inverted) of a grayscale image."""
    return 255 - to_uint8(img)


def register_ecc(moving: np.ndarray, fixed: np.ndarray, params: RegSegParams) -> np.ndarray:
    """Register *moving* to *fixed* using OpenCV ECC with an affine model."""
    if params.gausBlurDif > 0:
        kf = ksize_from_sigma(params.gausBlurDif)
        fixed_blur = cv2.GaussianBlur(fixed, (kf, kf), params.gausBlurDif) if kf > 0 else fixed
    else:
        fixed_blur = fixed
    if params.gausBlurIn > 0:
        km = ksize_from_sigma(params.gausBlurIn)
        moving_blur = cv2.GaussianBlur(moving, (km, km), params.gausBlurIn) if km > 0 else moving
    else:
        moving_blur = moving
    fb = to_uint8(fixed_blur).astype(np.float32) / 255.0
    mb = to_uint8(moving_blur).astype(np.float32) / 255.0
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, params.maxIter, params.epsilon)
    try:
        cv2.findTransformECC(fb, mb, warp_matrix, cv2.MOTION_AFFINE, criteria)
    except cv2.error as exc:
        logger.warning("ECC registration failed: %s", exc)
    h, w = fixed.shape
    registered = cv2.warpAffine(moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return registered


def segment_image(img: np.ndarray, params: RegSegParams) -> np.ndarray:
    """Segment an image using thresholding, morphology and optional Chan–Vese."""
    g = to_uint8(img)
    if params.bwThresh >= 0:
        _, BW = cv2.threshold(g, int(params.bwThresh), 255, cv2.THRESH_BINARY)
    else:
        _, BW = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - BW
    h, w = inv.shape
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood = inv.copy()
    cv2.floodFill(flood, mask, (0, 0), 0)
    holes = inv - flood
    BW_filled = BW.copy()
    BW_filled[holes > 0] = 255
    k = max(1, int(params.recMaSize))
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    BW_open = cv2.morphologyEx(BW_filled, cv2.MORPH_OPEN, rect, iterations=1)
    if SKIMAGE_AVAILABLE and params.segIter > 0:
        img01 = g.astype(np.float32) / 255.0
        init_ls = BW_open.astype(bool)
        cv_res = morphological_chan_vese(img01, iterations=params.segIter, init_level_set=init_ls, smoothing=1)
        BW_refined = (cv_res.astype(np.uint8) * 255)
    else:
        BW_refined = BW_open
    return BW_refined


def connected_component_areas(binary: np.ndarray) -> List[int]:
    """Compute the areas of connected components in a binary mask (excluding background)."""
    b = (binary > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    return [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
