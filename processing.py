from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp  # noqa: F401  # Cupy may be used by future GPU extensions
    CUPY_AVAILABLE = True
except Exception as exc:  # pragma: no cover - best effort detection
    logger.warning("cupy not available: %s", exc)
    CUPY_AVAILABLE = False

try:
    CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception as exc:  # pragma: no cover - best effort detection
    logger.warning("cv2.cuda not available: %s", exc)
    CUDA_AVAILABLE = False

from io_utils import to_uint8

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


def gpu_enabled(use_gpu: bool | None = None) -> bool:
    """Return True if GPU processing should be used."""
    if use_gpu is None:
        return CUDA_AVAILABLE
    return bool(use_gpu and CUDA_AVAILABLE)


def gaussian_blur(img: np.ndarray, ksize: int, sigma: float, use_gpu: bool | None = None) -> np.ndarray:
    """Gaussian blur that optionally executes on the GPU."""
    if ksize <= 0:
        return img
    if gpu_enabled(use_gpu):
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(img)
            blurred = cv2.cuda_GaussianBlur(gpu, (ksize, ksize), sigma)
            return blurred.download()
        except Exception as exc:  # pragma: no cover - fall back to CPU
            logger.warning("CUDA GaussianBlur failed, falling back to CPU: %s", exc)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def warp_affine(
    img: np.ndarray,
    matrix: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Affine warp that optionally executes on the GPU."""
    if gpu_enabled(use_gpu):
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(img)
            warped = cv2.cuda.warpAffine(gpu, matrix, dsize, flags=flags, borderMode=border_mode)
            return warped.download()
        except Exception as exc:  # pragma: no cover - fall back to CPU
            logger.warning("CUDA warpAffine failed, falling back to CPU: %s", exc)
    return cv2.warpAffine(img, matrix, dsize, flags=flags, borderMode=border_mode)


def clahe_equalize(img: np.ndarray) -> np.ndarray:
    """Apply contrast limited adaptive histogram equalization."""
    img8 = to_uint8(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img8)


def complement(img: np.ndarray) -> np.ndarray:
    """Return the complement (inverted) of a grayscale image."""
    return 255 - to_uint8(img)


def register_ecc(
    moving: np.ndarray,
    fixed: np.ndarray,
    params: RegSegParams,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Register *moving* to *fixed* using OpenCV ECC with an affine model.

    If ``use_gpu`` is ``True`` (or ``None`` and CUDA is available), Gaussian blur and
    affine warping are executed via :mod:`cv2.cuda` with automatic fallback to CPU.
    """
    use_gpu = gpu_enabled(use_gpu)
    if params.gausBlurDif > 0:
        kf = ksize_from_sigma(params.gausBlurDif)
        fixed_blur = gaussian_blur(fixed, kf, params.gausBlurDif, use_gpu) if kf > 0 else fixed
    else:
        fixed_blur = fixed
    if params.gausBlurIn > 0:
        km = ksize_from_sigma(params.gausBlurIn)
        moving_blur = gaussian_blur(moving, km, params.gausBlurIn, use_gpu) if km > 0 else moving
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
    registered = warp_affine(
        moving,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REPLICATE,
        use_gpu=use_gpu,
    )
    return registered


def segment_image(
    img: np.ndarray, params: RegSegParams, use_gpu: bool | None = None
) -> np.ndarray:
    """Segment an image using thresholding, morphology and optional Chan–Vese.

    When ``use_gpu`` is ``True`` (or ``None`` and CUDA is available) thresholding and
    morphology are attempted on the GPU with transparent fallback to CPU.
    """
    g = to_uint8(img)
    use_gpu = gpu_enabled(use_gpu)
    if use_gpu:
        try:
            g_gpu = cv2.cuda_GpuMat()
            g_gpu.upload(g)
            if params.bwThresh >= 0:
                _, BW_gpu = cv2.cuda.threshold(g_gpu, float(int(params.bwThresh)), 255, cv2.THRESH_BINARY)
            else:
                _, BW_gpu = cv2.cuda.threshold(g_gpu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            BW = BW_gpu.download()
        except Exception as exc:  # pragma: no cover - fall back to CPU
            logger.warning("CUDA threshold failed, falling back to CPU: %s", exc)
            if params.bwThresh >= 0:
                _, BW = cv2.threshold(g, int(params.bwThresh), 255, cv2.THRESH_BINARY)
            else:
                _, BW = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
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
    if use_gpu:
        try:
            bw_gpu = cv2.cuda_GpuMat()
            bw_gpu.upload(BW_filled)
            morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, bw_gpu.type(), rect)
            BW_open = morph.apply(bw_gpu).download()
        except Exception as exc:  # pragma: no cover - fall back to CPU
            logger.warning("CUDA morphology failed, falling back to CPU: %s", exc)
            BW_open = cv2.morphologyEx(BW_filled, cv2.MORPH_OPEN, rect, iterations=1)
    else:
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
