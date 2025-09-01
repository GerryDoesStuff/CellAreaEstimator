from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from openpyxl import Workbook, load_workbook
from PyQt6.QtGui import QImage


def imread_gray(path: Path | str) -> np.ndarray:
    """Read an image from *path* and ensure it is grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize an array to uint8 (0–255)."""
    if img.dtype == np.uint8:
        return img
    img = np.asarray(img, dtype=np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def qimage_from_gray(img: np.ndarray) -> QImage:
    """Convert a grayscale array to a QImage."""
    g = to_uint8(img)
    h, w = g.shape
    return QImage(g.data, w, h, w, QImage.Format.Format_Grayscale8)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_jpgs(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() == ".jpg"]


def num2xlcol(col_num: int) -> str:
    """Convert a 1‑based column index to an Excel column name (A, B, ...)."""
    n = col_num
    col_chars = []
    while n > 0:
        n, r = divmod(n - 1, 26)
        col_chars.append(chr(r + 65))
    return ''.join(reversed(col_chars))


def write_sorted_areas_xlsx(xlsx_path: Path, column_index: int, areas: List[int]) -> None:
    """Write a list of integer areas into *xlsx_path* at column *column_index* (1‑based)."""
    if xlsx_path.exists():
        wb = load_workbook(xlsx_path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
    row = 1
    for area in areas:
        ws.cell(row=row, column=column_index, value=int(area))
        row += 1
    wb.save(xlsx_path)
