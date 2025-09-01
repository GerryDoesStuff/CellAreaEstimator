import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")

import cv2
import numpy as np
from pathlib import Path

from processing import register_ecc, RegSegParams, difference_mask, mask_from_rects

DATA = Path(__file__).parent / "data"

def load(name: str):
    return cv2.imread(str(DATA / name), cv2.IMREAD_UNCHANGED)

def test_registration_cropping_and_mask():
    fixed = load("fixed.pgm")
    moving = load("moving.pgm")
    params = RegSegParams(maxIter=50)
    reg, mask = register_ecc(moving, fixed, params)
    ys, xs = np.nonzero(mask)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    reg_crop = reg[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]
    # Expect shift of 2 pixels leading to 18x18 overlap
    assert reg_crop.shape == (18, 18)
    assert mask_crop.min() == 1 and mask_crop.max() == 1
    diff = cv2.subtract(fixed, reg)
    diff_masked = diff * mask
    assert np.all(diff_masked[mask == 0] == 0)

def test_mask_preparation_dm_bm():
    img_a = load("fixed.pgm")
    img_b = load("moving.pgm")
    dm = difference_mask(img_a, img_b)
    assert dm.shape == img_a.shape
    assert dm.min() >= 0 and dm.max() <= 255
    bm = mask_from_rects(dm.shape, [(2, 2, 5, 5)])
    assert bm.shape == dm.shape
    assert bm.min() >= 0 and bm.max() <= 255
    assert set(np.unique(bm)).issubset({0, 255})


def test_registration_with_roi():
    fixed = load("fixed.pgm")
    moving = load("moving.pgm")
    params = RegSegParams(maxIter=50)
    roi = (2, 2, 10, 10)
    reg, mask = register_ecc(moving, fixed, params, roi=roi)
    assert reg.shape == (10, 10)
    assert mask.shape == (10, 10)
    assert mask.min() == 1 and mask.max() == 1
    fixed_roi = fixed[2:12, 2:12]
    diff = cv2.subtract(fixed_roi, reg)
    assert diff.max() == 0


def test_register_ecc_uses_roi_metadata():
    import json
    fixed = load("syn_full_ascii.pgm")
    params = RegSegParams(maxIter=50)
    with open(DATA / "syn_dm_ascii.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    x, y = meta["offset"]
    w, h = meta["size"]
    roi = (x, y, w, h)
    reg, mask = register_ecc(fixed, fixed, params, roi=roi)
    assert reg.shape == (h, w)
    assert mask.shape == (h, w)
    assert mask.min() == 1 and mask.max() == 1
    assert np.array_equal(reg, fixed[y:y + h, x:x + w])


def test_load_dm_with_sidecar(tmp_path, monkeypatch):
    """MainWindow.load_dm should populate dm_roi from JSON sidecar."""
    import json
    import pytest

    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from gui import MainWindow

    # Ensure headless Qt
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    # Create dummy difference mask and accompanying metadata
    dm_path = tmp_path / "dm.png"
    cv2.imwrite(str(dm_path), np.zeros((5, 5), dtype=np.uint8))
    roi = (1, 2, 3, 4)
    with open(dm_path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump({"offset": [roi[0], roi[1]], "size": [roi[2], roi[3]]}, fh)

    # Load via MainWindow, patching file dialog
    mw = MainWindow()
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(dm_path), "")
    )
    mw.load_dm()
    assert mw.dm_roi == roi

    # Cleanup temporary files
    dm_path.unlink()
    dm_path.with_suffix(".json").unlink()
    app.quit()


def test_load_bm_with_sidecar(tmp_path, monkeypatch):
    """MainWindow.load_bm should populate dm_roi from JSON sidecar."""
    import json
    import pytest

    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from gui import MainWindow

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    bm_path = tmp_path / "bm.png"
    cv2.imwrite(str(bm_path), np.zeros((5, 5), dtype=np.uint8))
    roi = (1, 2, 3, 4)
    with open(bm_path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump({"offset": [roi[0], roi[1]], "size": [roi[2], roi[3]]}, fh)

    mw = MainWindow()
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(bm_path), "")
    )
    mw.load_bm()
    assert mw.dm_roi == roi

    bm_path.unlink()
    bm_path.with_suffix(".json").unlink()
    app.quit()


def test_register_ecc_matches_roi_dataset():
    full = load("roi_full.pgm")
    dm_small = load("roi_dm.pgm")
    roi = (5, 7, 8, 8)
    dm_full = np.zeros_like(full)
    x, y, w, h = roi
    dm_full[y:y + h, x:x + w] = dm_small
    params = RegSegParams(maxIter=50)
    reg, mask = register_ecc(full, dm_full, params, roi=roi)
    assert mask.shape == (h, w)
    assert mask.min() == 1 and mask.max() == 1
    assert np.array_equal(reg, full[y:y + h, x:x + w])
