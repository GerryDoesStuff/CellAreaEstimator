import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")
pytest.importorskip("PyQt6")

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QFileDialog
from gui import MaskPrepDialog


def test_save_sidecar(tmp_path):
    dlg = MaskPrepDialog()
    meta_path = tmp_path / "meta.json"
    meta = {"offset": [1, 2], "size": [3, 4]}
    dlg._save_sidecar(meta_path, meta)
    with open(meta_path, "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved == meta


def test_save_dm_writes_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    dlg = MaskPrepDialog()
    dlg.dm = np.zeros((5, 5), dtype=np.uint8)
    dlg.dm_roi = (1, 2, 3, 4)
    out_path = tmp_path / "dm.png"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(out_path), "")
    )
    dlg.save_dm()
    with open(out_path.with_suffix(".json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["offset"] == [1, 2]
    assert meta["size"] == [3, 4]
    app.quit()


def test_save_bm_writes_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    dlg = MaskPrepDialog()
    dm_path = tmp_path / "dm.png"
    cv2.imwrite(str(dm_path), np.zeros((4, 4), dtype=np.uint8))
    dlg.dm_path = dm_path
    dlg.dm_roi = (1, 2, 3, 4)

    class DummyDraw:
        def get_mask(self):
            return np.ones((4, 4), dtype=np.uint8)

        def get_roi(self):
            return (1, 2, 3, 4)

    dlg.draw = DummyDraw()
    out_path = tmp_path / "bm.png"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(out_path), "")
    )
    dlg.save_bm()
    with open(dm_path.with_suffix(".json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["offset"] == [1, 2]
    assert meta["size"] == [3, 4]
    app.quit()
