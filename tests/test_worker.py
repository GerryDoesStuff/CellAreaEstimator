import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("PyQt6")
pytest.importorskip("openpyxl")

import cv2
import numpy as np
from pathlib import Path

import worker
from worker import _apply_roi, _process_file, ProcessorWorker
from processing import RegSegParams, complement

DATA = Path(__file__).parent / "data"


def load(name: str):
    return cv2.imread(str(DATA / name), cv2.IMREAD_UNCHANGED)


def test_apply_roi_with_roi():
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    dm = arr + 1
    bm = arr + 2
    top = arr + 3
    bot = arr + 4
    roi = (1, 1, 2, 2)
    cur_c, dm_c, bm_c, top_c, bot_c = _apply_roi(arr, dm, bm, top, bot, roi)
    expected = arr[1:3, 1:3]
    assert np.array_equal(cur_c, expected)
    assert np.array_equal(dm_c, dm[1:3, 1:3])
    assert np.array_equal(bm_c, bm[1:3, 1:3])
    assert np.array_equal(top_c, top[1:3, 1:3])
    assert np.array_equal(bot_c, bot[1:3, 1:3])


def test_apply_roi_without_roi():
    arr = np.arange(9, dtype=np.uint8).reshape(3, 3)
    dm = arr + 1
    bm = arr + 2
    top = arr + 3
    bot = arr + 4
    cur_c, dm_c, bm_c, top_c, bot_c = _apply_roi(arr, dm, bm, top, bot, None)
    assert np.array_equal(cur_c, arr)
    assert np.array_equal(dm_c, dm)
    assert np.array_equal(bm_c, bm)
    assert np.array_equal(top_c, top)
    assert np.array_equal(bot_c, bot)


def test_process_file_with_roi(tmp_path):
    fixed = load("fixed.pgm")
    moving = load("moving.pgm")
    roi = (2, 2, 10, 10)
    dm = np.zeros_like(fixed)
    bm = np.zeros_like(fixed)
    dm[2:12, 2:12] = fixed[2:12, 2:12]
    binDif_top = cv2.subtract(dm, bm)
    binDif_bot = cv2.subtract(dm, complement(bm))
    img_path = tmp_path / "cur.pgm"
    cv2.imwrite(str(img_path), moving)
    top_dir = tmp_path / "top"
    topbw_dir = tmp_path / "topBW"
    bot_dir = tmp_path / "bottom"
    botbw_dir = tmp_path / "bottomBW"
    for d in (top_dir, topbw_dir, bot_dir, botbw_dir):
        d.mkdir()
    params = RegSegParams(maxIter=50)
    _, _, _, topDiff, botDiff, _, _ = _process_file(
        1,
        img_path,
        dm,
        bm,
        params,
        top_dir,
        topbw_dir,
        bot_dir,
        botbw_dir,
        binDif_top,
        binDif_bot,
        roi,
    )
    assert topDiff.shape == (10, 10)
    assert botDiff.shape == (10, 10)


def test_process_file_crops_to_mask(tmp_path):
    full = load("roi_full.pgm")
    dm_small = load("roi_dm.pgm")
    bm_small = load("roi_bm.pgm")
    roi = (5, 7, 8, 8)
    dm = np.zeros_like(full)
    bm = np.zeros_like(full)
    x, y, w, h = roi
    dm[y:y + h, x:x + w] = dm_small
    bm[y:y + h, x:x + w] = bm_small
    binDif_top = cv2.subtract(dm, bm)
    binDif_bot = cv2.subtract(dm, complement(bm))
    img_path = tmp_path / "cur.pgm"
    cv2.imwrite(str(img_path), full)
    top_dir = tmp_path / "top"; top_dir.mkdir()
    topbw_dir = tmp_path / "topBW"; topbw_dir.mkdir()
    bot_dir = tmp_path / "bottom"; bot_dir.mkdir()
    botbw_dir = tmp_path / "bottomBW"; botbw_dir.mkdir()
    params = RegSegParams(maxIter=50)
    _, _, _, topDiff, botDiff, _, _ = _process_file(
        1,
        img_path,
        dm,
        bm,
        params,
        top_dir,
        topbw_dir,
        bot_dir,
        botbw_dir,
        binDif_top,
        binDif_bot,
        roi,
    )
    assert topDiff.shape == (h, w)
    assert botDiff.shape == (h, w)


def test_process_file_complements_roi_only(tmp_path, monkeypatch):
    full = load("roi_full.pgm")
    dm_small = load("roi_dm.pgm")
    bm_small = load("roi_bm.pgm")
    roi = (5, 7, 8, 8)
    dm = np.zeros_like(full)
    bm = np.zeros_like(full)
    x, y, w, h = roi
    dm[y:y + h, x:x + w] = dm_small
    bm[y:y + h, x:x + w] = bm_small
    binDif_top = cv2.subtract(dm, bm)
    from processing import complement as proc_complement
    binDif_bot = cv2.subtract(dm, proc_complement(bm))
    img_path = tmp_path / "cur.pgm"
    cv2.imwrite(str(img_path), full)
    top_dir = tmp_path / "top"; top_dir.mkdir()
    topbw_dir = tmp_path / "topBW"; topbw_dir.mkdir()
    bot_dir = tmp_path / "bottom"; bot_dir.mkdir()
    botbw_dir = tmp_path / "bottomBW"; botbw_dir.mkdir()
    params = RegSegParams(maxIter=50)

    def fake_register(cur, dm_roi, params):
        return cur, np.ones_like(cur, dtype=np.uint8)

    monkeypatch.setattr(worker, "register_ecc", fake_register)

    shapes: list[tuple[int, int]] = []

    def spy_complement(img: np.ndarray) -> np.ndarray:
        shapes.append(img.shape)
        return proc_complement(img)

    monkeypatch.setattr(worker, "complement", spy_complement)

    _process_file(
        1,
        img_path,
        dm,
        bm,
        params,
        top_dir,
        topbw_dir,
        bot_dir,
        botbw_dir,
        binDif_top,
        binDif_bot,
        roi,
    )

    assert shapes == [bm_small.shape]


def test_worker_run_handles_roi(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    import json

    app = QApplication.instance() or QApplication([])
    dm_path = DATA / "syn_dm_ascii.pgm"
    bm_path = DATA / "syn_bm_ascii.pgm"
    with open(DATA / "syn_dm_ascii.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    roi = (meta["offset"][0], meta["offset"][1], meta["size"][0], meta["size"][1])
    full = load("syn_full_ascii.pgm")
    img_path = tmp_path / "cur.pgm"
    cv2.imwrite(str(img_path), full)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    captured: dict[str, tuple[int, int]] = {}

    def fake_process_file(idx, path, dm, bm, params, top_dir, topbw_dir, bot_dir, botbw_dir, binDif_top, binDif_bot, roi_arg):
        captured["dm_shape"] = dm.shape
        captured["bm_shape"] = bm.shape
        captured["roi"] = roi_arg
        return idx, path.name, dm, dm, dm, [], []

    monkeypatch.setattr(worker, "_process_file", fake_process_file)

    params = RegSegParams(maxIter=1)
    worker_obj = ProcessorWorker(tmp_path, out_dir, dm_path, bm_path, params, [img_path], dm_roi=roi)
    worker_obj.run()

    assert captured["dm_shape"] == full.shape
    assert captured["bm_shape"] == full.shape
    assert captured["roi"] == roi
    app.quit()
