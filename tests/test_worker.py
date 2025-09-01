import cv2
import numpy as np
from pathlib import Path

from worker import _process_file
from processing import RegSegParams, complement

DATA = Path(__file__).parent / "data"


def load(name: str):
    return cv2.imread(str(DATA / name), cv2.IMREAD_UNCHANGED)


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
