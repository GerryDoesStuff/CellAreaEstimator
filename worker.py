from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from io_utils import (
    imread_gray,
    ensure_dir,
    list_jpgs,
    write_sorted_areas_xlsx,
    to_uint8,
)
from processing import (
    RegSegParams,
    register_ecc,
    segment_image,
    connected_component_areas,
    clahe_equalize,
    complement,
)


class ProcessorWorker(QObject):
    """Worker object that processes images in a separate thread."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    imagePreviews = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    diffPreviews = pyqtSignal(np.ndarray, np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, in_dir: Path, out_dir: Path, dm_path: Path, bm_path: Path, params: RegSegParams):
        super().__init__()
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.dm_path = dm_path
        self.bm_path = bm_path
        self.params = params
        self._stop = False
        self._pause = False

    def stop(self) -> None:
        self._stop = True

    def toggle_pause(self, value: bool) -> None:
        self._pause = value

    def _check_pause_stop(self) -> None:
        while self._pause and not self._stop:
            time.sleep(0.2)
        if self._stop:
            raise RuntimeError("Processing stopped by user.")

    @pyqtSlot()
    def run(self) -> None:
        """Entry point for the worker thread."""
        try:
            dm = imread_gray(self.dm_path)
            bm = imread_gray(self.bm_path)
            self.imagePreviews.emit(dm, bm, dm)
            top_dir = self.out_dir / "top"
            topbw_dir = self.out_dir / "topBW"
            bot_dir = self.out_dir / "bottom"
            botbw_dir = self.out_dir / "bottomBW"
            for d in (top_dir, topbw_dir, bot_dir, botbw_dir):
                ensure_dir(d)
            binDif_top = cv2.subtract(dm, bm)
            binDif_bot = cv2.subtract(dm, complement(bm))
            files = list_jpgs(self.in_dir)
            if not files:
                self.error.emit("No .jpg files found in the input directory.")
                self.finished.emit()
                return
            total = len(files)
            top_xlsx = self.out_dir / "top.xlsx"
            bot_xlsx = self.out_dir / "bottom.xlsx"
            for idx, path in enumerate(files, start=1):
                self._check_pause_stop()
                self.status.emit(f"Processing {idx}/{total}: {path.name}")
                cur = imread_gray(path)
                self.imagePreviews.emit(dm, bm, cur)
                reg = register_ecc(cur, dm, self.params)
                binReg_top = cv2.subtract(reg, bm)
                topDiff = cv2.subtract(clahe_equalize(binDif_top), clahe_equalize(binReg_top))
                self.diffPreviews.emit(topDiff, topDiff)
                cv2.imwrite(str(top_dir / path.name), to_uint8(topDiff))
                topBW = segment_image(topDiff, self.params)
                cv2.imwrite(str(topbw_dir / path.name), to_uint8(topBW))
                areas_top = connected_component_areas(topBW)
                areas_top.sort(reverse=True)
                write_sorted_areas_xlsx(top_xlsx, idx, areas_top)
                binReg_bot = cv2.subtract(reg, complement(bm))
                botDiff = cv2.subtract(clahe_equalize(binDif_bot), clahe_equalize(binReg_bot))
                self.diffPreviews.emit(topDiff, botDiff)
                cv2.imwrite(str(bot_dir / path.name), to_uint8(botDiff))
                botBW = segment_image(botDiff, self.params)
                cv2.imwrite(str(botbw_dir / path.name), to_uint8(botBW))
                areas_bot = connected_component_areas(botBW)
                areas_bot.sort(reverse=True)
                write_sorted_areas_xlsx(bot_xlsx, idx, areas_bot)
                progress_pct = int(idx / total * 100)
                self.progress.emit(progress_pct)
            self.status.emit("Done.")
            self.progress.emit(100)
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit()
