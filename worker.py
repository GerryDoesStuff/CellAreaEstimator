from __future__ import annotations

import os

import cv2
import numpy as np
from PyQt6.QtCore import (
    QObject,
    pyqtSignal,
    pyqtSlot,
    QMutex,
    QWaitCondition,
)

from io_utils import (
    imread_gray,
    ensure_dir,
    list_jpgs,
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
from openpyxl import Workbook, load_workbook


class ProcessorWorker(QObject):
    """Worker object that processes images in a separate thread."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    imagePreviews = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    diffPreviews = pyqtSignal(np.ndarray, np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, in_dir: str, out_dir: str, dm_path: str, bm_path: str, params: RegSegParams):
        super().__init__()
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.dm_path = dm_path
        self.bm_path = bm_path
        self.params = params
        self._stop = False
        self._pause = False
        self._mutex = QMutex()
        self._wait_condition = QWaitCondition()

    def stop(self) -> None:
        """Request the worker thread to stop processing."""
        self._mutex.lock()
        try:
            self._stop = True
            self._wait_condition.wakeAll()
        finally:
            self._mutex.unlock()

    def toggle_pause(self, value: bool) -> None:
        """Pause or resume processing."""
        self._mutex.lock()
        try:
            self._pause = value
            if not self._pause:
                self._wait_condition.wakeAll()
        finally:
            self._mutex.unlock()

    def _check_pause_stop(self) -> None:
        """Wait while paused and raise if a stop was requested."""
        self._mutex.lock()
        try:
            while self._pause and not self._stop:
                self._wait_condition.wait(self._mutex)
            if self._stop:
                raise RuntimeError("Processing stopped by user.")
        finally:
            self._mutex.unlock()

    @pyqtSlot()
    def run(self) -> None:
        """Entry point for the worker thread."""
        top_wb = bot_wb = None
        try:
            dm = imread_gray(self.dm_path)
            bm = imread_gray(self.bm_path)
            self.imagePreviews.emit(dm, bm, dm)
            top_dir = os.path.join(self.out_dir, "top")
            topbw_dir = os.path.join(self.out_dir, "topBW")
            bot_dir = os.path.join(self.out_dir, "bottom")
            botbw_dir = os.path.join(self.out_dir, "bottomBW")
            for d in (top_dir, topbw_dir, bot_dir, botbw_dir):
                ensure_dir(d)
            binDif_top = cv2.subtract(dm, bm)
            binDif_bot = cv2.subtract(dm, complement(bm))
            files = list_jpgs(self.in_dir)
            if not files:
                self.error.emit("No .jpg files found in the input directory.")
                return
            total = len(files)
            top_xlsx = os.path.join(self.out_dir, "top.xlsx")
            bot_xlsx = os.path.join(self.out_dir, "bottom.xlsx")
            if os.path.exists(top_xlsx):
                top_wb = load_workbook(top_xlsx)
                top_ws = top_wb.active
            else:
                top_wb = Workbook()
                top_ws = top_wb.active
            if os.path.exists(bot_xlsx):
                bot_wb = load_workbook(bot_xlsx)
                bot_ws = bot_wb.active
            else:
                bot_wb = Workbook()
                bot_ws = bot_wb.active
            for idx, fname in enumerate(files, start=1):
                self._check_pause_stop()
                self.status.emit(f"Processing {idx}/{total}: {fname}")
                path = os.path.join(self.in_dir, fname)
                cur = imread_gray(path)
                self.imagePreviews.emit(dm, bm, cur)
                reg = register_ecc(cur, dm, self.params)
                binReg_top = cv2.subtract(reg, bm)
                topDiff = cv2.subtract(clahe_equalize(binDif_top), clahe_equalize(binReg_top))
                self.diffPreviews.emit(topDiff, topDiff)
                cv2.imwrite(os.path.join(top_dir, fname), to_uint8(topDiff))
                topBW = segment_image(topDiff, self.params)
                cv2.imwrite(os.path.join(topbw_dir, fname), to_uint8(topBW))
                areas_top = connected_component_areas(topBW)
                areas_top.sort(reverse=True)
                for row, area in enumerate(areas_top, start=1):
                    top_ws.cell(row=row, column=idx, value=int(area))
                binReg_bot = cv2.subtract(reg, complement(bm))
                botDiff = cv2.subtract(clahe_equalize(binDif_bot), clahe_equalize(binReg_bot))
                self.diffPreviews.emit(topDiff, botDiff)
                cv2.imwrite(os.path.join(bot_dir, fname), to_uint8(botDiff))
                botBW = segment_image(botDiff, self.params)
                cv2.imwrite(os.path.join(botbw_dir, fname), to_uint8(botBW))
                areas_bot = connected_component_areas(botBW)
                areas_bot.sort(reverse=True)
                for row, area in enumerate(areas_bot, start=1):
                    bot_ws.cell(row=row, column=idx, value=int(area))
                progress_pct = int(idx / total * 100)
                self.progress.emit(progress_pct)
            self.status.emit("Done.")
            self.progress.emit(100)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if top_wb is not None:
                top_wb.save(top_xlsx)
            if bot_wb is not None:
                bot_wb.save(bot_xlsx)
            self.finished.emit()
