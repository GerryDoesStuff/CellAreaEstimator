from __future__ import annotations

from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

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

logger = logging.getLogger(__name__)


def _process_file(
    idx: int,
    path: Path,
    dm: np.ndarray,
    bm: np.ndarray,
    params: RegSegParams,
    top_dir: Path,
    topbw_dir: Path,
    bot_dir: Path,
    botbw_dir: Path,
    binDif_top: np.ndarray,
    binDif_bot: np.ndarray,
):
    """Process a single image file.

    Returns the index, filename, current image, and processing results.
    """
    cur = imread_gray(path)
    reg = register_ecc(cur, dm, params)
    binReg_top = cv2.subtract(reg, bm)
    topDiff = cv2.subtract(clahe_equalize(binDif_top), clahe_equalize(binReg_top))
    cv2.imwrite(str(top_dir / path.name), to_uint8(topDiff))
    topBW = segment_image(topDiff, params)
    cv2.imwrite(str(topbw_dir / path.name), to_uint8(topBW))
    areas_top = connected_component_areas(topBW)
    areas_top.sort(reverse=True)
    binReg_bot = cv2.subtract(reg, complement(bm))
    botDiff = cv2.subtract(clahe_equalize(binDif_bot), clahe_equalize(binReg_bot))
    cv2.imwrite(str(bot_dir / path.name), to_uint8(botDiff))
    botBW = segment_image(botDiff, params)
    cv2.imwrite(str(botbw_dir / path.name), to_uint8(botBW))
    areas_bot = connected_component_areas(botBW)
    areas_bot.sort(reverse=True)
    return idx, path.name, cur, topDiff, botDiff, areas_top, areas_bot


class ProcessorWorker(QObject):
    """Worker object that processes images in a separate thread."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    imagePreviews = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    diffPreviews = pyqtSignal(np.ndarray, np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, in_dir: Path | str, out_dir: Path | str, dm_path: Path | str, bm_path: Path | str, params: RegSegParams):
        super().__init__()
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.dm_path = Path(dm_path)
        self.bm_path = Path(bm_path)
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
                msg = "No .jpg files found in the input directory."
                logger.warning(msg)
                self.error.emit(msg)
                return
            total = len(files)
            top_xlsx = self.out_dir / "top.xlsx"
            bot_xlsx = self.out_dir / "bottom.xlsx"
            if top_xlsx.exists():
                top_wb = load_workbook(top_xlsx)
                top_ws = top_wb.active
            else:
                top_wb = Workbook()
                top_ws = top_wb.active
            if bot_xlsx.exists():
                bot_wb = load_workbook(bot_xlsx)
                bot_ws = bot_wb.active
            else:
                bot_wb = Workbook()
                bot_ws = bot_wb.active
            logger.info("Processing %d files", len(files))
            results_top: dict[int, list[int]] = {}
            results_bot: dict[int, list[int]] = {}
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        _process_file,
                        idx,
                        path,
                        dm,
                        bm,
                        self.params,
                        top_dir,
                        topbw_dir,
                        bot_dir,
                        botbw_dir,
                        binDif_top,
                        binDif_bot,
                    )
                    for idx, path in enumerate(files, start=1)
                ]
                completed = 0
                for future in as_completed(futures):
                    self._check_pause_stop()
                    (
                        idx,
                        fname,
                        cur,
                        topDiff,
                        botDiff,
                        areas_top,
                        areas_bot,
                    ) = future.result()
                    results_top[idx] = areas_top
                    results_bot[idx] = areas_bot
                    self.status.emit(f"Processing {idx}/{total}: {fname}")
                    self.imagePreviews.emit(dm, bm, cur)
                    self.diffPreviews.emit(topDiff, botDiff)
                    completed += 1
                    progress_pct = int(completed / total * 100)
                    self.progress.emit(progress_pct)
            for idx in range(1, total + 1):
                areas_top = results_top.get(idx, [])
                for row, area in enumerate(areas_top, start=1):
                    top_ws.cell(row=row, column=idx, value=int(area))
                areas_bot = results_bot.get(idx, [])
                for row, area in enumerate(areas_bot, start=1):
                    bot_ws.cell(row=row, column=idx, value=int(area))
            self.status.emit("Done.")
            self.progress.emit(100)
        except Exception as exc:
            logger.exception("Processing failed")
            self.error.emit(str(exc))
        finally:
            if top_wb is not None:
                top_wb.save(top_xlsx)
            if bot_wb is not None:
                bot_wb.save(bot_xlsx)
            self.finished.emit()
