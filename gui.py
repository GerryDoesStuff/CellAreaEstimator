from __future__ import annotations

from pathlib import Path
import logging
from typing import Optional
import json

import numpy as np
import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSlot, QPoint, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QAction,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from processing import RegSegParams, difference_mask, mask_from_rects
from worker import ProcessorWorker
from io_utils import imread_gray, qimage_from_gray, list_jpgs

logger = logging.getLogger(__name__)


class ParamDialog(QDialog):
    """Dialog window for editing registration/segmentation parameters."""

    def __init__(self, params: RegSegParams, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Registration/Segmentation Parameters")
        self.params = params
        form = QFormLayout(self)
        self.sp_numSpaSam = QSpinBox(); self.sp_numSpaSam.setRange(1, 1_000_000); self.sp_numSpaSam.setValue(params.numSpaSam)
        self.sp_numHisBin = QSpinBox(); self.sp_numHisBin.setRange(1, 4096); self.sp_numHisBin.setValue(params.numHisBin)
        self.cb_allPix = QCheckBox(); self.cb_allPix.setChecked(params.allPix)
        self.dsp_groFac = QDoubleSpinBox(); self.dsp_groFac.setRange(0.0, 10.0); self.dsp_groFac.setSingleStep(0.1); self.dsp_groFac.setValue(params.groFac)
        self.dsp_epsilon = QDoubleSpinBox(); self.dsp_epsilon.setDecimals(10); self.dsp_epsilon.setRange(1e-10, 1e-1); self.dsp_epsilon.setValue(params.epsilon)
        self.dsp_iniRad = QDoubleSpinBox(); self.dsp_iniRad.setRange(0.0, 50.0); self.dsp_iniRad.setSingleStep(0.1); self.dsp_iniRad.setValue(params.iniRad)
        self.sp_maxIter = QSpinBox(); self.sp_maxIter.setRange(1, 10000); self.sp_maxIter.setValue(params.maxIter)
        form.addRow("[Reg] Num Spatial Samples", self.sp_numSpaSam)
        form.addRow("[Reg] Num Hist Bins", self.sp_numHisBin)
        form.addRow("[Reg] Use All Pixels", self.cb_allPix)
        form.addRow("[Reg] Growth Factor", self.dsp_groFac)
        form.addRow("[Reg] Epsilon", self.dsp_epsilon)
        form.addRow("[Reg] Initial Radius", self.dsp_iniRad)
        form.addRow("[Reg] Max Iterations", self.sp_maxIter)
        self.dsp_gbd = QDoubleSpinBox(); self.dsp_gbd.setRange(0.0, 50.0); self.dsp_gbd.setSingleStep(0.1); self.dsp_gbd.setValue(params.gausBlurDif)
        self.dsp_gbi = QDoubleSpinBox(); self.dsp_gbi.setRange(0.0, 50.0); self.dsp_gbi.setSingleStep(0.1); self.dsp_gbi.setValue(params.gausBlurIn)
        form.addRow("[Blur] Gaussian sigma (DM)", self.dsp_gbd)
        form.addRow("[Blur] Gaussian sigma (Input)", self.dsp_gbi)
        self.sp_thresh = QSpinBox(); self.sp_thresh.setRange(-1, 255); self.sp_thresh.setValue(params.bwThresh)
        self.sp_rect = QSpinBox(); self.sp_rect.setRange(1, 99); self.sp_rect.setSingleStep(1); self.sp_rect.setValue(params.recMaSize)
        self.sp_segIter = QSpinBox(); self.sp_segIter.setRange(0, 10000); self.sp_segIter.setValue(params.segIter)
        form.addRow("[Seg] Threshold (−1=Otsu)", self.sp_thresh)
        form.addRow("[Seg] Rect Mask Size", self.sp_rect)
        form.addRow("[Seg] Chan–Vese Iterations", self.sp_segIter)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK"); ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel"); cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        form.addRow(btn_layout)

    def apply(self) -> None:
        """Update the bound RegSegParams with the current UI values."""
        self.params.numSpaSam = int(self.sp_numSpaSam.value())
        self.params.numHisBin = int(self.sp_numHisBin.value())
        self.params.allPix = bool(self.cb_allPix.isChecked())
        self.params.groFac = float(self.dsp_groFac.value())
        self.params.epsilon = float(self.dsp_epsilon.value())
        self.params.iniRad = float(self.dsp_iniRad.value())
        self.params.maxIter = int(self.sp_maxIter.value())
        self.params.gausBlurDif = float(self.dsp_gbd.value())
        self.params.gausBlurIn = float(self.dsp_gbi.value())
        self.params.bwThresh = int(self.sp_thresh.value())
        self.params.recMaSize = int(self.sp_rect.value())
        self.params.segIter = int(self.sp_segIter.value())


class DrawLabel(QLabel):
    """Label widget allowing rectangular drawing to create a binary mask."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.base_pix: Optional[QPixmap] = None
        self.rects: list[QRect] = []
        self.drawing = False
        self.start_pos: Optional[QPoint] = None
        self.end_pos: Optional[QPoint] = None

    def load_image(self, img: np.ndarray) -> None:
        qimg = qimage_from_gray(img)
        pix = QPixmap.fromImage(qimg)
        self.base_pix = pix
        self.setPixmap(pix)
        self.setFixedSize(pix.size())
        self.rects.clear()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self.base_pix is not None:
            self.drawing = True
            self.start_pos = event.position().toPoint()
            self.end_pos = self.start_pos

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.drawing:
            self.end_pos = event.position().toPoint()
            self._update_display(temp=True)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            if self.start_pos and self.end_pos:
                rect = QRect(self.start_pos, self.end_pos).normalized()
                self.rects.append(rect)
            self._update_display(temp=False)

    def _update_display(self, temp: bool) -> None:
        if self.base_pix is None:
            return
        pix = self.base_pix.copy()
        painter = QPainter(pix)
        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QColor(255, 0, 0, 80))
        for r in self.rects:
            painter.drawRect(r)
        if temp and self.start_pos and self.end_pos:
            painter.drawRect(QRect(self.start_pos, self.end_pos).normalized())
        painter.end()
        self.setPixmap(pix)

    def get_mask(self) -> np.ndarray:
        if self.base_pix is None:
            return np.zeros((1, 1), dtype=np.uint8)
        h = self.base_pix.height()
        w = self.base_pix.width()
        rect_list = [(r.x(), r.y(), r.width(), r.height()) for r in self.rects]
        return mask_from_rects((h, w), rect_list)


class MaskPrepDialog(QDialog):
    """Dialog to prepare difference and binary masks from baseline images."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mask Preparation")
        self.img_a: Optional[np.ndarray] = None
        self.img_b: Optional[np.ndarray] = None
        self.dm: Optional[np.ndarray] = None
        self.dm_path: Optional[Path] = None
        self.bm_path: Optional[Path] = None
        self.dm_roi: Optional[tuple[int, int, int, int]] = None
        load_layout = QHBoxLayout()
        self.btn_load_a = QPushButton("Load Image A")
        self.btn_load_b = QPushButton("Load Image B")
        self.btn_compute = QPushButton("Compute DM")
        load_layout.addWidget(self.btn_load_a)
        load_layout.addWidget(self.btn_load_b)
        load_layout.addWidget(self.btn_compute)
        self.lbl_a = QLabel(); self.lbl_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_b = QLabel(); self.lbl_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_dm = QLabel(); self.lbl_dm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.draw = DrawLabel()
        save_layout = QHBoxLayout()
        self.btn_save_dm = QPushButton("Save DM")
        self.btn_save_bm = QPushButton("Save BM")
        save_layout.addWidget(self.btn_save_dm)
        save_layout.addWidget(self.btn_save_bm)
        close_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_layout.addWidget(close_btn)
        vbox = QVBoxLayout(self)
        vbox.addLayout(load_layout)
        vbox.addWidget(self.lbl_a)
        vbox.addWidget(self.lbl_b)
        vbox.addWidget(self.lbl_dm)
        vbox.addWidget(self.draw)
        vbox.addLayout(save_layout)
        vbox.addLayout(close_layout)
        self.btn_load_a.clicked.connect(self.load_img_a)
        self.btn_load_b.clicked.connect(self.load_img_b)
        self.btn_compute.clicked.connect(self.compute_dm)
        self.btn_save_dm.clicked.connect(self.save_dm)
        self.btn_save_bm.clicked.connect(self.save_bm)
        close_btn.clicked.connect(self.accept)

    def _compute_roi(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(mask)
        if ys.size == 0 or xs.size == 0:
            h, w = mask.shape
            return 0, 0, w, h
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    def _save_sidecar(self, base: Path, roi: tuple[int, int, int, int]) -> None:
        sidecar = base.with_suffix(base.suffix + ".json")
        with sidecar.open("w", encoding="utf-8") as fh:
            json.dump({"roi": roi}, fh)

    def load_img_a(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Baseline Image A", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if path:
            self.img_a = imread_gray(path)
            self.lbl_a.setPixmap(QPixmap.fromImage(qimage_from_gray(self.img_a)))

    def load_img_b(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Baseline Image B", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if path:
            self.img_b = imread_gray(path)
            self.lbl_b.setPixmap(QPixmap.fromImage(qimage_from_gray(self.img_b)))

    def compute_dm(self) -> None:
        if self.img_a is None or self.img_b is None:
            QMessageBox.warning(self, "Missing Images", "Please load both baseline images first.")
            return
        self.dm = difference_mask(self.img_a, self.img_b)
        self.lbl_dm.setPixmap(QPixmap.fromImage(qimage_from_gray(self.dm)))
        self.draw.load_image(self.dm)

    def save_dm(self) -> None:
        if self.dm is None:
            QMessageBox.warning(self, "No DM", "Compute the difference mask first.")
            return
        mask = self.draw.get_mask()
        roi = self._compute_roi(mask)
        x, y, w, h = roi
        dm_crop = self.dm[y:y + h, x:x + w]
        path, _ = QFileDialog.getSaveFileName(self, "Save Difference Mask", "dm.png", "Images (*.png *.jpg *.tif *.tiff)")
        if path:
            cv2.imwrite(path, dm_crop)
            self.dm_path = Path(path)
            self.dm_roi = roi
            self._save_sidecar(self.dm_path, roi)

    def save_bm(self) -> None:
        mask = self.draw.get_mask()
        if mask.size == 1:
            QMessageBox.warning(self, "No BM", "Draw on the difference mask to create a binary mask.")
            return
        roi = self._compute_roi(mask)
        x, y, w, h = roi
        bm_crop = mask[y:y + h, x:x + w]
        path, _ = QFileDialog.getSaveFileName(self, "Save Binary Mask", "bm.png", "Images (*.png *.jpg *.tif *.tiff)")
        if path:
            cv2.imwrite(path, bm_crop)
            self.bm_path = Path(path)
            self.dm_roi = roi
            self._save_sidecar(self.bm_path, roi)


class MainWindow(QMainWindow):
    """The main window of the cell area estimator."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cell Area Estimator (PyQt6)")
        self.params = RegSegParams()
        self.dm_roi: Optional[tuple[int, int, int, int]] = None
        menubar = self.menuBar()
        tools_menu = menubar.addMenu("Tools")
        params_action = QAction("Registration/Segmentation Setup", self)
        tools_menu.addAction(params_action)
        params_action.triggered.connect(self.open_param_dialog)
        mask_action = QAction("Prepare Masks", self)
        tools_menu.addAction(mask_action)
        mask_action.triggered.connect(self.open_mask_prep_dialog)
        toolbar = QToolBar("Tools", self)
        self.addToolBar(toolbar)
        toolbar.addAction(params_action)
        toolbar.addAction(mask_action)
        self.dm_label = QLabel("Difference Mask:")
        self.dm_path = QLineEdit(); self.dm_path.setReadOnly(True)
        self.btn_dm = QPushButton("Load DM"); self.btn_dm.clicked.connect(self.load_dm)
        self.bm_label = QLabel("Binary Mask:")
        self.bm_path = QLineEdit(); self.bm_path.setReadOnly(True)
        self.btn_bm = QPushButton("Load BM"); self.btn_bm.clicked.connect(self.load_bm)
        self.in_label = QLabel("Input Image Directory:")
        self.in_path = QLineEdit(); self.in_path.setReadOnly(True)
        self.btn_in = QPushButton("Select Dir"); self.btn_in.clicked.connect(self.select_input_dir)
        self.out_label = QLabel("Save Difference Images To Directory:")
        self.out_path = QLineEdit(); self.out_path.setReadOnly(True)
        self.btn_out = QPushButton("Select Dir"); self.btn_out.clicked.connect(self.select_output_dir)
        self.btn_process = QPushButton("Process (Single Type)")
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self.toggle_pause)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setStyleSheet("color: red; font-weight: bold;")
        self.progress_label = QLabel("Current Progress")
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        self.img_dm = QLabel(); self.img_dm.setMinimumSize(240, 240); self.img_dm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_bm = QLabel(); self.img_bm.setMinimumSize(240, 240); self.img_bm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_input = QLabel(); self.img_input.setMinimumSize(240, 240); self.img_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_top = QLabel(); self.img_top.setMinimumSize(240, 240); self.img_top.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_bottom = QLabel(); self.img_bottom.setMinimumSize(240, 240); self.img_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meta_layout = QGridLayout()
        meta_layout.addWidget(self.dm_label, 0, 0); meta_layout.addWidget(self.dm_path, 0, 1); meta_layout.addWidget(self.btn_dm, 0, 2)
        meta_layout.addWidget(self.bm_label, 1, 0); meta_layout.addWidget(self.bm_path, 1, 1); meta_layout.addWidget(self.btn_bm, 1, 2)
        meta_layout.addWidget(self.in_label, 2, 0); meta_layout.addWidget(self.in_path, 2, 1); meta_layout.addWidget(self.btn_in, 2, 2)
        meta_layout.addWidget(self.out_label, 3, 0); meta_layout.addWidget(self.out_path, 3, 1); meta_layout.addWidget(self.btn_out, 3, 2)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_process)
        btn_layout.addWidget(self.progress_label)
        btn_layout.addWidget(self.progress_bar)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        grid_layout = QGridLayout()
        grid_layout.addWidget(self.img_dm, 0, 0)
        grid_layout.addWidget(self.img_bm, 0, 1)
        grid_layout.addWidget(self.img_input, 1, 0)
        grid_layout.addWidget(self.img_top, 1, 1)
        grid_layout.addWidget(self.img_bottom, 1, 2)
        central = QWidget()
        vbox = QVBoxLayout(central)
        vbox.addLayout(meta_layout)
        vbox.addLayout(btn_layout)
        vbox.addLayout(grid_layout)
        self.setCentralWidget(central)
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[ProcessorWorker] = None

    def open_param_dialog(self) -> None:
        dlg = ParamDialog(self.params, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dlg.apply()

    def open_mask_prep_dialog(self) -> None:
        dlg = MaskPrepDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if dlg.dm_path and dlg.dm is not None:
                self.dm_path.setText(str(dlg.dm_path))
                self.set_label_image(self.img_dm, dlg.dm)
                self.dm_roi = dlg.dm_roi
            if dlg.bm_path:
                bm = imread_gray(dlg.bm_path)
                self.bm_path.setText(str(dlg.bm_path))
                self.set_label_image(self.img_bm, bm)

    def load_dm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Difference Mask", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if path:
            self.dm_path.setText(path)
            try:
                im = imread_gray(Path(path))
                self.set_label_image(self.img_dm, im)
                self.dm_roi = None
                sidecar = Path(path).with_suffix(Path(path).suffix + ".json")
                if sidecar.exists():
                    with sidecar.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                        roi = data.get("roi")
                        if roi and len(roi) == 4:
                            self.dm_roi = tuple(int(v) for v in roi)
                logger.info("Loaded difference mask from %s", path)
            except Exception as exc:
                logger.error("Failed to load difference mask", exc_info=exc)
                QMessageBox.critical(self, "Error", str(exc))

    def load_bm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Binary Mask", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if path:
            self.bm_path.setText(path)
            try:
                im = imread_gray(Path(path))
                self.set_label_image(self.img_bm, im)
                logger.info("Loaded binary mask from %s", path)
            except Exception as exc:
                logger.error("Failed to load binary mask", exc_info=exc)
                QMessageBox.critical(self, "Error", str(exc))

    def select_input_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.in_path.setText(directory)

    def select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.out_path.setText(directory)

    def start_processing(self) -> None:
        in_dir = Path(self.in_path.text().strip())
        out_dir = Path(self.out_path.text().strip())
        dm = Path(self.dm_path.text().strip())
        bm = Path(self.bm_path.text().strip())
        if not (in_dir.is_dir() and out_dir.is_dir() and dm.is_file() and bm.is_file()):
            QMessageBox.warning(self, "Missing Input", "Please provide valid DM, BM, input directory and output directory.")
            logger.warning("Missing input paths: dm=%s bm=%s in=%s out=%s", dm, bm, in_dir, out_dir)
            return
        files = list_jpgs(in_dir)
        if not files:
            QMessageBox.warning(self, "No Images", f"No image files found in {in_dir}")
            logger.warning("No image files found in %s", in_dir)
            return
        self.progress_bar.setValue(0)
        logger.info("Starting processing for %d images in %s", len(files), in_dir)
        self.worker_thread = QThread()
        self.worker = ProcessorWorker(in_dir, out_dir, dm, bm, self.params, files, dm_roi=self.dm_roi)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.imagePreviews.connect(self.update_image_previews)
        self.worker.topPreview.connect(self.update_top_preview)
        self.worker.bottomPreview.connect(self.update_bottom_preview)
        self.worker.error.connect(lambda msg: QMessageBox.critical(self, "Error", msg))
        self.worker_thread.start()

    def toggle_pause(self, checked: bool) -> None:
        if self.worker:
            self.worker.toggle_pause(checked)
        self.btn_pause.setText("Resume" if checked else "Pause")

    def stop_processing(self) -> None:
        if self.worker:
            self.worker.stop()

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_image_previews(self, dm: np.ndarray, bm: np.ndarray, cur: np.ndarray) -> None:
        self.set_label_image(self.img_dm, dm)
        self.set_label_image(self.img_bm, bm)
        self.set_label_image(self.img_input, cur)

    @pyqtSlot(np.ndarray)
    def update_top_preview(self, top: np.ndarray) -> None:
        self.set_label_image(self.img_top, top)

    @pyqtSlot(np.ndarray)
    def update_bottom_preview(self, bottom: np.ndarray) -> None:
        self.set_label_image(self.img_bottom, bottom)

    def set_label_image(self, label: QLabel, img: np.ndarray) -> None:
        qimg = qimage_from_gray(img)
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(
            pix.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
