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

from processing import RegSegParams, difference_mask, mask_from_rects, register_ecc
from worker import ProcessorWorker
from io_utils import imread_gray, qimage_from_gray, list_jpgs, to_uint8

logger = logging.getLogger(__name__)


class ParamDialog(QDialog):
    """Dialog window for editing registration/segmentation parameters."""

    def __init__(self, params: RegSegParams, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Registration/Segmentation Parameters")
        self.params = params
        form = QFormLayout(self)
        self.dsp_epsilon = QDoubleSpinBox(); self.dsp_epsilon.setDecimals(10); self.dsp_epsilon.setRange(1e-10, 1e-1); self.dsp_epsilon.setValue(params.epsilon)
        self.sp_maxIter = QSpinBox(); self.sp_maxIter.setRange(1, 10000); self.sp_maxIter.setValue(params.maxIter)
        form.addRow("[Reg] Epsilon", self.dsp_epsilon)
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
        self.params.epsilon = float(self.dsp_epsilon.value())
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

    def get_roi(self) -> tuple[int, int, int, int] | None:
        """Return the bounding rectangle of all drawn rectangles.

        The returned tuple is ``(x, y, w, h)`` or ``None`` if no rectangles
        have been drawn.  This is useful for cropping the underlying image to
        the selected region of interest.
        """
        if not self.rects:
            return None
        x1 = min(r.x() for r in self.rects)
        y1 = min(r.y() for r in self.rects)
        x2 = max(r.x() + r.width() for r in self.rects)
        y2 = max(r.y() + r.height() for r in self.rects)
        return x1, y1, x2 - x1, y2 - y1


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
        # Bounding box of the ROI selected during mask creation.  This is
        # propagated back to the main window so that processing can crop to the
        # same region even if the difference mask itself is not saved.
        self.dm_roi: Optional[tuple[int, int, int, int]] = None
        load_layout = QHBoxLayout()
        self.btn_load_a = QPushButton("Load Image A")
        self.btn_load_b = QPushButton("Load Image B")
        self.btn_compute = QPushButton("Compute DM")
        self.cb_register = QCheckBox("Register")
        load_layout.addWidget(self.btn_load_a)
        load_layout.addWidget(self.btn_load_b)
        load_layout.addWidget(self.btn_compute)
        load_layout.addWidget(self.cb_register)
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
        img_a = self.img_a
        img_b = self.img_b
        if img_a is None or img_b is None:
            return  # pragma: no cover - guarded above
        if self.cb_register.isChecked():
            params = getattr(self.parent(), "params", RegSegParams())
            reg, mask = register_ecc(img_b, img_a, params)
        else:
            reg = img_b
            mask = np.ones_like(img_a, dtype=np.uint8)
        disp = to_uint8(reg * mask)
        x, y, w, h = cv2.selectROI("Crop Registered", disp, False, False)
        cv2.destroyWindow("Crop Registered")
        if w == 0 or h == 0:
            ys, xs = np.nonzero(mask)
            if xs.size > 0 and ys.size > 0:
                x1, x2 = xs.min(), xs.max() + 1
                y1, y2 = ys.min(), ys.max() + 1
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
            else:
                h, w = img_a.shape
                x = y = 0
        self.dm_roi = (int(x), int(y), int(w), int(h))
        img_a_crop = img_a[y:y + h, x:x + w]
        reg_crop = reg[y:y + h, x:x + w]
        mask_crop = mask[y:y + h, x:x + w]
        dm_raw = difference_mask(img_a_crop, reg_crop)
        self.dm = dm_raw * mask_crop
        self.lbl_dm.setPixmap(QPixmap.fromImage(qimage_from_gray(self.dm)))
        self.draw.load_image(self.dm)

    def _save_sidecar(self, path: Path, meta: dict) -> None:
        """Persist JSON metadata to a sidecar file.

        Any :class:`OSError` during writing triggers a critical message box and is
        logged for troubleshooting.
        """
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
        except OSError as exc:
            logger.error("Failed to save metadata sidecar %s", path, exc_info=exc)
            QMessageBox.critical(self, "Save Error", f"Could not save metadata: {exc}")

    def save_dm(self) -> None:
        if self.dm is None:
            QMessageBox.warning(self, "No DM", "Compute the difference mask first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Difference Mask", "dm.png", "Images (*.png *.jpg *.tif *.tiff)")
        if path:
            cv2.imwrite(path, self.dm)
            meta_path = Path(path).with_suffix(".json")
            if self.dm_roi is None:
                roi = self.draw.get_roi()
                if roi is None:
                    h, w = self.dm.shape
                    roi = (0, 0, w, h)
                self.dm_roi = roi
            x, y, w, h = self.dm_roi
            meta = {"offset": [int(x), int(y)], "size": [int(w), int(h)]}
            self._save_sidecar(meta_path, meta)
            self.dm_path = Path(path)

    def save_bm(self) -> None:
        mask = self.draw.get_mask()
        if mask.size == 1 or np.max(mask) == 0:
            QMessageBox.warning(self, "No BM", "Draw on the difference mask to create a binary mask.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Binary Mask", "bm.png", "Images (*.png *.jpg *.tif *.tiff)")
        if path:
            cv2.imwrite(path, mask)
            if self.dm_path:
                meta_path = self.dm_path.with_suffix(".json")
            else:
                meta_path = Path(path).with_suffix(".json")
            if self.dm_roi is None:
                roi = self.draw.get_roi()
                if roi is None:
                    h, w = mask.shape
                    roi = (0, 0, w, h)
                self.dm_roi = roi
            x, y, w, h = self.dm_roi
            meta = {"offset": [int(x), int(y)], "size": [int(w), int(h)]}
            self._save_sidecar(meta_path, meta)
            self.bm_path = Path(path)


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
            # Always carry over the ROI if the dialog produced one, even if the
            # difference mask itself was not saved.  ``dlg.dm_roi`` is populated
            # when either mask is written to disk and represents the crop
            # applied to both DM and BM.
            if getattr(dlg, "dm_roi", None) is not None:
                self.dm_roi = dlg.dm_roi
            if dlg.dm_path and dlg.dm is not None:
                self.dm_path.setText(str(dlg.dm_path))
                self.set_label_image(self.img_dm, dlg.dm)
                try:
                    self.load_dm_metadata(dlg.dm_path)
                except (OSError, json.JSONDecodeError) as exc:
                    logger.error(
                        "Failed to load DM metadata from %s", dlg.dm_path.with_suffix(".json"), exc_info=exc
                    )
                    QMessageBox.warning(self, "Metadata Error", f"Failed to load DM metadata: {exc}")
            if dlg.bm_path:
                bm = imread_gray(dlg.bm_path)
                self.bm_path.setText(str(dlg.bm_path))
                self.set_label_image(self.img_bm, bm)

    def load_dm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Difference Mask", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)"
        )
        if not path:
            return
        self.dm_path.setText(path)
        try:
            im = imread_gray(Path(path))
        except Exception as exc:
            logger.error("Failed to load difference mask image %s", path, exc_info=exc)
            QMessageBox.critical(self, "Error", f"Failed to load difference mask: {exc}")
            return
        self.set_label_image(self.img_dm, im)
        logger.info("Loaded difference mask from %s", path)
        try:
            self.load_dm_metadata(Path(path))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to load DM metadata from %s", Path(path).with_suffix(".json"), exc_info=exc
            )
            QMessageBox.warning(self, "Metadata Error", f"Failed to load DM metadata: {exc}")
            self.dm_roi = None

    def load_bm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Binary Mask", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if path:
            self.bm_path.setText(path)
            try:
                im = imread_gray(Path(path))
                self.set_label_image(self.img_bm, im)
                logger.info("Loaded binary mask from %s", path)
                try:
                    # The binary mask shares the same ROI metadata format as the
                    # difference mask.  When loading a BM we therefore attempt to
                    # read the accompanying JSON sidecar so that processing can
                    # crop to the correct region even if no DM is provided.
                    self.load_dm_metadata(Path(path))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.error(
                        "Failed to load BM metadata from %s",
                        Path(path).with_suffix(".json"),
                        exc_info=exc,
                    )
                    QMessageBox.warning(self, "Metadata Error", f"Failed to load BM metadata: {exc}")
                    self.dm_roi = None
            except Exception as exc:
                logger.error("Failed to load binary mask", exc_info=exc)
                QMessageBox.critical(self, "Error", str(exc))

    def load_dm_metadata(self, path: Path) -> None:
        """Load ROI metadata from a JSON sidecar next to ``path``."""
        self.dm_roi = None
        meta = path.with_suffix(".json")
        if not meta.exists():
            return
        with open(meta, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        offset = data.get("offset")
        size = data.get("size")
        if offset and size and len(offset) == 2 and len(size) == 2:
            self.dm_roi = (
                int(offset[0]),
                int(offset[1]),
                int(size[0]),
                int(size[1]),
            )

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
        dm_text = self.dm_path.text().strip()
        dm = Path(dm_text) if dm_text else None
        bm = Path(self.bm_path.text().strip())
        if not (in_dir.is_dir() and out_dir.is_dir() and bm.is_file()):
            QMessageBox.warning(
                self,
                "Missing Input",
                "Please provide valid BM, input directory and output directory.",
            )
            logger.warning("Missing input paths: dm=%s bm=%s in=%s out=%s", dm, bm, in_dir, out_dir)
            return
        if dm is not None and not dm.is_file():
            QMessageBox.warning(self, "Missing Input", "Difference mask path is invalid.")
            logger.warning("Invalid DM path: %s", dm)
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
