import base64
import json
import os.path as osp
from PIL import Image

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QProgressDialog,
    QDialog,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
    QApplication,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QComboBox,
    QHeaderView,
    QAbstractItemView,
    QSpinBox,
)

from anylabeling.app_info import __version__
from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.services.auto_labeling import (
    _BATCH_PROCESSING_INVALID_MODELS,
    _BATCH_PROCESSING_POINT_PROMPT_MODELS,
    _BATCH_PROCESSING_TEXT_PROMPT_MODELS,
    _BATCH_PROCESSING_VIDEO_MODELS,
    _SKIP_DET_MODELS,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils._io import io_open
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import get_msg_box_style
from anylabeling.views.labeling.widgets.popup import Popup

__all__ = ["run_all_images"]


class TextInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Enter Text Prompt"))
        self.setFixedSize(400, 180)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.MSWindowsFixedSizeDialogHint
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        prompt_label = QLabel(self.tr("Please enter your text prompt:"))
        prompt_label.setStyleSheet(
            f"font-size: 13px; color: {get_theme()['text']}; font-weight: 500;"
        )
        layout.addWidget(prompt_label)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(self.tr("Enter prompt here..."))
        layout.addWidget(self.text_input)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        t = get_theme()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {t["background"]};
                border-radius: 10px;
            }}

            QLineEdit {{
                border: 1px solid {t["border"]};
                border-radius: 8px;
                background-color: {t["background_secondary"]};
                font-size: 13px;
                height: 36px;
                padding: 0 12px;
                color: {t["text"]};
            }}

            QLineEdit:hover {{
                background-color: {t["background_hover"]};
            }}

            QLineEdit:focus {{
                border: 2px solid {t["highlight"]};
                background-color: {t["background_secondary"]};
            }}

            QPushButton {{
                min-width: 100px;
                height: 36px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 13px;
            }}

            QPushButton[text="OK"] {{
                background-color: {t["primary"]};
                color: white;
                border: none;
            }}

            QPushButton[text="OK"]:hover {{
                background-color: {t["primary_hover"]};
            }}

            QPushButton[text="OK"]:pressed {{
                background-color: {t["primary"]};
            }}

            QPushButton[text="Cancel"] {{
                background-color: {t["surface"]};
                color: {t["text"]};
                border: 1px solid {t["border"]};
            }}

            QPushButton[text="Cancel"]:hover {{
                background-color: {t["background_hover"]};
            }}

            QPushButton[text="Cancel"]:pressed {{
                background-color: {t["surface"]};
            }}
        """)

    def get_input_text(self):
        if self.exec() == QDialog.DialogCode.Accepted:
            return self.text_input.text().strip()
        return ""


class PointInputDialog(QDialog):
    """Dialog for entering fixed point/rectangle prompts for batch SAM segmentation."""

    # Column indices
    _COL_SHAPE = 0
    _COL_X1 = 1
    _COL_Y1 = 2
    _COL_X2 = 3
    _COL_Y2 = 4
    _COL_TYPE = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Batch SAM - Prompts"))
        self.setMinimumWidth(620)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.MSWindowsFixedSizeDialogHint
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        t = get_theme()

        info_label = QLabel(
            self.tr(
                "Enter the prompts to apply to every image.\n"
                "Points: click a single XY location.  "
                "Rectangles: define a bounding box with top-left (X1,Y1) and bottom-right (X2,Y2).\n"
                "Positive prompts indicate the object; negative prompts exclude regions."
            )
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            f"font-size: 13px; color: {t['text']};"
        )
        layout.addWidget(info_label)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Shape"),
                self.tr("X1"),
                self.tr("Y1"),
                self.tr("X2"),
                self.tr("Y2"),
                self.tr("Type"),
            ]
        )
        for col in range(5):
            self.table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setMinimumHeight(180)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {t["background_secondary"]};
                border: 1px solid {t["border"]};
                border-radius: 6px;
                gridline-color: {t["border"]};
                color: {t["text"]};
                font-size: 13px;
            }}
            QHeaderView::section {{
                background-color: {t["surface"]};
                color: {t["text"]};
                border: none;
                border-bottom: 1px solid {t["border"]};
                padding: 4px 8px;
                font-weight: 500;
            }}
        """)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.add_point_btn = QPushButton(self.tr("Add Point"))
        self.add_rect_btn = QPushButton(self.tr("Add Rectangle"))
        self.remove_btn = QPushButton(self.tr("Remove Selected"))
        self.add_point_btn.clicked.connect(lambda: self._add_row("point"))
        self.add_rect_btn.clicked.connect(lambda: self._add_row("rectangle"))
        self.remove_btn.clicked.connect(self._remove_row)
        btn_style = f"""
            QPushButton {{
                background-color: {t["surface"]};
                border: 1px solid {t["border"]};
                border-radius: 6px;
                font-size: 13px;
                color: {t["text"]};
                height: 32px;
                padding: 0 12px;
            }}
            QPushButton:hover {{
                background-color: {t["background_hover"]};
            }}
        """
        self.add_point_btn.setStyleSheet(btn_style)
        self.add_rect_btn.setStyleSheet(btn_style)
        self.remove_btn.setStyleSheet(btn_style)
        btn_layout.addWidget(self.add_point_btn)
        btn_layout.addWidget(self.add_rect_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {t["background"]};
                border-radius: 10px;
            }}
            QPushButton[text="OK"] {{
                background-color: {t["primary"]};
                color: white;
                border: none;
                min-width: 100px;
                height: 36px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton[text="OK"]:hover {{
                background-color: {t["primary_hover"]};
            }}
            QPushButton[text="Cancel"] {{
                background-color: {t["surface"]};
                color: {t["text"]};
                border: 1px solid {t["border"]};
                min-width: 100px;
                height: 36px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton[text="Cancel"]:hover {{
                background-color: {t["background_hover"]};
            }}
            QSpinBox {{
                border: 1px solid {t["border"]};
                border-radius: 4px;
                background-color: {t["background_secondary"]};
                color: {t["text"]};
                font-size: 13px;
                padding: 2px 4px;
            }}
            QSpinBox:disabled {{
                background-color: {t["surface"]};
                color: {t["border"]};
            }}
            QComboBox {{
                border: 1px solid {t["border"]};
                border-radius: 4px;
                background-color: {t["background_secondary"]};
                color: {t["text"]};
                font-size: 13px;
                padding: 2px 4px;
            }}
        """)

        # Start with one default point row
        self._add_row("point")

    def _make_spinbox(self):
        spin = QSpinBox()
        spin.setRange(0, 99999)
        spin.setValue(0)
        return spin

    def _add_row(self, shape_type):
        row = self.table.rowCount()
        self.table.insertRow(row)

        shape_combo = QComboBox()
        shape_combo.addItems([self.tr("Point"), self.tr("Rectangle")])
        shape_combo.setCurrentIndex(0 if shape_type == "point" else 1)
        self.table.setCellWidget(row, self._COL_SHAPE, shape_combo)

        x1_spin = self._make_spinbox()
        y1_spin = self._make_spinbox()
        x2_spin = self._make_spinbox()
        y2_spin = self._make_spinbox()
        self.table.setCellWidget(row, self._COL_X1, x1_spin)
        self.table.setCellWidget(row, self._COL_Y1, y1_spin)
        self.table.setCellWidget(row, self._COL_X2, x2_spin)
        self.table.setCellWidget(row, self._COL_Y2, y2_spin)

        type_combo = QComboBox()
        type_combo.addItems([self.tr("Positive"), self.tr("Negative")])
        self.table.setCellWidget(row, self._COL_TYPE, type_combo)

        # Wire shape change to enable/disable X2/Y2
        shape_combo.currentIndexChanged.connect(
            lambda idx, r=row: self._on_shape_changed(r, idx)
        )
        self._on_shape_changed(row, shape_combo.currentIndex())

    def _on_shape_changed(self, row, index):
        is_rect = index == 1  # 0=Point, 1=Rectangle
        for col in (self._COL_X2, self._COL_Y2):
            widget = self.table.cellWidget(row, col)
            if widget:
                widget.setEnabled(is_rect)

    def _remove_row(self):
        selected_rows = sorted(
            set(idx.row() for idx in self.table.selectedIndexes()),
            reverse=True,
        )
        if selected_rows:
            for row in selected_rows:
                self.table.removeRow(row)
        elif self.table.rowCount() > 0:
            self.table.removeRow(self.table.rowCount() - 1)

    def get_marks(self):
        marks = []
        for row in range(self.table.rowCount()):
            shape_w = self.table.cellWidget(row, self._COL_SHAPE)
            x1_w = self.table.cellWidget(row, self._COL_X1)
            y1_w = self.table.cellWidget(row, self._COL_Y1)
            x2_w = self.table.cellWidget(row, self._COL_X2)
            y2_w = self.table.cellWidget(row, self._COL_Y2)
            type_w = self.table.cellWidget(row, self._COL_TYPE)
            if not all([shape_w, x1_w, y1_w, x2_w, y2_w, type_w]):
                continue
            label = 1 if type_w.currentIndex() == 0 else 0
            if shape_w.currentIndex() == 0:  # Point
                marks.append(
                    {
                        "type": "point",
                        "data": [x1_w.value(), y1_w.value()],
                        "label": label,
                    }
                )
            else:  # Rectangle
                marks.append(
                    {
                        "type": "rectangle",
                        "data": [
                            x1_w.value(),
                            y1_w.value(),
                            x2_w.value(),
                            y2_w.value(),
                        ],
                        "label": label,
                    }
                )
        return marks

    def exec_and_get_marks(self):
        if self.exec() == QDialog.DialogCode.Accepted:
            return self.get_marks()
        return []


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def load_existing_shapes(image_file):
    """
    Loads existing shapes from the JSON file for skip detection.

    Args:
        image_file (str): The path to the image file.

    Returns:
        list: A list of Shape objects loaded from the JSON file, or None if
              the file does not exist or contains no shapes.
    """
    label_file = osp.splitext(image_file)[0] + ".json"
    if not osp.exists(label_file):
        return None

    try:
        with io_open(label_file, "r") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        if not shapes:
            return None

        existing_shapes = []
        for shape_data in shapes:
            shape = Shape()
            shape.load_from_dict(shape_data, close=False)
            if shape.shape_type in ["rectangle", "rotation", "polygon"]:
                shape.selected = True
                existing_shapes.append(shape)

        return existing_shapes if existing_shapes else None

    except Exception as e:
        logger.warning(f"Failed to load existing shapes: {e}")
        return None


def finish_processing(self, progress_dialog):
    target_index = self.current_index
    target_file = self.image_list[self.current_index]
    self.import_image_folder(osp.dirname(target_file), load=False)
    self.file_list_widget.setCurrentRow(target_index)

    del self.text_prompt
    del self.run_tracker
    del self.image_index
    del self.current_index
    if hasattr(self, "point_marks"):
        del self.point_marks

    progress_dialog.close()

    popup = Popup(
        self.tr("Processing completed successfully!"),
        self,
        icon=new_icon_path("copy-green", "svg"),
    )
    popup.show_popup(self, position="center")


def cancel_operation(self):
    self.cancel_processing = True


def save_auto_labeling_result(self, image_file, auto_labeling_result):
    try:
        label_file = osp.splitext(image_file)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))

        if auto_labeling_result is None:
            new_shapes = []
            new_description = ""
            replace = True
        else:
            new_shapes = [
                shape.to_dict() for shape in auto_labeling_result.shapes
            ]
            new_description = auto_labeling_result.description
            replace = auto_labeling_result.replace

        if osp.exists(label_file):
            with io_open(label_file, "r") as f:
                data = json.load(f)

            if replace:
                data["shapes"] = new_shapes
                data["description"] = new_description
            else:
                data["shapes"].extend(new_shapes)
                if "description" in data:
                    data["description"] += new_description
                else:
                    data["description"] = new_description
        else:
            if self._config["store_data"]:
                with open(image_file, "rb") as f:
                    image_data = f.read()
                image_data = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data = None

            image_path = osp.basename(image_file)
            image_width, image_height = get_image_size(image_file)

            data = {
                "version": __version__,
                "flags": {},
                "shapes": new_shapes,
                "imagePath": image_path,
                "imageData": image_data,
                "imageHeight": image_height,
                "imageWidth": image_width,
                "description": new_description,
            }

        with io_open(label_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(
            f"Failed to save auto labeling result for image file '{image_file}': {str(e)}"
        )


class BatchProcessingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    processing_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        app,
        image_list,
        image_index,
        model_type,
        text_prompt,
        run_tracker,
        skip_detection,
        point_marks=None,
    ):
        super().__init__()
        self.app = app
        self.image_list = image_list
        self.image_index = image_index
        self.model_type = model_type
        self.text_prompt = text_prompt
        self.run_tracker = run_tracker
        self.skip_detection = skip_detection
        self.point_marks = point_marks or []

    def run(self):
        total_images = len(self.image_list)
        try:
            while (
                self.image_index < total_images
                and not self.app.cancel_processing
            ):
                image_file = self.image_list[self.image_index]
                current = self.image_index + 1
                self.progress_updated.emit(
                    current, f"Progress: {current}/{total_images}"
                )

                if self.point_marks:
                    self.app.auto_labeling_widget.model_manager.set_auto_labeling_marks(
                        self.point_marks
                    )

                if self.text_prompt:
                    result = self.app.auto_labeling_widget.model_manager.predict_shapes(
                        self.app.image,
                        image_file,
                        text_prompt=self.text_prompt,
                        batch=True,
                    )
                elif self.run_tracker:
                    result = self.app.auto_labeling_widget.model_manager.predict_shapes(
                        self.app.image,
                        image_file,
                        run_tracker=self.run_tracker,
                        batch=True,
                    )
                else:
                    existing_shapes = None
                    if (
                        self.model_type in _SKIP_DET_MODELS
                        and self.skip_detection
                    ):
                        existing_shapes = load_existing_shapes(image_file)
                    result = self.app.auto_labeling_widget.model_manager.predict_shapes(
                        self.app.image,
                        image_file,
                        batch=True,
                        existing_shapes=existing_shapes,
                    )

                save_auto_labeling_result(self.app, image_file, result)
                self.image_index += 1

            self.app.image_index = self.image_index
            self.processing_finished.emit()
        except Exception as e:
            self.app.image_index = self.image_index
            self.error_occurred.emit(str(e))


def process_next_image(self, progress_dialog, batch=True):
    """Process images in batch mode.

    Args:
        progress_dialog: Progress dialog widget for displaying progress.
        batch: If True, results are saved directly without updating canvas.
               If False, results trigger UI updates and canvas refresh.
               Defaults to True for batch processing mode.
    """
    model_type = self.auto_labeling_widget.model_manager.loaded_model_config[
        "type"
    ]
    model = self.auto_labeling_widget.model_manager.loaded_model_config[
        "model"
    ]
    total_images = len(self.image_list)
    self._progress_dialog = progress_dialog

    batch_processing_mode = "default"
    if model_type == "remote_server":
        batch_processing_mode = model.get_batch_processing_mode()

    if (
        model_type not in _BATCH_PROCESSING_VIDEO_MODELS
        and batch_processing_mode != "video"
    ):
        skip_detection = (
            self.auto_labeling_widget.button_skip_detection.isChecked()
        )
        self._batch_thread = BatchProcessingThread(
            self,
            self.image_list,
            self.image_index,
            model_type,
            self.text_prompt,
            self.run_tracker,
            skip_detection,
            point_marks=getattr(self, "point_marks", []),
        )

        def _on_progress(value, label):
            progress_dialog.setValue(value)
            progress_dialog.setLabelText(label)

        def _on_error(msg):
            progress_dialog.close()
            logger.error(f"Error occurred while processing images: {msg}")
            popup = Popup(
                self.tr("Error occurred while processing images!"),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

        self._batch_thread.progress_updated.connect(_on_progress)
        self._batch_thread.processing_finished.connect(
            lambda: finish_processing(self, progress_dialog)
        )
        self._batch_thread.error_occurred.connect(_on_error)
        self._batch_thread.start()
        return

    try:
        while (self.image_index < total_images) and (
            not self.cancel_processing
        ):
            image_file = self.image_list[self.image_index]
            current_progress = self.image_index + 1
            progress_dialog.setValue(current_progress)
            progress_dialog.setLabelText(
                f"Progress: {current_progress}/{total_images}"
            )
            QApplication.processEvents()

            batch_processing_mode = "default"
            if model_type == "remote_server":
                batch_processing_mode = model.get_batch_processing_mode()
                if batch_processing_mode == "video":
                    model._widget = self
                    self.filename = image_file
                    self.load_file(self.filename)
                    batch = False
            elif model_type in _BATCH_PROCESSING_VIDEO_MODELS:
                self.filename = image_file
                self.load_file(self.filename)
                batch = False

            if self.text_prompt:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        text_prompt=self.text_prompt,
                        batch=batch,
                    )
                )
            elif self.run_tracker:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        run_tracker=self.run_tracker,
                        batch=batch,
                    )
                )
                if batch_processing_mode == "video":
                    logger.info("Video propagation completed, breaking loop")
                    self.image_index = total_images
                    break
            else:
                existing_shapes = None
                if (
                    model_type in _SKIP_DET_MODELS
                    and self.auto_labeling_widget.button_skip_detection.isChecked()
                ):
                    existing_shapes = load_existing_shapes(image_file)

                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        batch=batch,
                        existing_shapes=existing_shapes,
                    )
                )

            if batch:
                save_auto_labeling_result(
                    self, image_file, auto_labeling_result
                )

            self.image_index += 1

        finish_processing(self, progress_dialog)

    except Exception as e:
        progress_dialog.close()

        logger.error(f"Error occurred while processing images: {e}")
        popup = Popup(
            self.tr("Error occurred while processing images!"),
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def show_progress_dialog_and_process(self):
    self.cancel_processing = False

    progress_dialog = QProgressDialog(
        self.tr("Processing..."),
        self.tr("Cancel"),
        0,
        len(self.image_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Batch Processing"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)

    initial_progress = (
        self.image_index + 1
        if self.image_index < len(self.image_list)
        else len(self.image_list)
    )
    progress_dialog.setValue(initial_progress)
    progress_dialog.setLabelText(
        f"Progress: {initial_progress}/{len(self.image_list)}"
    )
    progress_bar = progress_dialog.findChild(QtWidgets.QProgressBar)

    if progress_bar:
        model_type = (
            self.auto_labeling_widget.model_manager.loaded_model_config.get(
                "type", ""
            )
        )
        batch_processing_mode = "default"
        if model_type == "remote_server":
            model = self.auto_labeling_widget.model_manager.loaded_model_config.get(
                "model"
            )
            batch_processing_mode = model.get_batch_processing_mode()

        def update_progress(value):
            if batch_processing_mode != "video":
                progress_dialog.setLabelText(f"{value}/{len(self.image_list)}")

        progress_bar.valueChanged.connect(update_progress)

    t = get_theme()
    progress_dialog.setStyleSheet(f"""
        QProgressDialog {{
            background-color: {t["background"]};
            border-radius: 12px;
            min-width: 280px;
            min-height: 120px;
            padding: 20px;
        }}
        QProgressBar {{
            border: none;
            border-radius: 4px;
            background-color: {t["surface"]};
            text-align: center;
            color: {t["text"]};
            font-size: 13px;
            min-height: 20px;
            max-height: 20px;
            margin: 16px 0;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {t["primary"]},
                stop:0.5 {t["highlight"]},
                stop:1 {t["primary"]});
            border-radius: 3px;
        }}
        QLabel {{
            color: {t["text"]};
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 8px;
        }}
        QPushButton {{
            background-color: {t["surface"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
            color: {t["primary"]};
            min-width: 82px;
            height: 36px;
            padding: 0 16px;
            margin-top: 16px;
        }}
        QPushButton:hover {{
            background-color: {t["background_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["surface"]};
        }}
    """)
    progress_dialog.canceled.connect(lambda: cancel_operation(self))
    progress_dialog.show()

    QTimer.singleShot(200, lambda: process_next_image(self, progress_dialog))


def run_all_images(self):
    if len(self.image_list) < 1:
        return

    if self.auto_labeling_widget.model_manager.loaded_model_config is None:
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr("Model is not loaded. Choose a mode to continue.")
        )
        return

    if (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in _BATCH_PROCESSING_INVALID_MODELS
    ):
        logger.warning(
            f"The model `{self.auto_labeling_widget.model_manager.loaded_model_config['type']}`"
            f" is not supported for this action."
            f" Please choose a valid model to execute."
        )
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr(
                "Invalid model type, please choose a valid model_type to run."
            )
        )
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    response.setWindowTitle(self.tr("Confirmation"))
    response.setText(self.tr("Do you want to process all images?"))
    response.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Cancel
        | QtWidgets.QMessageBox.StandardButton.Ok
    )
    response.setStyleSheet(get_msg_box_style())
    if response.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
        return

    logger.info("Start running all images...")

    self.current_index = self.fn_to_index[str(self.filename)]
    self.image_index = self.current_index
    self.text_prompt = ""
    self.run_tracker = False
    self.point_marks = []

    model_type = self.auto_labeling_widget.model_manager.loaded_model_config[
        "type"
    ]

    if model_type == "remote_server":
        batch_processing_mode = "default"
        model = self.auto_labeling_widget.model_manager.loaded_model_config[
            "model"
        ]
        if hasattr(model, "get_batch_processing_mode"):
            batch_processing_mode = model.get_batch_processing_mode()
        else:
            batch_processing_mode = "default"
        if batch_processing_mode is None:
            self.auto_labeling_widget.model_manager.new_model_status.emit(
                self.tr(
                    "Batch processing is not supported for the current task."
                )
            )
            return
        if batch_processing_mode == "video":
            self.run_tracker = True
            show_progress_dialog_and_process(self)
        elif batch_processing_mode == "text_prompt":
            text_input_dialog = TextInputDialog(parent=self)
            self.text_prompt = text_input_dialog.get_input_text()
            if self.text_prompt:
                show_progress_dialog_and_process(self)
        else:
            show_progress_dialog_and_process(self)
    elif model_type in _BATCH_PROCESSING_POINT_PROMPT_MODELS:
        point_input_dialog = PointInputDialog(parent=self)
        self.point_marks = point_input_dialog.exec_and_get_marks()
        if self.point_marks:
            show_progress_dialog_and_process(self)
    elif model_type in _BATCH_PROCESSING_TEXT_PROMPT_MODELS:
        text_input_dialog = TextInputDialog(parent=self)
        self.text_prompt = text_input_dialog.get_input_text()
        if self.text_prompt or model_type == "yoloe":
            show_progress_dialog_and_process(self)
    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        == "florence2"
    ):
        self.text_prompt = self.auto_labeling_widget.edit_text.text()
        show_progress_dialog_and_process(self)
    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in _BATCH_PROCESSING_VIDEO_MODELS
    ):
        self.run_tracker = True
        show_progress_dialog_and_process(self)
    else:
        show_progress_dialog_and_process(self)
