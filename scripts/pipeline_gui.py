"""
Graphical User Interface for the RCP Analysis Pipeline
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

#PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSplitter, QCheckBox, QListWidget,
                               QListWidgetItem, QGridLayout, QMessageBox, QAbstractItemView,
                               QComboBox, QProgressBar, QFrame, QPlainTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

from RCP_analysis.python.functions.params_loading import load_experiment_params
from RCP_analysis.python.functions.br_preproc import list_br_sessions
from run_pipeline import run_scripts

def list_sessions(data_root:str) -> list[dict]:
    """List all sessions and the locations for a given data_root.
    Args:
        data_root (str): The root directory of the data.

    Returns:
        list[dict]: A list of session dictionaries.
    """
    status_csv = Path(data_root) / "data_status_reaching.csv"

    if not status_csv.exists():
        raise FileNotFoundError(f"data_status_reaching.csv not found: {status_csv}")

    with status_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return rows


def list_br_indices(data_root: str, location: str, session: str) -> list[int]:
    """List all BR indices for a given data_root, location, and session.

    Only files/folders whose name starts with `session` are considered --
    a Blackrock folder can also contain unrelated recordings (e.g. "test_001"
    calibration files) that don't belong to any session and would otherwise
    crash the int() parse below.

    Args:
        data_root (str): The root directory of the data.
        location (str): The location to filter by.
        session (str): The session name; only entries prefixed with this are kept.

    Returns:
        list[int]: A list of BR indices.
    """
    br_root = Path(data_root) / location / "Blackrock"
    paths = list_br_sessions(br_root)

    indices = set()
    for p in paths:
        if not p.name.startswith(session):
            continue
        try:
            indices.add(int(p.name.split("_")[-1]))
        except ValueError:
            continue

    return sorted(indices)


SCRIPT_CATALOG: list[tuple[str, str]] = [
    ("Preprocessing", "preprocessing_scripts/OCR_frame_correction.py"),
    ("Preprocessing", "preprocessing_scripts/align_dlc_two_cams_to_br.py"),
    ("Preprocessing", "preprocessing_scripts/align_VOG_to_br.py"),
    ("Preprocessing", "preprocessing_scripts/NPRW_Intan_analysis_mf.py"),
    ("Preprocessing", "preprocessing_scripts/compute_br_to_intan_shifts.py"),
    ("Preprocessing", "preprocessing_scripts/UA_BR_analysis_mf.py"),
    ("Preprocessing", "preprocessing_scripts/UA_BR_analysis_ssmf.py"),
    ("Preprocessing", "preprocessing_scripts/make_aligned_npz_and_mat.py"),
    ("Preprocessing", "preprocessing_scripts/extract_peri_stim.py"),
    ("Preprocessing", "preprocessing_scripts/inspect_kinematics_trajectories.py"),
    ("Preprocessing", "preprocessing_scripts/analyze_lfp_bands.py"),
    ("Analysis", "analysis_scripts/plot_plateau_analysis.py"),
    ("Analysis", "analysis_scripts/RSA_calculation.py"),
    ("Analysis", "analysis_scripts/plot_complete_shaded_BT.py"),
    ("Analysis", "analysis_scripts/plot_peri_stim_raster.py"),
    ("Analysis", "analysis_scripts/plot_stim_group_responses.py"),
    ("Analysis", "analysis_scripts/plot_stim_response_overlays.py"),
    ("Analysis", "analysis_scripts/plot_peak_csv_summaries.py"),
    ("Nikita Scripts", "scripts/nikita_scripts/lfp_processing/plot_lfp_cleaner.py"),
    ("Nikita Scripts", "scripts/nikita_scripts/plotting_scripts/combine_UA_gifs.py"),
]

# Number of condition checkboxes per row in the Conditions strip.
CONDITIONS_PER_ROW = 15


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RCP Analysis Pipeline")

        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_root: str | None = None
        self.data_root_parent: Path | None = None
        self.sessions_data: list[dict] = []
        self.thread: QThread | None = None
        self.worker: "PipelineWorker | None" = None

        # condition_checkboxes: (br_index, QCheckBox) pairs for whichever single
        # session is currently selected -- rebuilt by _on_session_selection_changed.
        self.condition_checkboxes: list[tuple[int, QCheckBox]] = []

        # queue_items: session -> QListWidgetItem, built fresh at the start of each run.
        self.queue_items: dict[str, QListWidgetItem] = {}
        self.run_order: list[str] = []

        # --- Top bar: Animal + data_root + Run button ---
        self.animal_label = QLabel("Animal:")
        self.monkey_combo = QComboBox()
        self.monkey_combo.addItems(["Nike", "Ada", "Bert"])
        self.monkey_combo.currentTextChanged.connect(self._on_monkey_changed)

        self.data_root_label = QLabel("data_root: (not loaded)")

        self.run_button = QPushButton("Run Selected Scripts")
        self.run_button.setEnabled(False)
        # Named _on_run_clicked (not run_scripts) so it doesn't shadow the
        # run_scripts() imported from run_pipeline.
        self.run_button.clicked.connect(self._on_run_clicked)

        # --- Conditions strip: "Run all" toggle + per-BR-index checkboxes ---
        self.run_all_checkbox = QCheckBox("Run all conditions")
        self.run_all_checkbox.setChecked(True)
        self.run_all_checkbox.stateChanged.connect(self._on_run_all_changed)

        self.conditions_container = QWidget()
        self.conditions_layout = QGridLayout()
        self.conditions_layout.setContentsMargins(0, 0, 0, 0)
        self.conditions_container.setLayout(self.conditions_layout)

        # --- Sessions column ---
        self.sessions_count_label = QLabel("")
        self.session_list_widget = QListWidget()
        self.session_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.session_list_widget.itemSelectionChanged.connect(self._on_session_selection_changed)
        self.session_list_widget.itemSelectionChanged.connect(self._validate)

        # --- Scripts column ---
        self.scripts_count_label = QLabel("")
        self.clear_scripts_button = QPushButton("Clear")
        self.clear_scripts_button.clicked.connect(self._on_clear_scripts_clicked)
        self.script_list_widget = QListWidget()
        self.script_list_widget.itemChanged.connect(self._validate)

        # --- Run column: progress bar, per-session queue, compact log preview ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%v / %m sessions")
        self.queue_list_widget = QListWidget()
        self.log_preview = QPlainTextEdit()
        self.log_preview.setReadOnly(True)
        self.log_preview.setMaximumBlockCount(300)
        self.log_preview.setMaximumHeight(110)

        # --- Footer status bar ---
        self.status_label = QLabel("")

        self._populate_script_catalog()
        self.setup_layout()

        self._load_data_root()
        self._populate_sessions()

    # ---------------------------------------------------------------
    # Layout
    # ---------------------------------------------------------------
    def setup_layout(self):
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.animal_label)
        top_bar.addWidget(self.monkey_combo)
        top_bar.addWidget(self.data_root_label, stretch=1)
        top_bar.addWidget(self.run_button)
        top_bar_widget = QWidget()
        top_bar_widget.setLayout(top_bar)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)

        conditions_bar = QHBoxLayout()
        conditions_bar.addWidget(QLabel("Conditions"))
        conditions_bar.addWidget(self.run_all_checkbox)
        conditions_bar.addWidget(divider)
        conditions_bar.addWidget(self.conditions_container, stretch=1)
        conditions_bar_widget = QWidget()
        conditions_bar_widget.setLayout(conditions_bar)

        sessions_header = QHBoxLayout()
        sessions_header.addWidget(QLabel("Sessions"))
        sessions_header.addWidget(self.sessions_count_label)
        sessions_header.addStretch(1)
        sessions_layout = QVBoxLayout()
        sessions_layout.addLayout(sessions_header)
        sessions_layout.addWidget(self.session_list_widget)
        sessions_column = QWidget()
        sessions_column.setLayout(sessions_layout)

        scripts_header = QHBoxLayout()
        scripts_header.addWidget(QLabel("Scripts"))
        scripts_header.addWidget(self.scripts_count_label)
        scripts_header.addStretch(1)
        scripts_header.addWidget(self.clear_scripts_button)
        scripts_layout = QVBoxLayout()
        scripts_layout.addLayout(scripts_header)
        scripts_layout.addWidget(self.script_list_widget)
        scripts_column = QWidget()
        scripts_column.setLayout(scripts_layout)

        run_layout = QVBoxLayout()
        run_layout.addWidget(QLabel("Run"))
        run_layout.addWidget(self.progress_bar)
        run_layout.addWidget(self.queue_list_widget)
        run_layout.addWidget(self.log_preview)
        run_column = QWidget()
        run_column.setLayout(run_layout)

        columns_layout = QHBoxLayout()
        columns_layout.addWidget(sessions_column)
        columns_layout.addWidget(scripts_column)
        columns_layout.addWidget(run_column)
        columns_widget = QWidget()
        columns_widget.setLayout(columns_layout)

        footer_layout = QHBoxLayout()
        footer_layout.addWidget(self.status_label)
        footer_widget = QWidget()
        footer_widget.setLayout(footer_layout)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(columns_widget)

        central_layout = QVBoxLayout()
        central_layout.addWidget(top_bar_widget)
        central_layout.addWidget(conditions_bar_widget)
        central_layout.addWidget(splitter, stretch=1)
        central_layout.addWidget(footer_widget)
        central_widget = QWidget()
        central_widget.setLayout(central_layout)

        self.setCentralWidget(central_widget)
        self.resize(1040, 700)

    def _populate_script_catalog(self):
        current_group = None
        for group, script_path in SCRIPT_CATALOG:
            if group != current_group:
                header = QListWidgetItem(f"-- {group} --")
                header.setFlags(Qt.NoItemFlags)
                self.script_list_widget.addItem(header)
                current_group = group

            item = QListWidgetItem(Path(script_path).name)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, script_path)
            self.script_list_widget.addItem(item)

    # ---------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------
    def _load_data_root(self):
        try:
            params_path = self.base_dir / "config" / "params.yaml"
            params = load_experiment_params(params_path, repo_root=self.base_dir, first_run=True)
            self.data_root = params.data_root
            self.data_root_parent = Path(self.data_root).parent
            # Reflect the resolved animal in the combo without re-triggering a reload.
            self.monkey_combo.blockSignals(True)
            self.monkey_combo.setCurrentText(params.monkey)
            self.monkey_combo.blockSignals(False)
        except Exception as exc:
            self.data_root = None
            self.data_root_parent = None
            self.status_label.setText(f"Could not resolve data_root: {exc}")

        self.data_root_label.setText(f"data_root: {self.data_root or '(unavailable)'}")

    def _on_monkey_changed(self, new_monkey: str):
        if self.data_root_parent is None:
            return

        self.data_root = str(self.data_root_parent / new_monkey)
        self.data_root_label.setText(f"data_root: {self.data_root}")

        self._rebuild_conditions([])
        self._populate_sessions()

    def _populate_sessions(self):
        self.session_list_widget.clear()
        self.sessions_data = []

        if self.data_root is None:
            self.sessions_count_label.setText("")
            return

        try:
            self.sessions_data = list_sessions(self.data_root)
        except FileNotFoundError as exc:
            self.status_label.setText(str(exc))
            self.sessions_count_label.setText("")
            return

        for row in self.sessions_data:
            self.session_list_widget.addItem(row["Session"])

        self.sessions_count_label.setText(f"0 / {len(self.sessions_data)}")

    def _on_session_selection_changed(self):
        selected_items = self.session_list_widget.selectedItems()
        self.sessions_count_label.setText(f"{len(selected_items)} / {self.session_list_widget.count()}")

        # Item 5 stopgap: specific-condition picking is only allowed for a single
        # selected session, since process_only is one global list applied to every
        # selected session in the run -- multi-select locks to "run all conditions".
        if len(selected_items) > 1:
            self.run_all_checkbox.setChecked(True)
            self.run_all_checkbox.setEnabled(False)
            self.run_all_checkbox.setToolTip(
                "Multiple sessions selected -- specific conditions can only be picked "
                "for a single session at a time."
            )
        else:
            self.run_all_checkbox.setEnabled(True)
            self.run_all_checkbox.setToolTip("")

        if not selected_items or self.data_root is None:
            self._rebuild_conditions([])
            return

        selected_name = selected_items[0].text()
        matching_row = next(
            (row for row in self.sessions_data if row["Session"] == selected_name),
            None,
        )
        if matching_row is None:
            self._rebuild_conditions([])
            return

        location = matching_row["Location"]
        try:
            indices = list_br_indices(self.data_root, location, selected_name)
        except FileNotFoundError as exc:
            self.status_label.setText(str(exc))
            self._rebuild_conditions([])
            return

        self._rebuild_conditions(indices)

    def _rebuild_conditions(self, indices: list[int]):
        # Clear out any checkboxes from the previously selected session.
        while self.conditions_layout.count():
            child = self.conditions_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()
        self.condition_checkboxes = []

        run_all = self.run_all_checkbox.isChecked()
        for i, idx in enumerate(indices):
            checkbox = QCheckBox(str(idx))
            checkbox.setEnabled(not run_all)
            checkbox.stateChanged.connect(self._validate)
            self.conditions_layout.addWidget(checkbox, i // CONDITIONS_PER_ROW, i % CONDITIONS_PER_ROW)
            self.condition_checkboxes.append((idx, checkbox))

    def _on_run_all_changed(self, _state):
        run_all = self.run_all_checkbox.isChecked()
        for _idx, checkbox in self.condition_checkboxes:
            checkbox.setEnabled(not run_all)
        self._validate()

    def _on_clear_scripts_clicked(self):
        for i in range(self.script_list_widget.count()):
            item = self.script_list_widget.item(i)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Unchecked)
        self._validate()

    # ---------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------
    def _collect_run_params(self):
        sessions = [item.text() for item in self.session_list_widget.selectedItems()]

        if self.run_all_checkbox.isChecked():
            process_only = []
        else:
            process_only = [idx for idx, checkbox in self.condition_checkboxes if checkbox.isChecked()]

        scripts = []
        for i in range(self.script_list_widget.count()):
            item = self.script_list_widget.item(i)
            if item.flags() & Qt.ItemIsUserCheckable and item.checkState() == Qt.Checked:
                scripts.append(item.data(Qt.UserRole))

        return sessions, process_only, scripts

    def _validate(self):
        sessions, _, scripts = self._collect_run_params()
        self.scripts_count_label.setText(f"{len(scripts)} selected")
        self.run_button.setEnabled(bool(sessions) and bool(scripts))

    def _on_run_clicked(self):
        sessions, process_only, scripts = self._collect_run_params()

        if not sessions or not scripts:
            QMessageBox.warning(self, "Missing selection", "Select at least one session and one script.")
            return

        if self.data_root is None:
            QMessageBox.critical(
                self,
                "Data root unavailable",
                "Cannot resolve data_root -- check that params.yaml/machines.yaml are set up "
                "and the data drive is mounted.",
            )
            return

        self._set_inputs_enabled(False)
        self.run_button.setText("Running...")
        self.progress_bar.setMaximum(len(sessions))
        self.progress_bar.setValue(0)
        self.log_preview.clear()
        self.status_label.setText("Running...")

        self.run_order = list(sessions)
        self.queue_list_widget.clear()
        self.queue_items = {}
        for session in self.run_order:
            item = QListWidgetItem(f"{session} — queued")
            self.queue_list_widget.addItem(item)
            self.queue_items[session] = item
        if self.run_order:
            self.queue_items[self.run_order[0]].setText(f"{self.run_order[0]} — running")

        self.thread = QThread()
        self.worker = PipelineWorker(self.base_dir, sessions, scripts, process_only)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log_line.connect(self.log_preview.appendPlainText)
        self.worker.session_done.connect(self._on_session_done)
        self.worker.finished.connect(self._on_run_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _on_session_done(self, session: str, ok: bool):
        self.progress_bar.setValue(self.progress_bar.value() + 1)
        state = "OK" if ok else "FAILED"
        self.status_label.setText(f"{session}: {state}")

        item = self.queue_items.get(session)
        if item is not None:
            item.setText(f"{session} — {state}")

        # Mark the next queued session (if any) as running.
        if session in self.run_order:
            next_index = self.run_order.index(session) + 1
            if next_index < len(self.run_order):
                next_session = self.run_order[next_index]
                next_item = self.queue_items.get(next_session)
                if next_item is not None:
                    next_item.setText(f"{next_session} — running")

    def _on_run_finished(self, results: dict):
        self._set_inputs_enabled(True)
        self.run_button.setText("Run Selected Scripts")
        if results:
            summary = ", ".join(f"{session}: {'OK' if ok else 'FAILED'}" for session, ok in results.items())
        else:
            summary = "No sessions were run."
        self.status_label.setText(summary)

    def _set_inputs_enabled(self, enabled: bool):
        self.session_list_widget.setEnabled(enabled)
        self.script_list_widget.setEnabled(enabled)
        self.clear_scripts_button.setEnabled(enabled)
        self.monkey_combo.setEnabled(enabled)
        self.run_all_checkbox.setEnabled(enabled and len(self.session_list_widget.selectedItems()) <= 1)
        run_all = self.run_all_checkbox.isChecked()
        for _idx, checkbox in self.condition_checkboxes:
            checkbox.setEnabled(enabled and not run_all)
        if enabled:
            self._validate()
        else:
            self.run_button.setEnabled(False)


class PipelineWorker(QObject):
    log_line = pyqtSignal(str)
    session_done = pyqtSignal(str, bool)
    finished = pyqtSignal(dict)

    def __init__(self, base_dir: Path, sessions: list[str], scripts: list[str], process_only: list[int]):
        super().__init__()
        self.base_dir = base_dir
        self.sessions = sessions
        self.scripts = scripts
        self.process_only = process_only

    def run(self):
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

        with log_path.open("w", encoding="utf-8") as log_file:
            def log(message: str) -> None:
                log_file.write(message + "\n")
                log_file.flush()
                self.log_line.emit(message)

            results = run_scripts(
                self.base_dir,
                self.base_dir,
                sessions=self.sessions,
                scripts=self.scripts,
                process_only=self.process_only,
                log=log,
                on_session_complete=self.session_done.emit,
            )

        self.finished.emit(results)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
