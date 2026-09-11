"""
Graphical User Interface for the RCP Analysis Pipeline
"""

import sys
import csv
from pathlib import Path

#PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSplitter, QCheckBox, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QScrollArea, QGroupBox,QLabel,
                               QPushButton, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

from RCP_analysis.python.functions.params_loading import load_experiment_params
from RCP_analysis.python.functions.br_preproc import list_br_sessions
from run_pipeline import run_scripts 

def list_sessions(data_root:str) -> list[dict]:
    """List all sessions for a given data_root.
    Args:
        data_root (str): The root directory of the data.

    Returns:
        list[dict]: A list of session dictionaries.
    """
    status_csv = Path(data_root) / "data_status_reaching.csv"

    if not status_csv.exists():
        raise FileNotFoundError(f"Status CSV not found at {status_csv}")

    with status_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sessions = reader.fieldnames  # Get the header row (field names)

        if fieldnames is None:
            raise ValueError("CSV file is empty or malformed.") 

        required_cols = {"Session"}
        missing_cols = required_cols - set(sessions)
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}") 

        rows = list(reader)  # Read all rows into a list of dictionaries

    #keep directories that contain blackrock files and use directory name for sessions, specifically the trailing text after date if applicable
    #return a list of Sessions 

def list_br_indices(data_root:str, location:str) -> list[int]:
    """List all BR indices for a given data_root and location.
    Args:
        data_root (str): The root directory of the data.
        location (str): The location to filter by.

    Returns:
        list[int]: A list of BR indices.
    """
    #open data_root/sessions/blackrock
    #find unique numbers in file names 
    #return list of indices 



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
    ("Analysis", "analysis_scripts/plot_plateau_analysis.py"),
    ("Analysis", "analysis_scripts/RSA_calculation.py"),
    ("Analysis", "analysis_scripts/plot_complete_shaded_BT.py"),
    ("Analysis", "analysis_scripts/plot_peri_stim_raster.py"),
    ("Analysis", "analysis_scripts/plot_stim_group_responses.py"),
    ("Analysis", "analysis_scripts/plot_stim_response_overlays.py"),
    ("Analysis", "analysis_scripts/plot_peak_csv_summaries.py"),
    ("Preprocessing", "preprocessing_scripts/analyze_lfp_bands.py"),
    ("Scripts", "scripts/nikita_scripts/lfp_processing/plot_lfp_cleaner.py"),
    ("Scripts", "scripts/nikita_scripts/plotting_scripts/combine_UA_gifs.py"),
]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RCP Analysis Pipeline")
        #QListWidget to display the list of sessions
        self.session_list_widget = QListWidget()
        #QListWidget to display the list of scripts
        self.script_list_widget = QListWidget()
        #QListWidget to display the list of BR indices (conditions)
        self.br_index_list_widget = QListWidget()
        #QPlainTextEdit to display the log output
        self.log_output = QPlainTextEdit()
        #QPushButton to run the selected scripts
        self.run_button = QPushButton("Run Selected Scripts")
        #Connect the run_button to the run_scripts method
        self.run_button.clicked.connect(self.run_scripts)  
        #Set up the layout with 3 columns: sessions, scripts, and BR indices
        self.setup_layout()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
        