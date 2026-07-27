"""
Run scripts in order across multiple sessions.
Automatically updates params.yaml for each session (preserves comments).
"""

import subprocess
from pathlib import Path
import re
import csv
from datetime import datetime
from RCP_analysis.python.functions.params_loading import load_experiment_params 

SESSIONS_TO_RUN = [
    # "NRR_RW035",
    "NRR_RW034",
    "NRR_RW032",
    "NRR_RW029",
    "NRR_RW026",
    "NRR_RW022",
    "NRR_RW019",
    "NRR_RW018",
    "NRR_RW017",
    "NRR_RW016",
    "NRR_RW015",
    "NRR_RW014",
    "NRR_RW013",
    "NRR_RW012",
    "NRR_RW011",
]

BATCH_SCRIPTS = [
    # "OCR_frame_correction.py",
    # "align_dlc_two_cams_to_br.py",
    # "align_VOG_to_br.py",
    "NPRW_Intan_analysis_mf.py",
    "compute_br_to_intan_shifts.py",
    "UA_BR_analysis_mf.py", 
    "make_aligned_npz_and_mat.py",
    "extract_peri_stim.py",
    "inspect_kinematics_trajectories.py",
    # "plot_plateau_analysis.py",
    # "RSA_calculation.py",

    ### RUN ^ inspect_kinematics_trajectories.py to check for remaining bad traces -> add bad traces to /config/manual_trial_remove.csv
    ### RERUN extract_peri_stim.py

    # "generate_stim_summary.py",
    # "plot_complete_shaded_BT.py",
    # "plot_peri_stim_raster.py",
    # "analyze_lfp_bands.py",
    # "quantify_stim_kinematics.py",
    # "quantify_smoothed_position.py",
]



# Mapping script name to columns in CSV
SCRIPT_STATUS_COLUMNS = {
    "OCR_frame_mapping_BT_edit.py": "OCR",
    "align_dlc_two_cams_to_br.py": "DLC",
    "NPRW_Intan_analysis_mf.py": "NPRW_Intan_analysis",
    "compute_br_to_intan_shifts.py": "compute_shifts",
    "UA_BR_analysis_mf.py": "UA_BR_analysis",
    "make_aligned_npz_and_mat.py": "make_aligned",
    "extract_peri_stim.py": "extract_peri",
    "inspect_kinematics_trajectories.py": "manual_inspection",
    "plot_plateau_analysis.py": "plateau",
    "RSA_calculation.py": "rsa",
}

def _update_script_status_for_session(
    data_root: str,
    session: str,
    status_column: str,
    value: str,
) -> None:

    status_csv = Path(data_root) / "Nike" / "data_status_reaching.csv"

    if not status_csv.exists():
        raise FileNotFoundError(f"data_status_reaching.csv not found: {status_csv}")

    with status_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError(f"{status_csv} appears to be empty or has no header row.")

        required_cols = {"Session"}
        missing = required_cols - set(fieldnames)
        if missing:
            raise KeyError(
                f"Missing required column(s) in {status_csv}: {sorted(missing)}"
            )

        rows = list(reader)

    # Add the status column if it does not exist yet
    if status_column not in fieldnames:
        fieldnames.append(status_column)
        for row in rows:
            row[status_column] = ""

    target_session = str(session).strip()
    found = False

    for row in rows:
        row_session = str(row.get("Session", "")).strip()

        if row_session == target_session:
            row[status_column] = value
            found = True
            break

    if not found:
        raise ValueError(f"Session '{target_session}' was not found in {status_csv}")

    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[RAS] Updated {status_column} for session {target_session}: {value}")


def _set_process_which_for_session(data_root: str, session: str) -> None:
    """
    Read data_root/data_status_reaching.csv, find the row whose 'Session'
    matches `session`, set that row's 'Process Session?' to 'Yes', and set all
    other rows' 'Process Session?' values to 'No'.

    This modifies the CSV file in place.
    """
    status_csv = Path(data_root)  / "data_status_reaching.csv"

    if not status_csv.exists():
        raise FileNotFoundError(f"data_status_reaching.csv not found: {status_csv}")

    with status_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError(f"{status_csv} appears to be empty or has no header row.")

        required_cols = {"Process Session?", "Session"}
        missing = required_cols - set(fieldnames)
        if missing:
            raise KeyError(
                f"Missing required column(s) in {status_csv}: {sorted(missing)}"
            )

        rows = list(reader)

    target_session = str(session).strip()
    found = False

    for row in rows:
        row_session = str(row.get("Session", "")).strip()

        if row_session == target_session:
            row["Process Session?"] = "Yes"
            found = True
        else:
            row["Process Session?"] = "No"

    if not found:
        raise ValueError(
            f"Session '{target_session}' was not found in {status_csv}"
        )

    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_scripts(base_dir: Path, batch_scripts_folder: Path):
    params_path = base_dir / "config" / "params.yaml"

    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found: {params_path}")

    # Load params once so we can get the machine-specific data_root.
    PARAMS = load_experiment_params(params_path, repo_root=base_dir, first_run=True)
    data_root = PARAMS.data_root

    print("Running Across Sessions = RAS")
    print(f"{'=' * 60}")

    for session in SESSIONS_TO_RUN:

        print(f"\n{'=' * 60}")
        print(f"[RAS] Processing session: {session}")
        print(f"{'=' * 60}")

        # Update data_status_reaching.csv for this session.
        # This replaces the old update_params_yaml(...) behavior.
        _set_process_which_for_session(data_root, session)

        # Run all scripts for this session
        for batch_script in BATCH_SCRIPTS:
            script_path = batch_scripts_folder / batch_script

            if not script_path.exists():
                print(f"[RAS: ERROR] Script not found: {script_path}")
                break

            print(f"\n[RAS] Running {script_path}\n")

            status_column = SCRIPT_STATUS_COLUMNS.get(batch_script)

            try:
                subprocess.run(
                    ["python", str(script_path)],
                    check=True,
                    cwd=str(base_dir),
                )

                # If this script has a corresponding CSV status column, write finish time
                if status_column is not None:
                    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    _update_script_status_for_session(
                        data_root=data_root,
                        session=session,
                        status_column=status_column,
                        value=finished_at,
                    )

            except subprocess.CalledProcessError as e:
                print(
                    f"[RAS: ERROR] Script {batch_script} failed for "
                    f"{session} with exit code {e.returncode}"
                )

                # If this script has a corresponding CSV status column, write FAIL
                if status_column is not None:
                    _update_script_status_for_session(
                        data_root=data_root,
                        session=session,
                        status_column=status_column,
                        value="FAIL",
                    )

                print("[RAS] Skipping to next session...")
                break
        else:
            print(f"\n[RAS: SUCCESS] Completed all scripts for {session}")

    print(f"\n{'=' * 60}")
    print("[RAS] All sessions completed!")
    print(f"{'=' * 60}")


def main():
    BASE = Path(__file__).resolve().parents[1]
    BATCH_SCRIPTS_FOLDER = BASE / "batch_scripts"
    run_scripts(BASE, BATCH_SCRIPTS_FOLDER)
    
if __name__ == "__main__":
    main()