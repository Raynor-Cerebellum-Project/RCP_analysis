"""
VELES - Versatile Electrophysiology and Limb-motion Evaluation Suite

Veles: a slavic diety of magic, knowledge, divination, and poetry, 
        providing wisdom and guidance to his shamans, 
        governing the world outside the fences of human dwellings.

Like the god Veles, may this script provide us a view into the unknown!


CHOOSE THE MONKEY + SESSION(s) + SCRIPT(s) via Commenting/Uncommenting
"""

# MONKEY = "Ada"
# MONKEY = "Bert"
MONKEY = "Nike"


SESSIONS_TO_RUN = [
    # "NRR_RW035",
    # "NRR_RW034",
    # "NRR_RW032",
    # "NRR_RW029",
    # "NRR_RW026",
    # "NRR_RW022",
    # "NRR_RW019",
    # "NRR_RW018",
    # "NRR_RW017",
    # "NRR_RW016",
    # "NRR_RW015",
    # "NRR_RW014",
    # "NRR_RW013",
    "NRR_RW012",
    # "NRR_RW011",
]

PROCESS_ONLY = [17]

SCRIPTS = [
    "preprocessing_scripts/OCR_frame_correction.py",
    # "preprocessing_scripts/align_dlc_two_cams_to_br.py",
    # "preprocessing_scripts/align_VOG_to_br.py",
    "preprocessing_scripts/NPRW_Intan_analysis_mf.py",
    "preprocessing_scripts/compute_br_to_intan_shifts.py",
    # "preprocessing_scripts/UA_BR_analysis_mf.py", 
    "preprocessing_scripts/UA_BR_analysis_ssmf.py",
    "preprocessing_scripts/make_aligned_npz_and_mat.py",
    "preprocessing_scripts/extract_peri_stim.py",
    "preprocessing_scripts/inspect_kinematics_trajectories.py",
    ### RUN ^ inspect_kinematics_trajectories.py to check for remaining bad traces -> add bad traces to /config/manual_trial_remove.csv
    ### RERUN extract_peri_stim.py

    # "analysis_scripts/plot_plateau_analysis.py",
    # "analysis_scripts/RSA_calculation.py",
    # "analysis_scripts/plot_firing_rates.py",
    # "analysis_scripts/plot_peri_stim_raster.py",
    # "analysis_scripts/plot_stim_group_responses.py",
    # "analysis_scripts/plot_stim_response_overlays.py",
    # "analysis_scripts/plot_peak_csv_summaries.py",


    # "preprocessing_scripts/analyze_lfp_bands.py",
    # "scripts/nikita_scripts/lfp_processing/plot_lfp_cleaner.py",
    # "scripts/nikita_scripts/plotting_scripts/combine_UA_gifs.py",
]


import subprocess
from pathlib import Path
import csv
import sys
from datetime import datetime
from RCP_analysis.python.functions.params_loading import load_experiment_params 
from RCP_analysis.python.functions.pipeline_hierarchy import (
    SCRIPT_STATUS_COLUMNS,
    check_and_confirm_dependencies,
)
import os
import json
from typing import Any, Callable

LOG_FILE = Path(__file__).resolve().parent / "logs" / "VELES.log"


def _init_veles_run_log(log_file: Path, monkey: str, sessions: list[str], process_only: list[Any], scripts: list[str],) -> None:
    """Initialize a run block in VELES.log with execution parameters."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    lines = [
        "=" * 80,
        f"VELES RUN START: {ts}",
        f"Monkey: {monkey}",
        f"Sessions to run: {sessions}",
        f"Process only: {process_only}",
        "Scripts:",
    ]
    for s in scripts:
        lines.append(f"  - {s}")
    lines.append("=" * 80)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

def _log_veles_event(log_file: Path, session: str, script: str, status_text: str,) -> None:
    """Append a timestamped event line to VELES.log and flush immediately."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    entry = f"[{ts}] [{session}] [{script}] {status_text}\n"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(entry)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

def _update_script_status_for_session(data_root: str, session: str, status_column: str, value: str) -> None:
    """Update data_status_reaching.csv for the specified session and column."""
    status_csv = Path(data_root) / "data_status_reaching.csv"
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
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

    print(f"[VELES] Updated {status_column} for session {target_session}: {value}")

def _get_location_for_session(data_root: str, session: str) -> str:
    """
    Read data_root/data_status_reaching.csv and return the Location for `session`.

    This does NOT modify 'Process Session?'.
    It only uses the CSV as a lookup table.
    """
    status_csv = Path(data_root) / "data_status_reaching.csv"

    if not status_csv.exists():
        raise FileNotFoundError(f"data_status_reaching.csv not found: {status_csv}")

    with status_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError(f"{status_csv} appears to be empty or has no header row.")

        required_cols = {"Location", "Session"}
        missing = required_cols - set(fieldnames)
        if missing:
            raise KeyError(
                f"Missing required column(s) in {status_csv}: {sorted(missing)}"
            )

        rows = list(reader)

    target_session = str(session).strip()
    matches = []

    for row in rows:
        row_session = str(row.get("Session", "")).strip()
        if row_session == target_session:
            matches.append(row)

    if not matches:
        raise ValueError(f"Session '{target_session}' was not found in {status_csv}")

    if len(matches) > 1:
        raise ValueError(f"Session '{target_session}' appears multiple times in {status_csv}")

    location = str(matches[0].get("Location", "")).strip()

    if not location:
        raise ValueError(
            f"Session '{target_session}' has empty Location in {status_csv}"
        )

    return location


def run_scripts(
    base_dir: Path,
    scripts_folder: Path,
    sessions: list[str] | None = None,
    scripts: list[str] | None = None,
    process_only: list[int] | None = None,
    log: Callable[[str], None] = print,
    on_session_complete: Callable[[str, bool], None] | None = None,
    monkey: str | None = None,
) -> dict[str, bool]:
    monkey = MONKEY if monkey is None else monkey
    sessions = SESSIONS_TO_RUN if sessions is None else sessions
    scripts = SCRIPTS if scripts is None else scripts
    process_only = PROCESS_ONLY if process_only is None else process_only
    results: dict[str, bool] = {}

    params_path = base_dir / "config" / "params.yaml"

    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found: {params_path}")

    # Load params once so we can get the machine-specific data_root.
    PARAMS = load_experiment_params(params_path, repo_root=base_dir, first_run=True)
    root = Path(PARAMS.data_root)
    data_root_parent = root.parent if PARAMS.monkey else root
    data_root = str(data_root_parent / monkey)

    # Initialize run in VELES.log
    _init_veles_run_log(
        log_file=LOG_FILE,
        monkey=monkey,
        sessions=sessions,
        process_only=process_only,
        scripts=scripts,
    )

    for session in sessions:
        results[session] = True

        log(f"\n{'=' * 60}")
        log(f"[VELES] Processing session: {session}")
        log(f"{'=' * 60}")

        # Look up this session's location without modifying "Process Session?"
        location = _get_location_for_session(data_root, session)

        # Per-subprocess session context.
        # This is private to scripts launched by this run_pipeline.py process.
        env = os.environ.copy()
        env["RCP_MONKEY"] = monkey
        env["RCP_SESSION"] = session
        env["RCP_LOCATION"] = location
        env["RCP_PROCESS_ONLY"] = json.dumps(process_only)
        env["RCP_VELES_RUN"] = "1"

        log(f"[VELES] Session context: RCP_MONKEY={monkey}, RCP_SESSION={session}, RCP_LOCATION={location}, RCP_PROCESS_ONLY={process_only}")

        # Run all scripts for this session
        for script in scripts:
            script_path = scripts_folder / script

            if not script_path.exists():
                log(f"[VELES: ERROR] Script not found: {script_path}")
                failed_at = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
                _log_veles_event(LOG_FILE, session, script, f"FAIL: {failed_at} (Script not found)")
                results[session] = False
                break

            status_column = SCRIPT_STATUS_COLUMNS.get(Path(script).name)

            # Check pipeline hierarchy dependencies before running
            dep_ok = check_and_confirm_dependencies(
                script_name=script,
                session=session,
                data_root=data_root,
                planned_batch_scripts=scripts,
            )
            if not dep_ok:
                log(f"[VELES] Dependency check not confirmed for {script}; skipping.")
                _log_veles_event(LOG_FILE, session, script, "SKIPPED: Dependency discrepancy not approved by user")
                results[session] = False
                continue

            # Record IN-PROGRESS in log file and CSV immediately
            _log_veles_event(LOG_FILE, session, script, "IN-PROGRESS")
            if status_column is not None:
                _update_script_status_for_session(
                    data_root=data_root,
                    session=session,
                    status_column=status_column,
                    value="IN-PROGRESS",
                )

            log(f"\n[VELES] Running {script_path}\n")

            try:
                subprocess.run(
                    [sys.executable, str(script_path)],
                    check=True,
                    cwd=str(base_dir),
                    env=env,
                )

                # Script finished successfully
                finished_at = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
                _log_veles_event(LOG_FILE, session, script, f"FINISHED: {finished_at}")

                # If this script has a corresponding CSV status column, write finish time
                if status_column is not None:
                    _update_script_status_for_session(
                        data_root=data_root,
                        session=session,
                        status_column=status_column,
                        value=finished_at,
                    )

            except subprocess.CalledProcessError as e:
                failed_at = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
                log(f"[VELES: ERROR] Script failed for {session} with exit code {e.returncode}")
                _log_veles_event(LOG_FILE, session, script, f"FAIL: {failed_at} (exit code {e.returncode})")

                # If this script has a corresponding CSV status column, write FAIL
                if status_column is not None:
                    _update_script_status_for_session(
                        data_root=data_root,
                        session=session,
                        status_column=status_column,
                        value="FAIL",
                    )

                log("[VELES] Skipping to next session...")
                results[session] = False
                break

            except KeyboardInterrupt:
                log(f"\n[VELES: INTERRUPT] Execution interrupted by user. Status retained as IN-PROGRESS for {script}.")
                raise
        else:
            if results[session]:
                log(f"\n[VELES: SUCCESS] Completed all scripts for {session}")

        if on_session_complete is not None:
            on_session_complete(session, results[session])

    log(f"\n{'=' * 60}")
    log("[VELES] All sessions completed!")
    log(f"{'=' * 60}")

    return results


def main():
    BASE = Path(__file__).resolve().parents[0]
    SCRIPTS_FOLDER = BASE
    run_scripts(BASE, SCRIPTS_FOLDER)
    
if __name__ == "__main__":
    main()
