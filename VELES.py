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

PROCESS_ONLY = [7, 8, 9, 10, 11, 12]

SCRIPTS = [
    # "preprocessing_scripts/OCR_frame_correction.py",
    # "preprocessing_scripts/align_dlc_two_cams_to_br.py",
    # "preprocessing_scripts/align_VOG_to_br.py",
    # "preprocessing_scripts/NPRW_Intan_analysis_mf.py",
    # "preprocessing_scripts/compute_br_to_intan_shifts.py",
    # "preprocessing_scripts/UA_BR_analysis_mf.py", 
    # "preprocessing_scripts/UA_BR_analysis_ssmf.py",
    # "preprocessing_scripts/make_aligned_npz_and_mat.py",
    # "preprocessing_scripts/extract_peri_stim.py",
    # "preprocessing_scripts/inspect_kinematics_trajectories.py",
    ### RUN ^ inspect_kinematics_trajectories.py to check for remaining bad traces -> add bad traces to /config/manual_trial_remove.csv
    ### RERUN extract_peri_stim.py

    # "analysis_scripts/plot_plateau_analysis.py",
    # "analysis_scripts/RSA_calculation.py",
    # "analysis_scripts/plot_firing_rates.py",
    # "analysis_scripts/plot_peri_stim_raster.py",
    # "analysis_scripts/plot_stim_group_responses.py",
    # "analysis_scripts/plot_stim_response_overlays.py",
    "analysis_scripts/plot_cluster_stim_responses.py",

    # "scripts/nikita_scripts/plotting_scripts/plot_nprw_hpf_traces.py"


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
import argparse
from typing import Any, List, Optional

LOG_FILE = Path(__file__).resolve().parent / "logs" / "VELES.log"


def _get_veles_log_file(base_dir: Path, monkey: str) -> Path:
    """Generate a unique log file path formatted as VELES_{monkey}_{yyyymmdd}_{hhmmss}.log."""
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"VELES_{monkey}_{timestamp}.log"
    log_file = logs_dir / filename

    counter = 1
    while log_file.exists():
        log_file = logs_dir / f"VELES_{monkey}_{timestamp}_{counter}.log"
        counter += 1

    return log_file


def parse_process_only_inputs(raw_inputs: Any) -> Optional[List[Any]]:
    """
    Parse flexible CLI inputs for PROCESS_ONLY.
    Supports formats like:
      -po "7, 8, 9, 10, 11, 12"
      -po "7 8 9 10"
      -po 7 8 9 10 11 12
      -po 7, 8, 9
      -po "[7, 8, 9]"
    """
    if raw_inputs is None:
        return None

    if isinstance(raw_inputs, str):
        raw_inputs = [raw_inputs]

    items: List[Any] = []
    for item in raw_inputs:
        if isinstance(item, int):
            items.append(item)
            continue
        s = str(item).strip()
        if not s:
            continue

        # Check if enclosed in brackets JSON-style
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed_json = json.loads(s)
                if isinstance(parsed_json, list):
                    for sub in parsed_json:
                        items.append(int(sub) if str(sub).strip().lstrip("-").isdigit() else str(sub).strip())
                    continue
            except Exception:
                pass

        # Split by comma
        parts = s.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Handle space-separated tokens within a part
            subparts = part.split()
            for subpart in subparts:
                clean = subpart.strip().strip("'\"")
                if not clean:
                    continue
                if clean.lstrip("-").isdigit():
                    items.append(int(clean))
                else:
                    items.append(clean)

    return items if items else None


def _init_veles_run_log(log_file: Path, monkey: str, sessions: list[str], process_only: list[Any], scripts: list[str],) -> None:
    """Initialize a run block in log file with execution parameters."""
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
    """Append a timestamped event line to log file and flush immediately."""
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
    """Update data_status_reaching.csv for the specified session and column using an atomic write."""
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

    # Use atomic write via temp file replacement to prevent race conditions during concurrent runs
    temp_csv = status_csv.with_suffix(".csv.tmp")
    with temp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

    os.replace(temp_csv, status_csv)

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
    monkey: Optional[str] = None,
    sessions: Optional[List[str]] = None,
    process_only: Optional[List[Any]] = None,
):
    active_monkey = monkey if monkey is not None else MONKEY
    active_sessions = sessions if sessions is not None else SESSIONS_TO_RUN
    active_process_only = process_only if process_only is not None else PROCESS_ONLY

    params_path = base_dir / "config" / "params.yaml"

    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found: {params_path}")

    # Load params once so we can get the machine-specific data_root.
    PARAMS = load_experiment_params(params_path, repo_root=base_dir, first_run=True)
    data_root = f"{PARAMS.data_root}/{active_monkey}"

    # Generate unique per-instance log file for this run
    log_file = _get_veles_log_file(base_dir, active_monkey)
    print(f"[VELES] Logging run to: {log_file}")

    # Initialize run in instance log file
    _init_veles_run_log(
        log_file=log_file,
        monkey=active_monkey,
        sessions=active_sessions,
        process_only=active_process_only,
        scripts=SCRIPTS,
    )

    for session in active_sessions:

        print(f"\n{'=' * 60}")
        print(f"[VELES] Processing session: {session}")
        print(f"{'=' * 60}")

        # Look up this session's location without modifying "Process Session?"
        location = _get_location_for_session(data_root, session)

        # Per-subprocess session context.
        # This is private to scripts launched by this run_pipeline.py process.
        env = os.environ.copy()
        env["RCP_MONKEY"] = active_monkey
        env["RCP_SESSION"] = session
        env["RCP_LOCATION"] = location
        env["RCP_PROCESS_ONLY"] = json.dumps(active_process_only)
        env["RCP_VELES_RUN"] = "1"

        print(f"[VELES] Session context: RCP_MONKEY={active_monkey}, RCP_SESSION={session}, RCP_LOCATION={location}, RCP_PROCESS_ONLY={active_process_only}")

        # Run all scripts for this session
        for script in SCRIPTS:
            script_path = scripts_folder / script

            if not script_path.exists():
                print(f"[VELES: ERROR] Script not found: {script_path}")
                failed_at = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
                _log_veles_event(log_file, session, script, f"FAIL: {failed_at} (Script not found)")
                break

            status_column = SCRIPT_STATUS_COLUMNS.get(Path(script).name)

            # Check pipeline hierarchy dependencies before running
            dep_ok = check_and_confirm_dependencies(
                script_name=script,
                session=session,
                data_root=data_root,
                planned_batch_scripts=SCRIPTS,
            )
            if not dep_ok:
                print(f"[VELES] Dependency check not confirmed for {script}; skipping.")
                _log_veles_event(log_file, session, script, "SKIPPED: Dependency discrepancy not approved by user")
                continue

            # Record IN-PROGRESS in log file and CSV immediately
            _log_veles_event(log_file, session, script, "IN-PROGRESS")
            if status_column is not None:
                _update_script_status_for_session(
                    data_root=data_root,
                    session=session,
                    status_column=status_column,
                    value="IN-PROGRESS",
                )

            print(f"\n[VELES] Running {script_path}\n")

            try:
                subprocess.run(
                    [sys.executable, str(script_path)],
                    check=True,
                    cwd=str(base_dir),
                    env=env,
                )

                # Script finished successfully
                finished_at = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
                _log_veles_event(log_file, session, script, f"FINISHED: {finished_at}")

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
                print(f"[VELES: ERROR] Script failed for {session} with exit code {e.returncode}")
                _log_veles_event(log_file, session, script, f"FAIL: {failed_at} (exit code {e.returncode})")

                # If this script has a corresponding CSV status column, write FAIL
                if status_column is not None:
                    _update_script_status_for_session(
                        data_root=data_root,
                        session=session,
                        status_column=status_column,
                        value="FAIL",
                    )

                print("[VELES] Skipping to next session...")
                break

            except KeyboardInterrupt:
                print(f"\n[VELES: INTERRUPT] Execution interrupted by user. Status retained as IN-PROGRESS for {script}.")
                raise
        else:
            print(f"\n[VELES: SUCCESS] Completed all scripts for {session}")

    print(f"\n{'=' * 60}")
    print("[VELES] All sessions completed!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="VELES - Versatile Electrophysiology and Limb-motion Evaluation Suite"
    )
    parser.add_argument(
        "-po", "--process-only",
        nargs="+",
        dest="process_only",
        default=None,
        help="Optional conditions/BR indices to process (e.g. -po '7, 8, 9, 10, 11, 12' or -po 7 8 9 10). Overrides PROCESS_ONLY in VELES.py.",
    )
    parser.add_argument(
        "-m", "--monkey",
        type=str,
        dest="monkey",
        default=None,
        help="Optional monkey name (e.g. Nike, Ada, Bert). Overrides MONKEY in VELES.py.",
    )
    parser.add_argument(
        "-s", "--sessions",
        nargs="+",
        dest="sessions",
        default=None,
        help="Optional session(s) to process (e.g. -s NRR_RW012). Overrides SESSIONS_TO_RUN in VELES.py.",
    )
    parsed_args, _ = parser.parse_known_args()

    parsed_po = parse_process_only_inputs(parsed_args.process_only)

    BASE = Path(__file__).resolve().parents[0]
    SCRIPTS_FOLDER = BASE
    run_scripts(
        BASE,
        SCRIPTS_FOLDER,
        monkey=parsed_args.monkey,
        sessions=parsed_args.sessions,
        process_only=parsed_po,
    )
    
if __name__ == "__main__":
    main()