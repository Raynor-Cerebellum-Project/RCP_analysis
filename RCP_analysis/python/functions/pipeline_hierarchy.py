"""
Pipeline hierarchy and dependency validation for electrophysiology and behavioral analysis scripts.
Tracks prerequisite relationships between preprocessing and analysis stages, validates
timestamp staleness across data_status_reaching.csv, and provides interactive confirmation
for pipeline discrepancies.
"""

from pathlib import Path
import csv
from datetime import datetime
from typing import Any

# Mapping from script file name (or path basename) to column name in data_status_reaching.csv
SCRIPT_STATUS_COLUMNS: dict[str, str] = {
    # Preprocessing scripts
    "OCR_frame_correction.py": "OCR",
    "align_dlc_two_cams_to_br.py": "DLC",
    "align_VOG_to_br.py": "VOG",
    "NPRW_Intan_analysis_mf.py": "NPRW_Intan_analysis",
    "compute_br_to_intan_shifts.py": "compute_shifts",
    "UA_BR_analysis_mf.py": "UA_BR_analysis",
    "UA_BR_analysis_ssmf.py": "UA_BR_analysis",
    "make_aligned_npz_and_mat.py": "make_aligned",
    "extract_peri_stim.py": "extract_peri",
    "inspect_kinematics_trajectories.py": "manual_inspection",
    "analyze_lfp_bands.py": "lfp_bands",
    "plot_lfp_cleaner.py": "plot_lfp",
    "combine_UA_gifs.py": "combine_UA_gifs",

    # Analysis scripts
    "plot_plateau_analysis.py": "plateau",
    "RSA_calculation.py": "rsa",
    "plot_firing_rates.py": "plot_FRs",
    "plot_peri_stim_raster.py": "plot_raster",
    "plot_stim_group_responses.py": "plot_group_responses",
    "plot_stim_response_overlays.py": "plot_PSTH_overlays",
    "plot_peak_csv_summaries.py": "plot_peak_csv_summaries",
    "plot_bin_counts_per_target.py": "plot_bin_counts",
    "plot_complete_shaded_BT.py": "plot_complete_shaded",
}

# Direct upstream dependencies for each status column
COLUMN_DEPENDENCIES: dict[str, list[str]] = {
    # Preprocessing DAG
    "OCR": [],
    "DLC": ["OCR"],
    "VOG": ["OCR"],
    "NPRW_Intan_analysis": [],
    "compute_shifts": ["NPRW_Intan_analysis", "DLC"],
    "UA_BR_analysis": ["compute_shifts"],
    "make_aligned": ["UA_BR_analysis"],
    "extract_peri": ["make_aligned"],

    # LFP scripts depend only on OCR and DLC
    "lfp_bands": ["OCR", "DLC"],
    "plot_lfp": ["OCR", "DLC"],

    # UA gifs depend on UA analysis
    "combine_UA_gifs": ["UA_BR_analysis"],

    # Downstream analysis & inspection scripts all depend on extract_peri
    "manual_inspection": ["extract_peri"],
    "plateau": ["extract_peri"],
    "rsa": ["extract_peri"],
    "plot_FRs": ["extract_peri"],
    "plot_raster": ["extract_peri"],
    "plot_group_responses": ["extract_peri"],
    "plot_PSTH_overlays": ["extract_peri"],
    "plot_peak_csv_summaries": ["extract_peri"],
    "plot_bin_counts": ["extract_peri"],
    "plot_complete_shaded": ["extract_peri"],
}


def get_script_column(script_name: str) -> str | None:
    """Return the CSV status column for a given script path or script name."""
    name = Path(script_name).name
    return SCRIPT_STATUS_COLUMNS.get(name)


def parse_status_timestamp(val: str | None) -> datetime | None:
    """
    Parse a timestamp string from data_status_reaching.csv into a datetime object.
    Returns None if the value is empty, boolean, or a non-date status string (e.g. 'done', 'FAIL', 'IN-PROGRESS').
    """
    if not val:
        return None
    cleaned = str(val).strip()
    if not cleaned or cleaned.upper() in (
        "DONE", "FAIL", "IN-PROGRESS", "ICE BOXED", "PENDING", "NO", "YES", "TRUE", "FALSE"
    ):
        return None

    # Supported timestamp formats across legacy and current runs
    date_formats = [
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%m-%d-%Y %H:%M:%S",
        "%m/%d/%Y",
        "%Y-%m-%d",
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass

    return None


def get_all_ancestor_columns(column: str) -> list[str]:
    """
    Return all ancestor columns for a given column in topological order (upstream first).
    """
    visited: set[str] = set()
    order: list[str] = []

    def dfs(col: str):
        for dep in COLUMN_DEPENDENCIES.get(col, []):
            if dep not in visited:
                visited.add(dep)
                dfs(dep)
                order.append(dep)

    dfs(column)
    return order


def load_session_status_row(data_root: str | Path, session: str) -> dict[str, str]:
    """
    Load the status dictionary for a specific session from data_status_reaching.csv.
    """
    status_csv = Path(data_root) / "data_status_reaching.csv"
    if not status_csv.exists():
        raise FileNotFoundError(f"data_status_reaching.csv not found at: {status_csv}")

    target_session = str(session).strip()
    with status_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("Session", "")).strip() == target_session:
                return {k: str(v).strip() for k, v in row.items()}

    raise ValueError(f"Session '{target_session}' not found in {status_csv}")


def validate_script_dependencies(
    script_name: str,
    session: str,
    data_root: str | Path,
    planned_batch_scripts: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate whether all upstream prerequisites for `script_name` have completed and are not stale.

    Parameters:
        script_name: Filename or path of the script to be run.
        session: The session ID (e.g. 'NRR_RW012').
        data_root: Root path to the monkey data folder containing data_status_reaching.csv.
        planned_batch_scripts: Optional list of scripts scheduled in the current batch.
            Prerequisites scheduled *earlier* in this batch are assumed to be refreshed.

    Returns:
        (is_valid, list_of_warning_messages)
    """
    target_col = get_script_column(script_name)
    if not target_col:
        # Script is not tracked in the hierarchy
        return True, []

    ancestors = get_all_ancestor_columns(target_col)
    if not ancestors:
        return True, []

    # Map scripts planned to run earlier in the current batch to their columns
    planned_earlier_cols: set[str] = set()
    if planned_batch_scripts:
        script_base = Path(script_name).name
        for s in planned_batch_scripts:
            s_base = Path(s).name
            if s_base == script_base:
                break
            col = SCRIPT_STATUS_COLUMNS.get(s_base)
            if col:
                planned_earlier_cols.add(col)

    row = load_session_status_row(data_root, session)
    warnings: list[str] = []

    # 1. Check completion status of all ancestors
    for anc in ancestors:
        if anc in planned_earlier_cols:
            continue

        raw_val = row.get(anc, "").strip()
        val_upper = raw_val.upper()

        if not raw_val:
            warnings.append(
                f"Prerequisite step '{anc}' has not been run for session '{session}'."
            )
        elif val_upper == "FAIL":
            warnings.append(
                f"Prerequisite step '{anc}' previously FAILED for session '{session}'."
            )
        elif val_upper == "IN-PROGRESS":
            warnings.append(
                f"Prerequisite step '{anc}' is marked IN-PROGRESS (interrupted or currently running) for session '{session}'."
            )

    # 2. Check timestamps along all dependency edges in the ancestor DAG
    # For every edge (u -> v), if u was run AFTER v, then v was not re-run after u updated.
    all_nodes_in_chain = ancestors + [target_col]
    for v in all_nodes_in_chain:
        if v in planned_earlier_cols:
            continue

        v_raw = row.get(v, "").strip()
        v_dt = parse_status_timestamp(v_raw)

        # Check all direct upstream parents of v
        for u in COLUMN_DEPENDENCIES.get(v, []):
            u_raw = row.get(u, "").strip()
            u_dt = parse_status_timestamp(u_raw)

            if u_dt and v_dt:
                # If upstream was executed > 2 seconds after downstream, downstream is stale
                if (u_dt - v_dt).total_seconds() > 2.0:
                    u_str = u_dt.strftime("%m/%d/%Y %H:%M:%S")
                    v_str = v_dt.strftime("%m/%d/%Y %H:%M:%S")
                    warnings.append(
                        f"Upstream step '{u}' ({u_str}) is NEWER than downstream prerequisite '{v}' ({v_str}). "
                        f"Step '{v}' was not re-run after '{u}' was updated."
                    )

    is_valid = len(warnings) == 0
    return is_valid, warnings


def prompt_dependency_warning(
    warnings: list[str],
    session: str,
    script_name: str,
) -> bool:
    """
    Display a formatted dependency warning in the console and prompt user for confirmation.
    Returns True if user typed 'RUN' (case-insensitive), False otherwise.
    """
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"[PIPELINE DEPENDENCY WARNING]")
    print(f"Session: {session}")
    print(f"Script:  {Path(script_name).name}")
    print("The following discrepancy / staleness issues were detected in data_status_reaching.csv:")
    for w in warnings:
        print(f"  * {w}")
    print("-" * 80)
    print("Downstream outputs may be out-of-sync or missing prerequisites.")
    
    try:
        user_input = input("Type 'RUN' to proceed anyway, or press Enter to skip/abort: ")
        confirmed = user_input.strip().upper() == "RUN"
    except (EOFError, KeyboardInterrupt):
        print("\nAborting on user cancel.")
        return False

    if confirmed:
        print(f"[PIPELINE] Proceeding with {Path(script_name).name} as requested by user.\n{sep}\n")
    else:
        print(f"[PIPELINE] Skipped {Path(script_name).name}.\n{sep}\n")

    return confirmed


def check_and_confirm_dependencies(
    script_name: str,
    session: str,
    data_root: str | Path,
    planned_batch_scripts: list[str] | None = None,
) -> bool:
    """
    Convenience function to validate dependencies and prompt user if discrepancies are found.
    Returns True if valid or user entered 'RUN', False if user declined.
    """
    is_valid, warnings = validate_script_dependencies(
        script_name=script_name,
        session=session,
        data_root=data_root,
        planned_batch_scripts=planned_batch_scripts,
    )
    if is_valid:
        return True

    return prompt_dependency_warning(
        warnings=warnings,
        session=session,
        script_name=script_name,
    )
