"""
Run scripts in order across multiple sessions.
Automatically updates params.yaml for each session (preserves comments).
"""

import subprocess
from pathlib import Path
import re

BATCH_SCRIPTS = [
    # run curation_scripts/OCR_frame_mapping_BT_edit.py on raw videos first -> copy into Video/OCR separate from Video/DLC
    # "align_dlc_two_cams_to_br.py",
    # "align_VOG_to_br.py",
    # "NPRW_Intan_analysis_mf.py",
    # "compute_br_to_intan_shifts.py",
    # "UA_BR_analysis_mf.py", 
    # "make_aligned_npz_and_mat.py",
    "extract_peri_stim.py",
    "inspect_kinematics_trajectories.py",

    ### RUN ^ inspect_kinematics_trajectories.py to check for remaining bad traces -> add bad traces to /config/manual_trial_remove.csv
    ### RERUN extract_peri_stim.py

    # "generate_stim_summary.py",
    # "plot_complete_shaded_BT.py",
    # "plot_peri_stim_raster.py",
    # "analyze_lfp_bands.py",
    # "quantify_stim_kinematics.py",
    # "quantify_smoothed_position.py",
]

# Define the sessions you want to process
SESSIONS_TO_RUN = [
    {"location": "Nike/20260311_NRR_RW022", "session": "NRR_RW022"},
    {"location": "Nike/20260116_NRR_RW012", "session": "NRR_RW012"},
    {"location": "Nike/20251203_NRR_RW011", "session": "NRR_RW011"},
    {"location": "Nike/20260416_NRR_RW032", "session": "NRR_RW032"},
    {"location": "Nike/20260324_NRR_RW026", "session": "NRR_RW026"},
    # {"location": "Nike/20260129_NRR_RW013", "session": "NRR_RW013"},
    # {"location": "Nike/20260130_NRR_RW014", "session": "NRR_RW014"},
    # {"location": "Nike/20260202_NRR_RW015", "session": "NRR_RW015"},
    # {"location": "Nike/20260206_NRR_RW016", "session": "NRR_RW016"},
    # {"location": "Nike/20260218_NRR_RW017", "session": "NRR_RW017"},
    # {"location": "Nike/20260220_NRR_RW018", "session": "NRR_RW018"},
    # {"location": "Nike/20260226_NRR_RW019", "session": "NRR_RW019"},
    # {"location": "Nike/20260409_NRR_RW029", "session": "NRR_RW029"},
]

def update_params_yaml(params_path: Path, location: str, session: str):
    """
    Update params.yaml by commenting out current active location/session
    and uncommenting the target session (preserves all comments).
    """
    with open(params_path, 'r') as f:
        content = f.read()
    
    # Step 1: Comment out any currently active (uncommented) location line
    # Matches lines like: '  location: "Nike/..."' (not starting with #)
    content = re.sub(
        r'^(\s*)location: "(.+)"',
        r'\1# location: "\2"',
        content,
        flags=re.MULTILINE
    )
    
    # Step 2: Comment out any currently active (uncommented) session line
    content = re.sub(
        r'^(\s*)session: "(.+)"',
        r'\1# session: "\2"',
        content,
        flags=re.MULTILINE
    )
    
    # Step 3: Uncomment the target location line
    content = re.sub(
        rf'^(\s*)# location: "{re.escape(location)}"',
        rf'\1location: "{location}"',
        content,
        flags=re.MULTILINE
    )
    
    # Step 4: Uncomment the target session line
    content = re.sub(
        rf'^(\s*)# session: "{re.escape(session)}"',
        rf'\1session: "{session}"',
        content,
        flags=re.MULTILINE
    )
    
    with open(params_path, 'w') as f:
        f.write(content)
    
    print(f"[INFO] Updated params.yaml: location={location}, session={session}")

def run_scripts(base_dir: Path, batch_scripts_folder: Path):
    params_path = base_dir / "config/params.yaml"
    
    for session_info in SESSIONS_TO_RUN:
        location = session_info["location"]
        session = session_info["session"]
        
        print(f"\n{'='*60}")
        print(f"[INFO] Processing session: {session}")
        print(f"{'='*60}")
        
        # Update params.yaml for this session
        update_params_yaml(params_path, location, session)
        
        # Run all scripts for this session
        for batch_script in BATCH_SCRIPTS:
            script_path = batch_scripts_folder / batch_script
            print(f"\n[INFO] Running {script_path} for {session}...\n")
            try:
                subprocess.run(
                    ["python", str(script_path)],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Script {batch_script} failed for {session} with exit code {e.returncode}")
                print(f"[INFO] Skipping to next session...")
                break
        else:
            print(f"\n[SUCCESS] Completed all scripts for {session}")
    
    print(f"\n{'='*60}")
    print("[INFO] All sessions completed!")
    print(f"{'='*60}")

def main():
    BASE = Path(__file__).resolve().parents[1]
    BATCH_SCRIPTS_FOLDER = BASE / "batch_scripts"
    run_scripts(BASE, BATCH_SCRIPTS_FOLDER)
    
if __name__ == "__main__":
    main()