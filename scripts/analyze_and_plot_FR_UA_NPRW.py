"""
Run scripts in order. Comment out ones you don't want to run.
"""

import subprocess
from pathlib import Path

BATCH_SCRIPTS = [
    # run curation_scripts/OCR_frame_mapping_BT_edit.py on raw videos first -> copy into Video/OCR separate from Video/DLC
    # "align_dlc_two_cams_to_br.py",
    # "align_VOG_to_br.py",
    "NPRW_Intan_analysis_mf.py",
    "compute_br_to_intan_shifts.py",
    "UA_BR_analysis_mf.py",
    "make_aligned_npz_and_mat.py",
    "extract_peri_stim.py",
    # "generate_stim_summary.py",
    # "plot_complete_shaded_BT.py",
    # "quantify_stim_kinematics.py",
]

def run_scripts(base_dir: Path):
    for batch_script in BATCH_SCRIPTS:
        script_path = base_dir / batch_script
        print(f"\n[INFO] Running {script_path} ...\n")
        try:
            subprocess.run(
                ["python", str(script_path)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Script {batch_script} failed with exit code {e.returncode}")
            break

def main():
    BASE = Path(__file__).resolve().parents[1]
    BATCH_SCRIPTS_FOLDER = BASE / "batch_scripts"
    run_scripts(BATCH_SCRIPTS_FOLDER)
    
if __name__ == "__main__":
    main()
