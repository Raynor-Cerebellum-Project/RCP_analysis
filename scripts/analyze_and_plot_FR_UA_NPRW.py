"""
Run all batch scripts in order.
"""

import subprocess
from pathlib import Path

# List of scripts you want to run
BATCH_SCRIPTS = [
    "align_dlc_one_cam_to_br.py"
    "NPRW_Intan_analysis_threshold.py",
    "compute_br_to_intan_shifts.py",
    "UA_BR_analysis_threshold.py",
    "make_aligned_npz_from_shifts_with_behv.py",
    "NPRW_BEHV_UA_FR_plotting.py",
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
