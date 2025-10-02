"""
Run all batch scripts in order.
"""

import subprocess
from pathlib import Path

# List of scripts you want to run
SCRIPTS = [
    "NPRW_Intan_analysis_threshold.py",
    "compute_br_to_intan_shifts.py",
    "make_aligned_npz_from_shifts.py",
    "UA_BR_analysis_threshold.py",
    "NPRW_UA_FR_plotting.py",
]

def run_scripts(base_dir: Path):
    for script in SCRIPTS:
        script_path = base_dir / script
        print(f"\n[INFO] Running {script_path} ...\n")
        try:
            subprocess.run(
                ["python", str(script_path)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Script {script} failed with exit code {e.returncode}")
            break

if __name__ == "__main__":
    # assumes this script is saved in the same folder as your batch scripts
    base = Path(__file__).resolve().parent
    run_scripts(base)
