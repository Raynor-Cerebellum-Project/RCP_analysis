import sys
import numpy as np
from pathlib import Path
sys.path.append(r'e:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis')
import RCP_analysis as rcp

def debug_stim(condition=10):
    repo_root = Path(r'e:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis')
    METADATA_CSV = Path(r'Y:\Current Project Databases - NHP\2025 Cerebellum prosthesis\Nike\20260116_NRR_RW012\Metadata\NRR_RW012_metadata.csv')
    NPRW_AUX_DATA = Path(r'Y:\Current Project Databases - NHP\2025 Cerebellum prosthesis\Nike\20260116_NRR_RW012\results\aux_data\NPRW')
    
    stim_npz_path, session = rcp.stim_npz_path_from_br_idx(condition, METADATA_CSV, NPRW_AUX_DATA)
    print(f"Loading {stim_npz_path} (session={session})")
    
    if stim_npz_path and stim_npz_path.exists():
        z = np.load(stim_npz_path, allow_pickle=True)
        block_bounds = z.get('block_bounds_samples', np.empty((0, 2), dtype=np.int64))
        pulse_sizes = z.get('pulse_sizes', np.array([]))
        trigger_pairs = z.get('trigger_pairs', np.empty((0, 2), dtype=np.int64))
        fs_nprw = 30000.0
        
        starts = trigger_pairs[:, 0]
        ends = trigger_pairs[:, 1]
        gaps = np.diff(starts)
        
        print(f'=== Condition {condition} ===')
        print(f'Number of trigger pairs/pulses: {len(starts)}')
        print(f'Number of blocks found: {len(block_bounds)}')
        if len(gaps) > 0:
            print(f'Median gap (ms): {np.median(gaps) / fs_nprw * 1000.0:.2f}')
        if len(block_bounds) > 0:
            med_pulse = np.median(pulse_sizes)
            print(f'Median pulse size (samples): {med_pulse:.1f}')
            print(f'Pulse size ref threshold in samples (50 * med_pulse): {50 * med_pulse:.1f}')
            bad_gaps = gaps[gaps > 50 * med_pulse]
            print(f'Number of gaps > threshold (which cause block splits): {len(bad_gaps)}')
            print(f'Bad gaps in ms: {bad_gaps / fs_nprw * 1000.0}')
    else:
        print(f'stim_npz_path not found: {stim_npz_path}')

if __name__ == "__main__":
    debug_stim(10)
