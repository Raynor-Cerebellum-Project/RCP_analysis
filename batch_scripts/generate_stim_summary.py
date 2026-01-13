import csv
import numpy as np
import pandas as pd
from pathlib import Path
import json
import RCP_analysis.python.functions.config_loading as cfg
import RCP_analysis.python.functions.utils as rcp

"""
Description:
    This script iterates through all aligned sessions (as defined in br_to_intan_shifts.csv),
    loads the stimulation stream data (stim_stream.npz), and extracts or calculates
    key stimulation parameters:
    - Stimulated Channels (from active_channels)
    - Stimulation Frequency (calculated from triggers, rounded to 130Hz or 400Hz if close)
    - Stimulation Amplitude (retrieved from metadata.csv)

Output:
    stimulation_summary.csv' containing the summary of stimulation parameters for each session.
    saved in the 'results' directory of the session
"""

def main():
    print("--- Generating Stimulation Summary CSV ---")

    # 1. Output File
    out_csv = cfg.OUT_BASE / "stimulation_summary.csv"
    
    # 2. Load Shifts (Session List)
    shifts_csv = cfg.METADATA_ROOT / "br_to_intan_shifts.csv"
    if not shifts_csv.exists():
        print(f"Error: {shifts_csv} not found.")
        return

    print(f"Loading shifts from {shifts_csv}")
    df_shifts = pd.read_csv(shifts_csv)
    
    # 3. Load Metadata for Amplitude Lookup
    meta_csv = cfg.METADATA_CSV
    print(f"Loading metadata from {meta_csv}")
    
    # Try to load metadata to get Amplitude
    meta_map_amp = {}
    meta_map_freq = {} # Sometimes freq is in metadata too
    meta_map_notes = {}
    
    if meta_csv.exists():
        try:
            df_meta = pd.read_csv(meta_csv)
            # Check for Amplitude column
            # Normalize column names slightly
            cols = {c.lower().strip(): c for c in df_meta.columns}
            
            amp_col = cols.get('amplitude') or cols.get('stim_amplitude') or cols.get('amp')
            freq_col = cols.get('frequency') or cols.get('freq') or cols.get('stim_frequency')
            
            # We map BR_File (index) to these values if possible
            if 'BR_File' in df_meta.columns:
                for idx, row in df_meta.iterrows():
                    br_idx = row['BR_File']
                    try:
                        br_idx = int(br_idx)
                    except:
                        continue
                        
                    if amp_col: meta_map_amp[br_idx] = row[amp_col]
                    if freq_col: meta_map_freq[br_idx] = row[freq_col]
                    if 'Notes' in row: meta_map_notes[br_idx] = row['Notes']
                    
        except Exception as e:
            print(f"Warning: Could not parse metadata CSV for amplitudes: {e}")
    else:
        print(f"Warning: Metadata CSV {meta_csv} not found.")

    # 4. Iterate Sessions
    results = []
    
    for i, row in df_shifts.iterrows():
        intan_session = row['session']
        br_idx_str = row['br_idx']
        
        try:
            br_idx = int(br_idx_str)
        except:
            br_idx = -1
            
        print(f"Processing: {intan_session} (BR {br_idx})")
        
        if intan_session:
            # We can extract intan index from the shift row if available, or just leave it
            intan_idx_val = row.get('intan_idx', "N/A")
        
        # Paths
        # Stim stream is in results/aux_data/NPRW/{session}_Intan_streams/stim_stream.npz
        stim_npz = cfg.NPRW_AUX_DATA / f"{intan_session}_Intan_streams" / "stim_stream.npz"
        
        active_chs = "N/A"
        frequency = "N/A"
        
        if stim_npz.exists():
            try:
                # Load stim data
                stim_data = rcp.load_stim_detection(stim_npz)
                
                # Channels
                chs = stim_data.get('active_channels')
                if chs is not None and chs.size > 0:
                    # Format as list string or range
                    chs = np.sort(chs)
                    if len(chs) > 2 and np.all(np.diff(chs) == 1):
                        active_chs = f"{chs[0]}-{chs[-1]}"
                    else:
                        active_chs = str(list(chs))
                else:
                    active_chs = "None"
                    
                # Frequency Calculation
                # triggers: (n, 2) start, end
                trigs = stim_data.get('trigger_pairs')
                if trigs is not None and trigs.shape[0] > 1:
                    starts = trigs[:, 0]
                    diffs = np.diff(starts) 
                    
                    fs = 30000.0
                    if 'fs_intan' in row:
                        fs = float(row['fs_intan'])
                    
                    median_diff_samp = np.median(diffs)
                    
                    if median_diff_samp > 0:
                        freq_val = fs / median_diff_samp
                        
                        # Rounding logic
                        # 130 Hz check
                        if abs(freq_val - 130) < 15: # +/- 15 Hz tolerance
                            frequency = "130 Hz"
                        elif abs(freq_val - 400) < 20: # +/- 20 Hz tolerance (394 -> 400)
                            frequency = "400 Hz"
                        else:
                            frequency = f"{freq_val:.1f} Hz"
                    else:
                        frequency = "Error"
                else:
                     frequency = "No Triggers"

            except Exception as e:
                print(f"  Error reading {stim_npz.name}: {e}")
                active_chs = "Error"
                frequency = "Error"
        else:
             print(f"  {stim_npz.name} not found.")
             active_chs = "File Not Found"

        # Amplitude Lookup
        amplitude = meta_map_amp.get(br_idx, "N/A")
        
        # If freq is "Error" or "N/A", maybe fallback to metadata
        if frequency in ["N/A", "Error", "No Triggers"] and br_idx in meta_map_freq:
            frequency = f"{meta_map_freq[br_idx]} (Meta)"

        # Construct Row
        res = {
            "Intan_Session": intan_session,
            "Intan_File_Index": intan_idx_val,
            "Blackrock_File": f"NRR_RW_001_{br_idx:03d}", # Approximate naming convention
            "BR_Index": br_idx,
            "Stim_Channels": active_chs,
            "Stim_Amplitudes": amplitude,
            "Stim_Frequency": frequency,
            "Notes": meta_map_notes.get(br_idx, "")
        }
        results.append(res)

    # 5. Save Output
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(out_csv, index=False)
        print(f"\nSaved summary to: {out_csv}")
        print(df_out.head())
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
