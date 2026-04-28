
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import RCP_analysis as rcp

# ---- MODE SELECTION ----
# Options: "TOUCHSCREEN" or "KINEMATIC"
MODE = "TOUCHSCREEN" 

# ---- CONFIG ----
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"
EVENTS_OUT  = OUT_BASE / "events"; EVENTS_OUT.mkdir(parents=True, exist_ok=True)

# TOUCHSCREEN CONFIG
BR_ROOT = SESSION_LOC / "Blackrock"
UA_AUX_DATA = OUT_BASE / "aux_data" / "UA"
SYNC_PIN = 14
EXPECTED_LABELS = [f"ainp{SYNC_PIN}", f"analog {SYNC_PIN}", f"ai{SYNC_PIN}", f"14", f"{128+SYNC_PIN}"]
V_THRESH_A_LOW, V_THRESH_A_HIGH = 2.0, 3.5
V_THRESH_B_LOW, V_THRESH_B_HIGH = 4.0, 6.0

# KINEMATIC CONFIG
BEHV_CKPT_ROOT = OUT_BASE / "checkpoints" / "Behavior"
CAM_IDX = 1
X_THRESH_REACH = 250.0
Y_THRESH_A_MIN = 230.0
Y_THRESH_B_MAX = 100.0
JUMP_THRESH = 10.0
REFRACTORY_SEC = 0.5
FPS = 100.0

# ---- SHARED HELPERS ----

def _save_outputs(events_list, output_classes, fps, session_name, suffix_base):
    """
    Save both NPZ events and CSV states time-series.
    """
    # 1. Save CSV Time Series
    # Columns: Time, Numeric State (0/-1/1), Char State (N/A/B)
    times = np.arange(len(output_classes)) / fps
    
    numeric_classes = np.zeros(len(output_classes), dtype=int)
    numeric_classes[output_classes == 'A'] = -1
    numeric_classes[output_classes == 'B'] = 1
    
    out_csv_df = pd.DataFrame({
        f"{session_name}": times,
        "numeric_state": numeric_classes,
        "char_state": output_classes
    })
    
    out_csv_path = EVENTS_OUT / f"{session_name}_{suffix_base}_states.csv"
    out_csv_df.to_csv(out_csv_path, index=False)
    print(f"  > Saved CSV states: {out_csv_path}")

    # 2. Save NPZ Events
    out_npz_path = EVENTS_OUT / f"{session_name}_{suffix_base}_events.npz"
    dtype = [('start_sample', 'i8'), ('end_sample', 'i8'), ('start_sec', 'f8'), ('end_sec', 'f8'), ('type', 'U1'), ('mean_voltage', 'f4')]
    arr = np.zeros(len(events_list), dtype=dtype)
    for i, e in enumerate(events_list):
        arr[i] = (e['start_sample'], e['end_sample'], e['start_sec'], e['end_sec'], e['type'], e.get('mean_voltage', 0.0))
        
    np.savez_compressed(out_npz_path, events=arr, fs=fps, session=session_name, method=MODE)
    print(f"  > Saved NPZ events: {out_npz_path}")

# ---- KINEMATIC LOGIC ----

def _find_middle_cols(df, cam_idx):
    headers = list(df.columns)
    col_x, col_y = None, None
    for i, h in enumerate(headers):
        h_str = str(h).lower()
        if "middle_x" in h_str and (f"cam-{cam_idx}" in h_str or f"cam{cam_idx}" in h_str or f"camera {cam_idx}" in h_str):
            col_x = i
        if "middle_y" in h_str and (f"cam-{cam_idx}" in h_str or f"cam{cam_idx}" in h_str or f"camera {cam_idx}" in h_str):
            col_y = i
    if col_x is None and cam_idx == 1: col_x = 27
    if col_y is None and cam_idx == 1: col_y = 28
    return col_x, col_y

def _classify_kinematic(x, y):
    if pd.isna(x) or pd.isna(y): return 'N'
    if x <= X_THRESH_REACH: return 'N'
    if y > Y_THRESH_A_MIN: return 'A'
    elif y < Y_THRESH_B_MAX: return 'B'
    return 'N'

def filter_jumps(series, thresh):
    diff = series.diff().abs()
    return (diff <= thresh) & (series.diff(-1).abs() <= thresh)

def run_kinematics():
    print("--- Running KINEMATIC Extraction ---")
    csv_files = sorted(BEHV_CKPT_ROOT.rglob("*both_cams_aligned.csv"))
    if not csv_files:
        csv_files = sorted(BEHV_CKPT_ROOT.rglob("*_aligned.csv"))
    print(f"Found {len(csv_files)} alignment CSVs.")
    
    for p in csv_files:
        print(f"\nProcessing: {p.name}")
        try:
            df = pd.read_csv(p, header=[0, 1, 2])
            flat_headers = ['_'.join([str(c) for c in col if 'Unnamed' not in str(c)]).strip() for col in df.columns]
            df.columns = flat_headers
        except:
            df = pd.read_csv(p, header=0)
            
        cx_idx, cy_idx = _find_middle_cols(df, CAM_IDX)
        if cx_idx is None or cy_idx is None:
            print(f"  [SKIP] Could not locate middle finger columns for Cam {CAM_IDX}")
            continue
            
        ser_x = pd.to_numeric(df.iloc[:, cx_idx], errors='coerce')
        ser_y = pd.to_numeric(df.iloc[:, cy_idx], errors='coerce')
        
        valid_x = filter_jumps(ser_x, JUMP_THRESH)
        valid_y = filter_jumps(ser_y, JUMP_THRESH)
        valid_mask = valid_x & valid_y
        
        classes = []
        for i in range(len(df)):
            if not valid_mask.iloc[i]:
                classes.append('N')
            else:
                classes.append(_classify_kinematic(ser_x.iloc[i], ser_y.iloc[i]))
        classes = np.array(classes)
        
        # Block processing
        output_classes = np.array(['N'] * len(classes))
        num_classes = np.zeros(len(classes), dtype=int)
        num_classes[classes == 'A'] = 1
        num_classes[classes == 'B'] = 2
        
        onsets = []
        last_event_end = -999999
        refractory_samples = int(REFRACTORY_SEC * FPS)
        
        in_event = False
        event_start = 0
        event_type = 0
        
        for i, val in enumerate(num_classes):
            if val != 0:
                if not in_event:
                    if (i - last_event_end) >= refractory_samples:
                        in_event = True
                        event_start = i
                        event_type = val
                elif val != event_type:
                    # Switch type
                    onsets.append((event_start, i, event_type))
                    last_event_end = i
                    in_event = True
                    event_start = i
                    event_type = val
            else:
                if in_event:
                    onsets.append((event_start, i, event_type))
                    last_event_end = i
                    in_event = False
        if in_event:
             onsets.append((event_start, len(num_classes), event_type))
             
        events_list = []
        for s, e, t in onsets:
            label = 'A' if t == 1 else 'B'
            output_classes[s:e] = label
            events_list.append({
                "start_sample": s, "end_sample": e,
                "start_sec": s / FPS, "end_sec": e / FPS,
                "type": label
            })
            
        session_name = p.stem.split("_both")[0].split("_Cam")[0]
        _save_outputs(events_list, output_classes, FPS, session_name, "kinematic")

# ---- TOUCHSCREEN LOGIC ----

def _find_touchscreen_channel(sigs, channel_ids):
    cids_str = [str(c).lower() for c in channel_ids]
    for cand in EXPECTED_LABELS:
        if cand.lower() in cids_str:
            return cids_str.index(cand.lower()), channel_ids[cids_str.index(cand.lower())]
            
    # Voltage fallback
    possible = []
    for i, sig in enumerate(sigs):
        mx = np.max(sig)
        if 1.5 < mx < 7.0:
            possible.append(i)
    if len(possible) == 1:
        return possible[0], channel_ids[possible[0]]
    return None, None

def run_touchscreen():
    print("--- Running TOUCHSCREEN Extraction ---")
    br_sessions = rcp.list_br_sessions(BR_ROOT)
    
    for sess in br_sessions:
        print(f"\nProcessing: {sess.name}")
        aux_npz = UA_AUX_DATA / f"{sess.name}__BR_aux_data.npz"
        sig = None; fs = 1000.0; ch_id = None
        
        if aux_npz.exists():
            try:
                with np.load(aux_npz) as z:
                    if "ns2_aux_traces" in z:
                        idx, cid = _find_touchscreen_channel(z["ns2_aux_traces"], z["ns2_channel_ids"])
                        if idx is not None:
                            sig = z["ns2_aux_traces"][idx]
                            fs = float(z["ns2_fs_hz"]) if "ns2_fs_hz" in z else 1000.0
                            ch_id = cid
            except Exception as e:
                print(f"  [ERR] Aux load failed: {e}")
                
        if sig is None:
            # Raw fallback
            try:
                rec = se.read_blackrock(str(sess), stream_name="nsx2", all_annotations=True)
                fs = rec.get_sampling_frequency()
                traces = rec.get_traces(return_scaled=True).T
                idx, cid = _find_touchscreen_channel(traces, rec.get_channel_ids())
                if idx is not None:
                    sig = traces[idx]
                    ch_id = cid
            except:
                pass
                
        if sig is None:
            print("  [SKIP] No signal found.")
            continue
            
        # Processing
        states = np.zeros_like(sig, dtype=int)
        states[(sig >= V_THRESH_A_LOW) & (sig <= V_THRESH_A_HIGH)] = 1
        states[(sig >= V_THRESH_B_LOW) & (sig <= V_THRESH_B_HIGH)] = 2
        
        # Debounce/Event extraction
        # We can produce a sample-wise state array ('N', 'A', 'B') directly from states?
        # But we need to clean it up like kinematics? The source script used simple transitions.
        # Let's use the states array to generate the CSV, and extract events relative to it.
        
        # Original logic was simple transition. Let's stick to it but format outputs uniformly.
        events_list = []
        d = np.diff(states, prepend=0)
        changes = np.where(d != 0)[0]
        
        current_start = None
        current_type = 0
        
        # Output char array
        char_states = np.array(['N'] * len(states), dtype='U1')
        
        if changes.size > 0:
            for idx in changes:
                val = states[idx]
                if current_start is not None:
                    dur = (idx - current_start) / fs
                    if dur > 0.05:
                        label = 'A' if current_type == 1 else 'B'
                        events_list.append({
                           "start_sample": current_start, "end_sample": idx,
                           "start_sec": current_start / fs, "end_sec": idx / fs,
                           "type": label, "mean_voltage": np.mean(sig[current_start:idx])
                        })
                        char_states[current_start:idx] = label
                    current_start = None
                    current_type = 0
                if val in (1, 2):
                    current_start = idx
                    current_type = val
            # Tail
            if current_start is not None:
                idx = len(states)
                dur = (idx - current_start) / fs
                if dur > 0.05:
                     label = 'A' if current_type == 1 else 'B'
                     events_list.append({
                           "start_sample": current_start, "end_sample": idx,
                           "start_sec": current_start / fs, "end_sec": idx / fs,
                           "type": label, "mean_voltage": np.mean(sig[current_start:idx])
                     })
                     char_states[current_start:idx] = label

        _save_outputs(events_list, char_states, fs, sess.name, "touchscreen")

def main():
    if MODE == "KINEMATIC":
        run_kinematics()
    elif MODE == "TOUCHSCREEN":
        run_touchscreen()
    else:
        print(f"Unknown MODE: {MODE}")

if __name__ == "__main__":
    main()
