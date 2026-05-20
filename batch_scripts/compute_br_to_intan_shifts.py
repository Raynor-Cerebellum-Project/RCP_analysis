import numpy as np
import csv
import spikeinterface.extractors as se
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# ---------- Config ----------
# Base paths from config_loading
TEMPLATE = REPO_ROOT / "config" / "br_intan_align_template.mat"

# Probe params, sync channels
NPRW_CFG = PARAMS.probes.get("NPRW")
NPRW_CAMERA_SYNC_CH = int(NPRW_CFG.get("triangle_sync_ch", 1))
UA_CFG = PARAMS.probes.get("UA")
CAMERA_SYNC_CH = int(UA_CFG.get("camera_sync_ch", 134))
TRIANGLE_SYNC_CH = int(UA_CFG.get("triangle_sync_ch", 138))
PROCESS_ONLY = PARAMS.preprocessing.get("process_only")

LOC_REFINE_N = 3000        # increased to 100ms to catch larger shifts
        
def _xcorr_normalized(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Handle empty inputs early
    if x.size == 0 or y.size == 0:
        return np.array([], dtype=float), np.array([], dtype=int)

    # Z-score each signal
    x = x - x.mean()
    x /= x.std() + 1e-12

    y = y - y.mean()
    y /= y.std() + 1e-12

    # Full cross-correlation
    c = np.correlate(x, y, mode="full")
    if c.size:
        c /= (min(len(x), len(y)) + 1e-12)  # normalize by length so max is ~1

    # Lags: from -(len(y)-1) to (len(x)-1)
    lags = np.arange(-(y.size - 1), x.size)
    return c, lags

# Template matching on Intan
def load_template(template_mat_path: Path) -> np.ndarray:
    from scipy.io import loadmat
    m = loadmat(str(template_mat_path))
    if "template" not in m:
        raise KeyError("Template not in br_intan_align_template.mat")
    return np.asarray(m["template"], float).squeeze()

def find_locs_via_template(adc_lock: np.ndarray, template: np.ndarray, fs: float, peak=0.95) -> np.ndarray:
    corr, lags = _xcorr_normalized(adc_lock[:300000], template) # Hard coded to first 300k samples to speed up
    # Scale corr to max 1 for peak finding if it's very large
    if corr.size and corr.max() > 0:
        corr = corr / corr.max()
    try:
        from scipy.signal import find_peaks
        idx, _ = find_peaks(corr, height=peak)
        locs = lags[idx].astype(int)
    except Exception:
        mid = np.arange(1, corr.size - 1)
        mask = (corr[mid-1] < corr[mid]) & (corr[mid] >= corr[mid+1]) & (corr[mid] >= peak)
        locs = lags[mid[mask]].astype(int)
    return locs

def refine_shift_with_loc_and_br0(
    rec_br,
    br_ch,
    fs_br: float,
    adc_triangle_intan: np.ndarray,
    fs_intan: float,
    loc: int,                 # Intan offset where BR starts (from template on Intan)
    search_n_br: int,         # ± extra BR-sample shifts to try
    br_window_sec: float = 20.0,  # long window to capture triangle reversals
    normalize_both: bool = True  # changed default to True
) -> tuple[int, int, float, dict]:
    """
    BR origin is 0. Align BR[0:W) against Intan starting at time loc/fs_intan,
    then refine with additional ±n BR-sample shifts. Return:
      shift_sample (Intan), delta_intan_samples, dt_ms, debug dict.
    """
    # ---- BR window: start at 0 ----
    n_br_total = rec_br.get_num_frames()
    br_end = min(n_br_total, int(round(br_window_sec * fs_br)))
    if br_end < 8:
        br_end = min(n_br_total, 8)
    y_br = rec_br.get_traces(start_frame=0, end_frame=br_end, channel_ids=[br_ch], return_in_uV=False)
    y_br = y_br.squeeze().astype(float, copy=False)
    N_b = int(y_br.size)

    # ---- Get Intan Window (with padding for search) ----
    # To search shifts ±search_n_br (in BR time), we need extra samples in Intan time
    r = float(fs_intan) / float(fs_br)
    search_n_intan = int(round(search_n_br * r))
    
    # We want Intan from loc - search_n_intan to loc + N_b*r + search_n_intan
    i_start = loc - search_n_intan
    i_end = int(loc + round(N_b * r) + search_n_intan)
    
    # Bound checks
    N_i_total = adc_triangle_intan.size
    i_start_safe = max(0, i_start)
    i_end_safe = min(N_i_total, i_end)
    
    if i_end_safe <= i_start_safe or N_b < 8:
        return loc, 0, float("nan"), {"br_start": 0, "br_end": br_end, "best_n_br": 0, "best_rms": float("nan")}
        
    x_i_window = adc_triangle_intan[i_start_safe:i_end_safe].astype(float, copy=False)
    
    # Compute normalized cross correlation directly to find the best shift
    # This handles scaling and DC offset naturally, and we can take abs() to handle inverted polarities
    corr, lags = _xcorr_normalized(y_br, x_i_window)
    
    if corr.size == 0:
        return loc, 0, float("nan"), {"br_start": 0, "br_end": br_end, "best_n_br": 0, "best_rms": float("nan")}
        
    # We want to find the lag that maximizes the absolute correlation
    # lag = index_in_x_i_window - index_in_y_br
    # So index_in_x_i_window = lag + index_in_y_br
    # For BR origin (index_in_y_br = 0), this corresponds to index_in_x_i_window = lag
    abs_corr = np.abs(corr)
    
    # Restrict search to the allowed search_n_br window
    # lag corresponding to the center (no shift relative to 'loc') is:
    expected_lag = loc - i_start_safe
    
    valid_lags_mask = (lags >= expected_lag - search_n_intan) & (lags <= expected_lag + search_n_intan)
    
    if not np.any(valid_lags_mask):
        return loc, 0, float("nan"), {"br_start": 0, "br_end": br_end, "best_n_br": 0, "best_rms": float("nan")}
        
    best_idx = np.argmax(abs_corr[valid_lags_mask])
    best_lag = lags[valid_lags_mask][best_idx]
    max_corr = abs_corr[valid_lags_mask][best_idx]
    
    # The actual Intan index for BR[0] is i_start_safe + best_lag
    shift_sample_intan = i_start_safe + best_lag
    delta_i = shift_sample_intan - loc
    best_n = delta_i / r
    
    dt_ms = 1000.0 * delta_i / float(fs_intan)

    dbg = {
        "br_start": 0,
        "br_end": br_end,
        "best_n_br": int(best_n),
        "best_rms": float(1.0 - max_corr), # pseudo RMS
        "Li": int(max(8, round(N_b * r))),
    }
    return int(shift_sample_intan), int(delta_i), float(dt_ms), dbg

def main():
    # Load Intan sessions (from processed Intan rate files)
    rate_files = sorted(NPRW_CKPT_ROOT.rglob("rates__*.npz"))
    sessions = sorted({p.stem[len("rates__"):].split("__bin", 1)[0] for p in rate_files })
    if not sessions:
        raise SystemExit(f"[error] No NPRW rate files under {NPRW_CKPT_ROOT}")

    _, intan_idx_to_sess = rcp.build_session_index_map(sessions)

    # Load metadata
    if not METADATA_CSV.exists():
        raise SystemExit(f"[error] mapping CSV not found: {METADATA_CSV}")
    intan2br = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "BR_File")
    notes_col = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "Notes")
    movement_trigger_col = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "Movement_Trigger")

    # Filter to PROCESS_ONLY BR indices if specified
    if PROCESS_ONLY:
        br_set = set(int(x) for x in PROCESS_ONLY)
        intan2br = {k: v for k, v in intan2br.items() if int(v) in br_set}
        print(f"[map] PROCESS_ONLY: {len(intan2br)} sessions selected.")

    print(f"[map] Loaded Intan→BR rows: {len(intan2br)}")
    
    # Load template
    template = load_template(TEMPLATE)

    # iterate BR Intan pairs, compute shift, and write shifts to CSV
    summary_rows = []
    for intan_idx, br_str in sorted(intan2br.items()):
        intan_filename = intan_idx_to_sess.get(intan_idx)
        
        br_idx = int(br_str)
        br_ns5_file_name = f"{PARAMS.session}_{br_idx:03d}"
        
        # Flags
        note_val   = str(notes_col.get(intan_idx, ""))
        note_norm = note_val.strip().lower()

        is_control = (note_norm == "baseline")
        is_at_rest  = (note_norm == "rest")

        movement_trigger = str(movement_trigger_col.get(intan_idx, "")).strip().lower()
        is_continuous_stim = (movement_trigger == "sig gen" and note_norm != "rest")
        skip_session = (movement_trigger == "sig gen" and note_norm == "rest")

        if skip_session:
            print(f"[skip] Intan {intan_idx:03d} / BR {br_idx:03d}: sig gen + Rest, skipping.")
            continue

        if intan_filename is None:
            print(f"[warn] No session name for Intan_File={intan_idx} (skipping)")
            continue

        adc_npz = NPRW_AUX_DATA / f"{intan_filename}_Intan_streams" / "aux_streams.npz"
        if not adc_npz.exists():
            print(f"[warn] Intan AUX stream missing: {adc_npz} (skip Intan {intan_idx} → BR {br_idx})")
            continue

        # Load Intan ADC
        try:
            intan_triangle_signal, intan_BR_start_signal, fs_intan = rcp.load_intan_aux(adc_npz)
        except Exception as e:
            print(f"[error] load_intan_aux failed for {adc_npz}: {e}")
            continue

        # Template-match on BR_sync_signal to get block starts
        br_sync_locs = find_locs_via_template(intan_BR_start_signal, template, fs_intan, peak=0.95)
        print(f"[locs] Intan {intan_idx:03d} peaks ≥0.95: {br_sync_locs.size} | first 10: {br_sync_locs[:10].tolist() if br_sync_locs.size else []}")
        if br_sync_locs.size == 0:
            print(f"[warn] No template peaks found for {intan_filename}; skipping.")
            continue
        
        # ---- load BR (lazy) ----
        fs_br = float("nan")
        rec_br = None
        dur_intan_sec = len(intan_triangle_signal) / fs_intan
        dur_br_sec = float("nan")
        
        br_ns5_file_path = BR_ROOT / f"{br_ns5_file_name}.ns5"
        use_br = False
        if br_ns5_file_path.exists():
            try:
                rec_br = se.read_blackrock(br_ns5_file_path)  # does not load all data
                fs_br = rec_br.get_sampling_frequency()
                # map TRIANGLE_SYNC_CH to a valid channel id
                if str(TRIANGLE_SYNC_CH) not in rec_br.get_channel_ids():
                    print(f"[note] TRIANGLE_SYNC_CH={str(TRIANGLE_SYNC_CH)} not found in BR channels; skipping BR refinement.")
                else:
                    use_br = True # ENABLED triangle alignment
                try:
                    dur_br_sec = rec_br.get_num_frames() / fs_br
                except Exception:
                    pass
            except Exception as e:
                print(f"[note] BR open failed for {br_ns5_file_path}: {e}")
                rec_br, fs_br, use_br = None, float("nan"), False

        # refine each block using triangle sync channel TODO FIX THIS
        locs_rows = []
        locs_rows_dt = []
        
        for block, loc in enumerate(br_sync_locs): # loc is a candidate for Intan sample where BR recording starts before refinement
            if use_br:
                search_n_br = int(round(LOC_REFINE_N * fs_br / fs_intan)) # Width of searching through BR samples
                shift_sample_intan, delta_intan_samples, dt_ms, debug_dict = refine_shift_with_loc_and_br0(
                    rec_br=rec_br, br_ch=str(TRIANGLE_SYNC_CH), fs_br=fs_br,
                    adc_triangle_intan=intan_triangle_signal, fs_intan=fs_intan,
                    loc=loc, search_n_br=search_n_br, br_window_sec=20.0, normalize_both=True
                )
                # record ref_i as triangle_loc_sample and d_i as delta_samples

                locs_rows.append((block, loc, int(shift_sample_intan), int(delta_intan_samples)))
                locs_rows_dt.append(dt_ms)

                if block < 3:
                    print(f"[block {block}] BR[{debug_dict['br_start']}:{debug_dict['br_end']}) len={debug_dict['br_end']-debug_dict['br_start']} | "
                        f"Intan loc={loc} → ref_i={shift_sample_intan} (Δ_i={delta_intan_samples:+d} samp, n_br={debug_dict['best_n_br']:+d}, {dt_ms:+.3f} ms) "
                        f"| Li={debug_dict['Li']} best_rms={debug_dict['best_rms']:.6g}")
            else:
                locs_rows.append((block, loc, loc, 0))
                locs_rows_dt.append(float('nan'))
            break

        # write per-block CSV (you reference locs_csv later) - TODO this is writing the triangle fix outputs
        locs_csv = METADATA_ROOT / "template_locs" / f"intan_{intan_idx:03d}__template_locs.csv"
        locs_csv.parent.mkdir(parents=True, exist_ok=True)
        with locs_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["block_idx", "lock_loc_sample", "triangle_loc_sample", "delta_samples"])
            w.writerows(locs_rows)







        # TODO Just using the first one now
        # initial/ refined shift from block 0
        template_shift_sample    = locs_rows[0][1]
        adjusted_shift_sample     = locs_rows[0][2]
        adjustment_samples  = locs_rows[0][3]
        shift_sec = adjusted_shift_sample / fs_intan
        shift_ms  = shift_sec * 1000.0

        refined = np.array([r[2] for r in locs_rows])   # triangle_loc_sample (refined)
        deltas  = np.array([r[3] for r in locs_rows])   # delta_samples
        dt_ms   = np.array([d for d in locs_rows_dt if d == d])  # drop NaNs

        if refined.size:
            parts = [
                f"[refine] Intan {intan_idx:03d}: blocks={refined.size}",
                f"Δsamples min/med/max={deltas.min():+d}/{int(np.median(deltas)):+d}/{deltas.max():+d}"
            ]
            if dt_ms.size:
                parts.append(f"Δt(ms) med/min/max={np.median(dt_ms):+.3f}/{np.min(dt_ms):+.3f}/{np.max(dt_ms):+.3f}")
            parts.append(f"first 10 refined={refined[:10].tolist()}")
            print(" | ".join(parts))
        else:
            print(f"[refine] Intan {intan_idx:03d}: no blocks refined")
            
        # keep CSV summary row
        summary_rows.append(dict(
            intan_filename=intan_filename,
            intan_idx=intan_idx,
            br_filename=br_ns5_file_name,
            br_idx=br_idx,
            is_control=is_control,
            is_at_rest=is_at_rest,
            is_continuous_stim=is_continuous_stim,
            notes=note_val,
            # br_ns5=str(br_ns5_file_path),
            fs_intan=fs_intan,
            fs_br=fs_br,
            shift_sample=adjusted_shift_sample,
            shift_seconds=shift_sec,
            shift_ms=shift_ms,
            dur_intan_sec=dur_intan_sec,
            dur_br_sec=dur_br_sec,
            adc_npz=str(adc_npz),
            triangle_refined_from=template_shift_sample,
            triangle_refine_delta_samples=adjustment_samples,
            n_locs=len(locs_rows),
            locs_csv=str(locs_csv),
        ))
        print(f"[shift] Intan {intan_idx:03d} ↔ BR {br_idx:03d} : shift={adjusted_shift_sample:+d} samp ({shift_sec:+.6f} s)")
        print(f"[rec time] Intan {dur_intan_sec:05f} ↔ BR {dur_br_sec:05f}")

    # ---------- write alignment summary CSV ----------
    if summary_rows:
        out_csv = METADATA_ROOT / "br_to_intan_shifts.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[done] wrote shifts → {out_csv}")
    else:
        print("[done] no rows to write (no shifts).")
 
if __name__ == "__main__":
    main()