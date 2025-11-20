from pathlib import Path
import numpy as np
import csv
import spikeinterface.extractors as se
import RCP_analysis as rcp

# ---------- Config ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)
BR_ROOT = SESSION_LOC / "Blackrock"; BR_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"
NPRW_BUNDLES   = OUT_BASE / "bundles" / "NPRW"
NPRW_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW"
TEMPLATE = REPO_ROOT / "config" / "br_intan_align_template.mat"

# Probe params, sync channels
NPRW_CFG = PARAMS.probes.get("NPRW")
NPRW_CAMERA_SYNC_CH = int(NPRW_CFG.get("triangle_sync_ch", 1))
UA_CFG = PARAMS.probes.get("UA")
CAMERA_SYNC_CH = int(UA_CFG.get("camera_sync_ch", 134))
TRIANGLE_SYNC_CH = int(UA_CFG.get("triangle_sync_ch", 138))
MATCH_WINDOW_MS = 1000.0

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
        c /= (np.max(np.abs(c)) + 1e-12)

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
    corr, lags = _xcorr_normalized(adc_lock, template)
    try:
        from scipy.signal import find_peaks
        idx, _ = find_peaks(corr, height=peak)
        locs = lags[idx].astype(int)
    except Exception:
        mid = np.arange(1, corr.size - 1)
        mask = (corr[mid-1] < corr[mid]) & (corr[mid] >= corr[mid+1]) & (corr[mid] >= peak)
        locs = lags[mid[mask]].astype(int)
    return locs

def refine_anchor_with_loc_and_br0(
    rec_br,
    br_ch,
    fs_br: float,
    adc_triangle_intan: np.ndarray,
    fs_intan: float,
    loc: int,                 # Intan offset where BR starts (from template on Intan)
    search_n_br: int,         # ± extra BR-sample shifts to try
    br_window_sec: float = 20.0,  # long window to capture triangle reversals
    normalize_both: bool = False  # keep False to match MATLAB (normalize Intan only)
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

    # ---- Normalize Intan only (MATLAB behavior) ----
    x_i = adc_triangle_intan.astype(float, copy=False)
    x_i = (x_i - x_i.mean()) / (np.ptp(x_i) + 1e-12)
    if normalize_both:
        y_br = (y_br - y_br.mean()) / (np.ptp(y_br) + 1e-12)

    N_i = int(x_i.size)
    if N_i == 0 or N_b < 8:
        return loc, 0, float("nan"), {"br_start": 0, "br_end": br_end, "best_n_br": 0, "best_rms": float("nan")}

    # ---- Time bases (BR origin = 0) ----
    t_i = np.arange(N_i, dtype=float) / float(fs_intan)   # Intan absolute times
    t0  = float(loc) / float(fs_intan)                    # Intan time at 'loc'
    t_m = np.arange(N_b, dtype=float) / float(fs_br)      # BR time grid 0..(N_b-1)
    r   = float(fs_intan) / float(fs_br)                  # Intan samp / BR samp

    # ---- Search integer BR shifts around the coarse anchor ----
    best_rms = np.inf
    best_n   = 0
    for n in range(-int(search_n_br), int(search_n_br) + 1):
        # Shift on BR grid by n ⇒ add n/fs_br seconds on the Intan time axis
        t_query = t0 + (n / float(fs_br)) + t_m
        seg_on_br = np.interp(t_query, t_i, x_i, left=np.nan, right=np.nan)

        # Edge handling if we run off Intan
        if np.isnan(seg_on_br).any():
            mval = np.nanmean(seg_on_br)
            seg_on_br = np.where(np.isnan(seg_on_br), mval, seg_on_br)

        rms = np.sqrt(np.mean((y_br - seg_on_br) ** 2))
        if rms < best_rms:
            best_rms = rms
            best_n   = n

    # ---- Convert best BR shift to Intan samples; final anchor = loc + delta_i ----
    delta_i = int(round(best_n * r))
    shift  = int(np.clip(loc + delta_i, 0, N_i - 1))
    dt_ms   = 1000.0 * delta_i / float(fs_intan)

    dbg = {
        "br_start": 0,
        "br_end": br_end,
        "best_n_br": int(best_n),
        "best_rms": float(best_rms),
        "Li": int(max(8, round(N_b * r))),
    }
    return shift, int(delta_i), float(dt_ms), dbg

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
    intan_to_br = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "BR_File")
    print(f"[map] Loaded Intan→BR rows: {len(intan_to_br)}")

    # Load template
    template = load_template(TEMPLATE)

    # iterate BR Intan pairs, compute shift, and write shifts to CSV
    summary_rows = []
    for intan_idx, br_idx in sorted(intan_to_br.items()):
        intan_sess = intan_idx_to_sess.get(intan_idx)
        if intan_sess is None:
            print(f"[warn] No session name for Intan_File={intan_idx} (skipping)")
            continue

        adc_npz = NPRW_BUNDLES / f"{intan_sess}_Intan_bundle" / "aux_streams.npz"
        if not adc_npz.exists():
            print(f"[warn] Intan ADC bundle missing: {adc_npz} (skip Intan {intan_idx} → BR {br_idx})")
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
            print(f"[warn] No template peaks found for {intan_sess}; skipping.")
            continue
        
        # ---- load BR (lazy) ----
        fs_br = float("nan")
        rec_br = None
        dur_intan_sec = len(intan_triangle_signal) / fs_intan
        dur_br_sec = float("nan")

        br_ns5_file = BR_ROOT / f"{PARAMS.session}_{br_idx:03d}.ns5"
        use_br = False
        if br_ns5_file.exists():
            try:
                rec_br = se.read_blackrock(br_ns5_file)  # does not load all data
                fs_br = rec_br.get_sampling_frequency()
                # map TRIANGLE_SYNC_CH to a valid channel id
                if str(TRIANGLE_SYNC_CH) not in rec_br.get_channel_ids():
                    print(f"[note] TRIANGLE_SYNC_CH={str(TRIANGLE_SYNC_CH)} not found in BR channels; skipping BR refinement.")
                else:
                    use_br = False # TODO Set to true after fixing triangle alignment
                try:
                    dur_br_sec = rec_br.get_num_frames() / fs_br
                except Exception:
                    pass
            except Exception as e:
                print(f"[note] BR open failed for {br_ns5_file}: {e}")
                rec_br, fs_br, use_br = None, float("nan"), False






        # refine each block using triangle sync channel TODO FIX THIS
        LOC_REFINE_N = 50        # like MATLAB's N
        locs_rows = []
        locs_rows_dt = []
        
        for block, loc in enumerate(br_sync_locs): # loc is a candidate for Intan sample where BR recording starts before refinement
            if use_br:
                search_n_br = int(round(LOC_REFINE_N * fs_br / fs_intan)) # Width of searching through BR samples
                shift_sample_intan, delta_intan_samples, dt_ms, debug_dict = refine_anchor_with_loc_and_br0(
                    rec_br=rec_br, br_ch=str(TRIANGLE_SYNC_CH), fs_br=fs_br,
                    adc_triangle_intan=intan_triangle_signal, fs_intan=fs_intan,
                    loc=loc, search_n_br=search_n_br, br_window_sec=20.0, normalize_both=False
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
        # initial/ refined anchor from block 0
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
            session=intan_sess,
            intan_idx=intan_idx,
            br_idx=br_idx,
            adc_npz=str(adc_npz),
            br_ns5=str(br_ns5_file),
            fs_intan=fs_intan,
            fs_br=fs_br,
            anchor_sample=adjusted_shift_sample,
            anchor_seconds=shift_sec,
            anchor_ms=shift_ms,
            dur_intan_sec=dur_intan_sec,
            dur_br_sec=dur_br_sec,
            triangle_refined_from=template_shift_sample,
            triangle_refine_delta_samples=adjustment_samples,
            n_locs=len(locs_rows),
            locs_csv=str(locs_csv),
        ))
        print(f"[shift] Intan {intan_idx:03d} ↔ BR {br_idx:03d} : shift={adjusted_shift_sample:+d} samp ({shift_sec:+.6f} s)")

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