import gc, csv
from sklearn.decomposition import PCA

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

""" 
    This script preprocesses the Blackrock data.
    Input:
        .ns6 files from Intan
    Output:
        Checkpoint after preprocessing
        Checkpoint after thresholding and calculating MUA peak locations and firing rate
"""

# ---------- Config ----------
# Base paths from config_loading
SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"
BR_SESSION_FOLDERS = rcp.list_br_sessions(BR_ROOT)

RATES = PARAMS.UA_rate_est
BIN_MS     = RATES.get("bin_ms")
SIGMA_MS   = RATES.get("sigma_ms")
THRESH     = RATES.get("detect_threshold")
PEAK_SIGN  = RATES.get("peak_sign")
ARTRMV_MS_BEFORE = float(RATES.get("remove_ms_before", 5.0))
ARTRMV_TAIL_MS   = float(RATES.get("remove_tail_ms_after", 5.0))

FS_NS2 = 1000.0

XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None
if UA_MAP is None:
    raise RuntimeError("UA mapping required for mapping on NS6.")

# Sync channels
UA_CFG = PARAMS.probes.get("UA")
VOG_CH = int(UA_CFG.get("VOG_CH", 131))
CAMERA_SYNC_CH = int(UA_CFG.get("camera_sync_ch", 134))
HR_CH = int(UA_CFG.get("HR_ch", 136))
TRIANGLE_SYNC_CH = int(UA_CFG.get("triangle_sync_ch", 138))
TOUCHSCREEN_CH = int(UA_CFG.get("touchscreen_ch", 139))

TOUCHSCREEN_THRES_A = 1.0e6
TOUCHSCREEN_THRES_B = 3.0e6

UA_CKPT_OUT = UA_CKPT_ROOT


global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)

def _anchor_from_shifts_row(row: dict) -> tuple[int, float]:
    """
    From a shifts row, return (anchor_sample_intan, fs_intan).
    If anchor_sample is missing, derive from anchor_ms.
    """
    fs_intan = float(row.get("fs_intan", 30000.0))
    if row.get("anchor_sample", "") != "":
        return int(row["anchor_sample"]), fs_intan
    if row.get("anchor_ms", "") != "":
        anchor_ms = float(row["anchor_ms"])
        return int(round(anchor_ms * 1e-3 * fs_intan)), fs_intan
    return 0, fs_intan

# For triangle pulse corrected stuff, save for later
def load_anchor_for_session(out_base: Path, session: str) -> tuple[int, float]:
    """
    Return (anchor_sample_intan, fs_intan) from figures/align_BR_to_Intan/br_to_intan_shifts.csv.
    Falls back to deriving anchor_sample from anchor_ms if needed.
    """
    shifts_csv = out_base / "figures" / "align_BR_to_Intan" / "br_to_intan_shifts.csv"
    if not shifts_csv.exists():
        return 0, 30000.0  # safe fallback

    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("session") == session:
                fs_intan = float(row.get("fs_intan", 30000.0))
                if row.get("anchor_sample", "") != "":
                    return int(row["anchor_sample"]), fs_intan
                if row.get("anchor_ms", "") != "":
                    anchor_ms = float(row["anchor_ms"])
                    return int(round(anchor_ms * 1e-3 * fs_intan)), fs_intan
                break

    return 0, 30000.0

def main():
    sess_folders = BR_SESSION_FOLDERS
    
    print("Found session folders:", len(sess_folders))
    for sess in (sess_folders[:]): # Can tweak here to isolate sessions 
        print(f"=== Session: {sess.name} ===")
        _, touchscreen_sig, hr_sig, vog_sig = rcp.extract_br_aux_streams_npz(sess, UA_AUX_DATA, CAMERA_SYNC_CH, TRIANGLE_SYNC_CH, TOUCHSCREEN_CH, HR_CH, VOG_CH) # Extract sync pulses and stuff
        rec_ns6 = se.read_blackrock(sess, stream_name = 'nsx6', all_annotations=True) # Load neural data
        
        # Threshold to get touchscreen state
        ts_state_num = np.zeros(0, dtype=np.int8)
        ts_state_char = np.full(0, 'N', dtype='U1')

        if touchscreen_sig is not None:
            touchscreen_sig = np.asarray(touchscreen_sig, float)
            ts_state_num = np.zeros_like(touchscreen_sig, dtype=np.int8)
            ts_state_char = np.full(touchscreen_sig.shape, 'N', dtype='U1')

            ts_state_num[touchscreen_sig >= TOUCHSCREEN_THRES_A] = 1
            ts_state_num[touchscreen_sig >= TOUCHSCREEN_THRES_B] = 2

            ts_state_char[ts_state_num == 1] = 'A'
            ts_state_char[ts_state_num == 2] = 'B'
            print("Unique touchscreen states:", np.unique(ts_state_num))
        else:
            print("[touchscreen] No touchscreen signal; leaving ts_state_* empty.")
        
        br_idx = int(sess.name.split('_')[-1]) # resolve br index
        n_channels = rec_ns6.get_num_channels()
        
        rec_ns6, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = rcp.apply_ua_mapping_with_regions(rec_ns6, UA_MAP, br_idx, METADATA_CSV)
        UA_probe = ua_region.copy()

        rec_hp  = spre.highpass_filter(rec_ns6, freq_min=float(PARAMS.highpass_hz))
        region_idx, elec_count = np.unique(UA_probe, return_counts=True)
        print(" | ".join(f"{ua_region_names[v]}: {c}" for v, c in zip(region_idx, elec_count)))

        # artifact windows
        block_bounds = np.empty((0, 2), dtype=int)
        shifts_row = None
        blank_windows = None
        
        # default to cleaned=ref unless we actually remove artifact
        rec_artif_removed = rec_hp
        fs_ua = rec_hp.get_sampling_frequency()
        
        if br_idx is None:
            print(f"[WARN] Could not parse BR index from session folder '{sess.name}'. Skipping artifact removal.")
        else:
            # 1) find Intan session via metadata + NPRW rates
            stim_npz_path, intan_session_name = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
            # 2) and fetch the shift row for the same BR index
            shifts_row = rcp.load_shift_row_by_br_idx(SHIFT_CSV, br_idx)

            if not stim_npz_path or not stim_npz_path.exists():
                print(f"[WARN] stim_stream.npz not found for BR {br_idx:03d} (looked at {stim_npz_path}).")
            else:
                print(f"[map] BR {br_idx:03d} -> Intan session '{intan_session_name}'")

                stim = rcp.load_stim_detection(stim_npz_path)
                block_bounds = stim.get("block_bounds_samples", [])

        if block_bounds.size and shifts_row is not None: # TODO check if we can simplify this block
            # Anchor from the SAME shifts row (important!)
            anchor_samp_intan, fs_intan = _anchor_from_shifts_row(shifts_row)

            starts_intan = block_bounds[:, 0].astype(np.int64)
            ends_intan   = block_bounds[:, 1].astype(np.int64)

            # shift+scale into UA sample index space
            scale = fs_ua / fs_intan
            starts_ua = np.round((starts_intan - anchor_samp_intan) * scale).astype(np.int64)
            ends_ua   = np.round((ends_intan   - anchor_samp_intan) * scale).astype(np.int64)

            # validity + clipping
            n_total = rec_hp.get_num_samples()
            ends_ua = np.minimum(ends_ua, n_total)
            valid = (ends_ua > starts_ua) & (starts_ua >= 0) & (ends_ua <= n_total)
            starts_ua = starts_ua[valid]
            ends_ua   = ends_ua[valid]
            
            if starts_ua.size:
                dur_ms    = (ends_ua - starts_ua) * 1000.0 / fs_ua
                ms_after  = float(dur_ms.max() + ARTRMV_TAIL_MS)

                rec_artif_removed = spre.remove_artifacts(
                    rec_hp,
                    list_triggers=starts_ua.tolist(),
                    ms_before=ARTRMV_MS_BEFORE,
                    ms_after=ms_after,     # one window long enough for all spans
                    mode="zeros",          # or "linear"
                )
                pad_before_samp = int(round(ARTRMV_MS_BEFORE * fs_ua / 1000.0))
                pad_after_samp  = int(round(ARTRMV_TAIL_MS  * fs_ua / 1000.0))

                starts_exp = np.clip(starts_ua - pad_before_samp, 0, None)
                ends_exp   = np.clip(ends_ua   + pad_after_samp,  0, n_total)

                blank_windows = {0: np.column_stack([starts_exp, ends_exp])}  # seg 0
            else:
                print("[WARN] all artifact intervals invalid after shift; skipping artifact removal.")
        else:
            if not block_bounds.size:
                print("[WARN] no stim found, skipping artifact removal.")
            elif shifts_row is None:
                print("[WARN] no shift row found, skipping artifact removal.")
            
        out_npz_loc = UA_CKPT_OUT / f"pp__{sess.name}__NS6"; out_npz_loc.mkdir(parents=True, exist_ok=True)
        rec_artif_removed.save(folder=out_npz_loc, overwrite=True)
        print(f"[{sess.name}] (ns6) saved preprocessed -> {out_npz_loc}")

        # compute rates
        rate_hz, t_cat_ms, counts_cat, peaks, peak_t_ms = rcp.threshold_mua_rates( #TODO This needs to be reduced to detect peaks only, because fr estimation should be done later
            rec_artif_removed,
            detect_threshold=THRESH,
            peak_sign=PEAK_SIGN,
            bin_ms=BIN_MS,
            sigma_ms=SIGMA_MS,
            n_jobs=PARAMS.parallel_jobs,
            blank_windows_samples=blank_windows,
        )
        
        ts_state_num_binned = np.zeros(t_cat_ms.shape, dtype=np.int8)
        ts_state_char_binned = np.full(t_cat_ms.shape, 'N', dtype='U1')

        if ts_state_num.size > 0:
            # t_cat_ms is in milliseconds since start of BR recording
            # ts_state_num is per touchscreen sample at ts_fs_hz
            # Map each bin center to nearest touchscreen sample index
            ts_idx = np.round((t_cat_ms / 1000.0) * FS_NS2).astype(np.int64)

            # Clip to valid range
            ts_idx = np.clip(ts_idx, 0, ts_state_num.size - 1)

            ts_state_num_binned = ts_state_num[ts_idx]
            ts_state_char_binned = ts_state_char[ts_idx]

            print(
                f"[touchscreen] Binned states: shape={ts_state_num_binned.shape}, "
                f"unique={np.unique(ts_state_num_binned)}"
            )
        else:
            print("[touchscreen] No touchscreen states to bin; using all 'N'.")

        
        X = rate_hz.T  # (n_bins, n_channels)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_bins, n_ch = X.shape
        n_comp = min(5, n_bins, n_ch)

        total_var = np.var(X, axis=0).sum()
        if n_comp >= 1 and total_var > 0.0:
            pca = PCA(n_components=n_comp, random_state=0)
            pcs = pca.fit_transform(X)  # (n_bins, n_comp)
            explained_var = np.nan_to_num(
                pca.explained_variance_ratio_, nan=0.0
            ).astype(np.float32)
            pcs_T = pcs.T.astype(np.float32)  # (n_comp, n_bins)
        else:
            pcs_T = np.empty((0, n_bins), dtype=np.float32)
            explained_var = np.empty((0,), dtype=np.float32)

        out_npz = UA_CKPT_OUT / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"
        
        save = dict(
            rate_hz=rate_hz.astype(np.float32),
            t_ms=t_cat_ms.astype(np.float32),
            counts=counts_cat.astype(np.uint16),
            peaks=peaks,
            peak_t_ms=peak_t_ms.astype(np.float32),
            pcs=pcs_T,
            explained_var=explained_var.astype(np.float32),            
            ts_state_num=ts_state_num_binned,
            ts_state_char=ts_state_char_binned,

            # UA indices
            ua_index=idx_rows.astype(np.int16),
            ua_elec=ua_elec.astype(np.int16),
            ua_nsp=ua_nsp.astype(np.int16),
            ua_region=ua_region,
            ua_region_names=ua_region_names,
            ua_port=ua_port,
            hr_sig=hr_sig,
            vog_sig=vog_sig,

            meta=dict(
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                bin_ms=BIN_MS,
                sigma_ms=SIGMA_MS,
                fs=fs_ua,
                n_channels=rec_artif_removed.get_num_channels(),
                session=str(sess.name),
            ),
        )
        
        np.savez_compressed(out_npz, **save)
        print(f"[{sess.name}] saved rate matrix + PCA -> {out_npz}")

        # cleanup to keep memory stable on long batches
        del rec_ns6, rec_hp, rec_artif_removed, rate_hz, t_cat_ms, counts_cat
        gc.collect()
if __name__ == "__main__":
    main()
