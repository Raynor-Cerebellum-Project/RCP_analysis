from __future__ import annotations
from pathlib import Path
from typing import Optional
import gc
import numpy as np
import re, csv
from sklearn.decomposition import PCA

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import RCP_analysis as rcp
""" 
    This script preprocesses the Blackrock data.
    Input:
        .ns6 files from Intan
    Output:
        Checkpoint after preprocessing
        Checkpoint after thresholding and calculating MUA peak locations and firing rate
"""

# ---------- Config ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)
BR_ROOT = SESSION_LOC / "Blackrock"; BR_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"
SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"

BR_SESSION_FOLDERS = rcp.list_br_sessions(BR_ROOT)

RATES = PARAMS.UA_rate_est
BIN_MS     = float(RATES.get("bin_ms", 1.0))
SIGMA_MS   = float(RATES.get("sigma_ms", 50.0))
THRESH     = float(RATES.get("detect_threshold", 3))
PEAK_SIGN  = str(RATES.get("peak_sign", "neg"))
    
XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None
if UA_MAP is None:
    raise RuntimeError("UA mapping required for mapping on NS6.")

# Sync channels
UA_CFG = PARAMS.probes.get("UA")
CAMERA_SYNC_CH = int(UA_CFG.get("camera_sync_ch", 134))
TRIANGLE_SYNC_CH = int(UA_CFG.get("triangle_sync_ch", 138))

UA_BUNDLES = OUT_BASE / "bundles" / "UA"
UA_CKPT_OUT = OUT_BASE / "checkpoints" / "UA"; UA_CKPT_OUT.mkdir(parents=True, exist_ok=True)
NPRW_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW"

global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)

def _session_from_rates_path(p: Path) -> str:
    # rates__NRR_RW001_250915_141052__bin1ms_sigma25ms.npz -> NRR_RW001_250915_141052
    stem = p.stem
    body = stem[len("rates__"):]
    return body.split("__bin", 1)[0]

def _parse_intan_session_dtkey(session: str) -> int:
    # Sort sessions by YYMMDD_HHMMSS at end of name if present
    m = re.search(r"(\d{6})_(\d{6})$", session)
    if not m:
        return int("9"*12)
    return int(m.group(1) + m.group(2))

def _build_session_index_map(nprw_ckpt_root: Path) -> tuple[dict[str,int], dict[int,str]]:
    """Scan NPRW rates to build Intan session ↔ Intan_File index mapping."""
    rate_files = sorted(nprw_ckpt_root.rglob("rates__*.npz"))
    sessions = sorted({_session_from_rates_path(p) for p in rate_files},
                      key=_parse_intan_session_dtkey)
    sess_to_intan = {s: i+1 for i, s in enumerate(sessions)}
    intan_to_sess = {i+1: s for i, s in enumerate(sessions)}
    if not sessions:
        raise RuntimeError(f"No NPRW rate files under {nprw_ckpt_root}")
    return sess_to_intan, intan_to_sess

def _ua_arrays_from_idx_rows(n_ch: int, mapped_nsp: np.ndarray, idx_rows: np.ndarray):
    ua_elec_per_row = -np.ones(n_ch, dtype=int)
    ua_nsp_per_row  = -np.ones(n_ch, dtype=int)
    for elec0, row in enumerate(idx_rows):
        if 0 <= row < n_ch:
            ua_elec_per_row[row] = elec0 + 1
            ua_nsp_per_row[row]  = int(mapped_nsp[elec0])
    return ua_elec_per_row, ua_nsp_per_row

def _resolve_intan_session_for_br_idx(
    meta_csv: Path,
    br_idx: int,
    nprw_ckpt_root: Path
) -> Optional[str]:
    """
    Use the provided metadata CSV and NPRW rates to map BR index -> Intan session name.
    """
    if not meta_csv.exists():
        print(f"[WARN] mapping CSV not found: {meta_csv}")
        return None

    # 1) {Intan_File -> BR_File}
    intan_to_br = rcp.get_metadata_mapping(METADATA_CSV, "Intan_File", "BR_File")

    # 2) invert to {BR_File -> Intan_File}
    br_to_intan = {b: i for i, b in intan_to_br.items()}
    if br_idx not in br_to_intan:
        print(f"[WARN] BR {br_idx:03d} not in mapping CSV.")
        return None
    intan_idx = br_to_intan[br_idx]

    # 3) Build {Intan_File index -> session string} from NPRW rates
    _, intan_idx_to_sess = _build_session_index_map(nprw_ckpt_root)
    sess = intan_idx_to_sess.get(intan_idx)
    if not sess:
        print(f"[WARN] Intan_File {intan_idx} not found among NPRW rates.")
        return None
    return sess

def _stim_npz_for_br_idx(meta_csv: Path, out_base: Path, br_idx: int, nprw_ckpt_root: Path) -> Optional[Path]:
    sess = _resolve_intan_session_for_br_idx(meta_csv, br_idx, nprw_ckpt_root)
    if not sess:
        return None
    bundles_root = out_base / "bundles" / "NPRW"
    return bundles_root / f"{sess}_Intan_bundle" / "stim_stream.npz"

def _load_shift_row_by_br_idx(metadata_path: Path, br_idx: int) -> Optional[dict]:
    """
    Read br_to_intan_shifts.csv and return the row for this br_idx.
    Expected columns include: session (Intan), br_idx, anchor_ms, fs_intan, (optionally anchor_sample, etc.)
    """
    shifts_csv = metadata_path
    if not shifts_csv.exists():
        return None
    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                if int(row.get("br_idx", "-1")) == int(br_idx):
                    return row
            except Exception:
                continue
    return None

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

def main(limit_sessions: Optional[int] = None):
    sess_folders = BR_SESSION_FOLDERS
    # limit_sessions = [13, 14, 15, 16, 17]
    if limit_sessions:
        if isinstance(limit_sessions, int):
            # 1-based: 11 -> the 11th folder
            idx = limit_sessions - 1 if limit_sessions > 0 else limit_sessions
            sess_folders = [sess_folders[idx]]
        elif isinstance(limit_sessions, (list, tuple)):
            # 1-based indices: e.g., (2, 5, 11)
            idxs = [(i - 1 if i > 0 else i) for i in limit_sessions]
            sess_folders = [sess_folders[i] for i in idxs]
        else:
            pass

    print("Found session folders:", len(sess_folders))

    for sess in sess_folders:
        print(f"=== Session: {sess.name} ===")
        rcp.extract_blackrock_bundle(sess, UA_BUNDLES, CAMERA_SYNC_CH, TRIANGLE_SYNC_CH) # Extract sync pulses and stuff
        rec_ns6 = se.read_blackrock(sess, stream_name = 'nsx6', all_annotations=True) # Load BR file
        
        br_idx = int(sess.name.split('_')[-1]) # resolve br index
        n_ch = rec_ns6.get_num_channels()
        
        rec_ns6, idx_rows = rcp.apply_ua_mapping_by_renaming(rec_ns6, UA_MAP["mapped_nsp"], br_idx, METADATA_CSV)
        rec_hp  = spre.highpass_filter(rec_ns6, freq_min=float(PARAMS.highpass_hz))
        
        # --- build UA arrays from idx_rows (no SI properties) ---
        ua_elec_per_row, ua_nsp_per_row = _ua_arrays_from_idx_rows(n_ch, UA_MAP["mapped_nsp"], idx_rows)
        row_from_elec = idx_rows.astype(int, copy=False)
        ua_region_per_row = np.array([rcp.ua_region_from_elec(int(elec_num)) for elec_num in ua_elec_per_row], dtype=np.int8)
        ua_region_names   = np.array(["SMA", "Dorsal premotor", "M1 inferior", "M1 superior"], dtype=object)
        UA_probe = ua_region_per_row.copy()
        
        # --- convert labels -> list of channel-ID lists (what SI expects) ---
        ch_ids = np.asarray(rec_hp.get_channel_ids())
        groups = []
        for group in np.unique(UA_probe):
            idx = np.where(UA_probe == group)[0]
            if idx.size >= 2:
                groups.append(ch_ids[idx].tolist())
            else:
                print(f"[debug] region {group}: singleton -> left unreferenced")
        print("[debug] group sizes:", [len(group) for group in groups])

        # Default: no artifact windows
        block_bounds = np.empty((0, 2), dtype=int)
        shifts_row = None
        
        # default to cleaned=ref unless we actually remove
        rec_artif_removed = rec_hp
        fs_ua = float(rec_hp.get_sampling_frequency())
        blank_windows = None
        
        if br_idx is None:
            print(f"[WARN] Could not parse BR index from session folder '{sess.name}'. Skipping artifact removal.")
        else:
            # 1) find Intan session via metadata + NPRW rates
            stim_npz_path = _stim_npz_for_br_idx(METADATA_CSV, OUT_BASE, br_idx, NPRW_CKPT_ROOT)
            # 2) and fetch the shift row for the same BR index
            shifts_row = _load_shift_row_by_br_idx(SHIFT_CSV, br_idx)

            if not stim_npz_path or not stim_npz_path.exists():
                print(f"[WARN] stim_stream.npz not found for BR {br_idx:03d} (looked at {stim_npz_path}).")
            else:
                # show the resolved Intan session name (parent of stim file)
                intan_session = stim_npz_path.parent.name.replace("_Intan_bundle", "")
                print(f"[map] BR {br_idx:03d} -> Intan session '{intan_session}'")

                stim = rcp.load_stim_detection(stim_npz_path)
                block_bounds = np.asarray(stim.get("block_bounds_samples", []), dtype=int)

        if block_bounds.size and shifts_row is not None:
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
                ms_before = 5.0
                tail_ms   = 5.0
                dur_ms    = (ends_ua - starts_ua) * 1000.0 / fs_ua
                ms_after  = float(dur_ms.max() + tail_ms)

                rec_artif_removed = spre.remove_artifacts(
                    rec_hp,
                    list_triggers=[starts_ua.tolist()],
                    ms_before=ms_before,
                    ms_after=ms_after,     # one window long enough for all spans
                    mode="zeros",          # or "linear"
                )
                pad_before_samp = int(round(ms_before * fs_ua / 1000.0))
                pad_after_samp  = int(round(tail_ms  * fs_ua / 1000.0))

                starts_exp = np.clip(starts_ua - pad_before_samp, 0, None)
                ends_exp   = np.clip(ends_ua   + pad_after_samp,  0, n_total)

                blank_windows = {0: np.column_stack([starts_exp, ends_exp])}  # seg 0
            else:
                print("[WARN] all artifact intervals invalid after shift; skipping artifact removal.")
        else:
            if not block_bounds.size:
                print("[WARN] no block spans found; skipping artifact removal.")
            elif shifts_row is None:
                print("[WARN] no shift row found for this BR index; skipping artifact removal.")
            
        out_npz_loc = UA_CKPT_OUT / f"pp_global__{sess.name}__NS6"
        out_npz_loc.mkdir(parents=True, exist_ok=True)
        rec_artif_removed.save(folder=out_npz_loc, overwrite=True)
        print(f"[{sess.name}] (ns6) saved preprocessed -> {out_npz_loc}")

        # compute rates
        rate_hz, t_cat_ms, counts_cat, peaks, peaks_t_ms = rcp.threshold_mua_rates(
            rec_artif_removed,
            detect_threshold=THRESH,
            peak_sign=PEAK_SIGN,
            bin_ms=BIN_MS,
            sigma_ms=SIGMA_MS,
            n_jobs=PARAMS.parallel_jobs,
            blank_windows_samples=blank_windows,
        )
        
        X = rate_hz.T # transpose to (n_bins, n_channels)
        X_filled = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) # Zero-impute NaNs for PCA
        col_means = np.nanmean(X, axis=0) #Nan-aware means from the original data
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        Xc = X_filled - col_means[None, :] # Mean-center columns

        # If total variance is ~0, skip PCA
        total_var = np.var(Xc, axis=0).sum()
        n_comp = min(5, Xc.shape[0], Xc.shape[1])
        if n_comp >= 1 and total_var > 0.0:
            pca = PCA(n_components=n_comp, random_state=0)
            pcs = pca.fit_transform(Xc) # (n_bins, n_comp)
            explained_var = np.nan_to_num(
                pca.explained_variance_ratio_, nan=0.0
            ).astype(np.float32)
            pcs_T = pcs.T.astype(np.float32) # (n_comp, n_bins)
        else:
            pcs_T = np.empty((0, Xc.shape[0]), dtype=np.float32)
            explained_var = np.empty((0,), dtype=np.float32)

        out_npz = UA_CKPT_OUT / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"
        
        # infer common field names across SI versions
        names = getattr(peaks, "dtype", None)
        names = names.names if names is not None else ()
        samp_f = "sample_index" if "sample_index" in names else ("sample_ind" if "sample_ind" in names else None) #TODO shorten
        chan_f = "channel_index" if "channel_index" in names else ("channel_ind" if "channel_ind" in names else None)
        amp_f  = "amplitude" if "amplitude" in names else None

        peak_sample = peaks[samp_f].astype(np.int64)   if (samp_f and peaks.size) else None #TODO shorten
        peak_ch     = peaks[chan_f].astype(np.int16)   if (chan_f and peaks.size) else None
        peak_amp    = peaks[amp_f].astype(np.float32)  if (amp_f  and peaks.size) else None
        
        save = dict(
            rate_hz=rate_hz.astype(np.float32),
            t_ms=t_cat_ms.astype(np.float32),
            counts=counts_cat.astype(np.uint16),
            peaks=peaks,
            peak_t_ms=peaks_t_ms.astype(np.float32),
            pcs=pcs_T,
            explained_var=explained_var.astype(np.float32),

            # keep these if you still want mapping in the file, even though no SI properties:
            ua_row_to_elec  = ua_elec_per_row.astype(np.int16),
            ua_row_to_nsp   = ua_nsp_per_row.astype(np.int16),
            ua_row_to_region = ua_region_per_row,
            ua_region_names  = ua_region_names,
            ua_row_index_from_electrode = row_from_elec.astype(np.int16),

            meta=dict(
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                bin_ms=BIN_MS,
                sigma_ms=SIGMA_MS,
                fs=fs_ua,
                n_channels=int(n_ch),
                session=str(sess.name),
            ),
        )

        if peak_sample is not None: save["peak_sample"] = peak_sample
        if peak_ch is not None:     save["peak_ch"] = peak_ch
        if peak_amp is not None:    save["peak_amp"] = peak_amp
        
        np.savez_compressed(out_npz, **save)
        print(f"[{sess.name}] saved rate matrix + PCA -> {out_npz}")

        # cleanup to keep memory stable on long batches
        del rec_ns6, rec_hp, rec_artif_removed, rate_hz, t_cat_ms, counts_cat
        gc.collect()
if __name__ == "__main__":
    main(limit_sessions=None)
