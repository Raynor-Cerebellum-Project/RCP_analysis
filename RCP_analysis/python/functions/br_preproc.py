from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import re, csv
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from scipy.signal import fftconvolve

# ---------- BR utils ----------
def list_br_sessions(br_root: Path) -> list[Path]:
    """
    List Blackrock sessions under br_root.

    A session can be either:
      - a subdirectory, or
      - a base filename for NSx/NEV pairs in br_root.
    """
    root = br_root
    if not root.exists():
        raise FileNotFoundError(f"Session root not found: {root}")

    # Case 1: sessions are organized as subdirectories
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        return sorted(subdirs)

    # Case 2: sessions are loose NSx/NEV files in the root
    files = [p for p in root.iterdir() if p.is_file()]
    nsx_nev = [
        p for p in files
        if p.suffix.lower().startswith(".ns") or p.suffix.lower() == ".nev"
    ]
    if not nsx_nev:
        raise FileNotFoundError(f"No NSx/NEV files found under {root}.")
    bases = sorted({p.stem for p in nsx_nev})
    return [root / b for b in bases]

def ua_excel_path(repo_root: Path, probes_cfg: dict | None) -> Optional[Path]:
    ua_cfg = (probes_cfg).get("UA")
    rel = ua_cfg.get("mapping_mat_rel")
    if not rel:
        return None
    p = (repo_root / rel) if not str(rel).startswith("/") else Path(rel)
    return p if p.exists() else None


# bundle extraction
def extract_blackrock_bundle(sess: Path, out_dir, camera_sync_ch, triangle_sync_ch) -> dict:
    """
    Build and save BR bundle:
      - ns5: pull specific sync channels (Ex: 134=camera, 138=triangle) if present
      - ns2: pull ALL channels present
    """
    bundle: Dict[str, Any] = {}

    # ns5 (aux streams / sync)
    camera_sync = None
    triangle_sync = None
    fs_ns5 = None
    try:
        rec_ns5 = se.read_blackrock(sess, stream_name = 'nsx5', all_annotations=True)
        fs_ns5 = rec_ns5.get_sampling_frequency()
        ids_ns5 = rec_ns5.get_channel_ids().astype(int)

        if camera_sync_ch in ids_ns5:
            camera_sync = rec_ns5.get_traces(segment_index=0, channel_ids=[str(camera_sync_ch)]).ravel()
        else:
            print("[WARN] camera_sync channel not found in ns5.")

        if triangle_sync_ch in ids_ns5:
            triangle_sync = rec_ns5.get_traces(segment_index=0, channel_ids=[str(triangle_sync_ch)]).ravel()
        else:
            print("[WARN] triangle_sync channel not found in ns5.")

    except FileNotFoundError:
        print("[WARN] ns5 not found; syncs unavailable.")

    aux = {
        "fs": fs_ns5,
        "camera_sync": camera_sync,
        "triangle_sync": triangle_sync,
    }

    # extract ns2 (all channels)
    digi = {"fs": None, "channels": {}}
    try:
        rec_ns2 = se.read_blackrock(sess, stream_name = 'nsx2', all_annotations=True)
        digi["fs"] = rec_ns2.get_sampling_frequency()

        ids_ns2 = rec_ns2.get_channel_ids().astype(int)
        tr_ns2 = rec_ns2.get_traces()

        # add ALL columns present
        for col, cid in enumerate(ids_ns2):
            digi["channels"][f"ch{cid}"] = tr_ns2[:, col]
    except FileNotFoundError:
        print("[WARN] ns2 not found; no digital channels.")

    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        out_dir / f"{sess}_UA_bundle.npz",
        aux_fs=aux['fs'],
        camera_sync=aux["camera_sync"],
        triangle_sync=aux["triangle_sync"],
        digi_fs=digi["fs"],
        **digi["channels"],
    )
    print(f"[SAVED] {out_dir / f'{sess}_UA_bundle.npz'}")
    return bundle


# Parsing MAP file excel file (xlsm)
def _parse_nsp_channel(val) -> Optional[int]:
    """
    Parse excel mapping file into an NSP channel int
    Ex: converts 'ch-15' to 15
    """
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    m = re.search(r'(\d+)', str(val))
    return int(m.group(1)) if m else None

def load_UA_mapping_from_excel(xls_path: Path, sheet: str | int = 0, n_elec: int | None = None) -> Dict[str, Any]:
    """
    Read the excel sheet that has columns like:
      - 'NSP ch' (values like 'ch-01')
      - 'Elec#'  (1..N)
    This function resolves the non-sequential mapping ("smattering") between electrode
    numbers and NSP channel IDs. The output array is indexed by electrode position:
        mapped_nsp[i] = NSP channel wired to Electrode #(i+1)
    Example:
        If the MAP file indicates that Electrode #1 is connected to NSP ch-146,
        then mapped_nsp[0] == 146
    Input:
        xls_path: Path to the excel file
        sheet:    Sheet name or index (default 0)
        n_elec:   Optionally specify the number of electrodes (otherwise inferred)
    Returns:
      {
        'mapped_nsp': np.ndarray (N,), NSP channel numbers in **correct order**,
        'n_channels': int
      }
    """
    xls_path = Path(xls_path)
    if not xls_path.exists():
        raise FileNotFoundError(f"Excel not found: {xls_path}")

    df = pd.read_excel(xls_path, sheet_name=sheet)

    def find_col(names):
        cols = {re.sub(r'[^a-z0-9]+', '', c.lower()): c for c in map(str, df.columns)}
        for n in names:
            key = re.sub(r'[^a-z0-9]+', '', n.lower())
            if key in cols:
                return cols[key]
        return None

    nsp_col  = find_col(["NSP ch", "NSP", "NSP Channel", "Channel", "CH"])
    elec_col = find_col(["Elec#", "Elec #", "Electrode", "Electrode #"])
    if nsp_col is None or elec_col is None:
        raise ValueError(f"Could not find NSP/Electrode columns in {xls_path.name}. Columns: {list(df.columns)}")

    nsp  = df[nsp_col].map(_parse_nsp_channel)
    elec = pd.to_numeric(df[elec_col], errors="coerce").astype("Int64")
    mask = nsp.notna() & elec.notna()
    nsp  = nsp[mask].astype(int).to_numpy()
    elec = elec[mask].astype(int).to_numpy()

    max_elec = int(n_elec or elec.max())
    mapped_nsp = np.zeros(max_elec, dtype=int)
    mapped_nsp[elec - 1] = nsp
    return {"mapped_nsp": mapped_nsp, "n_channels": int(max_elec)}

def _lookup_ua_port(meta_csv: Path, br_idx: int) -> Optional[str]:
    """
    Return UA_port for a given BR index from the metadata CSV.
    Accepts column variants: UA_port / UA Port / port / uaport, and BR / BR_File / br_file.
    """
    if not meta_csv.exists():
        print(f"[WARN] metadata CSV not found at {meta_csv}")
        return None

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    with meta_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            print(f"[WARN] {meta_csv.name} has no header")
            return None
        fmap = {norm(k): k for k in rdr.fieldnames if k}

        # pick header names robustly
        def pick(*names):
            for n in names:
                if n in fmap:
                    return fmap[n]
            return None

        col_br = pick("br","brfile","br_file","brindex","br_fileindex")
        col_port = pick("uaport","ua_port","ua port","port")

        if col_br is None or col_port is None:
            print(f"[WARN] metadata missing BR and/or UA_port columns (have: {rdr.fieldnames})")
            return None

        for row in rdr:
            try:
                if int(str(row[col_br]).strip()) == int(br_idx):
                    val = str(row[col_port]).strip()
                    return val.upper() if val else None
            except Exception:
                continue

    return None

def _align_mapping_index_to_recording(recording, mapped_nsp: np.ndarray, br_idx, meta_csv) -> np.ndarray:
    """
    Build an electrode->row index map, using NSP channel ids from `mapped_nsp`.
    For UA Port A, channel IDs are 1..128.
    For UA Port B, channel IDs are 129..256 (i.e., local 1..128 + 128).
    We *offset the recording channel IDs* when port == 'B' so lookups happen in 129..256.
    """
    ua_port = _lookup_ua_port(meta_csv, br_idx)
    ch_ids_raw = np.array(recording.get_channel_ids())

    # Primary path: numeric channel_ids from the Recording
    try:
        ch_ids_num = ch_ids_raw.astype(int)
        if ua_port and ua_port.upper() == 'B':
            # make the keys 129..256 so they match mapped_nsp for Port B
            keys = ch_ids_num + 128
        else:
            keys = ch_ids_num
        id_to_row = {int(k): i for i, k in enumerate(keys)}
    except Exception:
        id_to_row = {}

    # Fallback: derive numeric ids from properties (often already NSP ids)
    if not id_to_row:
        for key in ("nsp_channel", "electrode_id", "nsx_chan_id", "channel_name"):
            if key in recording.get_property_keys():
                vals = recording.get_property(key)
                def v2i(v):
                    if isinstance(v, (int, np.integer)): return int(v)
                    m = re.search(r'(\d+)', str(v)); return int(m.group(1)) if m else None
                ints = np.array([v2i(v) for v in vals], dtype=object)
                # If these are local 1..128 and we're on Port B, offset them.
                ints_num = np.array([int(x) for x in ints if x is not None], dtype=int)
                # Heuristic: if max <= 128 and port B, add 128
                if ua_port and ua_port.upper() == 'B' and ints_num.size and ints_num.max() <= 128:
                    ints_adj = [int(x)+128 if x is not None else None for x in ints]
                else:
                    ints_adj = ints
                id_to_row = {int(iv): i for i, iv in enumerate(ints_adj) if iv is not None}
                break

    # Map each electrode's NSP id -> recording row, -1 if missing
    idx_rows = np.full(mapped_nsp.shape, -1, dtype=int)
    for elec_idx0, nsp_id in enumerate(mapped_nsp):
        try:
            nsp_id = int(nsp_id)
        except Exception:
            continue
        row = id_to_row.get(nsp_id, -1)
        idx_rows[elec_idx0] = row

    return idx_rows

# Mapping
def apply_ua_mapping_by_renaming(recording, mapped_nsp: np.ndarray, br_idx, meta_csv):
    """
    Rename channel IDs to include UA electrode + NSP channel mapping.

    Example new ID format for mapped rows:
        'UAe003_NSP017'   # electrode 3, NSP channel 17 (after port normalization)

    Unmapped rows retain their original channel_id.

    Returns
    -------
    renamed : BaseRecording
        New recording with renamed channel_ids.
    idx_rows : np.ndarray[int]
        For each electrode (1..len(mapped_nsp)), the recording row index or -1 if absent.
    """
    # Map each electrode -> row index on this recording (handles Port B → 1..128 normalization)
    idx_rows = _align_mapping_index_to_recording(recording, mapped_nsp, br_idx, meta_csv)  # shape (N_elec,)

    # Build the positional list of new channel IDs
    ch_ids = list(recording.get_channel_ids())
    new_ids = [str(ch) for ch in ch_ids]  # default: keep original for unmapped rows

    n_ch = recording.get_num_channels()
    for elec0, row in enumerate(idx_rows):
        if 0 <= row < n_ch:
            elec = elec0 + 1
            nsp = int(mapped_nsp[elec0])
            new_ids[row] = f"UAe{elec:03d}_NSP{nsp:03d}"

    # IMPORTANT: rename_channels returns a *new* recording
    renamed = recording.rename_channels(new_ids)

    mapped = sum(1 for r in idx_rows if 0 <= r < n_ch)
    print(f"[MAP] renamed {mapped}/{n_ch} rows with UA mapping ('UAe###_NSP###').")

    # If you still want the fast reverse lookup later, you can stash it as an annotation:
    renamed.set_annotation("ua_row_index_from_electrode", idx_rows.astype(int))

    return renamed, idx_rows

def ua_region_from_elec(e: int) -> int:
    if e <= 0:      return -1
    if e <= 64:     return 0  # SMA
    if e <= 128:    return 1  # Dorsal premotor
    if e <= 192:    return 2  # M1 inferior
    return 3                  # M1 superior

def _gauss_kernel_from_sigma_bins(sigma_bins: float, radius_mult: float = 4.0):
    sigma_bins = max(1e-9, float(sigma_bins))
    r = int(np.ceil(radius_mult * sigma_bins))
    x = np.arange(-r, r+1, dtype=float)
    k = np.exp(-(x**2) / (2.0 * sigma_bins**2))
    k /= k.sum()
    return k

## This function used by Intan too
def threshold_mua_rates(
    recording,
    detect_threshold: float,
    peak_sign: str,
    bin_ms: float,
    sigma_ms: float,
    n_jobs: int,
    blank_windows_samples: dict[int, np.ndarray] | None = None,
):
    """
    Threshold-crossing MUA → binned + Gaussian-smoothed firing rates.

    Returns
    -------
    rate_hz : (n_channels, n_bins) float
    t_ms    : (n_bins,) float   # bin-center times in ms (concatenated across segments)
    counts  : (n_channels, n_bins) int
    peaks   : np.ndarray        # structured array from SpikeInterface (unchanged)
    peak_t_ms : (n_peaks,) float
        Global peak times in ms (segment offsets applied).
    """
    
    fs = float(recording.get_sampling_frequency())
    n_ch = int(recording.get_num_channels())
    n_seg = int(recording.get_num_segments())

    bin_samps = int(round(bin_ms * 1e-3 * fs))
    if bin_samps < 1:
        raise ValueError("bin_ms too small for sampling rate")

    # 1) Detect peaks
    noise_levels = si.get_noise_levels(recording, method="mad", return_in_uV=False) # They didn't write return_in_uV in their documentation
    peaks = detect_peaks(
        recording,
        method="by_channel_torch",
        detect_threshold=detect_threshold,
        peak_sign=peak_sign,
        noise_levels=noise_levels,
        n_jobs=n_jobs,
    )
    # --- Deduplicate near-coincident peaks per (segment, channel): cluster + keep strongest ---
    dedup_ms = 0.5
    dedup_samp = max(1, int(round(dedup_ms * 1e-3 * fs)))

    # Build keys
    ch_field, samp_field, seg_field, amp_field = ("channel_index", "sample_index", "segment_index", "amplitude")
    seg_key = (peaks[seg_field] if seg_field in peaks.dtype.names
            else np.zeros(peaks.shape[0], dtype=np.int64))
    ch_key  = peaks[ch_field].astype(np.int32, copy=False)
    t_key   = peaks[samp_field].astype(np.int64, copy=False)
    amp_val = np.abs(peaks[amp_field].astype(np.float32)) if amp_field in peaks.dtype.names else np.ones(len(peaks))

    # Sort by (segment, channel, time)
    order = np.lexsort((t_key, ch_key, seg_key))
    peaks = peaks[order]
    seg_key, ch_key, t_key, amp_val = seg_key[order], ch_key[order], t_key[order], amp_val[order]

    keep = np.zeros(len(peaks), dtype=bool)

    # Iterate per segment–channel group
    for seg in np.unique(seg_key):
        for ch in np.unique(ch_key[seg_key == seg]):
            mask = (seg_key == seg) & (ch_key == ch)
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue

            # Walk through sorted times → cluster overlapping spikes
            t_local = t_key[idxs]
            a_local = amp_val[idxs]

            max_cluster_ms = 1.0     # maximum total cluster length in ms
            max_cluster_samp = int(round(max_cluster_ms * 1e-3 * fs))

            cluster_start = 0
            for i in range(1, len(idxs)):
                # gap between consecutive spikes
                gap = t_local[i] - t_local[i - 1]
                # check both: gap too large OR cluster too long overall
                if gap > dedup_samp or (t_local[i] - t_local[cluster_start]) > max_cluster_samp:
                    # finalize current cluster
                    cluster_slice = slice(cluster_start, i)
                    best = idxs[cluster_slice.start + np.argmax(a_local[cluster_slice])]
                    keep[best] = True
                    cluster_start = i

            # finalize last cluster
            cluster_slice = slice(cluster_start, len(idxs))
            best = idxs[cluster_slice.start + np.argmax(a_local[cluster_slice])]
            keep[best] = True
    peaks = peaks[keep]

    # 1b) Build global peak time array in ms
    # Compute cumulative sample offsets per segment: [0, n0, n0+n1, ...]
    seg_n_samps = [int(recording.get_num_frames(s)) for s in range(n_seg)]
    seg_offsets_samp = np.cumsum([0] + seg_n_samps[:-1]).astype(np.int64)

    if seg_field in peaks.dtype.names:
        seg_idx = peaks[seg_field].astype(np.int64, copy=False)
    else:
        if n_seg != 1:
            raise RuntimeError("No segment field in peaks for multi-segment recording.")
        seg_idx = np.zeros(peaks.shape[0], dtype=np.int64)

    peak_samp_global = peaks[samp_field].astype(np.int64, copy=False) + seg_offsets_samp[seg_idx]
    peak_t_ms = peak_samp_global.astype(np.float32) * (1000.0 / fs)

    # 2) Bin counts per segment
    bin_offset = 0
    counts_all, t_all, blank_masks = [], [], []
    sigma_bins = max(1e-9, sigma_ms / bin_ms)

    for seg in range(n_seg):
        n_samps = seg_n_samps[seg]
        seg_bins = int(np.ceil(n_samps / bin_samps))
        counts = np.zeros((n_ch, seg_bins), dtype=np.int32)

        # select peaks for this segment
        if seg_field is not None and seg_field in peaks.dtype.names:
            seg_peaks = peaks[peaks[seg_field] == seg]
        else:
            seg_peaks = peaks if n_seg == 1 else None
            if seg_peaks is None:
                raise RuntimeError("No segment field in peaks for multi-segment recording.")
        if seg_peaks.size > 0:
            ch_idx = seg_peaks[ch_field].astype(np.int64, copy=False)
            samp   = seg_peaks[samp_field].astype(np.int64, copy=False)
            bins   = np.clip(samp // bin_samps, 0, seg_bins - 1)
            np.add.at(counts, (ch_idx, bins), 1)

        counts_all.append(counts)

        # ---- build a bin-level blank mask from sample windows ----
        blank_mask_seg = np.zeros(seg_bins, dtype=bool)
        if blank_windows_samples is not None and seg in blank_windows_samples:
            win = np.asarray(blank_windows_samples[seg], dtype=np.int32)
            win = win[(win[:,1] > win[:,0]) & (win[:,0] < n_samps) & (win[:,1] > 0)]
            if win.size:
                # convert sample windows -> bin-range [b0,b1)
                b0 = np.clip(win[:,0] // bin_samps, 0, seg_bins)
                # inclusive end in samples -> exclusive end in bins
                b1 = np.clip((win[:,1] + bin_samps) // bin_samps, 0, seg_bins)
                for i0, i1 in zip(b0, b1):
                    blank_mask_seg[i0:i1] = True
        blank_masks.append(blank_mask_seg)

        # times
        t_ms = (np.arange(seg_bins, dtype=np.float32) + 0.5 + bin_offset) * bin_ms
        t_all.append(t_ms)
        bin_offset += seg_bins

    counts_cat = np.concatenate(counts_all, axis=1)
    t_cat_ms   = np.concatenate(t_all)
    blank_mask = np.concatenate(blank_masks) if blank_masks else np.zeros(counts_cat.shape[1], bool)
    fs_bins = 1000.0 / float(bin_ms)

    if counts_cat.shape[1] < 4:
        counts_smooth = counts_cat.astype(float, copy=False)
        if blank_mask.any():
            counts_smooth[:, blank_mask] = np.nan
    else:
        sigma_bins = float(sigma_ms) / float(bin_ms)
        k = _gauss_kernel_from_sigma_bins(max(1e-9, sigma_bins), radius_mult=4.0)
        r = (k.size - 1) // 2  # kernel radius in bins

        X = counts_cat.astype(float, copy=False)
        valid0 = np.isfinite(X)

        # ---- (avoid half-kernels) ----
        if blank_mask.any():
            pad = np.zeros(r, dtype=int)
            dil = np.convolve(np.r_[pad, blank_mask.astype(int), pad],
                            np.ones(2 * r + 1, dtype=int), mode="same") > 0
            blank_dil = dil[r:-r]
        else:
            blank_dil = np.zeros(X.shape[1], dtype=bool)

        # base validity for smoothing
        valid = valid0 & (~blank_dil)

        # ---- inpainting: fill only short NaN runs (≤ r) per channel ----
        def _interp_short_gaps(y: np.ndarray, max_gap: int) -> np.ndarray:
            out = y.copy()
            isn = ~np.isfinite(out)
            if not isn.any():
                return out
            starts = np.where(isn & ~np.r_[False, isn[:-1]])[0]
            ends   = np.where(isn & ~np.r_[isn[1:],  False])[0]
            for s, e in zip(starts, ends):
                gap = e - s + 1
                L, R = s - 1, e + 1
                if gap <= max_gap and L >= 0 and R < out.size and np.isfinite(out[L]) and np.isfinite(out[R]):
                    out[s:e+1] = np.interp(np.arange(s, e+1), [L, R], [out[L], out[R]])
            return out

        X_filled = X.copy()
        # don’t inpaint original hard blank bins; only regular NaNs from data
        hard_nan = ~valid0
        for i in range(X.shape[0]):
            xi = X_filled[i]
            # protect true blanks
            xi[blank_dil] = np.nan
            X_filled[i] = _interp_short_gaps(xi, r)

        # ---- normalized single-pass convolution with reflect padding ----
        Xp = np.pad(np.where(valid, X_filled, 0.0), ((0, 0), (r, r)), mode="reflect")
        Wp = np.pad(valid.astype(float),          ((0, 0), (r, r)), mode="reflect")

        num = fftconvolve(Xp, k[None, :], mode="same")[:, r:-r]
        den = fftconvolve(Wp, k[None, :], mode="same")[:, r:-r]

        counts_smooth = num / np.clip(den, 1e-12, None)

        # keep hard blanks strictly NaN in the output (use original blank_mask, not dilated)
        counts_smooth[:, blank_mask] = np.nan

    # Finally convert counts→Hz
    rate_hz = counts_smooth * fs_bins
    return rate_hz.astype(np.float32), t_cat_ms, counts_cat.astype(np.uint32), peaks, peak_t_ms

__all__ = [
    "list_br_sessions", "ua_excel_path", "load_UA_mapping_from_excel", "apply_ua_mapping_by_renaming",
    "build_blackrock_bundle", "save_UA_bundle_npz",
    "threshold_mua_rates", "ua_region_from_elec"
]