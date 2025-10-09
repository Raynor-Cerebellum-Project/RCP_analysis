from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import re
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from scipy.signal import butter, filtfilt

# ---------- BR utils ----------
def list_br_sessions(data_root: Path, blackrock_rel: str) -> list[Path]:
    """
    List Blackrock sessions under data_root/blackrock_rel.
    A session can be either a directory or a base filename for NSx/NEV pairs.
    """
    root = data_root / blackrock_rel
    if not root.exists():
        raise FileNotFoundError(f"Session root not found: {root}")

    # Case 1: sessions are organized as subdirectories
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        return sorted(subdirs)

    # Case 2: sessions are loose NSx/NEV files in the root
    files = [p for p in root.iterdir() if p.is_file()]
    nsx_nev = [p for p in files if p.suffix.lower().startswith(".ns") or p.suffix.lower() == ".nev"]
    if not nsx_nev:
        raise FileNotFoundError(f"No NSx/NEV files found under {root}.")
    bases = sorted({p.stem for p in nsx_nev})
    return [root / b for b in bases]

def ua_excel_path(repo_root: Path, probes_cfg: dict | None) -> Optional[Path]:
    ua_cfg = (probes_cfg or {}).get("UA", {})
    rel = ua_cfg.get("mapping_excel_rel") or ua_cfg.get("mapping_mat_rel")
    if not rel:
        return None
    p = (repo_root / rel) if not str(rel).startswith("/") else Path(rel)
    return p if p.exists() else None

# ---------- BR loaders ----------
def _load_nsx(sess: Path, ext: str):
    """
    Load a Blackrock .nsx file by extension (e.g. ns2, ns5, ns6).
    - sess is the session path
    Returns a SpikeInterface BlackrockRecordingExtractor
    """
    sess = Path(sess)
    root = sess if sess.is_dir() else sess.parent
    base = sess.name if not sess.is_dir() else sorted({p.stem for p in root.glob("*.ns*")})[0]
    f = root / f"{base}.{ext}"
    if not f.exists():
        raise FileNotFoundError(f"Missing {ext} for base {base} in {root}")
    print(f"[LOAD] {f.name}")
    return se.read_blackrock(str(f), all_annotations=True)

def load_ns6_spikes(sess: Path):
    rec = _load_nsx(sess, "ns6")
    ids = _ensure_int_ids(rec)
    print(f"[NS6] {len(ids)} channels → {ids.tolist()}")
    return rec

def load_ns5_aux(sess: Path):
    rec = _load_nsx(sess, "ns5")
    ids = _ensure_int_ids(rec)
    print(f"[NS5] {len(ids)} channels → {ids.tolist()}")
    return rec

def load_ns2_digi(sess: Path):
    rec = _load_nsx(sess, "ns2")
    ids = _ensure_int_ids(rec)
    print(f"[NS2] {len(ids)} channels → {ids.tolist()}")
    return rec

# ---------- Mapping helpers ----------
def _ensure_int_ids(rec) -> np.ndarray:
    """
    Convert channel IDs from a SI RecordingExtractor to integers.
    Ex: Cases where IDs are strings like 'chan-001'
    """
    ids = rec.get_channel_ids()
    out = []
    for cid in ids:
        try:
            out.append(int(cid))
        except Exception:
            m = re.search(r"(\d+)", str(cid))
            out.append(int(m.group(1)) if m else None)
    return np.array(out, dtype=int)

def pick_cols_by_ids(rec, wanted_ids) -> Tuple[list[int], list[int]]:
    ids = _ensure_int_ids(rec)
    id2col = {ch: i for i, ch in enumerate(ids)}
    cols = [id2col[c] for c in wanted_ids if c in id2col]
    missing = [c for c in wanted_ids if c not in id2col]
    return cols, missing

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

def align_mapping_index_to_recording(recording, mapped_nsp: np.ndarray) -> np.ndarray:
    """
    Align an array of NSP channel numbers (mapped_nsp) to the row
    indices of a SpikeInterface Recording.

    Returns an array of length len(mapped_nsp):
      - valid row index if that NSP channel is present in recording
      - -1 if not present
    """
    ch_ids = np.array(recording.get_channel_ids())
    id_to_row = {}

    # try direct numeric match
    try:
        id_to_row = {int(cid): i for i, cid in enumerate(ch_ids.astype(int))}
    except Exception:
        pass

    # fallback: check common property keys
    if not id_to_row:
        for key in ("electrode_id", "nsx_chan_id", "nsp_channel", "channel_name"):
            if key in recording.get_property_keys():
                vals = recording.get_property(key)
                def v2i(v):
                    if isinstance(v, (int, np.integer)):
                        return int(v)
                    m = re.search(r'(\d+)', str(v))
                    return int(m.group(1)) if m else None
                ints = [v2i(v) for v in vals]
                id_to_row = {iv: i for i, iv in enumerate(ints) if iv is not None}
                break

    # build output: -1 for missing
    idx_rows = np.full(mapped_nsp.shape, -1, dtype=int)
    for i, nsp in enumerate(mapped_nsp):
        if nsp in id_to_row:
            idx_rows[i] = id_to_row[nsp]

    return idx_rows

def apply_ua_mapping_properties(recording, mapped_nsp: np.ndarray):
    """
    Stamp UA mapping info onto the Recording without geometry.

    Per-channel properties (length == n_channels):
      - 'ua_electrode'   : Electrode number (1..N) for that recording row, or -1
      - 'ua_nsp_channel' : NSP channel id mapped to that row, or -1

    Per-recording annotation (any length):
      - 'ua_row_index_from_electrode' : np.ndarray len == len(mapped_nsp),
        where entry i is the recording row index for electrode (i+1), or -1 if absent.
    """
    idx_rows = align_mapping_index_to_recording(recording, mapped_nsp)  # shape = (N_elec,)
    n_ch = recording.get_num_channels()

    # Per-channel arrays
    ua_elec_per_row = -np.ones(n_ch, dtype=int)
    ua_nsp_per_row  = -np.ones(n_ch, dtype=int)

    # Fill only rows that exist in the recording
    for elec_idx0, row in enumerate(idx_rows):
        if 0 <= row < n_ch:
            ua_elec_per_row[row] = elec_idx0 + 1            # 1-based electrode number
            ua_nsp_per_row[row]  = int(mapped_nsp[elec_idx0])

    # Set per-channel properties (must be length n_ch)
    recording.set_property("ua_electrode", ua_elec_per_row)
    recording.set_property("ua_nsp_channel", ua_nsp_per_row)

    # Store the per-electrode -> row-index map as an annotation (any shape allowed)
    recording.set_annotation("ua_row_index_from_electrode", idx_rows.astype(int))

    mapped = int((ua_elec_per_row > 0).sum())
    print(f"[MAP] stamped UA mapping on {mapped}/{n_ch} rows (no geometry).")

# ---------- Bundles (NS5+NS2 only) ----------
def build_blackrock_bundle(sess: Path, camera_sync_ch, triangle_sync_ch) -> dict:
    """
    Build a bundle:
      - ns5: pull specific sync channels (Ex: 134=camera, 138=triangle) if present
      - ns2: pull ALL channels present
    """
    bundle: Dict[str, Any] = {}

    # ---------------- NS5 (syncs) ----------------
    intan_sync = None
    camera_sync = None
    triangle_sync = None
    fs_ns5 = None
    try:
        rec_ns5 = load_ns5_aux(sess)
        fs_ns5 = float(rec_ns5.get_sampling_frequency())

        # map channel id -> column index
        ids_ns5 = _ensure_int_ids(rec_ns5)
        id2col_ns5 = {int(ch): i for i, ch in enumerate(ids_ns5)}

        wanted_sync = {camera_sync_ch: "camera_sync", triangle_sync_ch: "triangle_sync"}
        present = [cid for cid in wanted_sync if cid in id2col_ns5]
        if not present:
            print("[WARN] ns5 present but camera_sync or triangle_sync not found.")

        if present:
            n_frames = int(rec_ns5.get_num_frames(0))
            tr_ns5 = rec_ns5.get_traces(0, 0, end_frame=n_frames).astype(np.float32)
            for cid in present:
                col = id2col_ns5[cid]
                if cid == camera_sync_ch:
                    camera_sync = tr_ns5[:, col]
                elif cid == triangle_sync_ch:
                    triangle_sync = tr_ns5[:, col]
    except FileNotFoundError:
        print("[WARN] ns5 not found; syncs unavailable.")
    bundle["aux"] = {
        "fs": fs_ns5,
        "intan_sync": intan_sync,
        "camera_sync": camera_sync,
        "triangle_sync": triangle_sync,
    }

    # ---------------- NS2 (all channels) ----------------
    digi = {"fs": None, "channels": {}}
    try:
        rec_ns2 = load_ns2_digi(sess)
        fs_ns2 = float(rec_ns2.get_sampling_frequency())
        digi["fs"] = fs_ns2

        ids_ns2 = _ensure_int_ids(rec_ns2)
        n_frames = int(rec_ns2.get_num_frames(0))
        tr_ns2 = rec_ns2.get_traces(0, 0, rec_ns2.get_num_frames(0)).astype(np.float32)

        # add ALL columns present
        for col, cid in enumerate(ids_ns2):
            digi["channels"][f"ch{int(cid)}"] = tr_ns2[:, col]
    except FileNotFoundError:
        print("[WARN] ns2 not found; no digital channels.")
    bundle["digi"] = digi

    return bundle

def save_UA_bundle_npz(sess_name: str, bundle: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    aux = bundle.get("aux", {})
    digi = bundle.get("digi", {})

    np.savez_compressed(
        out_dir / f"{sess_name}_UA_bundle.npz",
        aux_fs=np.array(aux.get("fs", np.nan), dtype=np.float64),
        intan_sync=aux.get("intan_sync", np.array([])),
        camera_sync=aux.get("camera_sync", np.array([])),
        triangle_sync=aux.get("triangle_sync", np.array([])),
        digi_fs=np.array(digi.get("fs", np.nan), dtype=np.float64),
        **{k: v for k, v in digi.get("channels", {}).items()},
    )
    print(f"[SAVED] {out_dir / f'{sess_name}_UA_bundle.npz'}")

## This function used by Intan too
def threshold_mua_rates(
    recording,
    detect_threshold: float,
    peak_sign: str,
    bin_ms: float,
    sigma_ms: float,
    n_jobs: int,
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

    # Hard-coded fields (OK if your SI version uses these names)
    ch_field   = "channel_index"
    samp_field = "sample_index"
    seg_field  = "segment_index"

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
    peak_t_ms = peak_samp_global.astype(np.float64) * (1000.0 / fs)

    # 2) Bin counts per segment
    counts_all, t_all = [], []
    sigma_bins = max(1e-9, sigma_ms / bin_ms)  # ms → bins
    bin_offset = 0

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
        t_ms = (np.arange(seg_bins, dtype=np.float64) + 0.5 + bin_offset) * bin_ms
        t_all.append(t_ms)
        bin_offset += seg_bins

    counts_cat = np.concatenate(counts_all, axis=1)
    t_cat_ms   = np.concatenate(t_all)

    # 3) Gaussian smoothing → Hz
    fs_bins = 1000.0 / float(bin_ms)  # bins per second
    if counts_cat.shape[1] < 4:
        counts_smooth = counts_cat.astype(float, copy=False)
    else:
        # Map Gaussian sigma to an approximate -3 dB cutoff:
        # For a Gaussian, H(f) = exp(-(2π f σ)^2 / 2); |H| = 1/√2 at f_c = sqrt(ln 2) / (2π σ)
        # Use sigma_ms to set a comparable Butterworth cutoff.
        sigma_sec = max(1e-9, float(sigma_ms) / 1000.0)
        fc_hz = (np.sqrt(np.log(2.0)) / (2.0 * np.pi * sigma_sec))  # ~0.132 / sigma_sec

        # Normalize for butter() (0..1 where 1 = Nyquist)
        nyq = 0.5 * fs_bins
        wn = min(0.999, max(1e-6, fc_hz / nyq))

        b, a = butter(N=4, Wn=wn, btype="low", analog=False)

        # filtfilt default padlen = 3*(max(len(a), len(b)) - 1)
        # If too short, reduce padlen safely.
        T = counts_cat.shape[1]
        default_padlen = 3 * (max(len(a), len(b)) - 1)
        padlen = default_padlen if T > default_padlen else max(0, T - 1)

        try:
            counts_smooth = filtfilt(b, a, counts_cat.astype(float, copy=False),
                                     axis=1, padlen=padlen)
        except ValueError:
            # If still too short/edgey, fall back to no filtering
            counts_smooth = counts_cat.astype(float, copy=False)

    rate_hz = counts_smooth * fs_bins  # counts/bin → Hz
    return rate_hz, t_cat_ms, counts_cat, peaks, peak_t_ms

__all__ = [
    "list_br_sessions", "ua_excel_path",
    "load_ns6_spikes", "load_ns5_aux", "load_ns2_digi",
    "load_UA_mapping_from_excel", "apply_ua_mapping_properties",
    "build_blackrock_bundle", "save_UA_bundle_npz",
    "threshold_mua_rates",
]