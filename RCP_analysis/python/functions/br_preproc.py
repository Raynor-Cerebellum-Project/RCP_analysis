from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import re
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from scipy.ndimage import gaussian_filter1d


# ---------- Session discovery ----------
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
    return _load_nsx(sess, "ns6")   # 30 kHz, 1..128

def load_ns5_aux(sess: Path):
    return _load_nsx(sess, "ns5")   # 30 kHz, 134 & 138

def load_ns2_digi(sess: Path):
    return _load_nsx(sess, "ns2")   # 1 kHz, 129..133,135..137,139..144

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
def build_blackrock_bundle(sess: Path) -> dict:
    """
    Build a bundle of non-NS6 streams -- TODO: Load this in NWB format
      - aux:  ns5 (camera_sync ch134, intan_sync ch138)
      - digi:  ns2 (channels 129..133,135..137,139..144)
    Returns a dict of signals + sampling rates
    """
    bundle: Dict[str, Any] = {}

    # ns5: camera_sync (134), intan_sync (138)
    camera_sync = None
    intan_sync  = None
    fs_ns5 = None
    try:
        rec_ns5 = load_ns5_aux(sess)
        fs_ns5 = float(rec_ns5.get_sampling_frequency())
        cols, missing = pick_cols_by_ids(rec_ns5, [134, 138])
        if missing:
            print(f"[WARN] ns5 missing channels: {missing}")
        if cols:
            tr = rec_ns5.get_traces(0, 0, rec_ns5.get_num_frames(0)).astype(np.float32)
            ids = _ensure_int_ids(rec_ns5)
            for col in cols:
                ch_id = int(ids[col])
                if ch_id == 134:
                    camera_sync = tr[:, col]
                elif ch_id == 138:
                    intan_sync = tr[:, col]
    except FileNotFoundError:
        print("[WARN] ns5 not found; aux syncs unavailable.")
    bundle["aux"] = {"fs": fs_ns5, "camera_sync": camera_sync, "intan_sync": intan_sync}

    # ns2: keep as ch129, ch130, ...
    digi_ch = {"fs": None, "channels": {}}
    try:
        rec_ns2 = load_ns2_digi(sess)
        fs_ns2 = float(rec_ns2.get_sampling_frequency())
        digi_ch["fs"] = fs_ns2
        wanted = [129,130,131,132,133,135,136,137,139,140,141,142,143,144]
        cols, missing = pick_cols_by_ids(rec_ns2, wanted)
        if missing:
            print(f"[WARN] ns2 missing channels: {missing}")
        if cols:
            tr = rec_ns2.get_traces(0, 0, rec_ns2.get_num_frames(0)).astype(np.float32)
            ids = _ensure_int_ids(rec_ns2)
            for col in cols:
                ch_id = int(ids[col])
                digi_ch["channels"][f"ch{ch_id}"] = tr[:, col]
    except FileNotFoundError:
        print("[WARN] ns2 not found; no digital channels.")
    bundle["digi"] = digi_ch

    return bundle

def save_UA_bundle_npz(sess_name: str, bundle: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{sess_name}_UA_bundle.npz",
        aux_fs=np.array(bundle["aux"]["fs"] if bundle["aux"]["fs"] is not None else np.nan, dtype=np.float64),
        camera_sync=bundle["aux"]["camera_sync"] if bundle["aux"]["camera_sync"] is not None else np.array([]),
        intan_sync=bundle["aux"]["intan_sync"] if bundle["aux"]["intan_sync"] is not None else np.array([]),
        digi_fs=np.array(bundle["digi"]["fs"] if bundle["digi"]["fs"] is not None else np.nan, dtype=np.float64),
        **{k: v for k, v in bundle["digi"]["channels"].items()},
    )
    print(f"[SAVED] {out_dir / f'{sess_name}_bundle.npz'}")

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
    rate_hz : np.ndarray
        (n_channels, n_bins) Gaussian-smoothed firing rate in Hz.
    t_ms : np.ndarray
        (n_bins,) bin-center times in ms.
    counts : np.ndarray
        (n_channels, n_bins) raw spike counts per bin.
    """
    fs = recording.get_sampling_frequency()
    n_ch = recording.get_num_channels()
    n_seg = recording.get_num_segments()
    bin_samps = int(round(bin_ms * 1e-3 * fs))
    if bin_samps < 1:
        raise ValueError("bin_ms too small for sampling rate")

    # 1) Detect peaks
    noise_levels = si.get_noise_levels(recording)
    peaks = detect_peaks(
        recording,
        method="by_channel_torch",
        detect_threshold=detect_threshold,
        peak_sign=peak_sign,
        noise_levels=noise_levels,
        n_jobs=n_jobs,
    )
    
    ch_field   = "channel_index"
    samp_field = "sample_index"
    seg_field  = "segment_index"

    # 2) Bin counts per segment
    counts_all, t_all = [], []
    sigma_bins = max(1e-9, sigma_ms / bin_ms)  # ms → bins
    bin_offset = 0

    for seg in range(n_seg):
        n_samps = recording.get_num_frames(seg)
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
            ch_idx = seg_peaks[ch_field].astype(np.int64)
            samp   = seg_peaks[samp_field].astype(np.int64)
            bins   = np.clip(samp // bin_samps, 0, seg_bins - 1)
            np.add.at(counts, (ch_idx, bins), 1)

        counts_all.append(counts)
        t_ms = (np.arange(seg_bins) + 0.5 + bin_offset) * bin_ms
        t_all.append(t_ms)
        bin_offset += seg_bins

    counts_cat = np.concatenate(counts_all, axis=1)
    t_cat_ms   = np.concatenate(t_all)

    # 3) Gaussian smoothing → Hz
    # Sampling rate in the binned domain:
    fs_bins = 1000.0 / float(bin_ms)  # bins per second

    if sigma_bins <= 0 or counts_cat.shape[1] < 2:
        # no smoothing or too short
        counts_smooth = counts_cat.astype(float, copy=False)
    else:
        # gaussian_filter1d is zero-phase (symmetric kernel) and fast
        # mode='nearest' avoids edge dips; adjust if you prefer 'reflect'
        counts_smooth = gaussian_filter1d(
            counts_cat.astype(float, copy=False),
            sigma=sigma_bins,
            axis=1,
            mode="nearest",
            truncate=4.0,   # ≈ 4σ kernel half-width (common default)
        )

    rate_hz = counts_smooth * fs_bins  # counts/bin → Hz
    return rate_hz, t_cat_ms, counts_cat


__all__ = [
    "list_br_sessions", "ua_excel_path",
    "load_ns6_spikes", "load_ns5_aux", "load_ns2_digi",
    "load_UA_mapping_from_excel", "apply_ua_mapping_properties",
    "build_blackrock_bundle", "save_UA_bundle_npz",
    "threshold_mua_rates",
]