from __future__ import annotations
from pathlib import Path
import numpy as np
import csv

# ---------- CONFIG ----------
from pathlib import Path
import spikeinterface.extractors as se
import RCP_analysis as rcp

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
DATA_ROOT = rcp.resolve_data_root(PARAMS)
OUT_BASE  = rcp.resolve_output_root(PARAMS); OUT_BASE.mkdir(parents=True, exist_ok=True)

# Template and output
TEMPLATE = REPO_ROOT / "config" / "br_intan_align_template.mat"

BR_ROOT       = (DATA_ROOT / PARAMS.blackrock_rel); BR_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = (DATA_ROOT / PARAMS.metadata_rel)
METADATA_ROOT = METADATA_CSV.parent
METADATA_ROOT.mkdir(parents=True, exist_ok=True)

NPRW_BUNDLES   = OUT_BASE / "bundles" / "NPRW"
NPRW_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW"

# Sync channels
NPRW_PROBE_CFG = (PARAMS.probes or {}).get("NPRW", {})
NPRW_CAMERA_SYNC_CH = int(NPRW_PROBE_CFG.get("triangle_sync_ch", 1))

UA_PROBE_CFG = (PARAMS.probes or {}).get("UA", {})
UA_CAMERA_SYNC_CH = int(UA_PROBE_CFG.get("camera_sync_ch", 134))
TRIANGLE_SYNC_CH = int(UA_PROBE_CFG.get("triangle_sync_ch", 138))
MATCH_WINDOW_MS = 1000.0

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)

def xcorr_normalized(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xz, yz = _z(x), _z(y)
    c = np.correlate(xz, yz, mode="full")
    if c.size:
        c /= (np.max(np.abs(c)) + 1e-12)
    lags = np.arange(-(y.size - 1), x.size)
    return c, lags

def peak_lags(corr: np.ndarray, lags: np.ndarray, height: float = 0.95) -> np.ndarray:
    try:
        from scipy.signal import find_peaks
        idx, _ = find_peaks(corr, height=height)
        return lags[idx].astype(int)
    except Exception:
        mid = np.arange(1, corr.size - 1)
        mask = (corr[mid-1] < corr[mid]) & (corr[mid] >= corr[mid+1]) & (corr[mid] >= height)
        return lags[mid[mask]].astype(int)

# ===========================
# Template matching on Intan (lock)
# ===========================

def load_template(template_mat_path: Path) -> np.ndarray:
    from scipy.io import loadmat
    m = loadmat(str(template_mat_path))
    if "template" not in m:
        raise KeyError("'template' not found in br_intan_align_template.mat")
    return np.asarray(m["template"], float).squeeze()

def find_locs_via_template(adc_lock: np.ndarray, template: np.ndarray, fs: float, peak=0.95) -> np.ndarray:
    corr, lags = xcorr_normalized(adc_lock, template)
    locs = peak_lags(corr, lags, height=peak)
    return locs

def build_br_window(
    rec_br,
    br_ch,
    fs_br: float,
    fs_intan: float,
    g: int,                 # Intan coarse anchor (loc)
    search_n: int,          # ± Intan samples you'll try
    big_window_sec: float = 20.0,   # make this large enough to include direction flips
    extra_pad_sec: float = 0.5,     # small safety pad on both sides
) -> tuple[int, int, np.ndarray]:
    """
    Build a large BR window centered at the mapped Intan time, with padding for the
    ±search_n Intan shifts (converted to BR samples). Returns [start:end) and br_win.
    """
    n_br = rec_br.get_num_frames()
    br_center = int(round(g * fs_br / fs_intan))

    # How much BR padding is needed so ±search_n (Intan samples) still fits:
    pad_br = int(np.ceil(search_n * (fs_br / fs_intan)))

    # Base half-width from the 'big' window + extra pad
    half = int(round((big_window_sec / 2.0) * fs_br))
    extra = int(round(extra_pad_sec * fs_br))
    half_total = half + pad_br + extra

    # Clamp to file bounds
    s = max(0, br_center - half_total)
    e = min(n_br, br_center + half_total)
    if e - s < 8:
        e = min(n_br, s + 8)

    br_win = rec_br.get_traces(start_frame=s, end_frame=e, channel_ids=[br_ch], return_in_uV=False)
    return s, e, br_win.squeeze().astype(float, copy=False)

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
      anchor_sample (Intan), delta_intan_samples, dt_ms, debug dict.
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
    anchor  = int(np.clip(loc + delta_i, 0, N_i - 1))
    dt_ms   = 1000.0 * delta_i / float(fs_intan)

    dbg = {
        "br_start": 0,
        "br_end": br_end,
        "best_n_br": int(best_n),
        "best_rms": float(best_rms),
        "Li": int(max(8, round(N_b * r))),
    }
    return anchor, int(delta_i), float(dt_ms), dbg

# ---------- discover Intan sessions (from NPRW rates) ----------
def session_from_rates_path(p: Path) -> str:
    stem = p.stem
    body = stem[len("rates__"):]
    return body.split("__bin", 1)[0]
    
def main():
    if not METADATA_CSV.exists():
        raise SystemExit(f"[error] mapping CSV not found: {METADATA_CSV}")

    rate_files = sorted(NPRW_CKPT_ROOT.rglob("rates__*.npz"))
    sessions = sorted({session_from_rates_path(p) for p in rate_files })
    if not sessions:
        raise SystemExit(f"[error] No NPRW rate files under {NPRW_CKPT_ROOT}")

    sess_to_intan_idx, intan_idx_to_sess = rcp.build_session_index_map(sessions)

    # ---------- mapping CSV ----------
    intan_to_br = rcp.read_intan_to_br_map(METADATA_CSV)
    print(f"[map] Loaded Intan→BR rows: {len(intan_to_br)}")

    template = load_template(TEMPLATE)

    # ---------- iterate pairs, compute anchor, write shifts CSV ----------
    summary_rows = []
    for intan_idx, br_idx in sorted(intan_to_br.items()):
        session = intan_idx_to_sess.get(intan_idx)
        if session is None:
            print(f"[warn] No session name for Intan_File={intan_idx} (skipping)")
            continue

        adc_npz = NPRW_BUNDLES / f"{session}_Intan_bundle" / "USB_board_ADC_input_channel.npz"
        if not adc_npz.exists():
            print(f"[warn] Intan ADC bundle missing: {adc_npz} (skip Intan {intan_idx} → BR {br_idx})")
            continue

        # Load Intan ADC
        try:
            adc_triangle, adc_lock, fs_intan = rcp.load_intan_adc(adc_npz)
        except Exception as e:
            print(f"[error] load_intan_adc failed for {adc_npz}: {e}")
            continue

        # Template-match on LOCK to get block starts
        locs = find_locs_via_template(adc_lock, template, fs_intan, peak=0.95)
        print(f"[locs] Intan {intan_idx:03d} peaks ≥0.95: {locs.size} | first 10: {locs[:10].tolist() if locs.size else []}")
        if locs.size == 0:
            print(f"[warn] No template peaks found for {session}; skipping.")
            continue
        
        # ---- load BR (lazy) ----
        fs_br = float("nan")
        rec_br = None
        br_ch = None
        dur_intan_sec = len(adc_triangle) / fs_intan
        dur_br_sec = float("nan")

        br_ns5 = BR_ROOT / f"{PARAMS.session}_{br_idx:03d}.ns5"
        use_br = False
        if br_ns5.exists():
            try:
                rec_br = se.read_blackrock(br_ns5)  # does not load all data
                fs_br = float(rec_br.get_sampling_frequency())
                # map TRIANGLE_SYNC_CH to a valid channel id
                ch_ids = np.array(rec_br.get_channel_ids())
                br_ch = None
                tri_id = str(TRIANGLE_SYNC_CH)

                if tri_id in ch_ids:
                    # TRIANGLE_SYNC_CH is an actual channel id (most robust path)
                    br_ch = tri_id
                elif 0 <= tri_id < ch_ids.size:
                    # treat it as a zero-based index into the list
                    br_ch = int(ch_ids[tri_id])

                use_br = (br_ch is not None)
                if not use_br:
                    print(f"[note] TRIANGLE_SYNC_CH={tri_id} not found in BR channels; skipping BR refinement.")
                try:
                    dur_br_sec = rec_br.get_num_frames() / fs_br
                except Exception:
                    pass
            except Exception as e:
                print(f"[note] BR open failed for {br_ns5}: {e}")
                rec_br, fs_br, use_br = None, float("nan"), False

        LOC_REFINE_N = 50        # like MATLAB's N
        locs_rows = []
        locs_rows_dt = []

        for b, loc in enumerate(locs.astype(int)):
            if use_br:
                g = int(loc)  # coarse Intan anchor from template on Intan
                search_n_br = int(round(LOC_REFINE_N * fs_br / fs_intan))  # same span but on BR grid
                ref_i, d_i, dt_ms, dbg = refine_anchor_with_loc_and_br0(
                    rec_br=rec_br, br_ch=br_ch, fs_br=fs_br,
                    adc_triangle_intan=adc_triangle, fs_intan=fs_intan,
                    loc=g, search_n_br=search_n_br, br_window_sec=20.0, normalize_both=False
                )
                # record ref_i as triangle_loc_sample and d_i as delta_samples

                locs_rows.append((b, g, int(ref_i), int(d_i)))
                locs_rows_dt.append(dt_ms)

                if b < 3:
                    print(f"[block {b}] BR[{dbg['br_start']}:{dbg['br_end']}) len={dbg['br_end']-dbg['br_start']} | "
                        f"Intan g={g} → ref_i={ref_i} (Δ_i={d_i:+d} samp, n_br={dbg['best_n_br']:+d}, {dt_ms:+.3f} ms) "
                        f"| Li={dbg['Li']} best_rms={dbg['best_rms']:.6g}")
            else:
                g = int(loc)
                locs_rows.append((b, g, g, 0))
                locs_rows_dt.append(float('nan'))
            break # TODO hacky fix

        # write per-block CSV (you reference locs_csv later)
        locs_csv = METADATA_ROOT / "template_locs" / f"intan_{intan_idx:03d}__template_locs.csv"
        locs_csv.parent.mkdir(parents=True, exist_ok=True)
        with locs_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["block_idx", "lock_loc_sample", "triangle_loc_sample", "delta_samples"])
            w.writerows(locs_rows)

        # initial/ refined anchor from block 0
        anchor0    = locs_rows[0][1]  # lock_loc_sample
        anchor     = locs_rows[0][2]  # triangle_loc_sample
        tri_delta  = locs_rows[0][3]  # delta_samples
        anchor_sec = anchor / fs_intan
        anchor_ms  = anchor_sec * 1000.0

        refined = np.array([r[2] for r in locs_rows], dtype=int)   # triangle_loc_sample (refined)
        deltas  = np.array([r[3] for r in locs_rows], dtype=int)   # delta_samples
        dt_ms   = np.array([d for d in locs_rows_dt if d == d], dtype=float)  # drop NaNs

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
            session=session,
            intan_idx=intan_idx,
            br_idx=br_idx,
            adc_npz=str(adc_npz),
            br_ns5=str(br_ns5),
            fs_intan=float(fs_intan),
            fs_br=float(fs_br),
            anchor_sample=int(anchor),
            anchor_seconds=float(anchor_sec),
            anchor_ms=float(anchor_ms),
            dur_intan_sec=float(dur_intan_sec),
            dur_br_sec=float(dur_br_sec),
            triangle_refined_from=int(anchor0),
            triangle_refine_delta_samples=int(tri_delta),
            n_locs=len(locs_rows),
            locs_csv=str(locs_csv),
        ))

        print(f"[anchor] Intan {intan_idx:03d} ↔ BR {br_idx:03d} : anchor={anchor:+d} samp ({anchor_sec:+.6f} s)")

    # ---------- write alignment summary CSV ----------
    if summary_rows:
        out_csv = METADATA_ROOT / "br_to_intan_shifts.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[done] wrote shifts → {out_csv}")
    else:
        print("[done] no rows to write (no anchors).")
 
if __name__ == "__main__":
    main()