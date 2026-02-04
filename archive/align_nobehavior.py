from typing import Any

import json, csv
import numpy as np
from pathlib import Path
from scipy.io import savemat

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# ---------- CONFIG ----------
SHIFTS_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"


IR_STREAM    = "USB board digital input channel"
def _load_br_to_intan_shifts(shifts_csv: Path) -> dict[int, dict]:
    """
    Return {BR_idx: {"session": <str>, "shifts_ms": <float or 0.0>}}.
    Accepts header variations.
    """
    if not shifts_csv.exists():
        raise SystemExit(f"[error] shifts CSV not found: {shifts_csv}")
    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise SystemExit(f"[error] {shifts_csv.name} has no header")
        # normalize header names -> original
        cols = { re.sub(r'[^a-z0-9]+','', c.lower()): c for c in rdr.fieldnames }

        def col(*cands):
            for cand in cands:
                k = re.sub(r'[^a-z0-9]+','', cand.lower())
                if k in cols: 
                    return cols[k]
            return None

        c_br    = col("br_idx")
        c_sess  = col("session")
        c_shift = col("shift_ms")

        if not c_br or not c_sess:
            raise SystemExit(
                f"[error] {shifts_csv.name} missing BR and/or Session columns "
                f"(have: {rdr.fieldnames})"
            )

        out: dict[int, dict] = {}
        for row in rdr:
            try:
                br = int(float(str(row[c_br]).strip()))
            except Exception:
                continue
            sess = str(row.get(c_sess, "")).strip()
            if not sess:
                continue
            try:
                shift_ms = float(str(row.get(c_shift, "0")).strip()) if c_shift else 0.0
            except Exception:
                shift_ms = 0.0
            out[br] = {"session": sess, "shift_ms": shift_ms}
    return out

# IR event loader
def _detect_IR_crossings(x: np.ndarray, fs: float | None, refractory_sec: float = 0.0005) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return np.array([], np.int64)

    # fill NaNs without changing indexing
    if not np.isfinite(x).all():
        med = np.nanmedian(x)
        x = np.nan_to_num(x, nan=med)

    lo, hi = float(np.min(x)), float(np.max(x))
    thr = 0.5 * (lo + hi)
    b = (x > thr).astype(np.int8)

    db = np.diff(b, prepend=b[0])
    edges = np.flatnonzero(db == -1)

    if edges.size == 0:
        return edges.astype(np.int64)

    refr = max(1, int(round(refractory_sec * fs))) if (fs and fs > 0) else 1
    keep = [edges[0]]
    for e in edges[1:]:
        if e - keep[-1] >= refr:
            keep.append(e)
    return np.asarray(keep, np.int64)

def _load_ir_ms_from_aligned(aligned_path: Path, meta: dict[str, Any]) -> np.ndarray:
    """
    Compute IR event times (in BR-aligned ms)
      - read the Intan IR stream for the matching Intan session
      - detect IR 1→0 edges
      - convert sample indices to seconds, then to BR-aligned ms
        using shift_ms from br_to_intan_shifts.csv.

    Returns:
        ir_ms_br : np.ndarray of shape (N,) in BR time
    """
    br_idx = int(meta.get("br_idx", -1))
    if br_idx < 0:
        print(f"[baseline-IR] {aligned_path.name}: meta has no valid br_idx; skipping IR.")
        return np.array([], float)

    shift_entry = BR_TO_INTAN_SHIFTS.get(br_idx)
    if shift_entry is None:
        print(f"[baseline-IR] BR {br_idx}: no entry in br_to_intan_shifts; skipping IR.")
        return np.array([], float)

    intan_session = shift_entry.get("session")
    shift_ms = float(shift_entry.get("shift_ms", 0.0))
    if not intan_session:
        print(f"[baseline-IR] BR {br_idx}: empty Intan session; skipping IR.")
        return np.array([], float)

    intan_dir = INTAN_ROOT / intan_session
    try:
        rec_ir = se.read_split_intan_files(
            intan_dir,
            mode="concatenate",
            stream_name=IR_STREAM,
            use_names_as_ids=True,
        )
        rec_ir = spre.unsigned_to_signed(rec_ir)  # UInt16 -> int16
    except Exception as e:
        print(f"[baseline-IR] Reading IR stream failed for Intan={intan_session}: {e}")
        return np.array([], float)

    fs = float(rec_ir.sampling_frequency)
    sig = np.asarray(rec_ir.get_traces()).squeeze()
    if sig is None or sig.size == 0:
        print(f"[baseline-IR] Intan={intan_session}: empty IR signal.")
        return np.array([], float)

    ir_idx = _detect_IR_crossings(sig, fs, refractory_sec=0.0005)
    if ir_idx.size == 0:
        print(f"[baseline-IR] Intan={intan_session}: no IR crossings.")
        return np.array([], float)

    # Intan seconds -> BR-aligned ms (same as original script: subtract shift_ms)
    ir_sec = ir_idx / fs
    ir_ms_br = ir_sec * 1000.0 - shift_ms

    print(
        f"[baseline-IR] {aligned_path.name}: BR={br_idx}, Intan={intan_session}, "
        f"shift_ms={shift_ms:g}, n_IR={ir_ms_br.size}"
    )
    return ir_ms_br.astype(float)



def _parse_SI_peaks(peaks) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    peak_times = {}
    peak_amp = {}
    for i in np.unique(peaks["channel_index"]):
        peak_ch_times = peaks["sample_index"][peaks["channel_index"] == i]
        peak_ch_amp   = peaks["amplitude"][peaks["channel_index"] == i]
        peak_times[int(i)] = peak_ch_times
        peak_amp[int(i)]   = peak_ch_amp
    return peak_times, peak_amp

def _ua_perm_by_region(
    ua_region: np.ndarray,
    ua_elec: np.ndarray,
    ua_nsp: np.ndarray,
    region_order=(0, 1, 2, 3),
    within="nsp",  # or "elec"
) -> np.ndarray:
    ua_region = np.asarray(ua_region).astype(int)
    ua_elec   = np.asarray(ua_elec).astype(int)
    ua_nsp    = np.asarray(ua_nsp).astype(int)

    if within == "nsp":
        key = ua_nsp
    elif within == "elec":
        key = ua_elec
    else:
        raise ValueError("within must be 'nsp' or 'elec'")

    valid = ua_region >= 0
    rows = np.flatnonzero(valid)

    perm_parts = []
    for r in region_order:
        rr = rows[ua_region[rows] == r]
        rr = rr[np.argsort(key[rr], kind="stable")]
        perm_parts.append(rr)

    perm = np.concatenate(perm_parts) if perm_parts else np.array([], dtype=int)

    inv = np.flatnonzero(~valid)
    if inv.size:
        perm = np.concatenate([perm, inv])

    return perm

def _change_none_for_mat(d):
    """Return a dict with all None values removed. Non-dicts -> empty array."""
    if d is None:
        return np.array([])
    if not isinstance(d, dict):
        return np.array([])
    out = {}
    for k, v in d.items():
        if v is not None:
            out[k] = v
    return out

def main():
    if not SHIFTS_CSV.exists():
        raise SystemExit(f"[error] shifts CSV not found: {SHIFTS_CSV}")

    # ---------- read shifts ----------
    with SHIFTS_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    if not rows:
        raise SystemExit("[error] shifts CSV has no rows")

    for row in rows:
        try:
            session   = row["session"]
            intan_idx = int(row["intan_idx"])
            br_idx    = int(row["br_idx"])
            fs_nprw   = float(row.get("fs_intan", 30000.0))
            shift_ms  = float(row["shift_ms"])

            # ---------- load NPRW npz ----------
            cands = sorted(NPRW_CKPT_ROOT.glob(f"rates__{session}__*.npz"))
            if not cands:
                print(f"[warn] No NPRW rates for session {session}")
                continue
            nprw_rates_npz = cands[0]

            nprw_npz   = np.load(nprw_rates_npz, allow_pickle=True)
            nprw_peaks = nprw_npz["peaks"]
            nprw_meta  = nprw_npz["meta"].item() if hasattr(nprw_npz["meta"], "item") else nprw_npz["meta"]

            # shift NPRW meta time bounds (if present)
            if "rec_start_ms" in nprw_meta:
                nprw_meta["rec_start_ms_aligned"] = nprw_meta["rec_start_ms"] - shift_ms
            if "rec_end_ms" in nprw_meta:
                nprw_meta["rec_end_ms_aligned"]   = nprw_meta["rec_end_ms"] - shift_ms

            nprw_peak_samps, nprw_peak_amps = _parse_SI_peaks(nprw_peaks)
            nprw_peak_ms = {
                ch: samps / fs_nprw * 1000.0 - shift_ms
                for ch, samps in nprw_peak_samps.items()
            }

            # ---------- load UA npz ----------
            cands = sorted(UA_CKPT_ROOT.glob(f"rates__NRR_RW*_{br_idx:03d}__*.npz"))
            if not cands:
                print(f"[warn] No UA rates for BR {br_idx:03d} (session {session})")
                continue
            ua_rates_npz_loc = cands[0]

            ua_npz   = np.load(ua_rates_npz_loc, allow_pickle=True)
            ua_peaks = ua_npz["peaks"]
            ua_meta  = ua_npz["meta"].item()
            fs_ua    = float(ua_meta.get("fs_ua", np.nan))

            ua_elec     = ua_npz["ua_elec"]
            ua_nsp      = ua_npz["ua_nsp"]
            ua_idx_rows = ua_npz["ua_index"]
            ua_region   = ua_npz["ua_region"]
            ua_region_names = ua_npz["ua_region_names"]
            ua_port     = ua_npz["ua_port"]

            # optional reordering by region
            perm = _ua_perm_by_region(ua_region, ua_elec, ua_nsp, region_order=(0,1,2,3), within="nsp")
            n_ch = int(len(ua_nsp))
            inv_perm = np.empty(n_ch, dtype=int)
            inv_perm[perm] = np.arange(n_ch)

            ua_peaks = ua_peaks.copy()
            ua_peaks["channel_index"] = inv_perm[ua_peaks["channel_index"].astype(int)]

            ua_elec     = np.asarray(ua_elec)[perm]
            ua_nsp      = np.asarray(ua_nsp)[perm]
            ua_region   = np.asarray(ua_region)[perm]
            ua_idx_rows = np.asarray(ua_idx_rows)[perm]

            ua_peak_samps, ua_peak_amps = _parse_SI_peaks(ua_peaks)
            ua_peak_ms = {
                ch: samps / fs_ua * 1000.0
                for ch, samps in ua_peak_samps.items()
            }

            # ---------- stim times (absolute Intan ms, shifted to aligned NPRW ms) ----------
            stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
            stim_ms = np.array([], dtype=np.float32)

            if stim_npz_path is not None and Path(stim_npz_path).exists():
                stim = rcp.load_stim_detection(stim_npz_path)
                stim_ms = stim["block_bounds_samples"][:, 0].astype(np.float32) * (1000.0 / fs_nprw) - shift_ms
            else:
                print(f"[warn] stim npz not found for BR {br_idx:03d}: {stim_npz_path}")

            aligned_meta = dict(
                session=session,
                intan_idx=intan_idx,
                br_idx=br_idx,
                nprw_rates=str(nprw_rates_npz),
                ua_rates=str(ua_rates_npz_loc),
                fs_nprw=float(fs_nprw),
                fs_ua=float(fs_ua) if np.isfinite(fs_ua) else None,
                shift_ms=float(shift_ms),
            )

            out_npz = ALIGNED_CKPT_ROOT / f"aligned__{session}__Intan_{intan_idx:03d}__BR_{br_idx:03d}.npz"
            np.savez_compressed(
                out_npz,
                nprw_meta=(nprw_meta.item() if hasattr(nprw_meta, "item") else nprw_meta),
                nprw_peak_ms=nprw_peak_ms,
                nprw_peak_amps=nprw_peak_amps,

                ua_meta=(ua_meta.item() if hasattr(ua_meta, "item") else ua_meta),
                ua_peak_ms=ua_peak_ms,
                ua_peak_amps=ua_peak_amps,

                ua_elec=ua_elec,
                ua_region=ua_region,
                ua_region_names=ua_region_names,
                ua_port=ua_port,
                ua_nsp=ua_nsp,
                ua_idx_rows=ua_idx_rows,

                stim_ms=stim_ms.astype(np.float32),
                shift_ms=float(shift_ms),

                align_meta=json.dumps(aligned_meta),
            )

            print(f"[write] combined aligned (no HR/VOG/behavior): {out_npz}")

        except Exception as e:
            print(f"[error] Failed for session {row.get('session','?')}: {e}")
            continue

if __name__ == "__main__":
    main()
