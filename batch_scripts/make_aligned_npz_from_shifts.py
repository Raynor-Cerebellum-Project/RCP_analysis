from __future__ import annotations
from pathlib import Path
import numpy as np
import json, csv
from RCP_analysis import load_experiment_params, resolve_output_root

# ---------- CONFIG ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE  = resolve_output_root(PARAMS)
OUT_BASE.mkdir(parents=True, exist_ok=True)

NPRW_BUNDLES   = OUT_BASE / "bundles" / "NPRW"
NPRW_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW"
UA_CKPT_ROOT   = OUT_BASE / "checkpoints" / "UA"
ALIGNED_CKPT_ROOT   = OUT_BASE / "checkpoints" / "Aligned"
ALIGNED_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

METADATA_ROOT        = OUT_BASE.parent / "Metadata"
METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    
# ===========================
# Helpers
# ===========================

def _meta_fs(meta):
    if meta is None:
        return None
    if hasattr(meta, "item"):
        meta = meta.item()
    if isinstance(meta, (str, bytes)):
        try:
            meta = json.loads(meta)
        except Exception:
            return None
    if isinstance(meta, dict):
        return float(meta.get("fs_hz", meta.get("fs", np.nan)))
    return None
            
def load_rate_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    rate_hz = d["rate_hz"]           # (n_ch, n_bins)
    t_ms    = d["t_ms"]              # (n_bins,)
    meta    = d.get("meta", None)
    pcs     = d.get("pcs", None)
    peaks     = d.get("peaks", None)
    peaks_t_ms     = d.get("peak_t_ms", None) # TODO change to peaks_t_ms later
    peaks_t_sample     = d.get("peak_sample", None)
    explained_var = d.get("explained_var", None)
    return rate_hz, t_ms, meta, pcs, peaks, peaks_t_ms, peaks_t_sample, explained_var

def find_intan_rates_for_session(nprw_ckpt_root: Path, session: str) -> Path | None:
    cands = sorted(nprw_ckpt_root.glob(f"rates__{session}__*.npz"))
    return cands[0] if cands else None

def find_ua_rates_by_index(ua_ckpt_root: Path, br_idx: int) -> Path | None:
    patt = f"rates__NRR_RW_001_{br_idx:03d}__*.npz"
    cands = sorted(ua_ckpt_root.glob(patt))
    if not cands:
        return None
    pref = [p for p in cands if "__sigma25ms" in p.stem]
    return pref[0] if pref else cands[-1]

def try_load_stim_ms_from_intan_bundle(bundles_root: Path, session: str) -> np.ndarray | None:
    stim_npz = bundles_root / f"{session}_Intan_bundle" / "stim_stream.npz"
    if not stim_npz.exists():
        return None
    with np.load(stim_npz, allow_pickle=False) as z:
        if "block_bounds_samples" not in z or "meta" not in z:
            return None
        blocks = z["block_bounds_samples"].astype(np.int64)
        meta_json = z["meta"].item() if hasattr(z["meta"], "item") else z["meta"]
        meta = json.loads(meta_json)
        fs_hz = float(meta.get("fs_hz", 30000.0))
        onset_samples = blocks[:,0]
        return onset_samples.astype(float) * (1000.0 / fs_hz)

def _resample_rate_to_time(rate_hz: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """
    Resample per-channel rate traces from t_src -> t_tgt using linear interpolation.
    rate_hz: (n_ch, n_src); t_src, t_tgt in ms, strictly increasing.
    Returns (n_ch, n_tgt).
    """
    n_ch = rate_hz.shape[0]
    out = np.empty((n_ch, t_tgt.size), dtype=np.float32)
    # Use float64 in interp for numerical stability
    r64 = rate_hz.astype(np.float64, copy=False)
    for ch in range(n_ch):
        out[ch] = np.interp(t_tgt, t_src, r64[ch])
    return out

def _overlap_and_resample_to_intan(i_rate, i_t, u_rate, u_t):
    """
    1) Find overlap [t0, t1] between Intan and UA time vectors (already aligned).
    2) Take Intan's bins inside the overlap as the common grid (common_t).
    3) Crop UA to cover that range and resample UA onto common_t.
    Returns:
        i_rate_trim  (n_ch_i, n_bins_common)
        common_t     (n_bins_common,)
        u_rate_on_common (n_ch_u, n_bins_common)
        stats dict   (for logging/metadata)
    """
    if i_t.size == 0 or u_t.size == 0:
        return i_rate[:, :0], i_t[:0], u_rate[:, :0], {
            "overlap_bins": 0, "dt_i_ms": np.nan, "dt_u_ms": np.nan
        }

    t0 = max(float(i_t[0]), float(u_t[0]))
    t1 = min(float(i_t[-1]), float(u_t[-1]))
    if not (t1 > t0):
        return i_rate[:, :0], i_t[:0], u_rate[:, :0], {
            "overlap_bins": 0, "dt_i_ms": np.nan, "dt_u_ms": np.nan
        }

    # Intan: keep bins fully inside the overlap and use them as the target grid
    mi = (i_t >= t0) & (i_t <= t1)
    common_t = i_t[mi]
    i_rate_trim = i_rate[:, mi]

    if common_t.size == 0:
        return i_rate[:, :0], i_t[:0], u_rate[:, :0], {
            "overlap_bins": 0, "dt_i_ms": np.nan, "dt_u_ms": np.nan
        }

    # UA: crop to at least cover [common_t[0], common_t[-1]] then resample
    mu = (u_t >= common_t[0]) & (u_t <= common_t[-1])
    u_rate_crop = u_rate[:, mu]
    u_t_crop = u_t[mu]
    if u_t_crop.size < 2:
        # not enough samples to interpolate
        return i_rate[:, :0], i_t[:0], u_rate[:, :0], {
            "overlap_bins": 0, "dt_i_ms": np.nan, "dt_u_ms": np.nan
        }

    u_rate_on_common = _resample_rate_to_time(u_rate_crop, u_t_crop, common_t)

    dt_i = float(np.median(np.diff(common_t))) if common_t.size > 1 else np.nan
    dt_u = float(np.median(np.diff(u_t_crop))) if u_t_crop.size > 1 else np.nan
    stats = {"overlap_bins": int(common_t.size), "dt_i_ms": dt_i, "dt_u_ms": dt_u}
    return i_rate_trim, common_t, u_rate_on_common, stats

def main():
    shifts_csv     = METADATA_ROOT / "br_to_intan_shifts.csv"
    if not shifts_csv.exists():
        raise SystemExit(f"[error] shifts CSV not found: {shifts_csv}")

    # ---------- read shifts ----------
    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    if not rows:
        raise SystemExit("[error] shifts CSV has no rows")

    for row in rows:
        try:
            session   = row["session"]
            intan_idx = int(row["intan_idx"])
            br_idx    = int(row["br_idx"])
            fs_intan  = float(row.get("fs_intan", 30000.0))
            anchor_ms = float(row["anchor_ms"])

            # --------- Load rates npz for BOTH sides ----------
            intan_rates_npz = find_intan_rates_for_session(NPRW_CKPT_ROOT, session)
            if intan_rates_npz is None:
                print(f"[warn] No Intan rates for session {session}")
                continue
            ua_rates_npz = find_ua_rates_by_index(UA_CKPT_ROOT, br_idx)
            if ua_rates_npz is None:
                print(f"[warn] No UA rates for BR_File {br_idx:03d}")
                continue
            # --- pull mapping arrays from the UA rates npz ---
            ua_row_to_elec = ua_row_to_region = ua_region_names = ua_row_to_nsp = ua_row_index_from_electrode = None
            with np.load(ua_rates_npz, allow_pickle=True) as z:
                ua_row_to_elec = z["ua_row_to_elec"].astype(np.int16) if "ua_row_to_elec" in z else np.array([], dtype=np.int16)
                ua_row_to_region = z["ua_row_to_region"].astype(np.int8) if "ua_row_to_region" in z else np.array([], dtype=np.int8)
                ua_region_names = z["ua_region_names"] if "ua_region_names" in z else np.array([], dtype=object)
                ua_row_to_nsp = z["ua_row_to_nsp"].astype(np.int16) if "ua_row_to_nsp" in z else np.array([], dtype=np.int16)
                ua_row_index_from_electrode = (
                    z["ua_row_index_from_electrode"].astype(np.int16)
                    if "ua_row_index_from_electrode" in z else np.array([], dtype=np.int16)
                )

            intan_rate_hz, intan_t_ms, intan_meta, intan_pcs, intan_peaks, intan_peaks_t_ms, intan_peaks_t_sample, intan_expl = load_rate_npz(intan_rates_npz)
            ua_rate_hz, ua_t_ms, ua_meta, ua_pcs, ua_peaks, ua_peaks_t_ms, ua_peaks_t_sample, ua_expl = load_rate_npz(ua_rates_npz)

            # Optional stim times (absolute Intan ms)
            stim_ms_abs = try_load_stim_ms_from_intan_bundle(NPRW_BUNDLES, session)

            # --------- Apply Intan anchor shift to timebase ----------
            intan_t_ms_aligned = intan_t_ms - anchor_ms
            ua_t_ms_aligned    = ua_t_ms  # no shift yet
            
            # Intan peaks → ms
            intan_peaks_t_ms_raw = None
            if intan_peaks_t_ms is not None:
                intan_peaks_t_ms_raw = np.asarray(intan_peaks_t_ms, dtype=np.float64)
            elif intan_peaks_t_sample is not None:
                # prefer fs_intan from CSV/row; else from meta
                _fs_i = fs_intan if np.isfinite(fs_intan) else _meta_fs(intan_meta)
                if _fs_i and np.isfinite(_fs_i):
                    intan_peaks_t_ms_raw = (np.asarray(intan_peaks_t_sample, dtype=np.float64) / float(_fs_i)) * 1000.0

            # UA peaks → ms
            ua_peaks_t_ms_raw = None
            if ua_peaks_t_ms is not None:
                ua_peaks_t_ms_raw = np.asarray(ua_peaks_t_ms, dtype=np.float64)
            elif ua_peaks_t_sample is not None:
                _fs_u = _meta_fs(ua_meta)
                if not (_fs_u and np.isfinite(_fs_u)):
                    # derive from UA time base if needed
                    if ua_t_ms.size > 1:
                        _fs_u = 1000.0 / float(np.median(np.diff(ua_t_ms)))
                if _fs_u and np.isfinite(_fs_u):
                    ua_peaks_t_ms_raw = (np.asarray(ua_peaks_t_sample, dtype=np.float64) / float(_fs_u)) * 1000.0

            # Aligned (Intan timebase is shifted by anchor_ms in your pipeline)
            intan_peaks_t_ms_aligned = (intan_peaks_t_ms_raw - float(anchor_ms)) if intan_peaks_t_ms_raw is not None else np.array([], dtype=np.float64)
            ua_peaks_t_ms_aligned    = ua_peaks_t_ms_raw if ua_peaks_t_ms_raw is not None else np.array([], dtype=np.float64)

            # Keep original peak payloads too (or empty arrays for consistency)
            _intan_peaks          = intan_peaks if intan_peaks is not None else np.array([], dtype=np.float32)
            _intan_peaks_t_ms     = intan_peaks_t_ms if intan_peaks_t_ms is not None else np.array([], dtype=np.float64)
            _intan_peaks_t_sample = intan_peaks_t_sample if intan_peaks_t_sample is not None else np.array([], dtype=np.float64)

            _ua_peaks             = ua_peaks if ua_peaks is not None else np.array([], dtype=np.float32)
            _ua_peaks_t_ms        = ua_peaks_t_ms if ua_peaks_t_ms is not None else np.array([], dtype=np.float64)
            _ua_peaks_t_sample    = ua_peaks_t_sample if ua_peaks_t_sample is not None else np.array([], dtype=np.float64)

            # ---- Overlap and resample UA onto Intan's aligned grid ----
            orig_i_len, orig_u_len = intan_t_ms_aligned.size, ua_t_ms_aligned.size
            (i_rate_trim, common_t, u_rate_on_common, stats) = _overlap_and_resample_to_intan(
                intan_rate_hz, intan_t_ms_aligned, ua_rate_hz, ua_t_ms_aligned
            )

            if common_t.size == 0:
                print(f"[warn] No overlapping time region after alignment for session {session}; skipping.")
                continue

            print(f"[resample] {session}: overlap bins={stats['overlap_bins']}, "
                  f"Intan dt≈{stats['dt_i_ms']:.3f} ms, UA dt≈{stats['dt_u_ms']:.3f} ms → UA→Intan grid")

            combined_meta = dict(
                session=session,
                intan_idx=intan_idx,
                br_idx=br_idx,
                intan_rates=str(intan_rates_npz),
                ua_rates=str(ua_rates_npz),
                fs_intan=float(fs_intan),

                # Anchor info (from CSV)
                anchor_ms=float(anchor_ms),

                # Overlap/resample info
                equal_time_grid=True,
                resampled_to="intan",
                orig_intan_bins=int(orig_i_len),
                orig_ua_bins=int(orig_u_len),
                trimmed_bins=int(common_t.size),
                dt_ms_target=float(stats["dt_i_ms"]),
            )

            out_npz = ALIGNED_CKPT_ROOT / f"aligned__{session}__Intan_{intan_idx:03d}__BR_{br_idx:03d}.npz"
            np.savez_compressed(
                out_npz,
                
                # Intan (on common grid)
                intan_rate_hz=i_rate_trim.astype(np.float32),
                intan_t_ms=intan_t_ms.astype(np.float64),                 # original (pre-shift)
                intan_t_ms_aligned=common_t.astype(np.float64),           # Intan-aligned/common grid
                intan_meta=(intan_meta.item() if hasattr(intan_meta, "item") else intan_meta),
                intan_pcs=intan_pcs if intan_pcs is not None else np.array([], dtype=np.float32),
                intan_explained_var=intan_expl if intan_expl is not None else np.array([], dtype=np.float32),

                # Intan peaks (raw + aligned)
                intan_peaks=_intan_peaks,
                intan_peaks_t_ms=_intan_peaks_t_ms,
                intan_peaks_t_sample=_intan_peaks_t_sample,
                intan_peaks_t_ms_aligned=intan_peaks_t_ms_aligned.astype(np.float64),

                # UA (resampled to common grid)
                ua_rate_hz=u_rate_on_common.astype(np.float32),
                ua_t_ms=ua_t_ms.astype(np.float64),                       # original
                ua_t_ms_aligned=common_t.astype(np.float64),              # same common grid
                ua_meta=(ua_meta.item() if hasattr(ua_meta, "item") else ua_meta),
                ua_pcs=ua_pcs if ua_pcs is not None else np.array([], dtype=np.float32),
                ua_explained_var=ua_expl if ua_expl is not None else np.array([], dtype=np.float32),

                ua_row_to_elec=ua_row_to_elec,
                ua_row_to_region=ua_row_to_region,
                ua_region_names=ua_region_names,
                ua_row_to_nsp=ua_row_to_nsp,
                ua_row_index_from_electrode=ua_row_index_from_electrode,

                # UA peaks (raw + "aligned" = same as raw in ms since UA wasn't shifted)
                ua_peaks=_ua_peaks,
                ua_peaks_t_ms=_ua_peaks_t_ms,
                ua_peaks_t_sample=_ua_peaks_t_sample,
                ua_peaks_t_ms_aligned=ua_peaks_t_ms_aligned.astype(np.float64),

                # Stim (absolute Intan ms; consumer should subtract anchor_ms as needed)
                stim_ms=(stim_ms_abs.astype(np.float64) if stim_ms_abs is not None else np.array([], dtype=np.float64)),

                # Alignment meta (as JSON)
                align_meta=json.dumps(combined_meta),
            )

            print(f"[write] combined aligned → {out_npz}")

        except Exception as e:
            print(f"[error] Failed for session {row.get('session','?')}: {e}")
            continue

if __name__ == "__main__":
    main()