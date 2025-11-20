from pathlib import Path
import numpy as np
import json, csv, re
import RCP_analysis as rcp

# ---------- CONFIG ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)
NPRW_BUNDLES   = OUT_BASE / "bundles" / "NPRW"
NPRW_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW"
UA_CKPT_ROOT   = OUT_BASE / "checkpoints" / "UA"
ALIGNED_CKPT_ROOT   = OUT_BASE / "checkpoints" / "Aligned"; ALIGNED_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
BEHV_CKPT_ROOT = OUT_BASE / "checkpoints" / "behavior"; BEHV_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

NUM_CAM = PARAMS.kinematics.get("num_camera", 1)

METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"
SHIFTS_CSV    = METADATA_ROOT / "br_to_intan_shifts.csv"

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

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def _load_br_to_video_map(meta_csv: Path) -> dict[int, int]:
    """
    Returns { br_idx -> video_idx } from METADATA_CSV.
    Tolerates Excel encodings and light header/name variation.
    If duplicate BR_File rows map to different Video_File values, the last row wins (and we warn).
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman")
    last_err = None
    for enc in encodings:
        try:
            with meta_csv.open("r", newline="", encoding=enc, errors="strict") as f:
                rdr = csv.DictReader(f)
                cols = { _norm(c): c for c in (rdr.fieldnames or []) }

                def _col(cand: str) -> str:
                    k = _norm(cand)
                    if k in cols:
                        return cols[k]
                    raise KeyError(f"Missing column {cand}; found {list(cols.values())}")

                col_br_file   = _col("br_file")
                col_videofile = _col("video_file")

                out: dict[int, int] = {}
                for row in rdr:
                    try:
                        br_raw  = str(row[col_br_file]).strip()
                        vid_raw = str(row[col_videofile]).strip()
                        if not br_raw or not vid_raw:
                            continue
                        br  = int(float(br_raw))
                        vid = int(float(vid_raw))
                        if br in out and out[br] != vid:
                            print(f"[warn] BR_File {br:03d} appears multiple times with different Video_File "
                                  f"({out[br]} -> {vid}); using the last one.")
                        out[br] = vid
                    except Exception:
                        continue
                if not out:
                    print(f"[warn] No BR_File→Video_File pairs found in {meta_csv.name} using {enc}.")
                else:
                    if enc != "utf-8":
                        print(f"[info] Read {meta_csv.name} with encoding={enc}.")
                return out
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise UnicodeDecodeError(
        f"Could not decode {meta_csv} with tried encodings {encodings}. Last error: {last_err}"
    )

# ---------- Behavior CSV helpers ----------
# Matches:
#   ..._<BRIDX>_both_cams_aligned.csv
#   ..._<BRIDX>_Cam-0_aligned.csv
#   ..._<BRIDX>_Cam-1_aligned.csv
_BOTH_RE = re.compile(r"_(\d{3})_both_cams_aligned\.csv$", re.IGNORECASE)
_CAM_RE  = re.compile(r"_(\d{3})_Cam-([01])_aligned\.csv$", re.IGNORECASE)

def _find_behavior_files_for_br_idx(behv_root: Path, br_idx: int) -> dict[str, Path | None]:
    """
    Return newest matching files for the given br_idx:
      {
        'both': Path | None,
        'cam0': Path | None,
        'cam1': Path | None
      }
    """
    out = {'both': None, 'cam0': None, 'cam1': None}
    newest = {'both': -1.0, 'cam0': -1.0, 'cam1': -1.0}

    for p in behv_root.rglob("*.csv"):
        name = p.name

        m = _BOTH_RE.search(name)
        if m and int(m.group(1)) == int(br_idx):
            mt = p.stat().st_mtime
            if mt > newest['both']:
                out['both'] = p; newest['both'] = mt
            continue

        m = _CAM_RE.search(name)
        if m and int(m.group(1)) == int(br_idx):
            cam = int(m.group(2))
            key = 'cam0' if cam == 0 else 'cam1'
            mt = p.stat().st_mtime
            if mt > newest[key]:
                out[key] = p; newest[key] = mt

    return out

def _choose_behavior_csv(files: dict[str, Path | None]) -> tuple[Path | None, str]:
    """
    Preference: both → cam0 → cam1. Returns (Path|None, tag: 'both'|'cam0'|'cam1'|'none')
    """
    if files.get('both') is not None:
        return files['both'], 'both'
    if files.get('cam0') is not None:
        return files['cam0'], 'cam0'
    if files.get('cam1') is not None:
        return files['cam1'], 'cam1'
    return None, 'none'

def _behavior_times_ms_from_sources(
    beh_ns5_sample: np.ndarray,
    fs_br: float,
    *,
    frame_index: np.ndarray | None = None,
    fps_hint: float | None = None,
    explicit_ms: np.ndarray | None = None,
) -> np.ndarray:
    """
    Priority:
      1) explicit per-frame timestamps in ms (if provided)
      2) ns5-sample indices / fs_br  (robust to variable FPS if samples really are BR samples)
      3) frame_index + fps_hint      (LAST resort; assumes constant FPS)

    Returns a float32 array (ms); empty if nothing usable.
    """
    # 1) Explicit timestamps (best)
    if explicit_ms is not None and explicit_ms.size:
        t = np.asarray(explicit_ms, dtype=np.float32)
        return t

    # 2) Blackrock sample indices (robust, preferred)
    if beh_ns5_sample is not None and beh_ns5_sample.size and np.isfinite(fs_br) and fs_br > 0:
        s = np.asarray(beh_ns5_sample, dtype=np.float64)
        # guard for “looks like frame counters” (tiny integers growing by 1)
        diffs = np.diff(s) if s.size > 1 else np.array([], dtype=np.float64)
        looks_like_samples = (s.max() > 10_000) or (np.median(diffs) > 10)  # heuristic
        if looks_like_samples:
            return (s * (1000.0 / float(fs_br))).astype(np.float32)
        # else fall through to 3

    # 3) Frame index + FPS (only if we must)
    if frame_index is not None and frame_index.size and fps_hint and np.isfinite(fps_hint) and fps_hint > 0:
        fi = np.asarray(frame_index, dtype=np.float64)
        t0 = 0.0
        return (t0 + (fi / float(fps_hint)) * 1000.0).astype(np.float32)

    # nothing usable
    return np.array([], dtype=np.float32)

def _log_frame_jitter(beh_ns5_sample: np.ndarray, fs_br: float, label: str = "behavior"):
    """Optional: log per-frame interval stats from BR samples → ms (shows FPS jitter)."""
    if beh_ns5_sample.size > 2 and np.isfinite(fs_br) and fs_br > 0:
        s = beh_ns5_sample.astype(np.float64)
        d_ms = np.diff(s) * (1000.0 / float(fs_br))
        med = float(np.median(d_ms)); avg = float(np.mean(d_ms))
        sd  = float(np.std(d_ms));    mn  = float(np.min(d_ms)); mx = float(np.max(d_ms))
        print(f"[{label}] dt_ms: median={med:.3f}, mean={avg:.3f}, std={sd:.3f}, "
              f"min={mn:.3f}, max={mx:.3f}, n={d_ms.size}")

def main():
    if not SHIFTS_CSV.exists():
        raise SystemExit(f"[error] shifts CSV not found: {SHIFTS_CSV}")

    # ---------- read shifts ----------
    with SHIFTS_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    br2video = _load_br_to_video_map(METADATA_CSV)

    if not rows:
        raise SystemExit("[error] shifts CSV has no rows")

    for row in rows:
        try:
            session   = row["session"]
            intan_idx = int(row["intan_idx"])
            br_idx    = int(row["br_idx"])
            fs_intan  = float(row.get("fs_intan", 30000.0))
            anchor_ms = float(row["anchor_ms"])
            video_idx = br2video.get(br_idx)
            if video_idx is None:
                print(f"[warn] Video_File not found in metadata for BR {br_idx:03d}; "
                    f"falling back to BR index.")
                video_idx = br_idx

            files = _find_behavior_files_for_br_idx(BEHV_CKPT_ROOT, video_idx)
            beh_csv, beh_kind = _choose_behavior_csv(files)

            # --------- Load rates npzs ----------
            cands = sorted(NPRW_CKPT_ROOT.glob(f"rates__{session}__*.npz"))
            if not cands: print(f"[warn] No Intan rates for session {session}"); continue
            intan_rates_npz = cands[0]
            
            cands = sorted(UA_CKPT_ROOT.glob(f"rates__{session}__*.npz"))
            if not cands: print(f"[warn] No UA rates for session {session}"); continue
            ua_rates_npz = cands[0]
            
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

            intan_rate_hz, intan_t_ms, intan_meta, intan_pcs, intan_peaks, intan_peaks_t_ms, intan_peaks_t_sample, intan_expl = rcp.load_rate_npz(intan_rates_npz)
            ua_rate_hz, ua_t_ms, ua_meta, ua_pcs, ua_peaks, ua_peaks_t_ms, ua_peaks_t_sample, ua_expl = rcp.load_rate_npz(ua_rates_npz)

            # Optional stim times (absolute Intan ms)
            stim_ms_abs = try_load_stim_ms_from_intan_bundle(NPRW_BUNDLES, session)

            # --------- Apply Intan anchor shift to timebase ----------
            intan_t_ms_aligned = intan_t_ms - anchor_ms
            ua_t_ms_aligned    = ua_t_ms  # no shift yet
            
            # Intan peaks → ms
            intan_peaks_t_ms_raw = None
            if intan_peaks_t_ms is not None:
                intan_peaks_t_ms_raw = np.asarray(intan_peaks_t_ms, dtype=np.float32)
            elif intan_peaks_t_sample is not None:
                # prefer fs_intan from CSV/row; else from meta
                _fs_i = fs_intan if np.isfinite(fs_intan) else _meta_fs(intan_meta)
                if _fs_i and np.isfinite(_fs_i):
                    intan_peaks_t_ms_raw = (np.asarray(intan_peaks_t_sample, dtype=np.float32) / float(_fs_i)) * 1000.0

            # UA peaks → ms
            ua_peaks_t_ms_raw = None
            if ua_peaks_t_ms is not None:
                ua_peaks_t_ms_raw = np.asarray(ua_peaks_t_ms, dtype=np.float32)
            elif ua_peaks_t_sample is not None:
                _fs_u = _meta_fs(ua_meta)
                if not (_fs_u and np.isfinite(_fs_u)):
                    # derive from UA time base if needed
                    if ua_t_ms.size > 1:
                        _fs_u = 1000.0 / float(np.median(np.diff(ua_t_ms)))
                if _fs_u and np.isfinite(_fs_u):
                    ua_peaks_t_ms_raw = (np.asarray(ua_peaks_t_sample, dtype=np.float32) / float(_fs_u)) * 1000.0

            # Aligned (Intan timebase is shifted by anchor_ms in your pipeline)
            intan_peaks_t_ms_aligned = (intan_peaks_t_ms_raw - float(anchor_ms)) if intan_peaks_t_ms_raw is not None else np.array([], dtype=np.float32)
            ua_peaks_t_ms_aligned    = ua_peaks_t_ms_raw if ua_peaks_t_ms_raw is not None else np.array([], dtype=np.float32)

            # Keep original peak payloads too (or empty arrays for consistency)
            _intan_peaks          = intan_peaks if intan_peaks is not None else np.array([], dtype=np.float32)
            _intan_peaks_t_ms     = intan_peaks_t_ms if intan_peaks_t_ms is not None else np.array([], dtype=np.float32)
            _intan_peaks_t_sample = intan_peaks_t_sample if intan_peaks_t_sample is not None else np.array([], dtype=np.float32)

            _ua_peaks             = ua_peaks if ua_peaks is not None else np.array([], dtype=np.float32)
            _ua_peaks_t_ms        = ua_peaks_t_ms if ua_peaks_t_ms is not None else np.array([], dtype=np.float32)
            _ua_peaks_t_sample    = ua_peaks_t_sample if ua_peaks_t_sample is not None else np.array([], dtype=np.float32)

            # ---- Overlap and resample UA onto Intan's aligned grid ----
            # ---- Keep native binning; no resampling, no trimming ----
            orig_i_len = intan_t_ms_aligned.size
            orig_u_len = ua_t_ms_aligned.size

            dt_i = float(np.median(np.diff(intan_t_ms_aligned))) if orig_i_len > 1 else np.nan
            dt_u = float(np.median(np.diff(ua_t_ms_aligned)))    if orig_u_len > 1 else np.nan

            stats = {
                "overlap_bins": 0,          # no common grid anymore
                "dt_i_ms": dt_i,
                "dt_u_ms": dt_u,
            }


            # ---------- PRINT RANGES: Intan & BR/UA (original vs trimmed overlap) ----------
            def _fmt_ms(x):
                try:
                    return f"{float(x):.3f}"
                except Exception:
                    return "nan"

            # Intan original range (aligned)
            if intan_t_ms_aligned.size:
                i_orig_start_idx = 0
                i_orig_end_idx   = intan_t_ms_aligned.size - 1
                i_orig_start_ms  = intan_t_ms_aligned[0]
                i_orig_end_ms    = intan_t_ms_aligned[-1]
            else:
                i_orig_start_idx = i_orig_end_idx = -1
                i_orig_start_ms = i_orig_end_ms = np.nan

            print(
                "[ranges] Intan original:   "
                f"indices {i_orig_start_idx}→{i_orig_end_idx}  "
                f"ms { _fmt_ms(i_orig_start_ms) }→{ _fmt_ms(i_orig_end_ms) }"
            )
            print(
                "[ranges] Intan aligned: "
                f"indices 0→{orig_i_len-1}  ms {intan_t_ms_aligned[0]:.3f}→{intan_t_ms_aligned[-1]:.3f}"
            )
            print(
                "[ranges] UA aligned:    "
                f"indices 0→{orig_u_len-1}  ms {ua_t_ms_aligned[0]:.3f}→{ua_t_ms_aligned[-1]:.3f}"
            )
            print(f"[bins] {session}: Intan dt≈{stats['dt_i_ms']:.3f} ms, "
                f"UA dt≈{stats['dt_u_ms']:.3f} ms")
            
            # ---- Behavior: find and load per-trial CSV (supports both or single cam) ----
            beh_ns5_sample = np.array([], dtype=np.int64)
            beh_cam0 = np.zeros((0, 0), dtype=np.float32)
            beh_cam1 = np.zeros((0, 0), dtype=np.float32)
            beh_cam0_cols: list[str] = []
            beh_cam1_cols: list[str] = []

            # --- defaults so saving never fails ---
            fs_br = np.nan
            beh_t_ms = np.array([], dtype=np.float32)

            if beh_csv is not None:
                try:
                    # rcp.load_behavior_npz should tolerate "both" or single-cam CSVs;
                    # it can return empty arrays/cols for the missing cam.
                    (beh_ns5_sample,
                     beh_cam0, beh_cam0_cols,
                     beh_cam1, beh_cam1_cols) = rcp.load_behavior_npz(beh_csv, NUM_CAM)

                    print(f"[behavior] attached from {beh_csv.name} [{beh_kind}] (N={beh_ns5_sample.size})")

                    fs_br = float(row.get("fs_br", "nan"))
                    if not np.isfinite(fs_br):
                        # fallback: try opening using shifts CSV
                        br_ns5_path = Path(row.get("br_ns5", ""))
                        try:
                            if br_ns5_path.exists():
                                _, fs_br = rcp.load_br_sync(br_ns5_path, 134, stream_id=5)
                        except Exception:
                            pass

                    # --- Convert behavior sample indices -> ms (same origin as UA) ---
                    if beh_ns5_sample.size and np.isfinite(fs_br):
                        # --- Convert behavior timestamps (robust to variable FPS) ---
                        beh_t_ms = _behavior_times_ms_from_sources(
                            beh_ns5_sample=beh_ns5_sample,
                            fs_br=fs_br,
                        )

                        # Optional jitter diagnostics (confirms variable-FPS handling came from BR samples)
                        _log_frame_jitter(beh_ns5_sample, fs_br, label="behavior")
                        
                        # ---------- PRINT RANGES: Behavior (video) ----------
                        if beh_ns5_sample.size:
                            b_orig_start_idx = 0
                            b_orig_end_idx   = beh_ns5_sample.size - 1
                            b_sample_min     = int(beh_ns5_sample.min())
                            b_sample_max     = int(beh_ns5_sample.max())
                            if np.isfinite(fs_br):
                                b_ms_min = float(b_sample_min) * (1000.0 / float(fs_br))
                                b_ms_max = float(b_sample_max) * (1000.0 / float(fs_br))
                            else:
                                b_ms_min = b_ms_max = np.nan

                            print(
                                "[ranges] Video/Behavior original: "
                                f"indices {b_orig_start_idx}→{b_orig_end_idx}  "
                                f"samples {b_sample_min}→{b_sample_max}  "
                                f"ms { _fmt_ms(b_ms_min) }→{ _fmt_ms(b_ms_max) }"
                            )
                        else:
                            print("[ranges] Video/Behavior: no frames found.")
                    else:
                        if not np.isfinite(fs_br):
                            print("[warn] fs_br unavailable; saving behavior as samples only (no time mapping).")

                except Exception as e:
                    print(f"[warn] could not load behavior for BR {br_idx:03d}: {e}")
            else:
                print(f"[warn] no behavior CSV (both or single-cam) found for BR {br_idx:03d} in {BEHV_CKPT_ROOT}")

            combined_meta = dict(
                session=session,
                intan_idx=intan_idx,
                br_idx=br_idx,
                intan_rates=str(intan_rates_npz),
                ua_rates=str(ua_rates_npz),
                fs_intan=float(fs_intan),
                anchor_ms=float(anchor_ms),

                equal_time_grid=False,
                resampled_to=None,
                trimmed_bins=0,

                orig_intan_bins=int(orig_i_len),
                orig_ua_bins=int(orig_u_len),
                dt_ms_target=float(stats["dt_i_ms"]),

                # behavior meta (safe even if none found)
                behavior_csv=str(beh_csv) if 'beh_csv' in locals() and beh_csv is not None else None,
                behavior_rows=int(beh_ns5_sample.size),
                beh_fs_br=float(fs_br) if np.isfinite(fs_br) else None,
                beh_align_tol_ms=(float(stats["dt_i_ms"]) * 0.6
                                if "dt_i_ms" in stats and np.isfinite(stats["dt_i_ms"]) else None),
            )

            out_npz = ALIGNED_CKPT_ROOT / f"aligned__{session}__Intan_{intan_idx:03d}__BR_{br_idx:03d}.npz"
            np.savez_compressed(
                out_npz,
                intan_rate_hz=intan_rate_hz.astype(np.float32),
                intan_t_ms=intan_t_ms.astype(np.float32),                 # original (pre-shift)
                intan_t_ms_aligned=intan_t_ms_aligned.astype(np.float32), # aligned but same binning
                intan_meta=(intan_meta.item() if hasattr(intan_meta, "item") else intan_meta),
                intan_pcs=intan_pcs if intan_pcs is not None else np.array([], dtype=np.float32),
                intan_explained_var=intan_expl if intan_expl is not None else np.array([], dtype=np.float32),

                # Intan peaks (raw + aligned)
                intan_peaks=_intan_peaks,
                intan_peaks_t_ms=_intan_peaks_t_ms,
                intan_peaks_t_sample=_intan_peaks_t_sample,
                intan_peaks_t_ms_aligned=intan_peaks_t_ms_aligned.astype(np.float32),

                ua_rate_hz=ua_rate_hz.astype(np.float32),
                ua_t_ms=ua_t_ms.astype(np.float32),                       # original
                ua_t_ms_aligned=ua_t_ms_aligned.astype(np.float32),       # same here
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
                ua_peaks_t_ms_aligned=ua_peaks_t_ms_aligned.astype(np.float32),

                # Stim (absolute Intan ms; need to subtract anchor_ms as needed)
                stim_ms=(stim_ms_abs.astype(np.float32) if stim_ms_abs is not None else np.array([], dtype=np.float32)),

                # Alignment meta (as JSON)
                align_meta=json.dumps(combined_meta),
                
                # ---- Behavior (ns5 samples; no anchor shift applied) ----
                beh_ns5_sample=beh_ns5_sample,
                beh_cam0=beh_cam0,
                beh_cam1=beh_cam1,
                beh_cam0_cols=np.array(beh_cam0_cols, dtype=object),
                beh_cam1_cols=np.array(beh_cam1_cols, dtype=object),
                beh_t_ms=beh_t_ms,
            )

            print(f"[write] combined aligned → {out_npz}")

        except Exception as e:
            print(f"[error] Failed for session {row.get('session','?')}: {e}")
            continue

if __name__ == "__main__":
    main()