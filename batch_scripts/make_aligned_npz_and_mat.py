import json, csv
from scipy.io import savemat
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# ---------- CONFIG ----------
# Base paths loaded from config_loading

FS_NS2 = 1000.0
NUM_CAM = PARAMS.kinematics.get("num_camera", 1)
SHIFTS_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"


def _find_aligned_DLC_for_br_idx(behv_root: Path, br_idx: int) -> tuple[Path | None, str]:
    """
    Find the newest aligned DLC CSV for the given BR index, with preference:
      both_cams → Cam-0 → Cam-1.

    Returns
    -------
    (path, kind)
      path : Path | None
      kind : 'both' | 'cam0' | 'cam1' | 'none'

    Matches filenames like:
      ..._<BRIDX>_both_cams_aligned.csnv
      ..._<BRIDX>_Cam-0_aligned.csv
      ..._<BRIDX>_Cam-1_aligned.csv
    """
    br = f"{int(br_idx):03d}"

    def _newest(glob_pattern: str) -> Path | None:
        cands = list(behv_root.rglob(glob_pattern))
        if not cands:
            return None
        return max(cands, key=lambda p: p.stat().st_mtime)

    # find newest of each type
    both = _newest(f"*_{br}_both_cams_aligned.csv")
    cam0 = _newest(f"*_{br}_Cam-0_aligned.csv")
    cam1 = _newest(f"*_{br}_Cam-1_aligned.csv")

    # apply preference: both → cam0 → cam1
    if both is not None:
        return both, "both"
    if cam0 is not None:
        return cam0, "cam0"
    if cam1 is not None:
        return cam1, "cam1"
    return None, "none"

def _parse_SI_peaks(peaks) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    peak_times = {}
    peak_amp = {}
    for i in np.unique(peaks['channel_index']):
        peak_ch_times = peaks['sample_index'][peaks['channel_index'] == i]
        peak_ch_amp = peaks['amplitude'][peaks['channel_index'] == i]
        peak_times[i] = peak_ch_times
        peak_amp[i] = peak_ch_amp
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

    # keep only valid rows (if you ever use -1 to mark invalid)
    valid = ua_region >= 0
    rows = np.flatnonzero(valid)

    perm_parts = []
    for r in region_order:
        rr = rows[ua_region[rows] == r]
        rr = rr[np.argsort(key[rr], kind="stable")]
        perm_parts.append(rr)

    perm = np.concatenate(perm_parts) if perm_parts else np.array([], dtype=int)

    # append any "invalid" rows at the end (optional; you can drop instead)
    inv = np.flatnonzero(~valid)
    if inv.size:
        perm = np.concatenate([perm, inv])

    return perm

def _clean_meta(meta):
    """
    Return a clean dictionary without complex objects (like recording extractors).
    Useful for saving to .npz/.mat without pickling heavy dependencies.
    """
    if not isinstance(meta, dict):
         if hasattr(meta, "item"):
             meta = meta.item()
         if not isinstance(meta, dict):
             return {} # Fallback
             
    clean = {}
    for k, v in meta.items():
        # Skip keys known to hold heavy objects
        if k in ("recording", "sorting", "extractor", "neo_reader"):
            continue
        # Skip if value is a module, class, or complex object
        if hasattr(v, "__module__") and not isinstance(v, (np.ndarray, np.generic)):
            # Keep only if it's a simple type we trust (optional check)
            pass
            
        clean[k] = v
    return clean

def main():
    if not SHIFTS_CSV.exists():
        raise SystemExit(f"[error] shifts CSV not found: {SHIFTS_CSV}")

    # ---------- read shifts ----------
    with SHIFTS_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    br2video = rcp.get_metadata_mapping(METADATA_CSV, "BR_File", "Video_File")
    br2vog = rcp.get_metadata_mapping(METADATA_CSV, "BR_File", "VOG_File")

    if not rows:
        raise SystemExit("[error] shifts CSV has no rows")

    for row in rows:
        try:
            session   = row["session"]
            intan_idx = int(row["intan_idx"])
            br_idx    = int(row["br_idx"])
            fs_nprw  = float(row.get("fs_intan", 30000.0))
            shift_ms = float(row["shift_ms"])
            video_idx = int(br2video.get(br_idx))
            if video_idx is None:
                print(f"[warn] Video_File not found in metadata for BR {br_idx:03d}; "
                    f"falling back to BR index.")
                video_idx = br_idx

            # load DLC csv
            beh_csv, _ = _find_aligned_DLC_for_br_idx(BEHV_CKPT_ROOT, video_idx)

            # load NPRW npz
            cands = sorted(NPRW_CKPT_ROOT.glob(f"rates__{session}__*.npz"))
            if not cands: print(f"[warn] No NPRW rates for session {session}"); continue
            nprw_rates_npz = cands[0]
            
            # load UA npz
            cands = sorted(UA_CKPT_ROOT.glob(f"rates__NRR_RW*_{br_idx:03d}__*.npz"))
            if not cands: print(f"[warn] No UA rates for session {session}"); continue
            ua_rates_npz_loc = cands[0]
            
    
            # UA
            ua_npz = np.load(ua_rates_npz_loc, allow_pickle=True)
            ua_peaks = ua_npz["peaks"]
            ua_meta = ua_npz["meta"].item()
            fs_ua = float(ua_meta.get("fs_ua", np.nan))

            ua_elec = ua_npz["ua_elec"]
            ua_nsp  = ua_npz["ua_nsp"]
            ua_idx_rows = ua_npz["ua_index"]
            ua_region = ua_npz["ua_region"]
            ua_region_names = ua_npz["ua_region_names"]
            ua_port = ua_npz["ua_port"]

            hr_sig  = ua_npz.get("hr_sig", None)
            vog_sig = ua_npz.get("vog_sig", None)

            # touchscreen states
            ts_state_num  = ua_npz.get("ts_state_num", np.array([], dtype=np.int8)).astype(np.int8)
            ts_state_char = ua_npz.get("ts_state_char", np.full(0, "N", dtype="U1"))

            # get mapping
            perm = _ua_perm_by_region(ua_region, ua_elec, ua_nsp, region_order=(0,1,2,3), within="nsp")
            n_ch = int(len(ua_nsp))
            inv_perm = np.empty(n_ch, dtype=int)
            inv_perm[perm] = np.arange(n_ch)
            
            # rewrite peaks
            ua_peaks = ua_peaks.copy()
            ua_peaks["channel_index"] = inv_perm[ua_peaks["channel_index"].astype(int)]

            # reorder labels
            ua_elec     = np.asarray(ua_elec)[perm]
            ua_nsp      = np.asarray(ua_nsp)[perm]
            ua_region   = np.asarray(ua_region)[perm]
            ua_idx_rows = np.asarray(ua_idx_rows)[perm]

            # parse peaks after channel_index rewrite
            ua_peak_samps, ua_peak_amps = _parse_SI_peaks(ua_peaks)
            ua_peak_ms = {ch: samps / fs_ua * 1000.0 for ch, samps in ua_peak_samps.items()}

            # NPRW
            nprw_npz = np.load(nprw_rates_npz, allow_pickle=True)
            nprw_peaks = nprw_npz["peaks"]
            nprw_meta = nprw_npz["meta"].item() if hasattr(nprw_npz["meta"], "item") else nprw_npz["meta"]
            nprw_meta['rec_start_ms_aligned'] = nprw_meta['rec_start_ms'] - shift_ms
            nprw_meta['rec_end_ms_aligned'] = nprw_meta['rec_end_ms'] - shift_ms
            
            nprw_peak_samps, nprw_peak_amps = _parse_SI_peaks(nprw_peaks)
            nprw_peak_ms = {ch: samps / fs_nprw * 1000.0 - shift_ms for ch, samps in nprw_peak_samps.items()}

            # Stim times (absolute Intan ms)
            stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
            stim_ms = np.array([], dtype=np.float32)
            if stim_npz_path is not None and Path(stim_npz_path).exists():
                stim = rcp.load_stim_detection(stim_npz_path)
                stim_ms = stim["block_bounds_samples"][:, 0] * 1000.0 / fs_nprw - shift_ms
            else:
                print(f"[warn] stim npz not found for BR {br_idx:03d}: {stim_npz_path}")

            # behavior alignment (supports both two cam and one cam)
            beh_ns5_sample = np.array([], dtype=np.int64)
            beh_cam0 = np.zeros((0, 0), dtype=np.float32)
            beh_cam1 = np.zeros((0, 0), dtype=np.float32)
            beh_cam0_cols: list[str] = []
            beh_cam1_cols: list[str] = []
            beh_t_ms = np.array([], dtype=np.float32)

            ns5_fs = 30000.0
            meta_ns5 = ua_meta.get("meta_ns5", {})
            if isinstance(meta_ns5, dict) and meta_ns5.get("ns5_fs_hz") is not None:
                ns5_fs = float(meta_ns5["ns5_fs_hz"])
            if beh_csv is not None:
                try:
                    (beh_ns5_sample,
                        beh_cam0, beh_cam0_cols,
                        beh_cam1, beh_cam1_cols) = rcp.load_behavior_npz(beh_csv, NUM_CAM)

                    beh_t_ms = beh_ns5_sample * 1000.0 / ns5_fs
                    if beh_ns5_sample.size:
                        behv_orig_start_idx = 0
                        behv_orig_end_idx   = beh_ns5_sample.size - 1
                        behv_sample_min     = int(beh_ns5_sample.min())
                        behv_sample_max     = int(beh_ns5_sample.max())
                        if np.isfinite(ns5_fs):
                            behv_ms_min = behv_sample_min * 1000.0 / ns5_fs
                            behv_ms_max = behv_sample_max * 1000.0 / ns5_fs
                        else:
                            behv_ms_min = behv_ms_max = np.nan

                        print(
                            "[ranges] Video/Behavior original: "
                            f"indices {behv_orig_start_idx}-{behv_orig_end_idx}  "
                            f"samples {behv_sample_min}-{behv_sample_max}  "
                            f"ms {behv_ms_min:.3f}-{behv_ms_max:.3f}"
                        )
                        
                    else:
                        print("[ranges] Video/Behavior: no frames found.")
                except Exception as e:
                    print(f"[warn] could not load behavior for BR {br_idx:03d}: {e}")
            else:
                print(f"[warn] no behavior CSV (both or single-cam) found for BR {br_idx:03d} in {BEHV_CKPT_ROOT}")

            # ---- VOG alignment ----
            vog_ns2_samp = np.array([], dtype=np.int64)
            vog_t_ms      = np.array([], dtype=np.float32)
            vog_cols      = np.zeros((0, 0), dtype=np.float32)
            vog_col_names = np.array([], dtype=object)

            vog_csv = None
            vog_file_idx = int(br2vog.get(br_idx)) if br2vog is not None else None

            if vog_file_idx is not None:
                # e.g. NRR_RW011_002_VOG_aligned.csv
                cands_vog = sorted(
                    VOG_CKPT_ROOT.glob(f"*_{vog_file_idx:03d}_VOG_aligned.csv")
                )
                if cands_vog:
                    vog_csv = cands_vog[0]

            if vog_csv is not None:
                try:
                    df_vog = pd.read_csv(vog_csv)

                    # ns2_sample → time (ms); assume 1 kHz for ns2
                    if "ns2_sample" in df_vog.columns:
                        vog_ns2_samp = pd.to_numeric(df_vog["ns2_sample"],
                                                errors="coerce").to_numpy(dtype=np.int64)
                        vog_t_ms = vog_ns2_samp * (1000.0 / FS_NS2)
                    else:
                        vog_ns2_samp = np.array([], dtype=np.int64)

                    # pack the four VOG columns into a single matrix
                    wanted = ["HPos_pix", "VPos_pix", "H_volt", "V_volt"]
                    present = [c for c in wanted if c in df_vog.columns]

                    if present:
                        vog_cols = df_vog[present].apply(
                            pd.to_numeric, errors="coerce"
                        ).to_numpy(dtype=np.float32)
                        vog_col_names = np.array(present, dtype=object)

                    if vog_t_ms.size:
                        print(
                            "[ranges] VOG: indices 0-{end}  ms {t0:.3f}-{t1:.3f}".format(
                                end=vog_t_ms.size - 1,
                                t0=float(vog_t_ms[0]),
                                t1=float(vog_t_ms[-1]),
                            )
                        )

                    print(f"[VOG] attached from {vog_csv.name} (N={vog_t_ms.size})")

                except Exception as e:
                    print(f"[warn] could not load VOG for BR {br_idx:03d}: {e}")
            else:
                print(f"[warn] no VOG CSV found for BR {br_idx:03d} in {VOG_CKPT_ROOT}")
    
            aligned_meta = dict(
                session=session,
                intan_idx=intan_idx,
                br_idx=br_idx,
                nprw_rates=str(nprw_rates_npz),
                ua_rates=str(ua_rates_npz_loc),
                fs_nprw=float(fs_nprw),
                fs_ua=float(fs_ua) if np.isfinite(fs_ua) else None,
                shift_ms=float(shift_ms),

                behavior_csv=str(beh_csv) if beh_csv is not None else None,
                behavior_rows=int(beh_ns5_sample.size),
                fs_ns5=float(ns5_fs) if beh_ns5_sample.size else None,

                equal_time_grid=False,
                resampled_to=None,
                trimmed_bins=0,
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
                
                hr_sig=hr_sig,
                vog_sig=vog_sig,
                
                stim_ms=(stim_ms.astype(np.float32) if stim_ms is not None else np.array([], dtype=np.float32)),
                shift_ms=shift_ms,

                # Alignment meta (as JSON)
                align_meta=json.dumps(aligned_meta),
                
                # behavior (ns5 samples; no shift shift applied)
                beh_ns5_sample=beh_ns5_sample,
                beh_cam0=beh_cam0,
                beh_cam1=beh_cam1,
                beh_cam0_cols=np.array(beh_cam0_cols, dtype=object),
                beh_cam1_cols=np.array(beh_cam1_cols, dtype=object),
                beh_t_ms=beh_t_ms,
                
                # # ---- VOG arrays ----
                # vog_ns2_samp=vog_ns2_samp,
                # vog_t_ms=vog_t_ms,
                # vog_cols=vog_cols,
                # vog_col_names=vog_col_names,
                
                ts_state_num=ts_state_num,
                ts_state_char=ts_state_char,
            )

            print(f"[write] combined aligned: {out_npz}")
            
            # ---- also save a MATLAB .mat version ----
            out_mat = out_npz.with_suffix(".mat")

            def _change_none_for_mat(d):
                """Return a dict with all None values removed. Non-dicts -> {}."""
                if d is None:
                    return np.array([])
                if not isinstance(d, dict):
                    # Don’t try to iterate weird objects; just skip
                    return np.array([])
                out = {}
                for k, v in d.items():
                    if v is not None:
                        out[k] = v
                return out

            # Use mostly the same contents; for meta we store the dict instead of JSON.
            mat_dict = dict(
                nprw_meta=_change_none_for_mat(nprw_meta),
                ua_meta=_change_none_for_mat(ua_meta),

                ua_elec=ua_elec,
                ua_region=ua_region,
                ua_region_names=np.array(ua_region_names, dtype=object),
                ua_port=ua_port,
                ua_nsp=ua_nsp,
                ua_idx_rows=ua_idx_rows,

                stim_ms=(stim_ms.astype(np.float32)
                         if stim_ms is not None
                         else np.array([], dtype=np.float32)),
                shift_ms=shift_ms,
                
                hr_sig=hr_sig,
                vog_sig=vog_sig,
                
                ts_state_num=ts_state_num,
                ts_state_char=ts_state_char,

                # alignment meta as a struct-like dict for MATLAB
                align_meta=_change_none_for_mat(aligned_meta),

                beh_ns5_sample=beh_ns5_sample,
                beh_cam0=beh_cam0,
                beh_cam1=beh_cam1,
                beh_cam0_cols=np.array(beh_cam0_cols, dtype=object),
                beh_cam1_cols=np.array(beh_cam1_cols, dtype=object),
                beh_t_ms=beh_t_ms,
                
                vog_ns2_samp=vog_ns2_samp,
                vog_t_ms=vog_t_ms,
                vog_cols=vog_cols,
                vog_col_names=vog_col_names,
            )

            savemat(out_mat, mat_dict, do_compression=True)

        except Exception as e:
            print(f"[error] Failed for session {row.get('session','?')}: {e}")
            continue

if __name__ == "__main__":
    main()