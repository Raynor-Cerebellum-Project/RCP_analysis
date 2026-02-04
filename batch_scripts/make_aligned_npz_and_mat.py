import json, csv
from scipy.io import savemat
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# CONFIG
# Base paths loaded from config_loading

FS_NS2 = 1000.0
NUM_CAM = PARAMS.kinematics.get("num_camera", 1)
SHIFTS_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"

CONTROL_ROOT = ALIGNED_CKPT_ROOT / "control_reaches"
AT_REST_ROOT  = ALIGNED_CKPT_ROOT / "at_rest"
STIM_ROOT  = ALIGNED_CKPT_ROOT / "stim_reaches"

CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
AT_REST_ROOT.mkdir(parents=True, exist_ok=True)
STIM_ROOT.mkdir(parents=True, exist_ok=True)

HAS_BR = PARAMS.preprocessing.get("has_BR")
HAS_KINEMATICS = PARAMS.preprocessing.get("has_kinematics")
HAS_VOG = PARAMS.preprocessing.get("has_VOG")
HAS_HR = PARAMS.preprocessing.get("has_HR")
SAVE_MAT_FILE = PARAMS.preprocessing.get("output_aligned_mat")
PROCESS_ONLY = PARAMS.preprocessing.get("process_only")  # list of BR indices to process; empty = all

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
    peak_times: dict[int, np.ndarray] = {}
    peak_amp: dict[int, np.ndarray] = {}

    ch_idx = peaks["channel_index"]
    for i in np.unique(ch_idx):
        ii = int(i)
        m = (ch_idx == i)
        peak_times[ii] = peaks["sample_index"][m]
        peak_amp[ii] = peaks["amplitude"][m]

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

def parse_bool(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

def main():
    br2video = rcp.get_metadata_mapping(METADATA_CSV, "BR_File", "Video_File")
    br2vog = rcp.get_metadata_mapping(METADATA_CSV, "BR_File", "VOG_File")
    
    if not SHIFTS_CSV.exists():
        raise SystemExit(f"[error] shifts CSV not found: {SHIFTS_CSV}")

    # Shifts
    with SHIFTS_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    if not rows:
        raise SystemExit("[error] shifts CSV has no rows")

    for row in rows:
        try:
            intan_filename   = row["intan_filename"]
            intan_idx = int(row["intan_idx"])
            br_idx    = int(row["br_idx"])
            fs_nprw  = float(row.get("fs_intan", 30000.0))
            shift_ms = float(row["shift_ms"])
            is_control = parse_bool(row["is_control"])
            is_at_rest  = parse_bool(row["is_at_rest"])

            if is_control and is_at_rest:
                print(f"[warn] BR {br_idx:03d} marked as both control reaches and at rest; treating as control")
                is_at_rest = False
                
            video_val = br2video.get(br_idx)
            if video_val is None:
                print(f"[warn] Video_File not found in metadata for BR {br_idx:03d}; falling back to BR index.")
                video_idx = br_idx
            else:
                video_idx = int(video_val)

            # load NPRW npz
            cands = sorted(NPRW_CKPT_ROOT.glob(f"rates__{intan_filename}__*.npz"))
            if not cands: print(f"[warn] No NPRW rates for session {intan_filename}"); continue
            nprw_rates_npz_loc = cands[0]
            
            # NPRW
            nprw_npz = np.load(nprw_rates_npz_loc, allow_pickle=True)
            nprw_peaks = nprw_npz["peaks"]
            nprw_meta = nprw_npz["meta"].item() if hasattr(nprw_npz["meta"], "item") else nprw_npz["meta"]
            nprw_meta['rec_start_ms_aligned'] = nprw_meta['rec_start_ms'] - shift_ms
            nprw_meta['rec_end_ms_aligned'] = nprw_meta['rec_end_ms'] - shift_ms
            
            nprw_peak_samps, nprw_peak_amps = _parse_SI_peaks(nprw_peaks)
            nprw_peak_ms = {int(ch): samps / fs_nprw * 1000.0 - shift_ms for ch, samps in nprw_peak_samps.items()}

            ir_ms = nprw_npz["ir_ms"] - shift_ms

            # ir_ms_br = ir_sec * 1000.0 - shift_ms
            # Stim times (absolute Intan ms)
            stim_dur = nprw_meta['stim_dur']
            recording_stim_dur = float(np.median(stim_dur))
            stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
            stim_ms = np.array([], dtype=np.float32)
            if stim_npz_path is not None and Path(stim_npz_path).exists():
                stim = rcp.load_stim_detection(stim_npz_path)
                stim_ms = stim["block_bounds_samples"][:, 0] * 1000.0 / fs_nprw - shift_ms
            else:
                print(f"[warn] stim npz not found for BR {br_idx:03d}: {stim_npz_path}")
                
            
            if HAS_BR is True:
                # load UA npz
                cands = sorted(UA_CKPT_ROOT.glob(f"rates__NRR_RW*_{br_idx:03d}__*.npz"))
                if not cands: print(f"[warn] No UA rates for session {intan_filename}"); continue
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
                ua_peak_ms   = {int(ch): samps / fs_ua * 1000.0 for ch, samps in ua_peak_samps.items()}

            if HAS_KINEMATICS is True:
                # load DLC csv
                beh_csv, _ = _find_aligned_DLC_for_br_idx(BEHV_CKPT_ROOT, video_idx)
                
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
                    
            if HAS_VOG is True:
                # ---- VOG alignment ----
                vog_ns2_samp = np.array([], dtype=np.int64)
                vog_t_ms      = np.array([], dtype=np.float32)
                vog_cols      = np.zeros((0, 0), dtype=np.float32)
                vog_col_names = np.array([], dtype=object)

                vog_csv = None
                vog_file_idx = int(br2vog.get(br_idx)) if br2vog is not None else None

                if vog_file_idx is not None:
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
                intan_filename=intan_filename,
                intan_idx=intan_idx,
                br_idx=br_idx,
                nprw_rates=str(nprw_rates_npz_loc),
                ua_rates=str(ua_rates_npz_loc) if HAS_BR is True else None,
                fs_nprw=float(fs_nprw),
                fs_ua=float(fs_ua) if HAS_BR else None,
                shift_ms=float(shift_ms),
                is_control=is_control,
                is_at_rest=is_at_rest,
                recording_stim_dur=recording_stim_dur,

                behavior_csv=str(beh_csv) if HAS_KINEMATICS else None,
                behavior_rows=int(beh_ns5_sample.size) if HAS_KINEMATICS else None,
                fs_ns5=float(ns5_fs) if HAS_KINEMATICS else None,
            )

            # Choose output directory
            if is_control:
                out_dir = CONTROL_ROOT
            elif is_at_rest:
                out_dir = AT_REST_ROOT
            else:
                out_dir = STIM_ROOT

            out_npz = out_dir / f"aligned__{intan_filename}__Intan_{intan_idx:03d}__BR_{br_idx:03d}.npz"

            to_save = dict(
                # NPRW
                nprw_meta=nprw_meta,
                nprw_peak_ms=nprw_peak_ms,
                nprw_peak_amps=nprw_peak_amps,
                ir_ms=(ir_ms.astype(np.float32) if ir_ms is not None else np.array([], dtype=np.float32)),
                stim_ms=(stim_ms.astype(np.float32) if stim_ms is not None else np.array([], dtype=np.float32)),
                shift_ms=np.float32(shift_ms),

                align_meta=json.dumps(aligned_meta),
            )

            if HAS_BR is True:
                to_save.update(dict(
                    ua_meta=ua_meta,
                    ua_peak_ms=ua_peak_ms,
                    ua_peak_amps=ua_peak_amps,

                    ua_elec=ua_elec,
                    ua_region=ua_region,
                    ua_region_names=ua_region_names,
                    ua_port=ua_port,
                    ua_nsp=ua_nsp,
                    ua_idx_rows=ua_idx_rows,

                    hr_sig=hr_sig,

                    ts_state_num=ts_state_num,
                    ts_state_char=ts_state_char,
                ))

            if HAS_KINEMATICS is True:
                to_save.update(dict(
                    beh_ns5_sample=beh_ns5_sample,
                    beh_cam0=beh_cam0,
                    beh_cam1=beh_cam1,
                    beh_cam0_cols=np.array(beh_cam0_cols, dtype=object),
                    beh_cam1_cols=np.array(beh_cam1_cols, dtype=object),
                    beh_t_ms=beh_t_ms,
                ))

            if HAS_VOG is True:
                to_save.update(dict(
                    vog_sig=vog_sig,
                    vog_ns2_samp=vog_ns2_samp,
                    vog_t_ms=vog_t_ms,
                    vog_cols=vog_cols,
                    vog_col_names=vog_col_names,
                ))

            np.savez_compressed(out_npz, **to_save)
            print(f"[write] combined aligned: {out_npz}")
            
            if SAVE_MAT_FILE:
                out_mat = out_npz.with_suffix(".mat")

                def _to_mat_value(x):
                    """
                    Convert Python objects to something savemat can save
                    """
                    if x is None:
                        return np.array([])
                    if isinstance(x, (np.ndarray, np.number, float, int, bool)):
                        return x
                    if isinstance(x, str):
                        return x
                    if isinstance(x, (dict, list, tuple)):
                        # safest: preserve structure via JSON
                        return json.dumps(x, default=str)
                    # fallback: stringify unknown objects
                    return str(x)

                mat_dict = dict(
                    # Always present
                    nprw_meta=_to_mat_value(nprw_meta),
                    nprw_peak_ms=_to_mat_value(nprw_peak_ms),
                    nprw_peak_amps=_to_mat_value(nprw_peak_amps),

                    ir_ms=(ir_ms.astype(np.float32) if ir_ms is not None else np.array([], dtype=np.float32)),
                    
                    stim_ms=(stim_ms.astype(np.float32) if stim_ms is not None else np.array([], dtype=np.float32)),
                    shift_ms=np.float32(shift_ms),

                    # aligned metadata (JSON)
                    align_meta=_to_mat_value(aligned_meta),
                )

                if HAS_BR is True:
                    mat_dict.update(dict(
                        ua_meta=_to_mat_value(ua_meta),
                        ua_peak_ms=_to_mat_value(ua_peak_ms),
                        ua_peak_amps=_to_mat_value(ua_peak_amps),

                        ua_elec=np.asarray(ua_elec),
                        ua_region=np.asarray(ua_region),
                        ua_region_names=np.array(ua_region_names, dtype=object),
                        ua_port=np.asarray(ua_port),
                        ua_nsp=np.asarray(ua_nsp),
                        ua_idx_rows=np.asarray(ua_idx_rows),

                        hr_sig=_to_mat_value(hr_sig),

                        ts_state_num=np.asarray(ts_state_num),
                        ts_state_char=np.asarray(ts_state_char),
                    ))

                if HAS_KINEMATICS is True:
                    mat_dict.update(dict(
                        beh_ns5_sample=np.asarray(beh_ns5_sample),
                        beh_cam0=np.asarray(beh_cam0),
                        beh_cam1=np.asarray(beh_cam1),
                        beh_cam0_cols=np.array(beh_cam0_cols, dtype=object),
                        beh_cam1_cols=np.array(beh_cam1_cols, dtype=object),
                        beh_t_ms=np.asarray(beh_t_ms),
                    ))

                if HAS_VOG is True:
                    mat_dict.update(dict(
                        vog_sig=_to_mat_value(vog_sig),
                        vog_ns2_samp=np.asarray(vog_ns2_samp),
                        vog_t_ms=np.asarray(vog_t_ms),
                        vog_cols=np.asarray(vog_cols),
                        vog_col_names=np.array(vog_col_names, dtype=object),
                    ))

                savemat(out_mat, mat_dict, do_compression=True)
                print(f"[write] MATLAB aligned: {out_mat}")

        except Exception as e:
            print(f"[error] Failed for session {row.get('session','?')}: {e}")
            continue

if __name__ == "__main__":
    main()