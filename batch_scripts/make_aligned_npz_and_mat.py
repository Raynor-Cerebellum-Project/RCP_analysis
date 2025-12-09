from pathlib import Path
import numpy as np
import pandas as pd
import json, csv
from typing import Tuple, Optional
import RCP_analysis as rcp
from scipy.io import savemat

# ---------- CONFIG ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)
NPRW_AUX_DATA   = OUT_BASE / "aux_data" / "NPRW"
NPRW_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW"
UA_CKPT_ROOT   = OUT_BASE / "checkpoints" / "UA"
ALIGNED_CKPT_ROOT   = OUT_BASE / "checkpoints" / "Aligned"; ALIGNED_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
BEHV_CKPT_ROOT = OUT_BASE / "checkpoints" / "Behavior"; BEHV_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
VOG_CKPT_ROOT = OUT_BASE / "checkpoints" / "VOG"; VOG_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

FS_NS2 = 1000.0
                        
NUM_CAM = PARAMS.kinematics.get("num_camera", 1)

METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"
SHIFTS_CSV    = METADATA_ROOT / "br_to_intan_shifts.csv"

def _find_aligned_DLC_for_br_idx(behv_root: Path, br_idx: int) -> Tuple[Optional[Path], str]:
    """
    Find the newest aligned DLC CSV for the given BR index, with preference:
      both_cams → Cam-0 → Cam-1.

    Returns
    -------
    (path, kind)
      path : Path | None
      kind : 'both' | 'cam0' | 'cam1' | 'none'

    Matches filenames like:
      ..._<BRIDX>_both_cams_aligned.csv
      ..._<BRIDX>_Cam-0_aligned.csv
      ..._<BRIDX>_Cam-1_aligned.csv
    """
    br = f"{int(br_idx):03d}"

    def _newest(glob_pattern: str) -> Optional[Path]:
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
            anchor_ms = float(row["anchor_ms"])
            video_idx = br2video.get(br_idx)
            if video_idx is None:
                print(f"[warn] Video_File not found in metadata for BR {br_idx:03d}; "
                    f"falling back to BR index.")
                video_idx = br_idx

            # load DLC csv
            beh_csv, beh_kind = _find_aligned_DLC_for_br_idx(BEHV_CKPT_ROOT, video_idx)

            # load NPRW npz
            cands = sorted(NPRW_CKPT_ROOT.glob(f"rates__{session}__*.npz"))
            if not cands: print(f"[warn] No NPRW rates for session {session}"); continue
            nprw_rates_npz = cands[0]
            
            # load UA npz
            cands = sorted(UA_CKPT_ROOT.glob(f"rates__NRR_RW*_{br_idx:03d}__*.npz"))
            if not cands: print(f"[warn] No UA rates for session {session}"); continue
            ua_rates_npz_loc = cands[0]
            
    
            # get array index from UA + touchscreen state
            ua_elec = ua_region = ua_region_names = ua_nsp = ua_idx_rows = ua_port = ts_state_num = ts_state_char = None
            
            ua_npz = np.load(ua_rates_npz_loc, allow_pickle=True)
            ua_rate_hz = ua_npz["rate_hz"]
            ua_counts = ua_npz["counts"]
            ua_t_ms    = ua_npz["t_ms"]
            ua_pcs   = ua_npz["pcs"]
            ua_peaks = ua_npz["peaks"]
            ua_peak_t_ms     = ua_npz["peak_t_ms"]
            ua_expl = ua_npz["explained_var"]
            ua_meta = ua_npz["meta"]
            fs_br = ua_npz["meta"].item()["fs"]
            
            ua_elec = ua_npz["ua_elec"]
            ua_nsp = ua_npz["ua_nsp"]
            ua_idx_rows = ua_npz["ua_index"]
            ua_region = ua_npz["ua_region"]
            ua_region_names = ua_npz["ua_region_names"]
            ua_port = ua_npz["ua_port"]

            # touchscreen states
            if "ts_state_num" in ua_npz:
                ts_state_num = ua_npz["ts_state_num"].astype(np.int8)
            else:
                ts_state_num = np.array([], dtype=np.int8)

            if "ts_state_char" in ua_npz:
                ts_state_char = ua_npz["ts_state_char"]
            else:
                ts_state_char = np.full(0, 'N', dtype='U1')

            # NPRW
            nprw_npz = np.load(nprw_rates_npz, allow_pickle=True)
            nprw_rate_hz = nprw_npz["rate_hz"]
            nprw_counts = nprw_npz["counts"]
            nprw_t_ms    = nprw_npz["t_ms"]
            nprw_pcs   = nprw_npz["pcs"]
            nprw_peaks = nprw_npz["peaks"]
            nprw_peak_t_ms     = nprw_npz["peak_t_ms"]
            nprw_expl = nprw_npz["explained_var"]
            nprw_meta = nprw_npz["meta"]

            # Stim times (absolute Intan ms)
            stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
            stim = rcp.load_stim_detection(stim_npz_path)
            stim_ms_abs = stim["block_bounds_samples"][:, 0] * 1000 / fs_nprw
            
            # apply Intan anchor shift to timebase
            nprw_t_ms_aligned = nprw_t_ms - anchor_ms
            ua_t_ms_aligned    = ua_t_ms  # no shift yet

            # Aligned (Intan timebase is shifted by anchor_ms in your pipeline)
            nprw_peak_t_ms_aligned = (nprw_peak_t_ms - float(anchor_ms)) if nprw_peak_t_ms is not None else np.array([], dtype=np.float32)
            ua_peak_t_ms_aligned    = ua_peak_t_ms if ua_peak_t_ms is not None else np.array([], dtype=np.float32)

            # Keep native binning; no resampling, no trimming
            num_bins_nprw = nprw_t_ms_aligned.size
            num_bins_ua = ua_t_ms_aligned.size

            bin_dt_nprw = float(np.median(np.diff(nprw_t_ms_aligned))) if num_bins_nprw > 1 else np.nan # bin width
            bin_dt_ua = float(np.median(np.diff(ua_t_ms_aligned)))    if num_bins_ua > 1 else np.nan

            # print time ranges
            # NPRW original range (pre-shift)
            if nprw_t_ms.size:
                nprw_orig_start_idx = 0
                nprw_orig_end_idx   = nprw_t_ms.size - 1
                nprw_orig_start_ms  = nprw_t_ms[0]
                nprw_orig_end_ms    = nprw_t_ms[-1]
            else:
                nprw_orig_start_idx = nprw_orig_end_idx = -1
                nprw_orig_start_ms = nprw_orig_end_ms = np.nan

            print(
                "[ranges] NPRW original:   "
                f"bin indices {nprw_orig_start_idx}-{nprw_orig_end_idx}  "
                f"ms {nprw_orig_start_ms:.3f}-{nprw_orig_end_ms:.3f}"
            )

            print(
                "[ranges] NPRW aligned: "
                f"indices 0-{num_bins_nprw-1}  "
                f"ms {nprw_t_ms_aligned[0]:.3f}-{nprw_t_ms_aligned[-1]:.3f}"
            )

            print(
                "[ranges] UA aligned:    "
                f"indices 0-{num_bins_ua-1}  ms {ua_t_ms_aligned[0]:.3f}-{ua_t_ms_aligned[-1]:.3f}"
            )
            print(f"[bins] {session}: NPRW dt≈{bin_dt_nprw} ms, "
                f"UA dt≈{bin_dt_ua} ms")
            
            # behavior alignment (supports both two cam and one cam)
            beh_ns5_sample = np.array([], dtype=np.int64)
            beh_cam0 = np.zeros((0, 0), dtype=np.float32)
            beh_cam1 = np.zeros((0, 0), dtype=np.float32)
            beh_cam0_cols: list[str] = []
            beh_cam1_cols: list[str] = []
            beh_t_ms = np.array([], dtype=np.float32)

            if beh_csv is not None:
                try:
                    (beh_ns5_sample,
                     beh_cam0, beh_cam0_cols,
                     beh_cam1, beh_cam1_cols) = rcp.load_behavior_npz(beh_csv, NUM_CAM) # TODO Make concise

                    print(f"[behavior] attached from {beh_csv.name} [{beh_kind}] (N={beh_ns5_sample.size})")

                    # convert behavior sample indices to ms (same time as UA)
                    if beh_ns5_sample.size and np.isfinite(fs_br):
                        beh_t_ms = beh_ns5_sample * 1000 / fs_br

                        # ---------- PRINT RANGES: Behavior (video) ----------
                        if beh_ns5_sample.size:
                            behv_orig_start_idx = 0
                            behv_orig_end_idx   = beh_ns5_sample.size - 1
                            behv_sample_min     = int(beh_ns5_sample.min())
                            behv_sample_max     = int(beh_ns5_sample.max())
                            if np.isfinite(fs_br):
                                behv_ms_min = float(behv_sample_min) * (1000.0 / float(fs_br))
                                behv_ms_max = float(behv_sample_max) * (1000.0 / float(fs_br))
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
                    else:
                        if not np.isfinite(fs_br):
                            print("[warn] fs_br unavailable; saving behavior as samples only (no time mapping).")

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
            vog_file_idx = br2vog.get(br_idx) if br2vog is not None else None

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
                fs_nprw=fs_nprw,
                anchor_ms=float(anchor_ms),

                equal_time_grid=False,
                resampled_to=None,
                trimmed_bins=0,

                orig_nprw_bins=int(num_bins_nprw),
                orig_ua_bins=int(num_bins_ua),
                dt_ms_target=bin_dt_nprw,

                # behavior meta (safe even if none found)
                behavior_csv=str(beh_csv) if 'beh_csv' in locals() and beh_csv is not None else None,
                behavior_rows=int(beh_ns5_sample.size),
                beh_fs_br=float(fs_br) if np.isfinite(fs_br) else None,
                beh_align_tol_ms=(bin_dt_nprw * 0.6 if np.isfinite(bin_dt_nprw) else None),
            )

            out_npz = ALIGNED_CKPT_ROOT / f"aligned__{session}__Intan_{intan_idx:03d}__BR_{br_idx:03d}.npz"
            np.savez_compressed(
                out_npz,
                nprw_rate_hz=nprw_rate_hz.astype(np.float32),
                nprw_counts=nprw_counts.astype(np.uint16),
                nprw_t_ms=nprw_t_ms.astype(np.float32),                 # original (pre-shift)
                nprw_t_ms_aligned=nprw_t_ms_aligned.astype(np.float32), # aligned but same binning
                nprw_meta=(nprw_meta.item() if hasattr(nprw_meta, "item") else nprw_meta),
                nprw_pcs=nprw_pcs if nprw_pcs is not None else np.array([], dtype=np.float32),
                nprw_explained_var=nprw_expl if nprw_expl is not None else np.array([], dtype=np.float32),

                # NPRW peaks (raw + aligned)
                nprw_peaks=(nprw_peaks if nprw_peaks is not None else np.array([], dtype=np.float32)),
                nprw_peak_t_ms=(nprw_peak_t_ms if nprw_peak_t_ms is not None else np.array([], dtype=np.float32)),
                nprw_peak_t_ms_aligned=(nprw_peak_t_ms_aligned.astype(np.float32) if nprw_peak_t_ms_aligned is not None else np.array([], dtype=np.float32)),
                
                ua_rate_hz=ua_rate_hz.astype(np.float32),
                ua_counts=ua_counts.astype(np.uint16),
                ua_t_ms=ua_t_ms.astype(np.float32),                       # original
                ua_t_ms_aligned=ua_t_ms_aligned.astype(np.float32),       # same here
                ua_meta=(ua_meta.item() if hasattr(ua_meta, "item") else ua_meta),
                ua_pcs=ua_pcs if ua_pcs is not None else np.array([], dtype=np.float32),
                ua_explained_var=ua_expl if ua_expl is not None else np.array([], dtype=np.float32),

                ua_elec=ua_elec,
                ua_region=ua_region,
                ua_region_names=ua_region_names,
                ua_port=ua_port,
                ua_nsp=ua_nsp,
                ua_idx_rows=ua_idx_rows,

                ua_peaks=(ua_peaks if ua_peaks is not None else np.array([], dtype=np.float32)),
                ua_peak_t_ms=(ua_peak_t_ms if ua_peak_t_ms is not None else np.array([], dtype=np.float32)),
                ua_peak_t_ms_aligned=(ua_peak_t_ms_aligned.astype(np.float32) if ua_peak_t_ms_aligned is not None else np.array([], dtype=np.float32)),

                stim_ms=(stim_ms_abs.astype(np.float32) if stim_ms_abs is not None else np.array([], dtype=np.float32)),

                # Alignment meta (as JSON)
                align_meta=json.dumps(aligned_meta),
                
                # behavior (ns5 samples; no anchor shift applied)
                beh_ns5_sample=beh_ns5_sample,
                beh_cam0=beh_cam0,
                beh_cam1=beh_cam1,
                beh_cam0_cols=np.array(beh_cam0_cols, dtype=object),
                beh_cam1_cols=np.array(beh_cam1_cols, dtype=object),
                beh_t_ms=beh_t_ms,
                
                # ---- VOG arrays ----
                vog_ns2_samp=vog_ns2_samp,
                vog_t_ms=vog_t_ms,
                vog_cols=vog_cols,
                vog_col_names=vog_col_names,
                
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
                nprw_rate_hz=nprw_rate_hz.astype(np.float32),
                nprw_counts=nprw_counts.astype(np.uint16),
                nprw_t_ms=nprw_t_ms.astype(np.float32),
                nprw_t_ms_aligned=nprw_t_ms_aligned.astype(np.float32),
                nprw_meta=_change_none_for_mat(nprw_meta),

                ua_rate_hz=ua_rate_hz.astype(np.float32),
                ua_counts=ua_counts.astype(np.uint16),
                ua_t_ms=ua_t_ms.astype(np.float32),
                ua_t_ms_aligned=ua_t_ms_aligned.astype(np.float32),
                ua_meta=_change_none_for_mat(ua_meta),

                ua_elec=ua_elec,
                ua_region=ua_region,
                ua_region_names=np.array(ua_region_names, dtype=object),
                ua_port=ua_port,
                ua_nsp=ua_nsp,
                ua_idx_rows=ua_idx_rows,

                stim_ms=(stim_ms_abs.astype(np.float32)
                         if stim_ms_abs is not None
                         else np.array([], dtype=np.float32)),

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
                
                ts_state_num=ts_state_num,
                ts_state_char=ts_state_char,
            )

            savemat(out_mat, mat_dict, do_compression=True)

        except Exception as e:
            print(f"[error] Failed for session {row.get('session','?')}: {e}")
            continue

if __name__ == "__main__":
    main()