from __future__ import annotations
from pathlib import Path
import argparse
import RCP_analysis as rcp

# ---------- CONFIG ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
DATA_ROOT = rcp.resolve_data_root(PARAMS)
OUT_BASE  = rcp.resolve_output_root(PARAMS); OUT_BASE.mkdir(parents=True, exist_ok=True)

BR_ROOT    = (DATA_ROOT / PARAMS.blackrock_rel); BR_ROOT.mkdir(parents=True, exist_ok=True)
VIDEO_ROOT = (DATA_ROOT / PARAMS.video_rel);     VIDEO_ROOT.mkdir(parents=True, exist_ok=True)

BEHV_CKPT_ROOT = OUT_BASE / "checkpoints" / "behavior"
BEHV_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

METADATA_CSV  = (DATA_ROOT / PARAMS.metadata_rel).resolve(); METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Align a single camera's DLC to OCR (and optionally BR time).")
    ap.add_argument("--cam", type=int, choices=(0, 1), default=int(getattr(PARAMS, "video_cam", 0)),
                    help="Which camera to process (0 or 1). Default comes from PARAMS.video_cam or 0.")
    args = ap.parse_args()
    CAM = int(args.cam)

    # Load metadata map once
    try:
        VIDEO_TO_BR = rcp.load_video_to_br_map(METADATA_CSV)
    except Exception as e:
        print(f"[warn] Could not load metadata map ({METADATA_CSV.name}): {e}")
        VIDEO_TO_BR = {}

    print(f"[scan] VIDEO_ROOT={VIDEO_ROOT}")
    print(f"[scan] BR_ROOT={BR_ROOT}")
    per_trial = rcp.find_per_trial_inputs(VIDEO_ROOT)
    if not per_trial:
        print("[warn] No OCR/DLC files found.")
        return

    # Keep only trials that have the requested cam for BOTH OCR and DLC
    valid = {
        trial: d for trial, d in per_trial.items()
        if (CAM in d.get("ocr", {})) and (CAM in d.get("dlc", {}))
    }

    if not valid:
        print(f"[warn] No trials with both OCR and DLC for Cam-{CAM}.")
        return

    print(f"[info] Found {len(valid)} trials with Cam-{CAM}.")

    for trial, d in sorted(valid.items()):
        print(f"\n=== Trial: {trial} (Cam-{CAM}) ===")

        # Load OCR + DLC
        ocr = rcp.load_ocr_map(d["ocr"][CAM]); print(f"[ocr]  {d['ocr'][CAM].name}")
        dlc = rcp.load_dlc(d["dlc"][CAM]);     print(f"[dlc]  {d['dlc'][CAM].name}")

        # Expand to corrected index
        try:
            aligned = rcp.expand_dlc_to_corrected(dlc, ocr)
        except Exception as e:
            print(f"[warn] expand_dlc_to_corrected failed for {trial}: {e}")
            continue
        
        # --- Choose NS5 via metadata (Video_File -> BR_File) ---
        ns_path = None
        vid_idx = rcp.extract_video_idx_from_trial(trial)
        br_idx  = None

        if vid_idx is not None and VIDEO_TO_BR:
            br_idx = VIDEO_TO_BR.get(vid_idx)
            if br_idx is not None:
                print(f"[pair] Video_File={vid_idx:03d} → BR_File={br_idx:03d} (from {METADATA_CSV.name})")
                ns_path = rcp.find_nsx_by_br_index(BR_ROOT, br_idx)
                if ns_path is None:
                    print(f"[warn] BR_File {br_idx:03d} not found by index; will fallback to trial-name search.")
            else:
                print(f"[warn] No BR_File mapping for Video_File={vid_idx:03d} in {METADATA_CSV.name}; fallback.")

        if ns_path is None:
            # Fallback: old behavior—trial-name substring
            ns_path = rcp.find_nsx_for_trial(BR_ROOT, trial)
            if ns_path is not None:
                print(f"[pair] Fallback: trial substring → {ns_path.name}")
            else:
                print(f"[warn] No BR file found by trial substring fallback for {trial}")

        # Attach NS5 time if found
        if ns_path is not None:
            print(f"[file] Using BR file: {ns_path.name}")
            try:
                n_corr = len(aligned.index)
                ns5_samples, _, _ = rcp.corrected_to_time_ns5(
                    n_corr, ns_path, sync_chan=int(getattr(PARAMS, "camera_sync_ch", 134))
                )
                aligned = aligned.reindex(range(n_corr))
                if "ns5_sample" in aligned.columns:
                    aligned["ns5_sample"] = ns5_samples
                else:
                    aligned.insert(0, "ns5_sample", ns5_samples)
                print(f"[map] Attached NS5 time ({n_corr} frames)")
            except Exception as e:
                print(f"[warn] Could not attach NS5 time for {trial}: {e}")
        else:
            print(f"[warn] Skipping ns5_sample attachment for {trial} (no BR file)")

        # Save CSV for the single camera
        out_csv = BEHV_CKPT_ROOT / f"{trial}_Cam-{CAM}_aligned.csv"
        aligned.to_csv(out_csv, index_label="CORRECTED_framenum")
        print(f"[write] {out_csv}")

if __name__ == "__main__":
    main()
