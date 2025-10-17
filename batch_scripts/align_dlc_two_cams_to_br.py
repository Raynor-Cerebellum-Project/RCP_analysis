from __future__ import annotations
from pathlib import Path
import pandas as pd
import RCP_analysis as rcp

# ---------- CONFIG ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
DATA_ROOT = rcp.resolve_data_root(PARAMS)
OUT_BASE  = rcp.resolve_output_root(PARAMS); OUT_BASE.mkdir(parents=True, exist_ok=True)

BR_ROOT    = (DATA_ROOT / PARAMS.blackrock_rel); BR_ROOT.mkdir(parents=True, exist_ok=True)
VIDEO_ROOT = (DATA_ROOT / PARAMS.video_rel);     VIDEO_ROOT.mkdir(parents=True, exist_ok=True)

BEHV_BUNDLES   = OUT_BASE / "bundles" / "behavior"
BEHV_CKPT_ROOT = OUT_BASE / "checkpoints" / "behavior"
BEHV_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

METADATA_CSV  = (DATA_ROOT / PARAMS.metadata_rel).resolve(); METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
# -------------------------- OCR / DLC / NSx helpers (styled like the other script) --------------------------

def main():
    print(f"[scan] VIDEO_ROOT={VIDEO_ROOT}")
    print(f"[scan] BR_ROOT={BR_ROOT}")
    per_trial = rcp.find_per_trial_inputs(VIDEO_ROOT)
    if not per_trial:
        print("[warn] No complete trials with {Cam-0, Cam-1} × {OCR, DLC} found.")
        return

    print(f"[info] Found {len(per_trial)} trials.")

    # load once
    try:
        VIDEO_TO_BR = rcp.load_video_to_br_map(METADATA_CSV)
    except Exception as e:
        print(f"[warn] Could not load metadata map ({METADATA_CSV.name}): {e}")
        VIDEO_TO_BR = {}
        
    for trial, d in sorted(per_trial.items()):
        print(f"\n=== Trial: {trial} ===")
        cam_files = d['dlc'].keys()

        # Load OCR and DLC
        ocr0 = rcp.load_ocr_map(d['ocr'][0]); ocr1 = rcp.load_ocr_map(d['ocr'][1])
        dlc0 = rcp.load_dlc(d['dlc'][0]);     dlc1 = rcp.load_dlc(d['dlc'][1])

        # Expand to corrected index
        e0 = rcp.expand_dlc_to_corrected(dlc0, ocr0)
        e1 = rcp.expand_dlc_to_corrected(dlc1, ocr1)

        # per-trial
        vid_idx = rcp.extract_video_idx_from_trial(trial)
        ns_path = None
        if vid_idx is not None and VIDEO_TO_BR:
            br_idx = VIDEO_TO_BR.get(vid_idx)
            if br_idx is not None:
                print(f"[pair] Video_File={vid_idx:03d} → BR_File={br_idx:03d} (from {METADATA_CSV.name})")
                ns_path = rcp.find_ns5_by_br_index(BR_ROOT, br_idx)
                if ns_path is None:
                    print(f"[warn] BR_File {br_idx:03d} not found by index; fallback to trial-name search.")
        if ns_path is None:
            ns_path = rcp.find_ns5_for_trial(BR_ROOT, trial)  # your existing fallback

        if ns_path is not None:
            try:
                n_corr = max(len(e0.index), len(e1.index))
                ns5_samples, _, _ = rcp.corrected_to_time_ns5(n_corr, ns_path,
                                          sync_chan=int(getattr(PARAMS, "camera_sync_ch", 134)))
                e0 = e0.reindex(range(n_corr)); e1 = e1.reindex(range(n_corr))
                e0.insert(0, "ns5_sample", ns5_samples);   e1.insert(0, "ns5_sample", ns5_samples)
                print(f"[map] Attached NS5 time from {ns_path.name}")
            except Exception as e:
                print(f"[warn] Could not attach NS5 time for {trial}: {e}")

        # Save combined CSV (side-by-side)
        e0_cam = pd.concat({"cam0": e0}, axis=1)
        e1_cam = pd.concat({"cam1": e1}, axis=1)
        both = pd.concat([e0_cam, e1_cam], axis=1)
        both_csv = BEHV_CKPT_ROOT / f"{trial}_both_cams_aligned.csv"
        both.to_csv(both_csv, index_label="CORRECTED_framenum")
        print(f"[write] {both_csv}")

if __name__ == "__main__":
    main()
