from pathlib import Path
import pandas as pd
import sys
import RCP_analysis as rcp
""" 
    This script aligns two camera perspectives to the Blackrock recording. It scans through all paired cam-0 and cam-1 .csv files and outputs aligned timing based on frame mappings identified from OCR_frame_mapping_BT_edit.py script.
    The alignment is based on the camera sync pulse, may need to double-check which channel the sync pulse is at.
    Input:
        Pairs of DLC .csv files (cam-0 and cam-1)
        Pairs of OCR frame mapping .csv file (cam-0 and cam-1)
    Output:
        Aligned .csv file for both perspective
"""
# ---------- Config ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)
BR_ROOT = SESSION_LOC / "Blackrock"; BR_ROOT.mkdir(parents=True, exist_ok=True)
VIDEO_ROOT = SESSION_LOC / "Video"; VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
BEHV_BUNDLES   = OUT_BASE / "bundles" / "behavior"
BEHV_CKPT_ROOT = OUT_BASE / "checkpoints" / "behavior"; BEHV_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"

def main():
    print(f"[scan] VIDEO_ROOT={VIDEO_ROOT}")
    print(f"[scan] BR_ROOT={BR_ROOT}")
    per_cond = rcp.find_per_cond_inputs(VIDEO_ROOT)
    if not per_cond:
        print("[warn] No complete conditions with {Cam-0, Cam-1} × {OCR, DLC} found")
        return
    print(f"[info] Found {len(per_cond)} trials")

    try: # Check if csv file exists
        VIDEO_TO_BR = rcp.get_metadata_mapping(METADATA_CSV, "Video_File", "BR_File")
    except Exception as e:
        print(f"[warn] Could not load metadata map ({METADATA_CSV.name}): {e}")
        sys.exit(1)
        
    for cond, d in sorted(per_cond.items()):
        print(f"\n=== Condition: {cond} ===")
        # Load OCR and DLC
        ocr0 = rcp.load_ocr_map(d['ocr'][0]); ocr1 = rcp.load_ocr_map(d['ocr'][1])
        dlc0 = rcp.load_dlc(d['dlc'][0]);     dlc1 = rcp.load_dlc(d['dlc'][1])

        # Align OCR -- eventually remove?
        aligned_dlc0 = rcp.align_dlc_to_corrected(dlc0, ocr0)
        aligned_dlc1 = rcp.align_dlc_to_corrected(dlc1, ocr1)

        try:
            vid_idx = int(cond.split("_")[2])  # split into NRR, RW###, ###
        except Exception:
            vid_idx = None
    
        # Now map vid_idx to br_idx
        ns5_path = None
        if vid_idx is not None and VIDEO_TO_BR:
            br_idx = VIDEO_TO_BR.get(vid_idx)
            if br_idx is not None:
                print(f"[pairing] Video_File={vid_idx:03d} → BR_File={br_idx:03d})")
                ns5_path = rcp.find_ns5_by_br_index(BR_ROOT, br_idx)
                if ns5_path is None:
                    print(f"[warn] BR_File {br_idx:03d} not found by index; fallback to condition-name search.")
        if ns5_path is None:
            ns5_path = rcp.find_ns5_for_cond(BR_ROOT, cond)  # fallback using condition ID, should be the same

        # If nothing weird with mapping vid_idx to br path, start outputting
        if ns5_path is not None:
            try:
                frames_corrected = max(len(aligned_dlc0.index), len(aligned_dlc1.index)) # use the max of the two as length
                ns5_samples = rcp.frame2sample_br_ns5_sync(frames_corrected, ns5_path,
                                          sync_chan=str(getattr(PARAMS, "camera_sync_ch", '134')))
                aligned_dlc0 = aligned_dlc0.reindex(range(frames_corrected)); aligned_dlc1 = aligned_dlc1.reindex(range(frames_corrected))
                aligned_dlc0.insert(0, "ns5_sample", ns5_samples);   aligned_dlc1.insert(0, "ns5_sample", ns5_samples)
                # Put in max lenght vector (What if mismatch?)
                print(f"[map] Attached NS5 time from {ns5_path.name}")
            except Exception as e:
                print(f"[warn] Could not attach NS5 time for {cond}: {e}")

        # Save combined CSV (side-by-side)
        aligned_samps_dlc0 = pd.concat({"cam0": aligned_dlc0}, axis=1)
        aligned_samps_dlc1 = pd.concat({"cam1": aligned_dlc1}, axis=1)
        both_samps_aligned = pd.concat([aligned_samps_dlc0, aligned_samps_dlc1], axis=1)
        both_csv = BEHV_CKPT_ROOT / f"{cond}_both_cams_aligned.csv"
        both_samps_aligned.to_csv(both_csv, index_label="CORRECTED_framenum")
        print(f"[write] {both_csv}")

if __name__ == "__main__":
    main()
