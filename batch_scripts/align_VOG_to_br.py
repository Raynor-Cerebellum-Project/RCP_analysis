from pathlib import Path
import pandas as pd
import sys
import RCP_analysis as rcp

"""
Attach BR ns2 sample times to VOG CSVs using the VOG_File ↔ BR_File mapping.

Input:
    VOG/*.csv  (e.g. NRR_RW011_002.csv)

Requires:
    Metadata CSV with columns: VOG_File, BR_File
    rcp.frame2sample_br_ns2_sync
    rcp.find_ns2_by_br_index

Output:
    results/checkpoints/VOG/<cond>_VOG_aligned.csv
    (same columns as original + 'ns2_sample')
"""

# ---------- Config ----------
REPO_ROOT   = Path(__file__).resolve().parents[1]
PARAMS      = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()

OUT_BASE    = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)
BR_ROOT     = SESSION_LOC / "Blackrock"; BR_ROOT.mkdir(parents=True, exist_ok=True)
VOG_ROOT    = SESSION_LOC / "VOG"; VOG_ROOT.mkdir(parents=True, exist_ok=True)

VOG_CKPT_ROOT = OUT_BASE / "checkpoints" / "VOG"
VOG_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"

UA_CFG   = PARAMS.probes.get("UA")
VOG_SYNC = int(UA_CFG.get("VOG_sync_ch", 135))  # BR ns2 sync channel for VOG

def main():
    print(f"[scan] VOG_ROOT={VOG_ROOT}")
    print(f"[scan] BR_ROOT={BR_ROOT}")

    vog_files = sorted(VOG_ROOT.glob("*.csv"))
    if not vog_files:
        print("[warn] No VOG CSV files found.")
        return
    print(f"[info] Found {len(vog_files)} VOG files")

    # --- metadata mapping: VOG_File -> BR_File ---
    try:
        VOG_TO_BR = rcp.get_metadata_mapping(METADATA_CSV, "VOG_File", "BR_File")
    except Exception as e:
        print(f"[warn] Could not load metadata map ({METADATA_CSV.name}): {e}")
        sys.exit(1)

    for p in vog_files:
        cond = p.stem  # e.g. "NRR_RW011_002"
        print(f"\n=== VOG file: {cond} ===")

        # parse VOG index from filename (last token)
        try:
            vog_idx = int(cond.split("_")[-1])
        except Exception:
            vog_idx = None

        ns2_path = None
        br_idx = None

        if vog_idx is not None and VOG_TO_BR:
            br_idx = VOG_TO_BR.get(vog_idx)
            if br_idx is not None:
                print(f"[pairing] VOG_File={vog_idx:03d} → BR_File={br_idx:03d}")
                # IMPORTANT: use ns2 here, not ns5
                ns2_path = rcp.find_ns2_by_br_index(BR_ROOT, br_idx)
                if ns2_path is None:
                    print(f"[warn] BR_File {br_idx:03d}: ns2 not found by index; cannot attach ns2 time.")
            else:
                print(f"[warn] VOG_File={vog_idx} not found in metadata mapping.")
        else:
            print(f"[warn] Could not parse VOG index from filename '{cond}'.")

        # Load VOG CSV
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] Could not read VOG CSV {p.name}: {e}")
            continue

        # Optional sanity check for expected columns
        expected_cols = [
            "TimeStamp_ns", "FrameNumber", "PCDateTime",
            "HPos_pix", "VPos_pix",
            "H_volt", "V_volt",
            "H_head", "V_head",
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"[warn] {p.name}: missing expected columns: {missing}")

        # Attach ns2_sample if we have a BR pairing + ns2 file
        if ns2_path is not None:
            try:
                frames_corrected = len(df)  # use number of rows as frame count
                ns2_samples = rcp.frame2sample_br_ns2_sync(
                    frames_corrected,
                    ns2_path,
                    sync_chan=str(VOG_SYNC),
                )
                if len(ns2_samples) != frames_corrected:
                    print(f"[warn] ns2_samples length ({len(ns2_samples)}) "
                          f"!= frames ({frames_corrected}); clipping to min.")
                    m = min(len(ns2_samples), frames_corrected)
                    ns2_samples = ns2_samples[:m]
                    df = df.iloc[:m].reset_index(drop=True)

                # insert ns2_sample as first column
                df.insert(0, "ns2_sample", ns2_samples)
                print(f"[map] Attached NS2 time from {ns2_path.name}")
            except Exception as e:
                print(f"[warn] Could not attach NS2 time for {cond}: {e}")
        else:
            print(f"[info] No ns2 mapping for {cond}; saving VOG CSV without ns2_sample.")

        # Save aligned VOG CSV
        out_csv = VOG_CKPT_ROOT / f"{cond}_VOG_aligned.csv"
        df.to_csv(out_csv, index=False)
        print(f"[write] {out_csv}")

if __name__ == "__main__":
    main()
