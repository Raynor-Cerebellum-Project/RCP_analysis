import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the repo root to sys.path so we can import RCP_analysis
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from RCP_analysis.python.functions.config_loading import BEHV_CKPT_ROOT

def process_file(csv_path: Path, out_dir: Path):
    print(f"Processing {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        
        # Clean up 'Unnamed' in headers for processing and saving -> artifact from pandas
        # (Though we only need to clean it before saving, it helps debugging too)
        new_levels = []
        for level_idx in range(df.columns.nlevels):
            new_level_vals = [
                str(x) if not str(x).startswith("Unnamed") else "" 
                for x in df.columns.get_level_values(level_idx)
            ]
            new_levels.append(new_level_vals)
        
        # Reassign clean index (not strictly necessary for logic but good for export)
        df.columns = pd.MultiIndex.from_arrays(new_levels, names=df.columns.names)

        modified_count = 0
        
        # Iterate over columns to find likelihood columns
        # We look match 'likelihood' in Level 1
        for col in df.columns:
            # col is a tuple (level0, level1, level2)
            l0, l1, l2 = col
            
            # Check if Level 1 string ends with 'likelihood'
            # Note: l1 might be case sensitive? Inspection showed lowercase 'likelihood'
            if isinstance(l1, str) and l1.endswith("likelihood"):
                
                # Identify corresponding x and y columns
                # Construct the matching Level 1 strings for x and y
                base = l1[:-len("likelihood")] # remove 'likelihood'
                l1_x = base + "x"
                l1_y = base + "y"
                
                # Construct full column keys
                # We assume Level 0 and Level 2 match exactly (usually 'cam0' and '')
                col_x = (l0, l1_x, l2)
                col_y = (l0, l1_y, l2)
                
                if col_x in df.columns and col_y in df.columns:
                    # Apply mask -> if likelihood < 0.35, set x and y to NaN
                    mask = df[col] < 0.35
                    
                    if mask.any():
                        n_masked = mask.sum()
                        df.loc[mask, col_x] = np.nan
                        df.loc[mask, col_y] = np.nan
                        modified_count += n_masked

        if modified_count > 0:
            print(f"  corrected {modified_count} points.")
            
        out_path = out_dir / csv_path.name
        # Save with na_rep='' to make empty cells empty
        df.to_csv(out_path, na_rep='')

    except Exception as e:
        print(f"  Error processing {csv_path.name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    if not BEHV_CKPT_ROOT.exists():
        print(f"Error: Directory {BEHV_CKPT_ROOT} not found.")
        return

    out_dir = BEHV_CKPT_ROOT / "likelihood_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    files = sorted(list(BEHV_CKPT_ROOT.glob("*.csv")))
    
    if not files:
        print("No CSV files found.")
        return
        
    print(f"Found {len(files)} CSV files to process.")
    
    for f in files:
        process_file(f, out_dir)
        
    print("Done.")

if __name__ == "__main__":
    main()
