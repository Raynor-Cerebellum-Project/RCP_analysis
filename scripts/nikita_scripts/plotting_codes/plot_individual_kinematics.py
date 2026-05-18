import math
from RCP_analysis.python.functions.config_loading import *

"""
Purpose:
  Visualizes individual kinematic traces for every trial in the dataset.
  Organizes plots by trial and camera (Cam0/Cam1) to allow manual inspection of trial quality.
  Useful for identifying outlier trials or tracking bugs.

Key Features:
  - Generates grid plots (one subplot per trial).
  - Filters displayed traces by keyword (e.g., "Wrist X", "Middle Y").
  - Groups plots by Camera source (Cam0 block, then Cam1 block).

Outputs:
  - Figures: Saved to `results/figures/individual_trials/`
    - One figure per condition per target.

Configuration (Global Constants):
  - TRACES_TO_PLOT (list): Substrings to filter which traces to show (e.g. ["Wrist X"]). 
                           Set to None or empty list to plot all available traces.
  - KINEMATICS_YLIM (tuple): Y-axis limits for the plots (e.g. (-4, 4)).
  - N_COLS (int): Number of columns in the subplot grid.

Adjustable Parameters:
  - FIG_WIDTH_PER_COL, FIG_HEIGHT_PER_ROW: Sizing control for the output image.
"""


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
KINEMATICS_YLIM = (-4, 4)
FIG_WIDTH_PER_COL = 3.0
FIG_HEIGHT_PER_ROW = 2.0
N_COLS = 6

# Traces to plot (case-insensitive substring match). 
# If empty list or None, plots ALL traces.
# Example: ["Wrist X", "Wrist Y", "Middle X", "Middle Y"]
TRACES_TO_PLOT = ["Wrist X", "Wrist Y", "Middle X", "Middle Y"]

# ---------------------------------------------------------------------
# ROOTS / PARAMS
# ---------------------------------------------------------------------
# Base paths loaded from config_loading
FIG_ROOT = OUT_BASE / "figures" / "individual_trials"
FIG_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _strip_cam_prefix(n: str) -> str:
    n = str(n)
    if n.startswith("cam0_"): return n[len("cam0_"):]
    if n.startswith("cam1_"): return n[len("cam1_"):]
    return n


def _simple_beh_labels(names: list[str], keypoints: tuple[str, ...] = KEYPOINTS_ORDER) -> list[str]:
    out: list[str] = []
    kps_lc = tuple(kp.lower() for kp in keypoints)
    for n in names:
        s = str(n).lower()
        kp = next((kp for kp in kps_lc if kp in s), None)
        axis = "x" if ("_x" in s or s.endswith("x")) else ("y" if ("_y" in s or s.endswith("y")) else None)
        if kp and axis:
            base = kp.replace("_", " ").title()
            ax = axis.upper()
            out.append(f"{base} {ax}")
        else:
            out.append(str(n))
    return out


def filter_traces(segs, names):
    """
    segs: (n_features, n_time)
    names: list of feature names
    Returns filtered_segs, filtered_names
    """
    if segs is None or segs.size == 0 or not names:
        return np.zeros((0, 0)), []
    
    if not TRACES_TO_PLOT:
        return segs, names

    idxs = []
    filtered_names = []
    
    # Pre-compute lower case filters
    filters_lc = [f.lower() for f in TRACES_TO_PLOT]

    for i, name in enumerate(names):
        name_lc = name.lower()
        # Check if ANY filter string is in the name
        if any(f in name_lc for f in filters_lc):
            idxs.append(i)
            filtered_names.append(name)
            
    if not idxs:
        return np.zeros((0, segs.shape[1])), []
        
    return segs[idxs, :], filtered_names


def plot_single_trial(ax, t, segs, names, title_prefix=""):
    """
    segs: (n_features, n_time)
    names: list of feature names
    """
    if segs is None or segs.size == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        return

    n_feats = segs.shape[0]
    for i in range(n_feats):
        trace = segs[i, :]
        if np.all(np.isnan(trace)):
            continue
        label = names[i] if i < len(names) else f"Feat {i}"
        ax.plot(t, trace, lw=1.0, label=label, alpha=0.8)

    ax.set_ylim(KINEMATICS_YLIM)
    ax.set_title(title_prefix, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, lw=1)


def find_all_peristim_files() -> list[tuple[Path, str, str]]:
    """
    Find all PeriStim .npz files in the updated directory structure.
    
    Returns:
        List of tuples: (file_path, condition_type, target_label)
        - condition_type: 'stim', 'control', or 'continuous_stim'
        - target_label: 'target_A', 'target_B', or 'continuous_stim'
    """
    files = []
    
    # 1. Stim reaches: stim_reaches/target_A/, stim_reaches/target_B/
    for target in ['target_A', 'target_B']:
        target_dir = PERI_ROOT / "stim_reaches" / target
        if target_dir.exists():
            for f in sorted(target_dir.glob("*.npz")):
                files.append((f, 'stim', target))
    
    # 2. Control reaches: control_reaches/target_A/, control_reaches/target_B/
    for target in ['target_A', 'target_B']:
        target_dir = PERI_ROOT / "control_reaches" / target
        if target_dir.exists():
            for f in sorted(target_dir.glob("*.npz")):
                files.append((f, 'control', target))
    
    # 3. Continuous stim: continuous_stim/
    continuous_dir = PERI_ROOT / "continuous_stim"
    if continuous_dir.exists():
        for f in sorted(continuous_dir.glob("*.npz")):
            files.append((f, 'continuous_stim', 'continuous_stim'))
    
    # 4. Legacy structure: Target_A/, Target_B/ (fallback for old data)
    for target in ['Target_A', 'Target_B']:
        target_dir = PERI_ROOT / target
        if target_dir.exists():
            for f in sorted(target_dir.glob("*.npz")):
                # Determine if stim or control from filename
                cond_type = 'control' if 'baseline' in f.name.lower() else 'stim'
                files.append((f, cond_type, target.lower()))
    
    return files


def process_single_file(peri_stim_npz_loc: Path, condition_type: str, target_label: str):
    """Process a single PeriStim file and generate the trial visualization."""
    
    print(f"Processing {peri_stim_npz_loc.name} ({condition_type}/{target_label})...")
    
    try:
        peri_stim_npz = np.load(peri_stim_npz_loc, allow_pickle=True)
        
        # Get BR index if available
        br_idx = int(peri_stim_npz["br_idx"]) if "br_idx" in peri_stim_npz else 0
        
        # Load Data
        beh_rel_t = peri_stim_npz["beh_rel_t"]
        
        # Segments: (n_trials, n_features, n_time)
        cam0_segs = peri_stim_npz["beh_cam0_segs"] if "beh_cam0_segs" in peri_stim_npz else None
        cam1_segs = peri_stim_npz["beh_cam1_segs"] if "beh_cam1_segs" in peri_stim_npz else None
        
        # Names
        raw_c0 = peri_stim_npz["beh_cam0_names"] if "beh_cam0_names" in peri_stim_npz.files else []
        raw_c1 = peri_stim_npz["beh_cam1_names"] if "beh_cam1_names" in peri_stim_npz.files else []
        cam0_names = _simple_beh_labels([_strip_cam_prefix(n) for n in (raw_c0.tolist() if isinstance(raw_c0, np.ndarray) else raw_c0)])
        cam1_names = _simple_beh_labels([_strip_cam_prefix(n) for n in (raw_c1.tolist() if isinstance(raw_c1, np.ndarray) else raw_c1)])

        # Determine N trials
        n_trials = 0
        has_cam0 = False
        has_cam1 = False

        if cam0_segs is not None and cam0_segs.ndim == 3 and cam0_segs.shape[0] > 0:
            n_trials = max(n_trials, cam0_segs.shape[0])
            has_cam0 = True
        
        if cam1_segs is not None and cam1_segs.ndim == 3 and cam1_segs.shape[0] > 0:
            n_trials = max(n_trials, cam1_segs.shape[0])
            has_cam1 = True

        if n_trials == 0:
            print(f"  No trials found for {peri_stim_npz_loc.name}")
            return

        # Layout Calculation
        n_rows_per_cam = math.ceil(n_trials / N_COLS)
        
        # Total rows depends on which cameras are present
        n_total_rows = 0
        if has_cam0: n_total_rows += n_rows_per_cam
        if has_cam1: n_total_rows += n_rows_per_cam
        
        fig, axes = plt.subplots(
            n_total_rows, N_COLS, 
            figsize=(FIG_WIDTH_PER_COL * N_COLS, FIG_HEIGHT_PER_ROW * n_total_rows), 
            constrained_layout=True
        )
        
        # Ensure axes is 2D array even if 1 row
        if n_total_rows == 1:
            axes = np.array([axes])
        if axes.ndim == 1:
            axes = axes.reshape(-1, N_COLS)
        
        axes_flat = axes.flatten()

        current_row_start = 0

        # ---------------------------
        # PLOT CAM 0
        # ---------------------------
        if has_cam0:
            start_idx = 0
            
            for i in range(n_trials):
                ax_idx = start_idx + i
                ax = axes_flat[ax_idx]
                
                if i < cam0_segs.shape[0]:
                    segs_trial = cam0_segs[i]
                    segs_filt, names_filt = filter_traces(segs_trial, cam0_names)
                    plot_single_trial(ax, beh_rel_t, segs_filt, names_filt, title_prefix=f"Cam0 | Trial {i}")
                else:
                    ax.axis("off")

            # Hide unused axes in this block
            for i in range(n_trials, n_rows_per_cam * N_COLS):
                axes_flat[start_idx + i].axis("off")
            
            current_row_start += n_rows_per_cam
        
        # ---------------------------
        # PLOT CAM 1
        # ---------------------------
        if has_cam1:
            start_idx = current_row_start * N_COLS
            
            for i in range(n_trials):
                ax_idx = start_idx + i
                ax = axes_flat[ax_idx]
                
                if i < cam1_segs.shape[0]:
                    segs_trial = cam1_segs[i]
                    segs_filt, names_filt = filter_traces(segs_trial, cam1_names)
                    plot_single_trial(ax, beh_rel_t, segs_filt, names_filt, title_prefix=f"Cam1 | Trial {i}")
                else:
                    ax.axis("off")

            # Hide unused axes in this block
            for i in range(n_trials, n_rows_per_cam * N_COLS):
                if start_idx + i < len(axes_flat):
                    axes_flat[start_idx + i].axis("off")

        # Titles / Saving
        if "overall_title" in peri_stim_npz.files:
            overall_title_raw = peri_stim_npz["overall_title"]
            overall_title = str(overall_title_raw.item()) if overall_title_raw.shape == () else str(overall_title_raw)
        else:
            overall_title = peri_stim_npz_loc.stem
            
        fig.suptitle(f"{overall_title} - {condition_type}/{target_label} - Individual Trials (n={n_trials})", fontsize=16)

        # Legends
        handles, labels = [], []
        for ax in axes_flat:
            if ax.has_data():
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
        
        if handles:
            fig.legend(handles, labels, loc='outside lower center', ncol=min(len(labels), 10), fontsize='small')

        # Output directory: organized by condition_type/target_label
        out_dir = FIG_ROOT / condition_type / target_label
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if 'baseline' in peri_stim_npz_loc.name.lower():
            file_name = f"{peri_stim_npz_loc.stem}_trials.jpg"
        else:
            file_name = f"BR_{br_idx:03d}_trials.jpg"
            
        out_path = out_dir / file_name
        print(f"  Saving to {out_path}...")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    except Exception as e:
        print(f"Error processing {peri_stim_npz_loc.name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    print(f"Session: {PARAMS.session}")
    print(f"PeriStim Root: {PERI_ROOT}")
    print(f"Output Root: {FIG_ROOT}")
    print()
    
    # Find all files using the new structure
    all_files = find_all_peristim_files()
    
    if not all_files:
        print(f"[warn] No .npz files found in {PERI_ROOT}")
        print("  Checked directories:")
        print(f"    - {PERI_ROOT / 'stim_reaches'}")
        print(f"    - {PERI_ROOT / 'control_reaches'}")
        print(f"    - {PERI_ROOT / 'continuous_stim'}")
        print(f"    - {PERI_ROOT / 'Target_A'} (legacy)")
        print(f"    - {PERI_ROOT / 'Target_B'} (legacy)")
        return
    
    print(f"Found {len(all_files)} PeriStim files to process:")
    
    # Group by condition type for summary
    by_condition = {}
    for f, cond_type, target in all_files:
        key = f"{cond_type}/{target}"
        if key not in by_condition:
            by_condition[key] = []
        by_condition[key].append(f)
    
    for key, files in sorted(by_condition.items()):
        print(f"  {key}: {len(files)} files")
    print()
    
    # Process each file
    for peri_stim_npz_loc, condition_type, target_label in all_files:
        process_single_file(peri_stim_npz_loc, condition_type, target_label)
    
    print(f"\n[done] Results saved to {FIG_ROOT}")


if __name__ == "__main__":
    main()