import math
from RCP_analysis.python.functions.config_loading import *

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
FIG_ROOT    = OUT_BASE / "figures" / "individual_trials"
FIG_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _strip_cam_prefix(n: str) -> str:
    n = str(n)
    if n.startswith("cam0_"): return n[len("cam0_"):]
    if n.startswith("cam1_"): return n[len("cam1_"):]
    return n

def _simple_beh_labels(names: list[str], keypoints: Tuple[str, ...] = KEYPOINTS_ORDER) -> list[str]:
    out: list[str] = []
    kps_lc = tuple(kp.lower() for kp in keypoints)
    for n in names:
        s = str(n).lower()
        kp = next((kp for kp in kps_lc if kp in s), None)
        axis = "x" if ("_x" in s or s.endswith("x")) else ("y" if ("_y" in s or s.endswith("y")) else None)
        if kp and axis:
            base = kp.replace("_", " ").title()
            ax   = axis.upper()
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

def main():
    target_dirs = [PERI_ROOT / "Target_A", PERI_ROOT / "Target_B"]
    files: list[Path] = []
    for d in target_dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.npz")))

    if not files:
        print(f"[warn] No .npz files found in {target_dirs}")
        return

    for peri_stim_npz_loc in files:
        if not peri_stim_npz_loc.exists():
            continue

        target_label = peri_stim_npz_loc.parent.name
        print(f"Processing {peri_stim_npz_loc.name} ({target_label})...")
        
        try:
            peri_stim_npz = np.load(peri_stim_npz_loc, allow_pickle=True)
            br_idx = int(peri_stim_npz["br_idx"]) # Condition index
            
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
                continue

            # Layout Calculation
            n_rows_per_cam = math.ceil(n_trials / N_COLS)
            
            # Total rows depends on which cameras are present
            n_total_rows = 0
            if has_cam0: n_total_rows += n_rows_per_cam
            if has_cam1: n_total_rows += n_rows_per_cam
            
            # Add spacer rows if we have both? Or just use constrained_layout w_pad/h_pad? 
            # Distinct groups is better visualization.
            
            fig, axes = plt.subplots(n_total_rows, N_COLS, figsize=(FIG_WIDTH_PER_COL * N_COLS, FIG_HEIGHT_PER_ROW * n_total_rows), constrained_layout=True)
            
            # Ensure axes is 2D array even if 1 row
            if n_total_rows == 1:
                axes = np.array([axes])
            if axes.ndim == 1:
                axes = axes.reshape(-1, N_COLS) # Should reliably reshape
            
            axes_flat = axes.flatten()

            # Helper to perform plotting for a chunk of axes
            current_row_start = 0

            # ---------------------------
            # PLOT CAM 0
            # ---------------------------
            if has_cam0:
                # Add a label for this section
                # We can't easily add a single title for a set of subplots in gridspec without more complex layout, 
                # but we can add text to the figure or just rely on the first subplot title.
                
                rows_slice = slice(0, n_rows_per_cam)
                # axes_chunk = axes[rows_slice].flatten() # This might be risky if axes is handled oddly by subplots with constrained_layout
                
                # Safer: iterate indices
                start_idx = 0
                
                # Label the section?
                # fig.text(0.01, 1.0 - (0.5 * n_rows_per_cam / n_total_rows), "Camera 0", rotation=90, va='center', fontsize=14, fontweight='bold')

                for i in range(n_trials):
                    ax_idx = start_idx + i
                    ax = axes_flat[ax_idx]
                    
                    if i < cam0_segs.shape[0]:
                        segs_trial = cam0_segs[i]
                        # Filter
                        segs_filt, names_filt = filter_traces(segs_trial, cam0_names)
                        plot_single_trial(ax, beh_rel_t, segs_filt, names_filt, title_prefix=f"Cam0 | Trial {i}")
                    else:
                        ax.axis("off") # Should not happen if n_trials is max of both

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
                        # Filter
                        segs_filt, names_filt = filter_traces(segs_trial, cam1_names)
                        plot_single_trial(ax, beh_rel_t, segs_filt, names_filt, title_prefix=f"Cam1 | Trial {i}")
                    else:
                        ax.axis("off")

                # Hide unused axes in this block
                for i in range(n_trials, n_rows_per_cam * N_COLS):
                    axes_flat[start_idx + i].axis("off")


            # Titles / Saving
            overall_title_raw = peri_stim_npz["overall_title"]
            overall_title = str(overall_title_raw.item()) if overall_title_raw.shape == () else str(overall_title_raw)
            fig.suptitle(f"{overall_title} - {target_label} - Individual Trials (n={n_trials})", fontsize=16)

            # Legends
            # Try to grab handles from the first valid axis we can find
            handles, labels = [], []
            for ax in axes_flat:
                if ax.has_data():
                    h, l = ax.get_legend_handles_labels()
                    if h:
                        handles, labels = h, l
                        break
            
            if handles:
                 fig.legend(handles, labels, loc='outside lower center', ncol=min(len(labels), 10), fontsize='small')

            out_dir = FIG_ROOT / target_label
            out_dir.mkdir(parents=True, exist_ok=True)
            
            if peri_stim_npz_loc.name.startswith('baseline'):
                file_name = f"{peri_stim_npz_loc.name}_trials.jpg"
            else:
                file_name = f"Cond_{br_idx:03d}_trials.jpg"
                
            out_path = out_dir / file_name
            print(f"  Saving to {out_path}...")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"Error processing {peri_stim_npz_loc.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
