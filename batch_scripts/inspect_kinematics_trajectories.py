"""
Plot Kinematics Trajectories for Trial Inpection And Manual Removal
================================================
Each subplot title shows:
  - "Trial:i (Raw:R)" where:
    - i = position in this plot (for reference)
    - R = stable raw stim-pulse index -> THIS is what goes in manual_trial_remove.csv
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pathlib import Path

from RCP_analysis.python.functions.config_loading import *

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
N_COLS = 6
FIG_WIDTH_PER_COL = 2.5
FIG_HEIGHT_PER_ROW = 2.5
CAMERA = "cam1"  # Which camera to use
KEYPOINT = "middle"  # Which keypoint (middle finger)

# ---------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------
FIG_ROOT = OUT_BASE / "figures" / "kinematics_trajectories"
FIG_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def find_xy_indices(names: list, keypoint: str = "middle") -> tuple:
    """Find X and Y indices for a keypoint in the names list."""
    kp_lower = keypoint.lower()
    idx_x, idx_y = None, None
    
    for i, name in enumerate(names):
        name_lower = str(name).lower()
        if kp_lower in name_lower:
            if '_x' in name_lower or name_lower.endswith('x'):
                idx_x = i
            elif '_y' in name_lower or name_lower.endswith('y'):
                idx_y = i
    
    return idx_x, idx_y


def plot_xy_trajectory(ax, x, y, t, title: str):
    """
    Plot X-Y trajectory with time coloring.
    
    Parameters:
        ax: matplotlib axis
        x, y: position arrays
        t: time array (for coloring)
        title: subplot title
    """
    # Handle NaN values
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=7)
        return
    
    # Create line segments for color mapping
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Color by time
    norm = Normalize(vmin=t.min(), vmax=t.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=1.5, alpha=0.8)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    
    # Mark start (green) and end (red)
    ax.scatter(x[0], y[0], c='green', s=30, zorder=5, label='Start')
    ax.scatter(x[-1], y[-1], c='red', s=30, zorder=5, label='End')
    
    # Mark t=0 if in range
    t0_idx = np.argmin(np.abs(t))
    if 0 < t0_idx < len(x) - 1:
        ax.scatter(x[t0_idx], y[t0_idx], c='blue', s=40, marker='x', zorder=6, linewidths=2)
    
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=5)


def find_all_peristim_files() -> list:
    """Find all PeriStim .npz files."""
    files = []
    
    # Stim and control reaches
    for reach_type in ['stim_reaches', 'control_reaches']:
        for target in ['target_A', 'target_B']:
            target_dir = PERI_ROOT / reach_type / target
            if target_dir.exists():
                cond_type = 'stim' if 'stim' in reach_type else 'control'
                for f in sorted(target_dir.glob("*.npz")):
                    files.append((f, cond_type, target))
    
    # Continuous stim
    continuous_dir = PERI_ROOT / "continuous_stim"
    if continuous_dir.exists():
        for f in sorted(continuous_dir.glob("*.npz")):
            files.append((f, 'continuous_stim', 'continuous_stim'))
    
    # # Legacy structure
    # for target in ['Target_A', 'Target_B']:
    #     target_dir = PERI_ROOT / target
    #     if target_dir.exists():
    #         for f in sorted(target_dir.glob("*.npz")):
    #             cond_type = 'control' if 'baseline' in f.name.lower() else 'stim'
    #             files.append((f, cond_type, target.lower()))
    
    return files


def process_single_file(npz_path: Path, condition_type: str, target_label: str):
    """Process one file and generate X-Y trajectory grid."""
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # Get metadata
        br_idx = int(data["br_idx"]) if "br_idx" in data else 0
        intan_filename = str(data["sess"]) if "sess" in data else "Unknown"
        raw_trial_indices = data["raw_trial_indices"] if "raw_trial_indices" in data.files else None

        print(f"[IKT] Processing BR {br_idx}, {intan_filename}, {condition_type}, {target_label}:")
        
        # Get time axis
        t = data["beh_rel_t"]
        
        # Get segments for selected camera
        cam_key = f"beh_{CAMERA}_segs"
        names_key = f"beh_{CAMERA}_names"
        
        if cam_key not in data:
            print(f"[IKT] No {CAMERA} data found, skipping")
            return
        
        segs = data[cam_key]  # (n_trials, n_features, n_time)
        names = data[names_key].tolist() if names_key in data else []
        
        if segs.ndim != 3 or segs.shape[0] == 0:
            print(f"[IKT] No valid trials, skipping")
            return
        
        n_trials = segs.shape[0]
        
        # Find X and Y indices for middle finger
        idx_x, idx_y = find_xy_indices(names, KEYPOINT)
        
        if idx_x is None or idx_y is None:
            print(f"[IKT] Could not find {KEYPOINT} X/Y, skipping")
            print(f"[IKT] Available: {names}")
            return
        
        # Create figure
        n_rows = math.ceil(n_trials / N_COLS)
        fig, axes = plt.subplots(
            n_rows, N_COLS,
            figsize=(FIG_WIDTH_PER_COL * N_COLS, FIG_HEIGHT_PER_ROW * n_rows),
            constrained_layout=True
        )
        
        # Ensure axes is 2D
        if n_rows == 1:
            axes = np.array([axes])
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten()
        
        # Plot each trial
        for i in range(n_trials):
            ax = axes_flat[i]
            
            x = segs[i, idx_x, :]
            y = segs[i, idx_y, :]
            
            # Raw index = stable absolute stim-pulse number -> use this in manual_trial_remove.csv
            raw_idx = int(raw_trial_indices[i]) if raw_trial_indices is not None else i
            title = f"{intan_filename[:15]}\nBR:{br_idx} | Trial:{i+1} (Raw:{raw_idx})"
            
            plot_xy_trajectory(ax, x, y, t, title)
        
        # Hide unused axes
        for i in range(n_trials, len(axes_flat)):
            axes_flat[i].axis('off')
        
        # Main title
        if "overall_title" in data.files:
            raw_title = data["overall_title"]
            overall_title = str(raw_title.item()) if raw_title.shape == () else str(raw_title)
        else:
            overall_title = npz_path.stem
        
        fig.suptitle(
            f"{overall_title}\n{KEYPOINT.title()} Finger X-Y | {condition_type} / {target_label} | n={n_trials}\n"
            f"Session: {intan_filename} | BR: {br_idx}  —  put Raw:# in manual_trial_remove.csv",
            fontsize=10
        )
        
        # Add colorbar for time
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=t.min(), vmax=t.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes_flat[:n_trials], shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label('Time (ms)', fontsize=8)
        
        # Save
        out_dir = FIG_ROOT / condition_type / target_label
        out_dir.mkdir(parents=True, exist_ok=True)
        
        file_name = f"BR_{br_idx:03d}_xy.jpg" if br_idx else f"{npz_path.stem}_xy.jpg"
        out_path = out_dir / file_name
        
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        
    except Exception as e:
        print(f"[IKT] Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print(f"\n[IKT] Session: {PARAMS.session}, Camera: {CAMERA}, Keypoint: {KEYPOINT}")
    
    all_files = find_all_peristim_files()
    
    if not all_files:
        print(f"[IKT] No .npz files found in {PERI_ROOT}")
        return
    
    print(f"\n[IKT] Found {len(all_files)} files\n")
    
    for npz_path, condition_type, target_label in all_files:
        process_single_file(npz_path, condition_type, target_label)
    
    print(f"\n[IKT] Done!")


if __name__ == "__main__":
    main()