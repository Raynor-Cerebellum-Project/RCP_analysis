"""
Plateau Analysis Batch Script
=============================

Batch processing script for reach endpoint / plateau detection analysis.

This script:
  1. Loads all peri-reach condition files for a session.
  2. Smooths X/Y traces using Butterworth low-pass filtering.
  3. Computes distance from start.
  4. Detects plateau onset per trial using configurable multi-signal stability.
  5. Generates per-condition summary plots.
  6. Compares plateau onset times across conditions by target.
  7. Runs selected-condition statistical analysis.

"""

from __future__ import annotations

import os
import sys
import re
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy import stats
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path(os.getcwd()).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from RCP_analysis.python.functions.config_loading import *  # noqa: F401,F403

    CONFIG_LOADED = True
    print("[info] Config loaded from RCP_analysis")
    print(f"[info] Session: {PARAMS.session}")  # noqa: F405
    print(f"[info] PERI_ROOT: {PERI_ROOT}")     # noqa: F405

except ImportError:
    CONFIG_LOADED = False
    print("[warn] Could not load RCP_analysis config.")
    print("[warn] Please manually set PERI_ROOT below.")

    # -------------------------------------------------------------------------
    # Manual fallback config
    # -------------------------------------------------------------------------
    PERI_ROOT = Path("/path/to/peri_data")  # CHANGE THIS IF CONFIG IMPORT FAILS


# Main analysis settings
CAMERA_TO_USE = "cam1"
KEYPOINT_BASE = "middle"

PRE_ZERO_MS = 500.0
POST_ZERO_MS = 500.0

ONSET_TIME_XLIM = (0, POST_ZERO_MS)  # ms

CONTROL_BR_INDICES = [1]

# Butterworth smoothing parameters
SMOOTHING_METHOD = "butterworth"
SMOOTHING_PARAMS = {
    "cutoff_hz": 8.0,
    "order": 4,
}
SMOOTH_XY = True

# Plateau detection parameters
DETECTION_PARAMS = {
    "use_x": True,
    "use_y": False,
    "use_dist": True,
    "x_max_change_pct": 4.0,
    "y_max_change_pct": 4.0,
    "dist_max_change_pct": 4.0,
    "min_duration_ms": 100,
    "search_start_ms": 100,
    "search_end_ms": None,
}

# Conditions used in selected statistical comparison
COMPARE_CONDITIONS = [1, 12, 13, 14, 15, 16, 17]

# Plot/save settings
PLOT_EACH_CONDITION = True
SAVE_FIGS = True
SAVE_STATS = False

RESULTS_ROOT = Path(PERI_ROOT).parents[1]
SAVE_DIR = RESULTS_ROOT / "figures" / "plateau_analysis"

PER_CONDITION_FIG_DIR = SAVE_DIR / "per_condition"
COMPARISON_FIG_DIR = SAVE_DIR / "comparisons"
STATS_FIG_DIR = SAVE_DIR / "stats"
TABLE_DIR = SAVE_DIR / "tables"

for d in [
    SAVE_DIR,
    PER_CONDITION_FIG_DIR,
    COMPARISON_FIG_DIR,
    STATS_FIG_DIR,
    TABLE_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Pulse width used for condition labels
PULSE_WIDTH_MS = 2.5


# =============================================================================
# Matplotlib defaults
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 10,
        "figure.figsize": (12, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


# =============================================================================
# Smoothing
# =============================================================================

class SmoothingMethods:
    """Smoothing methods. Only Butterworth is retained for batch analysis."""

    @staticmethod
    def butterworth(
        trace: np.ndarray,
        dt: float,
        cutoff_hz: float = 8.0,
        order: int = 4,
    ) -> np.ndarray:
        """
        Butterworth low-pass filter.

        Parameters
        ----------
        trace:
            1D signal.
        dt:
            Sampling interval in milliseconds.
        cutoff_hz:
            Low-pass cutoff in Hz.
        order:
            Butterworth filter order.
        """
        fs = 1000.0 / dt
        nyq = fs / 2.0

        if cutoff_hz >= nyq:
            cutoff_hz = nyq * 0.9

        b, a = butter(order, cutoff_hz / nyq, btype="low")

        valid_mask = np.isfinite(trace)
        smoothed = trace.copy()

        if np.sum(valid_mask) < 3 * order:
            return smoothed

        if np.all(valid_mask):
            smoothed = filtfilt(b, a, trace)
        else:
            trace_interp = trace.copy()
            nans = ~valid_mask

            if np.any(nans) and np.any(valid_mask):
                trace_interp[nans] = np.interp(
                    np.flatnonzero(nans),
                    np.flatnonzero(valid_mask),
                    trace[valid_mask],
                )

            smoothed = filtfilt(b, a, trace_interp)
            smoothed[nans] = np.nan

        return smoothed


# =============================================================================
# Plateau detection
# =============================================================================

# =============================================================================
# Plateau detection - Modified to return failure reasons
# =============================================================================

class PlateauDetectionMethods:
    """Plateau detection algorithms."""

    @staticmethod
    def compute_stability_trace(
        signal: np.ndarray,
        time_ms: np.ndarray,
        window_ms: float = 75,
        threshold_pct: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute whether a signal is stable at each timepoint.

        Stability is defined as:

            100 * (max(window) - min(window)) / mean(abs(window)) <= threshold_pct

        Returns
        -------
        stability:
            Boolean array, True where signal is stable.
        percent_change:
            Percent change value for each window start.
        """
        dt = np.mean(np.diff(time_ms))
        window_samples = max(1, int(window_ms / dt))
        n = len(signal)

        stability = np.zeros(n, dtype=bool)
        percent_change = np.full(n, np.nan)

        for i in range(n - window_samples):
            window = signal[i:i + window_samples]

            if np.any(np.isnan(window)):
                continue

            window_range = np.max(window) - np.min(window)
            window_mean = np.mean(np.abs(window))

            if window_mean == 0:
                pct = 0 if window_range == 0 else np.inf
            else:
                pct = (window_range / window_mean) * 100.0

            percent_change[i] = pct
            stability[i] = pct <= threshold_pct

        return stability, percent_change

    @staticmethod
    def stable_region_multi(
        x_smooth: np.ndarray,
        y_smooth: np.ndarray,
        dist_smooth: np.ndarray,
        time_ms: np.ndarray,
        min_duration_ms: float = 75,
        x_max_change_pct: float = 3.0,
        y_max_change_pct: float = 3.0,
        dist_max_change_pct: float = 3.0,
        use_x: bool = True,
        use_y: bool = False,
        use_dist: bool = True,
        search_start_ms: Optional[float] = None,
        search_end_ms: Optional[float] = None,
        return_stability_traces: bool = False,
    ) -> Dict:
        """
        Find the first region where selected signals are stable simultaneously.

        Selected signals are controlled by use_x/use_y/use_dist.
        """
        if not (use_x or use_y or use_dist):
            raise ValueError("Must select at least one signal for plateau detection.")

        if search_start_ms is not None:
            start_idx = np.searchsorted(time_ms, search_start_ms)
        else:
            start_idx = 0

        if search_end_ms is not None:
            end_idx = np.searchsorted(time_ms, search_end_ms)
        else:
            end_idx = len(time_ms)

        x_stable_trace, x_pct_trace = PlateauDetectionMethods.compute_stability_trace(
            x_smooth,
            time_ms,
            min_duration_ms,
            x_max_change_pct,
        )

        y_stable_trace, y_pct_trace = PlateauDetectionMethods.compute_stability_trace(
            y_smooth,
            time_ms,
            min_duration_ms,
            y_max_change_pct,
        )

        d_stable_trace, d_pct_trace = PlateauDetectionMethods.compute_stability_trace(
            dist_smooth,
            time_ms,
            min_duration_ms,
            dist_max_change_pct,
        )

        combined_stable = np.ones(len(time_ms), dtype=bool)
        signals_used = []

        if use_x:
            combined_stable &= x_stable_trace
            signals_used.append("X")

        if use_y:
            combined_stable &= y_stable_trace
            signals_used.append("Y")

        if use_dist:
            combined_stable &= d_stable_trace
            signals_used.append("Dist")

        onset_idx = None
        for i in range(start_idx, min(end_idx, len(combined_stable))):
            if combined_stable[i]:
                onset_idx = i
                break

        # Determine failure reason if not found
        failure_reason = None
        failure_details = {}
        
        if onset_idx is None:
            # Check which signals never stabilized in the search window
            search_slice = slice(start_idx, min(end_idx, len(time_ms)))
            
            x_ever_stable = np.any(x_stable_trace[search_slice]) if use_x else True
            y_ever_stable = np.any(y_stable_trace[search_slice]) if use_y else True
            d_ever_stable = np.any(d_stable_trace[search_slice]) if use_dist else True
            
            unstable_signals = []
            if use_x and not x_ever_stable:
                unstable_signals.append("X")
            if use_y and not y_ever_stable:
                unstable_signals.append("Y")
            if use_dist and not d_ever_stable:
                unstable_signals.append("Dist")
            
            if unstable_signals:
                failure_reason = f"Never stable: {', '.join(unstable_signals)}"
            else:
                # Signals were stable at different times but never simultaneously
                failure_reason = "Signals never simultaneously stable"
            
            # Get minimum percent change achieved for each signal
            if use_x:
                valid_x_pct = x_pct_trace[search_slice]
                valid_x_pct = valid_x_pct[~np.isnan(valid_x_pct)]
                failure_details["x_min_pct"] = np.min(valid_x_pct) if len(valid_x_pct) > 0 else np.nan
                failure_details["x_threshold"] = x_max_change_pct
            
            if use_y:
                valid_y_pct = y_pct_trace[search_slice]
                valid_y_pct = valid_y_pct[~np.isnan(valid_y_pct)]
                failure_details["y_min_pct"] = np.min(valid_y_pct) if len(valid_y_pct) > 0 else np.nan
                failure_details["y_threshold"] = y_max_change_pct
            
            if use_dist:
                valid_d_pct = d_pct_trace[search_slice]
                valid_d_pct = valid_d_pct[~np.isnan(valid_d_pct)]
                failure_details["dist_min_pct"] = np.min(valid_d_pct) if len(valid_d_pct) > 0 else np.nan
                failure_details["dist_threshold"] = dist_max_change_pct

        result = {
            "found": onset_idx is not None,
            "onset_idx": onset_idx,
            "onset_ms": time_ms[onset_idx] if onset_idx is not None else None,
            "x_value": x_smooth[onset_idx] if onset_idx is not None else None,
            "y_value": y_smooth[onset_idx] if onset_idx is not None else None,
            "dist_value": dist_smooth[onset_idx] if onset_idx is not None else None,
            "signals_used": signals_used,
            "method": "stable_region_multi",
            "failure_reason": failure_reason,
            "failure_details": failure_details,
        }

        if return_stability_traces:
            result["stability_traces"] = {
                "x_stable": x_stable_trace,
                "y_stable": y_stable_trace,
                "d_stable": d_stable_trace,
                "combined_stable": combined_stable,
                "x_pct_change": x_pct_trace,
                "y_pct_change": y_pct_trace,
                "d_pct_change": d_pct_trace,
                "use_x": use_x,
                "use_y": use_y,
                "use_dist": use_dist,
            }

        return result


# =============================================================================
# Plotting - Modified with new Panel 6 and axis limits
# =============================================================================

def plot_plateau_summary_comprehensive(
    processed_data: Dict,
    plateau_summary: Dict,
    figsize: Tuple[int, int] = (16, 12),
    onset_time_xlim: Tuple[float, float] = (0, 400),
):
    """
    Comprehensive 6-panel per-condition summary visualization.
    
    Parameters
    ----------
    processed_data : Dict
        Processed condition data.
    plateau_summary : Dict
        Plateau detection summary.
    figsize : Tuple[int, int]
        Figure size.
    onset_time_xlim : Tuple[float, float]
        X-axis limits for onset time plots (default 0-400 ms).
    """
    results = plateau_summary["results"]
    onset_times = plateau_summary["onset_times"]
    time = processed_data["aligned_time"]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    fig.suptitle(
        f"{processed_data['cond_str']} - Plateau Detection Summary\n"
        f"Detection Rate: {plateau_summary['n_found']}/{plateau_summary['n_valid']} "
        f"({plateau_summary['detection_rate']:.1%})",
        fontsize=14,
        fontweight="bold",
    )

    cmap = plt.cm.tab10
    colors_by_trial = [cmap(i % 10) for i in range(len(results))]

    # Panel 1: histogram
    ax1 = axes[0, 0]

    if len(onset_times) > 0:
        # Filter onset times to be within xlim for histogram
        filtered_onsets = [t for t in onset_times if onset_time_xlim[0] <= t <= onset_time_xlim[1]]
        
        if len(filtered_onsets) > 0:
            bins = np.linspace(onset_time_xlim[0], onset_time_xlim[1], 16)
            ax1.hist(
                filtered_onsets,
                bins=bins,
                edgecolor="black",
                alpha=0.7,
                color="steelblue",
            )
        
        if onset_time_xlim[0] <= plateau_summary["onset_mean"] <= onset_time_xlim[1]:
            ax1.axvline(
                plateau_summary["onset_mean"],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {plateau_summary['onset_mean']:.1f} ms",
            )
        if onset_time_xlim[0] <= plateau_summary["onset_median"] <= onset_time_xlim[1]:
            ax1.axvline(
                plateau_summary["onset_median"],
                color="orange",
                linestyle=":",
                linewidth=2,
                label=f"Median: {plateau_summary['onset_median']:.1f} ms",
            )
        ax1.legend(fontsize=9)

    ax1.set_xlabel("Onset Time (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Onset Time Distribution")
    ax1.set_xlim(onset_time_xlim)

    # Panel 2: onset by trial
    ax2 = axes[0, 1]

    trial_nums = []
    onset_vals = []

    for i, r in enumerate(results):
        if r.get("found", False):
            trial_nums.append(i)
            onset_vals.append(r["onset_ms"])

    if len(trial_nums) > 0:
        ax2.scatter(
            trial_nums,
            onset_vals,
            c="steelblue",
            s=50,
            alpha=0.7,
            edgecolors="black",
        )
        ax2.axhline(
            plateau_summary["onset_mean"],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {plateau_summary['onset_mean']:.1f} ms",
        )
        ax2.axhspan(
            plateau_summary["onset_mean"] - plateau_summary["onset_std"],
            plateau_summary["onset_mean"] + plateau_summary["onset_std"],
            alpha=0.2,
            color="red",
            label=f"±1 SD: {plateau_summary['onset_std']:.1f} ms",
        )
        ax2.legend(fontsize=9)

    no_detect = [i for i, r in enumerate(results) if not r.get("found", False)]

    if no_detect:
        # Plot failed trials at the top of the y-axis range
        ax2.scatter(
            no_detect,
            [onset_time_xlim[1]] * len(no_detect),
            c="red",
            marker="x",
            s=80,
            label=f"No detection (n={len(no_detect)})",
            zorder=5,
        )
        ax2.legend(fontsize=9)

    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("Onset Time (ms)")
    ax2.set_title("Onset Time by Trial")
    ax2.set_xlim(-0.5, len(results) - 0.5)
    ax2.set_ylim(onset_time_xlim)

    # Panel 3: X traces
    ax3 = axes[0, 2]
    x_data = processed_data["x_smooth_all"]

    for i in range(x_data.shape[0]):
        ax3.plot(time, x_data[i], alpha=0.5, linewidth=1, color=colors_by_trial[i])

        if results[i].get("found", False):
            onset_idx = np.argmin(np.abs(time - results[i]["onset_ms"]))
            ax3.scatter(
                results[i]["onset_ms"],
                x_data[i, onset_idx],
                color=colors_by_trial[i],
                s=40,
                edgecolors="black",
                zorder=5,
            )

    ax3.axvline(
        0,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Stim onset",
    )
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("X Position")
    ax3.set_title("X Traces with Plateau Onsets")
    ax3.legend(fontsize=9)

    # Panel 4: distance traces
    ax4 = axes[1, 0]
    dist_data = processed_data["distance_smooth_all"]

    for i in range(dist_data.shape[0]):
        ax4.plot(time, dist_data[i], alpha=0.5, linewidth=1, color=colors_by_trial[i])

        if results[i].get("found", False):
            onset_idx = np.argmin(np.abs(time - results[i]["onset_ms"]))
            ax4.scatter(
                results[i]["onset_ms"],
                dist_data[i, onset_idx],
                color=colors_by_trial[i],
                s=40,
                edgecolors="black",
                zorder=5,
            )

    ax4.axvline(
        0,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Stim onset",
    )
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Distance")
    ax4.set_title("Distance Traces with Plateau Onsets")
    ax4.legend(fontsize=9)

    # Panel 5: XY trajectories
    ax5 = axes[1, 1]

    x_data = processed_data["x_smooth_all"]
    y_data = processed_data["y_smooth_all"]
    norm = Normalize(vmin=time.min(), vmax=time.max())

    for i in range(x_data.shape[0]):
        x = x_data[i]
        y = y_data[i]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap="viridis",
            norm=norm,
            alpha=0.6,
            linewidth=1,
        )
        lc.set_array(time[:-1])
        ax5.add_collection(lc)

        if results[i].get("found", False):
            onset_idx = np.argmin(np.abs(time - results[i]["onset_ms"]))
            ax5.scatter(
                x[onset_idx],
                y[onset_idx],
                color="red",
                s=50,
                edgecolors="white",
                zorder=5,
                linewidth=1,
            )

    ax5.autoscale()
    ax5.set_xlabel("X Position")
    ax5.set_ylabel("Y Position")
    ax5.set_title("X-Y Trajectories (color = time)")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax5, shrink=0.8)
    cbar.set_label("Time (ms)", fontsize=9)

    # Panel 6: Failed Trial Traces (colored by failure reason)
    ax6 = axes[1, 2]
    
    failed_trials = [r for r in results if not r.get("found", False)]
    successful_count = len([r for r in results if r.get("found", False)])
    total_count = len(results)
    
    if failed_trials and processed_data is not None:
        # Define colors for each failure reason
        failure_colors = {
            "Never stable: X": "#e41a1c",           # Red
            "Never stable: Dist": "#377eb8",         # Blue
            "Never stable: X, Dist": "#984ea3",      # Purple
            "Signals never simultaneously stable": "#ff7f00",  # Orange
            "Unknown": "#999999"                     # Gray
        }
        
        # Get time array
        aligned_time = processed_data.get("aligned_time", None)
        distance_traces = processed_data.get("distance_smooth_all", None)
        
        if aligned_time is not None and distance_traces is not None:
            # Group failed trials by reason for legend
            plotted_reasons = set()
            
            for r in failed_trials:
                trial_idx = r.get("trial_idx", None)
                failure_reason = r.get("failure_reason", "Unknown")
                
                if trial_idx is not None and trial_idx < len(distance_traces):
                    trace = distance_traces[trial_idx]
                    color = failure_colors.get(failure_reason, "#999999")
                    
                    # Only add label for first trace of each reason (for legend)
                    if failure_reason not in plotted_reasons:
                        ax6.plot(aligned_time, trace, color=color, alpha=0.5, 
                                linewidth=0.8, label=failure_reason)
                        plotted_reasons.add(failure_reason)
                    else:
                        ax6.plot(aligned_time, trace, color=color, alpha=0.5, 
                                linewidth=0.8)
            
            ax6.axvline(x=0, color='green', linestyle='--', linewidth=1, alpha=0.7)
            ax6.set_xlabel("Time from stim (ms)")
            ax6.set_ylabel("Distance (deg)")
            ax6.set_title(f"Failed Trials (n={len(failed_trials)}/{total_count})")
            # ax6.set_xlim(0, 400)
            
            # Add legend with smaller font
            if plotted_reasons:
                ax6.legend(loc='upper right', fontsize=7, framealpha=0.8)
            
            # Add detection rate text
            detection_rate = (successful_count / total_count * 100) if total_count > 0 else 0
            ax6.text(0.02, 0.98, f"Detection rate: {detection_rate:.1f}%", 
                    transform=ax6.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax6.text(0.5, 0.5, "Trace data not available", ha='center', va='center',
                    transform=ax6.transAxes, fontsize=10)
            ax6.set_title("Failed Trial Traces")
            ax6.set_xlim(0, 400)
    else:
        # All trials detected successfully
        ax6.text(
            0.5,
            0.5,
            f"All {total_count} trials\nsuccessfully detected!\n\n"
            f"Detection rate: 100%",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            transform=ax6.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        )
        ax6.set_title("Failed Trial Traces")
        ax6.set_xlim(0, 400)

    plt.tight_layout()
    return fig

# =============================================================================
# Data loading helpers
# =============================================================================

def find_keypoint_index(kp_names: list, pattern: str) -> Optional[int]:
    """Find index of keypoint matching pattern."""
    pattern_lower = pattern.lower()

    for i, name in enumerate(kp_names):
        if pattern_lower in str(name).lower():
            return i

    return None


def get_keypoint_names_from_data(data: dict, camera: str) -> list:
    """Extract keypoint names from npz data."""
    possible_keys = [
        f"beh_{camera}_names",
        f"{camera}_names",
        "beh_names",
        "keypoint_names",
        "kp_names",
    ]

    for key in possible_keys:
        if key in data:
            names = data[key]
            return names.tolist() if hasattr(names, "tolist") else list(names)

    return []


def get_br_from_filename(filename: str) -> Optional[int]:
    """Extract BR/condition index from filename."""
    patterns = [
        r"BR_(\d+)",
        r"Cond_(\d+)",
        r"_(\d+)_target",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)

        if match:
            return int(match.group(1))

    return None


def discover_available_files(peri_root: Path) -> Dict[str, List[Path]]:
    """Discover available NPZ files organized by target."""
    files_by_target = {
        "A": [],
        "B": [],
    }

    search_patterns = [
        peri_root / "stim_reaches" / "target_A" / "*.npz",
        peri_root / "stim_reaches" / "target_B" / "*.npz",
        peri_root / "control_reaches" / "target_A" / "*.npz",
        peri_root / "control_reaches" / "target_B" / "*.npz",
        peri_root / "Target_A" / "*.npz",
        peri_root / "Target_B" / "*.npz",
        peri_root / "*.npz",
    ]

    for pattern in search_patterns:
        if not pattern.parent.exists():
            continue

        for f in pattern.parent.glob(pattern.name):
            fname = f.name.lower()
            fstr = str(f).lower()

            if "target_a" in fname or "/target_a/" in fstr:
                if f not in files_by_target["A"]:
                    files_by_target["A"].append(f)

            elif "target_b" in fname or "/target_b/" in fstr:
                if f not in files_by_target["B"]:
                    files_by_target["B"].append(f)

    for target in files_by_target:
        files_by_target[target] = sorted(files_by_target[target])

    return files_by_target


def load_condition_data(
    file_path: Path,
    camera: str = "cam1",
    keypoint_base: str = "middle",
) -> Dict:
    """Load and preprocess data from a single condition npz file."""
    with np.load(file_path, allow_pickle=True) as data:
        pos_key = None

        for key in [f"beh_{camera}_segs", f"{camera}_segs", "segs"]:
            if key in data:
                pos_key = key
                break

        if pos_key is None:
            raise ValueError(f"No position data found in {file_path.name}")

        pos_segs = data[pos_key]

        t_axis = None
        for key in ["beh_rel_t", "rel_t", "t"]:
            if key in data:
                t_axis = data[key]
                break

        if t_axis is None:
            raise ValueError(f"No time axis found in {file_path.name}")

        kp_names = get_keypoint_names_from_data(dict(data), camera)

        if not kp_names:
            kp_names = [f"kp_{i}" for i in range(pos_segs.shape[1])]

        idx_x = find_keypoint_index(kp_names, f"{keypoint_base}_x")
        idx_y = find_keypoint_index(kp_names, f"{keypoint_base}_y")

        if idx_x is None or idx_y is None:
            for i, name in enumerate(kp_names):
                name_l = name.lower()

                if "_x" in name_l and keypoint_base.lower() in name_l:
                    idx_x = i

                if "_y" in name_l and keypoint_base.lower() in name_l:
                    idx_y = i

        if idx_x is None or idx_y is None:
            raise ValueError(f"Could not find X/Y keypoints for {keypoint_base}")

        x_all = pos_segs[:, idx_x, :]
        y_all = pos_segs[:, idx_y, :]

        br_idx = get_br_from_filename(file_path.name)

        if "overall_title" in data:
            raw_title = data["overall_title"]
            cond_str = str(raw_title.item()) if raw_title.shape == () else str(raw_title)
        else:
            cond_str = f"BR{br_idx}" if br_idx else "Unknown"

    return {
        "x_all": x_all,
        "y_all": y_all,
        "t_axis": np.array(t_axis),
        "pos_segs": pos_segs,
        "kp_names": kp_names,
        "br_idx": br_idx,
        "cond_str": cond_str,
        "n_trials": pos_segs.shape[0],
        "file_path": file_path,
    }


def load_all_conditions(
    peri_root: Path,
    camera: str = "cam1",
    keypoint_base: str = "middle",
    control_br_indices: Optional[list] = None,
) -> Dict[str, Dict]:
    """
    Load all condition files from a session.

    Returns
    -------
    all_conditions:
        Dict mapping condition key to condition_data dict.
    """
    if control_br_indices is None:
        control_br_indices = [1]

    files_by_target = discover_available_files(peri_root)
    all_conditions = {}

    for target, files in files_by_target.items():
        for file_path in files:
            try:
                cond_data = load_condition_data(file_path, camera, keypoint_base)

                key = f"{cond_data['cond_str']}_Target{target}"

                if key in all_conditions:
                    key = f"{key}_{file_path.stem}"

                all_conditions[key] = cond_data
                print(f"  Loaded: {key} ({cond_data['n_trials']} trials)")

            except Exception as exc:
                print(f"  Error loading {file_path.name}: {exc}")

    return all_conditions


# =============================================================================
# Processing
# =============================================================================

def compute_distance_from_start(
    x: np.ndarray,
    y: np.ndarray,
    ref_idx: int = 0,
) -> np.ndarray:
    """Compute Euclidean distance from starting position."""
    x0 = x[ref_idx]
    y0 = y[ref_idx]

    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


def normalize_trace(
    trace: np.ndarray,
    method: str = "max",
) -> Tuple[np.ndarray, float]:
    """Normalize trace."""
    if method == "percentile":
        scale = np.nanpercentile(trace, 95)
    else:
        scale = np.nanmax(trace)

    scale = max(scale, 1e-6)
    return trace / scale, scale


def process_single_trial(
    x: np.ndarray,
    y: np.ndarray,
    t_axis: np.ndarray,
    smoothing_params: Optional[dict] = None,
    smooth_xy: bool = True,
) -> Dict:
    """
    Process a single trial using Butterworth smoothing.
    """
    dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 10.0

    # If t_axis is in seconds, convert dt to ms.
    if dt < 0.9:
        dt *= 1000.0

    params = smoothing_params or SMOOTHING_PARAMS

    if smooth_xy:
        x_smooth = SmoothingMethods.butterworth(x, dt, **params)
        y_smooth = SmoothingMethods.butterworth(y, dt, **params)

        distance_raw = compute_distance_from_start(x, y, ref_idx=0)
        distance_smooth = compute_distance_from_start(x_smooth, y_smooth, ref_idx=0)

    else:
        x_smooth = x.copy()
        y_smooth = y.copy()

        distance_raw = compute_distance_from_start(x, y, ref_idx=0)
        distance_smooth = SmoothingMethods.butterworth(distance_raw, dt, **params)

    velocity = np.gradient(distance_smooth, dt)

    return {
        "x_raw": x,
        "y_raw": y,
        "x_smooth": x_smooth,
        "y_smooth": y_smooth,
        "distance_raw": distance_raw,
        "distance_smooth": distance_smooth,
        "velocity": velocity,
        "dt": dt,
    }


def process_condition(
    condition_data: Dict,
    smoothing_params: Optional[dict] = None,
    smooth_xy: bool = True,
    time_window: Tuple[float, float] = (-500, 500),
) -> Dict:
    """
    Process all trials in a condition.
    """
    x_all = condition_data["x_all"]
    y_all = condition_data["y_all"]
    t_axis = condition_data["t_axis"]
    n_trials = condition_data["n_trials"]

    dt = t_axis[1] - t_axis[0] if len(t_axis) > 1 else 10.0

    if dt < 0.9:
        dt *= 1000.0

    t0_idx = np.argmin(np.abs(t_axis))

    pre_samples = int(abs(time_window[0]) / dt)
    post_samples = int(time_window[1] / dt)

    start_idx = max(0, t0_idx - pre_samples)
    end_idx = min(len(t_axis), t0_idx + post_samples)

    aligned_time = t_axis[start_idx:end_idx] - t_axis[t0_idx]
    T_aligned = len(aligned_time)

    distance_raw_all = np.zeros((n_trials, T_aligned))
    distance_smooth_all = np.zeros((n_trials, T_aligned))
    x_smooth_all = np.zeros((n_trials, T_aligned))
    y_smooth_all = np.zeros((n_trials, T_aligned))
    valid_mask = np.ones(n_trials, dtype=bool)

    for i in range(n_trials):
        try:
            result = process_single_trial(
                x_all[i, start_idx:end_idx],
                y_all[i, start_idx:end_idx],
                aligned_time,
                smoothing_params=smoothing_params,
                smooth_xy=smooth_xy,
            )

            distance_raw_all[i, :] = result["distance_raw"]
            distance_smooth_all[i, :] = result["distance_smooth"]
            x_smooth_all[i, :] = result["x_smooth"]
            y_smooth_all[i, :] = result["y_smooth"]

        except Exception:
            valid_mask[i] = False
            distance_raw_all[i, :] = np.nan
            distance_smooth_all[i, :] = np.nan
            x_smooth_all[i, :] = np.nan
            y_smooth_all[i, :] = np.nan

    dist_norm = np.zeros_like(distance_smooth_all)
    scale_factors = np.zeros(n_trials)

    for i in range(n_trials):
        if valid_mask[i]:
            dist_norm[i, :], scale_factors[i] = normalize_trace(distance_smooth_all[i, :])
        else:
            dist_norm[i, :] = np.nan
            scale_factors[i] = np.nan

    valid_dist = dist_norm[valid_mask, :]
    n_valid = np.sum(valid_mask)

    if n_valid > 0:
        dist_median = np.nanmedian(valid_dist, axis=0)
        dist_mean = np.nanmean(valid_dist, axis=0)
        dist_std = np.nanstd(valid_dist, axis=0)
        dist_sem = dist_std / np.sqrt(max(1, n_valid))
        dist_ci_lower = dist_median - 1.96 * dist_sem
        dist_ci_upper = dist_median + 1.96 * dist_sem
    else:
        dist_median = np.full(T_aligned, np.nan)
        dist_mean = np.full(T_aligned, np.nan)
        dist_std = np.full(T_aligned, np.nan)
        dist_sem = np.full(T_aligned, np.nan)
        dist_ci_lower = np.full(T_aligned, np.nan)
        dist_ci_upper = np.full(T_aligned, np.nan)

    return {
        "aligned_time": aligned_time,
        "distance_raw_all": distance_raw_all,
        "distance_smooth_all": distance_smooth_all,
        "dist_norm": dist_norm,
        "x_smooth_all": x_smooth_all,
        "y_smooth_all": y_smooth_all,
        "valid_mask": valid_mask,
        "scale_factors": scale_factors,
        "dist_median": dist_median,
        "dist_mean": dist_mean,
        "dist_std": dist_std,
        "dist_sem": dist_sem,
        "dist_ci_lower": dist_ci_lower,
        "dist_ci_upper": dist_ci_upper,
        "n_valid": n_valid,
        "n_total": n_trials,
        "dt": dt,
        "br_idx": condition_data.get("br_idx"),
        "cond_str": condition_data.get("cond_str"),
    }


def process_and_detect_condition(
    cond_data: Dict,
    smoothing_params: Optional[dict] = None,
    detection_params: Optional[dict] = None,
    time_window: Tuple[float, float] = (-500, 500),
) -> Dict:
    """
    Process a condition and run plateau detection on all trials.
    """
    if smoothing_params is None:
        smoothing_params = SMOOTHING_PARAMS

    if detection_params is None:
        detection_params = DETECTION_PARAMS

    processed = process_condition(
        cond_data,
        smoothing_params=smoothing_params,
        smooth_xy=SMOOTH_XY,
        time_window=time_window,
    )

    time_ms = processed["aligned_time"]
    n_valid = processed["n_valid"]

    results = []
    onset_times = []
    x_values = []
    y_values = []
    dist_values = []

    for i in range(processed["n_total"]):
        if not processed["valid_mask"][i]:
            results.append(
                {
                    "found": False,
                    "trial_idx": i,
                    "onset_ms": None,
                    "x_value": None,
                    "y_value": None,
                    "dist_value": None,
                }
            )
            continue

        x_smooth = processed["x_smooth_all"][i]
        y_smooth = processed["y_smooth_all"][i]
        dist_smooth = processed["distance_smooth_all"][i]

        det_result = PlateauDetectionMethods.stable_region_multi(
            x_smooth=x_smooth,
            y_smooth=y_smooth,
            dist_smooth=dist_smooth,
            time_ms=time_ms,
            min_duration_ms=detection_params["min_duration_ms"],
            x_max_change_pct=detection_params.get("x_max_change_pct", 2.0),
            y_max_change_pct=detection_params.get("y_max_change_pct", 2.0),
            dist_max_change_pct=detection_params.get("dist_max_change_pct", 2.0),
            use_x=detection_params["use_x"],
            use_y=detection_params["use_y"],
            use_dist=detection_params["use_dist"],
            search_start_ms=detection_params["search_start_ms"],
            search_end_ms=detection_params["search_end_ms"],
            return_stability_traces=False,
        )

        det_result["trial_idx"] = i
        results.append(det_result)

        if det_result["found"]:
            onset_times.append(det_result["onset_ms"])
            x_values.append(det_result["x_value"])
            y_values.append(det_result["y_value"])
            dist_values.append(det_result["dist_value"])

    n_found = len(onset_times)

    plateau_summary = {
        "n_trials": processed["n_total"],
        "n_valid": n_valid,
        "n_found": n_found,
        "n_not_found": n_valid - n_found,
        "detection_rate": n_found / max(1, n_valid),
        "onset_times": np.array(onset_times),
        "x_values": np.array(x_values),
        "y_values": np.array(y_values),
        "dist_values": np.array(dist_values),
        "results": results,
        "params": detection_params.copy(),
        "onset_mean": np.mean(onset_times) if n_found > 0 else np.nan,
        "onset_std": np.std(onset_times) if n_found > 0 else np.nan,
        "onset_median": np.median(onset_times) if n_found > 0 else np.nan,
    }

    if n_found > 0:
        plateau_summary["x_mean"] = np.mean(x_values)
        plateau_summary["y_mean"] = np.mean(y_values)
        plateau_summary["dist_mean"] = np.mean(dist_values)

    return {
        "processed_data": processed,
        "plateau_summary": plateau_summary,
    }


# =============================================================================
# Plotting
# =============================================================================

def save_and_close_fig(fig, save_path, dpi=300):
    """
    Save a matplotlib figure and close it to avoid memory warnings.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {save_path}")


# =============================================================================
# Batch analysis
# =============================================================================

def analyze_all_conditions(
    peri_root: Path,
    camera: str = "cam1",
    keypoint_base: str = "middle",
    control_br_indices: Optional[list] = None,
    smoothing_params: Optional[dict] = None,
    detection_params: Optional[dict] = None,
    time_window: Tuple[float, float] = (-500, 500),
    plot_each: bool = True,
    save_figs: bool = False,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Load all conditions and run plateau analysis.
    """
    if control_br_indices is None:
        control_br_indices = CONTROL_BR_INDICES

    if smoothing_params is None:
        smoothing_params = SMOOTHING_PARAMS

    if detection_params is None:
        detection_params = DETECTION_PARAMS

    print("=" * 60)
    print("LOADING ALL CONDITIONS")
    print("=" * 60)
    print(f"PERI_ROOT: {peri_root}")
    print(f"Camera: {camera}, Keypoint: {keypoint_base}")
    print(f"Control BR indices: {control_br_indices}\n")

    all_conditions = load_all_conditions(
        peri_root=Path(peri_root),
        camera=camera,
        keypoint_base=keypoint_base,
        control_br_indices=control_br_indices,
    )

    if len(all_conditions) == 0:
        print("[error] No conditions found.")
        return {}

    print(f"\nLoaded {len(all_conditions)} conditions\n")

    all_results = {}

    for cond_idx, (cond_key, cond_data) in enumerate(all_conditions.items()):
        print(f"\n{'=' * 60}")
        print(f"Processing {cond_idx + 1}/{len(all_conditions)}: {cond_key}")
        print("=" * 60)

        try:
            result = process_and_detect_condition(
                cond_data=cond_data,
                smoothing_params=smoothing_params,
                detection_params=detection_params,
                time_window=time_window,
            )

            result["processed_data"]["cond_str"] = cond_key

            ps = result["plateau_summary"]

            print(f"  Valid trials: {ps['n_valid']}")
            print(
                f"  Plateaus found: {ps['n_found']}/{ps['n_valid']} "
                f"({ps['detection_rate']:.1%})"
            )

            if ps["n_found"] > 0:
                print(f"  Mean onset: {ps['onset_mean']:.1f} ± {ps['onset_std']:.1f} ms")

            all_results[cond_key] = result

            if plot_each and ps["n_found"] > 0:
                fig = plot_plateau_summary_comprehensive(
                    result["processed_data"],
                    result["plateau_summary"],
                    onset_time_xlim=ONSET_TIME_XLIM,
                )

                if save_figs:
                    if save_dir is None:
                        save_dir = SAVE_DIR

                    save_dir = Path(save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)

                    safe_name = (
                        cond_key.replace("/", "_")
                        .replace("\\", "_")
                        .replace(" ", "_")
                        .replace(":", "")
                    )

                    fig_path = save_dir / f"plateau_summary_{safe_name}.png"
                    save_and_close_fig(fig, fig_path, dpi=150)
                else:
                    plt.show()

        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            all_results[cond_key] = {"error": str(exc)}

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 60)

    print(f"\n{'Condition':<35} {'Trials':<8} {'Found':<8} {'Rate':<10} {'Mean Onset':<12}")
    print("-" * 75)

    for cond_key, res in all_results.items():
        if "error" in res:
            print(f"{cond_key[:35]:<35} {'ERROR':<8}")
        else:
            ps = res["plateau_summary"]
            onset_str = f"{ps['onset_mean']:.1f} ms" if ps["n_found"] > 0 else "N/A"
            rate_str = f"{ps['detection_rate']:.1%}"
            print(
                f"{cond_key[:35]:<35} "
                f"{ps['n_valid']:<8} "
                f"{ps['n_found']:<8} "
                f"{rate_str:<10} "
                f"{onset_str:<12}"
            )

    return all_results


# =============================================================================
# Cross-condition comparison
# =============================================================================

def _extract_condition_data_for_target(
    all_results: Dict,
    target: str = "A",
    pulse_width_ms: float = 2.5,
    min_reaches: int = 3,
    conditions_to_compare: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Shared helper for extracting condition summary data.
    """
    target_suffix = f"_Target{target}"
    condition_data = []

    for key, data in all_results.items():
        if "error" in data:
            continue

        if not key.endswith(target_suffix):
            continue

        plateau_summary = data.get("plateau_summary", {})
        onset_times = plateau_summary.get("onset_times", np.array([]))

        if isinstance(onset_times, np.ndarray):
            onset_times = onset_times[~np.isnan(onset_times)].tolist()
        else:
            onset_times = [t for t in onset_times if not np.isnan(t)]

        n_reaches = len(onset_times)

        if n_reaches < min_reaches:
            print(f"Skipping {key[:50]}...: only {n_reaches} reaches, need ≥{min_reaches}")
            continue

        condition_str = key.replace(target_suffix, "")

        cond_match = re.search(r"Condition:\s*(\d+)", condition_str)
        dur_match = re.search(r"Duration:\s*([\d.]+)\s*ms", condition_str)
        freq_match = re.search(r"Freq:\s*([\d.]+)\s*Hz", condition_str)

        if cond_match:
            cond_num = int(cond_match.group(1))
        else:
            cond_num = 999

        if conditions_to_compare is not None and cond_num not in conditions_to_compare:
            continue

        if dur_match and freq_match:
            duration_ms = float(dur_match.group(1))
            freq_hz = float(freq_match.group(1))
            n_pulses = int(duration_ms / pulse_width_ms)
            label = f"C{cond_num}, {n_pulses}p {int(freq_hz)}Hz"
        elif dur_match:
            duration_ms = float(dur_match.group(1))
            n_pulses = int(duration_ms / pulse_width_ms)
            label = f"C{cond_num}, {n_pulses}p"
        else:
            label = f"C{cond_num}"

        mean_val = np.mean(onset_times)
        std_val = np.std(onset_times, ddof=1)
        sem_val = std_val / np.sqrt(n_reaches)
        t_crit = stats.t.ppf(0.975, df=n_reaches - 1)
        ci95 = t_crit * sem_val

        processed_data = data.get("processed_data", {})

        condition_data.append(
            {
                "label": label,
                "cond_num": cond_num,
                "mean": mean_val,
                "std": std_val,
                "sem": sem_val,
                "ci95": ci95,
                "n": n_reaches,
                "times": onset_times,
                "full_key": key,
                "aligned_time": processed_data.get("aligned_time", None),
                "dist_norm": processed_data.get("dist_norm", None),
            }
        )

    condition_data.sort(key=lambda x: x["cond_num"])
    return condition_data


def compare_conditions_by_target(
    all_results: Dict,
    target: str = "A",
    figsize: Tuple[int, int] = (16, 10),
    cmap: str = "cool",
    pulse_width_ms: float = 2.5,
):
    """
    Compare plateau onset times across all valid conditions for one target.
    """
    condition_data = _extract_condition_data_for_target(
        all_results,
        target=target,
        pulse_width_ms=pulse_width_ms,
        min_reaches=3,
        conditions_to_compare=None,
    )

    if not condition_data:
        print(f"No valid conditions found for Target {target}")
        return None, None

    labels = [d["label"] for d in condition_data]
    means = [d["mean"] for d in condition_data]
    stds = [d["std"] for d in condition_data]
    ci95s = [d["ci95"] for d in condition_data]
    ns = [d["n"] for d in condition_data]

    norm = Normalize(vmin=min(means), vmax=max(means))
    colormap = plt.colormaps.get_cmap(cmap)
    colors = [colormap(norm(m)) for m in means]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"Plateau Onset Time Analysis - Target {target}",
        fontsize=14,
        fontweight="bold",
    )

    x_pos = np.arange(len(labels))
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    # Panel 1: bar plot
    ax1 = axes[0, 0]
    bars = ax1.bar(x_pos, means, color=colors, alpha=0.8, edgecolor="black")

    ax1.errorbar(
        x_pos,
        means,
        yerr=stds,
        fmt="none",
        capsize=8,
        capthick=1.5,
        ecolor="gray",
        elinewidth=1.5,
        label="±1 SD",
        alpha=0.6,
    )

    ax1.errorbar(
        x_pos,
        means,
        yerr=ci95s,
        fmt="none",
        capsize=5,
        capthick=2,
        ecolor="darkred",
        elinewidth=2,
        label="95% CI",
    )

    ax1.set_xlabel("Condition")
    ax1.set_ylabel("Plateau Onset Time (ms)")
    ax1.set_title("Mean ± SD and 95% CI")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend(loc="upper right", fontsize=9)

    cbar1 = plt.colorbar(sm, ax=ax1, pad=0.02)
    cbar1.set_label("Mean Onset (ms)", fontsize=9)

    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds[i] + 5,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel 2: box plot
    ax2 = axes[0, 1]
    all_times = [d["times"] for d in condition_data]
    bp = ax2.boxplot(all_times, tick_labels=labels, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax2.set_xlabel("Condition")
    ax2.set_ylabel("Plateau Onset Time (ms)")
    ax2.set_title("Distribution")
    ax2.tick_params(axis="x", rotation=45)

    cbar2 = plt.colorbar(sm, ax=ax2, pad=0.02)
    cbar2.set_label("Mean Onset (ms)", fontsize=9)

    # Panel 3: individual points + mean CI
    ax3 = axes[1, 0]

    for i, d in enumerate(condition_data):
        jitter = np.random.normal(0, 0.08, len(d["times"]))

        ax3.scatter(
            np.full(len(d["times"]), i) + jitter,
            d["times"],
            color=colors[i],
            alpha=0.5,
            s=30,
            edgecolor="none",
        )

        ax3.errorbar(
            i,
            d["mean"],
            yerr=d["ci95"],
            fmt="D",
            markersize=10,
            color="black",
            capsize=6,
            capthick=2,
            elinewidth=2,
        )

    ax3.set_xlabel("Condition")
    ax3.set_ylabel("Plateau Onset Time (ms)")
    ax3.set_title("Individual Reaches + Mean ± 95% CI")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha="right")

    cbar3 = plt.colorbar(sm, ax=ax3, pad=0.02)
    cbar3.set_label("Mean Onset (ms)", fontsize=9)

    # Panel 4: normalized distance traces
    ax4 = axes[1, 1]

    for i, d in enumerate(condition_data):
        dist_norm = d["dist_norm"]
        aligned_time = d["aligned_time"]
        color = colors[i]

        if isinstance(dist_norm, np.ndarray) and dist_norm.ndim == 2:
            n_trials, n_timepoints = dist_norm.shape

            mean_dist = np.nanmean(dist_norm, axis=0)
            std_dist = np.nanstd(dist_norm, axis=0, ddof=1)
            n_valid = np.sum(~np.isnan(dist_norm), axis=0)
            sem_dist = std_dist / np.sqrt(np.maximum(n_valid, 1))

            t_crit = stats.t.ppf(0.975, df=max(n_trials - 1, 1))
            ci95_dist = t_crit * sem_dist

            time_ms = aligned_time if aligned_time is not None else np.arange(n_timepoints)

            ax4.plot(time_ms, mean_dist, color=color, linewidth=2, label=d["label"])
            ax4.fill_between(
                time_ms,
                mean_dist - ci95_dist,
                mean_dist + ci95_dist,
                color=color,
                alpha=0.2,
            )

    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Normalized Distance")
    ax4.set_title("Mean Normalized Distance ± 95% CI")
    ax4.legend(loc="lower right", fontsize=8, ncol=2)
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax4.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax4.set_ylim(-0.1, 1.2)

    cbar4 = plt.colorbar(sm, ax=ax4, pad=0.02)
    cbar4.set_label("Mean Onset (ms)", fontsize=9)

    plt.tight_layout()

    print("\nSUMMARY TABLE")
    print("=" * 80)
    print(f"{'Condition':<25} {'n':>5} {'Mean':>8} {'SD':>8} {'SEM':>8} {'95% CI':>10}")
    print("-" * 80)

    for d in condition_data:
        print(
            f"{d['label']:<25} "
            f"{d['n']:>5} "
            f"{d['mean']:>8.1f} "
            f"{d['std']:>8.1f} "
            f"{d['sem']:>8.1f} "
            f"±{d['ci95']:.1f}"
        )

    return condition_data, fig


def analyze_selected_conditions(
    all_results: Dict,
    conditions_to_compare: Optional[List[int]] = None,
    target: str = "A",
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = "cool",
    pulse_width_ms: float = 2.5,
):
    """
    Statistical analysis of selected conditions for one target.
    """
    condition_data = _extract_condition_data_for_target(
        all_results,
        target=target,
        pulse_width_ms=pulse_width_ms,
        min_reaches=3,
        conditions_to_compare=conditions_to_compare,
    )

    if len(condition_data) < 2:
        print(f"Need at least 2 valid conditions for comparison. Found: {len(condition_data)}")
        return None, None

    labels = [d["label"] for d in condition_data]
    means = [d["mean"] for d in condition_data]
    stds = [d["std"] for d in condition_data]
    ci95s = [d["ci95"] for d in condition_data]
    ns = [d["n"] for d in condition_data]
    all_times = [d["times"] for d in condition_data]

    norm_cmap = Normalize(vmin=min(means), vmax=max(means))
    colormap = plt.colormaps.get_cmap(cmap)
    colors = [colormap(norm_cmap(m)) for m in means]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    cond_str = ", ".join([str(c) for c in conditions_to_compare]) if conditions_to_compare else "All"

    fig.suptitle(
        f"Statistical Comparison - Target {target}\n"
        f"(Conditions: {cond_str})",
        fontsize=12,
        fontweight="bold",
    )

    x_pos = np.arange(len(labels))
    sm = ScalarMappable(cmap=colormap, norm=norm_cmap)
    sm.set_array([])

    # Panel 1: bar plot
    ax1 = axes[0, 0]
    bars = ax1.bar(x_pos, means, color=colors, alpha=0.8, edgecolor="black")

    ax1.errorbar(
        x_pos,
        means,
        yerr=stds,
        fmt="none",
        capsize=8,
        capthick=1.5,
        ecolor="gray",
        elinewidth=1.5,
        label="±1 SD",
        alpha=0.6,
    )

    ax1.errorbar(
        x_pos,
        means,
        yerr=ci95s,
        fmt="none",
        capsize=5,
        capthick=2,
        ecolor="darkred",
        elinewidth=2,
        label="95% CI",
    )

    ax1.set_xlabel("Condition")
    ax1.set_ylabel("Plateau Onset Time (ms)")
    ax1.set_title("Mean ± SD and 95% CI")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend(loc="upper right", fontsize=9)

    cbar1 = plt.colorbar(sm, ax=ax1, pad=0.02)
    cbar1.set_label("Mean Onset (ms)", fontsize=9)

    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds[i] + 5,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel 2: normalized distance traces
    ax2 = axes[0, 1]

    for i, d in enumerate(condition_data):
        dist_norm = d["dist_norm"]
        aligned_time = d["aligned_time"]

        if isinstance(dist_norm, np.ndarray) and dist_norm.ndim == 2:
            n_trials, n_timepoints = dist_norm.shape

            mean_dist = np.nanmean(dist_norm, axis=0)
            std_dist = np.nanstd(dist_norm, axis=0, ddof=1)
            n_valid = np.sum(~np.isnan(dist_norm), axis=0)
            sem_dist = std_dist / np.sqrt(np.maximum(n_valid, 1))

            t_crit = stats.t.ppf(0.975, df=max(n_trials - 1, 1))
            ci95_dist = t_crit * sem_dist

            time_ms = aligned_time if aligned_time is not None else np.arange(n_timepoints)

            ax2.plot(time_ms, mean_dist, color=colors[i], linewidth=2, label=d["label"])
            ax2.fill_between(
                time_ms,
                mean_dist - ci95_dist,
                mean_dist + ci95_dist,
                color=colors[i],
                alpha=0.2,
            )

    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Normalized Distance")
    ax2.set_title("Mean Normalized Distance ± 95% CI")
    ax2.legend(loc="lower right", fontsize=8, ncol=2)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax2.set_ylim(-0.1, 1.2)

    cbar2 = plt.colorbar(sm, ax=ax2, pad=0.02)
    cbar2.set_label("Mean Onset (ms)", fontsize=9)

    # Panel 3: statistical tests text
    ax3 = axes[1, 0]
    ax3.axis("off")

    stats_text = "STATISTICAL ANALYSIS\n" + "=" * 40 + "\n\n"

    f_stat, p_anova = stats.f_oneway(*all_times)
    stats_text += "One-way ANOVA:\n"
    stats_text += f"  F = {f_stat:.3f}, p = {p_anova:.4f}\n"
    stats_text += f"  {'** SIGNIFICANT **' if p_anova < 0.05 else 'Not significant'}\n\n"

    h_stat, p_kruskal = stats.kruskal(*all_times)
    stats_text += "Kruskal-Wallis:\n"
    stats_text += f"  H = {h_stat:.3f}, p = {p_kruskal:.4f}\n"
    stats_text += f"  {'** SIGNIFICANT **' if p_kruskal < 0.05 else 'Not significant'}\n\n"

    if p_anova < 0.05 and len(condition_data) > 2:
        stats_text += "Pairwise t-tests, uncorrected:\n"

        for i in range(len(condition_data)):
            for j in range(i + 1, len(condition_data)):
                t_stat, p_val = stats.ttest_ind(
                    condition_data[i]["times"],
                    condition_data[j]["times"],
                )

                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = ""

                stats_text += f"  {labels[i]} vs {labels[j]}: p={p_val:.4f} {sig}\n"

    ax3.text(
        0.05,
        0.95,
        stats_text,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 4: violin + points
    ax4 = axes[1, 1]

    for i, d in enumerate(condition_data):
        jitter = np.random.normal(0, 0.05, len(d["times"]))

        ax4.scatter(
            np.full(len(d["times"]), i) + jitter,
            d["times"],
            color=colors[i],
            alpha=0.6,
            s=25,
            edgecolor="none",
        )

        ax4.errorbar(
            i,
            d["mean"],
            yerr=d["ci95"],
            fmt="D",
            markersize=10,
            color="black",
            capsize=6,
            capthick=2,
            elinewidth=2,
        )

    vp = ax4.violinplot(
        all_times,
        positions=x_pos,
        showmeans=False,
        showmedians=False,
    )

    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.3)

    ax4.set_xlabel("Condition")
    ax4.set_ylabel("Plateau Onset Time (ms)")
    ax4.set_title("Distribution + Mean ± 95% CI")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45, ha="right")

    cbar4 = plt.colorbar(sm, ax=ax4, pad=0.02)
    cbar4.set_label("Mean Onset (ms)", fontsize=9)

    plt.tight_layout()
    plt.show()

    print("\nSUMMARY TABLE")
    print("=" * 80)
    print(f"{'Condition':<25} {'n':>5} {'Mean':>8} {'SD':>8} {'SEM':>8} {'95% CI':>10}")
    print("-" * 80)

    for d in condition_data:
        print(
            f"{d['label']:<25} "
            f"{d['n']:>5} "
            f"{d['mean']:>8.1f} "
            f"{d['std']:>8.1f} "
            f"{d['sem']:>8.1f} "
            f"±{d['ci95']:.1f}"
        )

    return condition_data, fig


# =============================================================================
# Optional export
# =============================================================================

def create_plateau_dataframe(all_results: Dict) -> pd.DataFrame:
    """
    Create one combined DataFrame with plateau detection results for every trial
    in every condition.
    """
    rows = []

    for cond_key, res in all_results.items():
        if "error" in res:
            continue

        plateau_summary = res["plateau_summary"]
        processed = res["processed_data"]

        for r in plateau_summary["results"]:
            rows.append(
                {
                    "condition": cond_key,
                    "br_idx": processed.get("br_idx"),
                    "trial_idx": r.get("trial_idx"),
                    "plateau_found": r.get("found", False),
                    "onset_ms": r.get("onset_ms"),
                    "x_at_onset": r.get("x_value"),
                    "y_at_onset": r.get("y_value"),
                    "dist_at_onset": r.get("dist_value"),
                    "signals_used": "+".join(r.get("signals_used", [])),
                    "n_valid_condition": plateau_summary.get("n_valid"),
                    "condition_detection_rate": plateau_summary.get("detection_rate"),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Run full batch plateau analysis.
    """
    print("\n" + "=" * 80)
    print("PLATEAU ANALYSIS BATCH SCRIPT")
    print("=" * 80)

    print("\nSettings:")
    print(f"  PERI_ROOT: {PERI_ROOT}")
    print(f"  Camera: {CAMERA_TO_USE}")
    print(f"  Keypoint: {KEYPOINT_BASE}")
    print(f"  Butterworth params: {SMOOTHING_PARAMS}")
    print(f"  Detection params: {DETECTION_PARAMS}")
    print(f"  Time window: {-PRE_ZERO_MS} to {POST_ZERO_MS} ms")

    all_results = analyze_all_conditions(
        peri_root=Path(PERI_ROOT),
        camera=CAMERA_TO_USE,
        keypoint_base=KEYPOINT_BASE,
        control_br_indices=CONTROL_BR_INDICES,
        smoothing_params=SMOOTHING_PARAMS,
        detection_params=DETECTION_PARAMS,
        time_window=(-PRE_ZERO_MS, POST_ZERO_MS),
        plot_each=PLOT_EACH_CONDITION,
        save_figs=SAVE_FIGS,
        save_dir=PER_CONDITION_FIG_DIR,
    )

    if not all_results:
        print("[error] No results to analyze.")
        return


    if SAVE_STATS:
        # Combined result DataFrame
        plateau_df = create_plateau_dataframe(all_results)
        print("\nCombined plateau DataFrame:")
        print(plateau_df.head())

        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = TABLE_DIR / "plateau_detection_results_all_conditions.csv"
        plateau_df.to_csv(csv_path, index=False)
        print(f"[saved] {csv_path}")

    # -------------------------------------------------------------------------
    # Cell 20 equivalent: compare conditions by target
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TARGET A ANALYSIS")
    print("=" * 60)
    target_a_condition_data, target_a_fig = compare_conditions_by_target(
        all_results,
        target="A",
        cmap="cool",
        pulse_width_ms=PULSE_WIDTH_MS,
    )

    if SAVE_FIGS and target_a_fig is not None:
        save_and_close_fig(
            target_a_fig,
            COMPARISON_FIG_DIR / "target_A_all_conditions_plateau_onset_comparison.png",
        )

    print("\n" + "=" * 60)
    print("TARGET B ANALYSIS")
    print("=" * 60)
    target_b_condition_data, target_b_fig = compare_conditions_by_target(
        all_results,
        target="B",
        cmap="cool",
        pulse_width_ms=PULSE_WIDTH_MS,
    )

    if SAVE_FIGS and target_b_fig is not None:
        save_and_close_fig(
            target_b_fig,
            COMPARISON_FIG_DIR / "target_B_all_conditions_plateau_onset_comparison.png",
        )

    # -------------------------------------------------------------------------
    # Cell 21 equivalent: selected condition statistics
    # -------------------------------------------------------------------------
    if SAVE_STATS:
        print("\n" + "=" * 60)
        print("SELECTED CONDITIONS ANALYSIS - Target A")
        print(f"Comparing: {COMPARE_CONDITIONS if COMPARE_CONDITIONS else 'ALL conditions'}")
        print("=" * 60)

        selected_a_data, selected_a_fig = analyze_selected_conditions(
            all_results,
            conditions_to_compare=COMPARE_CONDITIONS,
            target="A",
            cmap="cool",
            pulse_width_ms=PULSE_WIDTH_MS,
        )

        if SAVE_FIGS and selected_a_fig is not None:
            save_and_close_fig(
                selected_a_fig,
                STATS_FIG_DIR / "target_A_selected_conditions_statistics.png",
            )

        print("\n" + "=" * 60)
        print("SELECTED CONDITIONS ANALYSIS - Target B")
        print(f"Comparing: {COMPARE_CONDITIONS if COMPARE_CONDITIONS else 'ALL conditions'}")
        print("=" * 60)

        selected_b_data, selected_b_fig = analyze_selected_conditions(
            all_results,
            conditions_to_compare=COMPARE_CONDITIONS,
            target="B",
            cmap="cool",
            pulse_width_ms=PULSE_WIDTH_MS,
        )

        if SAVE_FIGS and selected_b_fig is not None:
            save_and_close_fig(
                selected_b_fig,
                STATS_FIG_DIR / "target_B_selected_conditions_statistics.png",
            )

    
    print("\n[done] Plateau analysis complete.")


if __name__ == "__main__":
    main()