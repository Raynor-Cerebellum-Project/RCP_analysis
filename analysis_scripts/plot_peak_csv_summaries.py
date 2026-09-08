#!/usr/bin/env python3
"""
plot_peak_csv_summaries.py

Analyze peak timing and rate CSVs from overlay script.
- Between-condition comparisons (REST/CTRL/STIM bar plots)
- Within-condition channel comparisons
- Channel classification by response pattern
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CSV_ROOT = "/home/keith/stim_response_overlays"
OUTPUT_DIR = "/home/keith/stim_response_overlays/_csv_summaries"

# Thresholds for classification (in Hz or ms)
RATE_CHANGE_THRESHOLD_HZ = 5.0  # Minimum change to count as increase/decrease
TIMING_CHANGE_THRESHOLD_MS = 10.0  # Minimum timing shift to count as significant

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def find_all_csvs(root_dir: str, pattern: str) -> List[Path]:
    """Find all CSVs matching pattern recursively."""
    csvs = []
    for path in Path(root_dir).rglob(pattern):
        csvs.append(path)
    return csvs


def extract_target_from_path(csv_path: Path) -> str:
    """Extract target (A or B) from CSV path."""
    path_str = str(csv_path).lower()
    if "target_a" in path_str or "targeta" in path_str:
        return "A"
    elif "target_b" in path_str or "targetb" in path_str:
        return "B"
    else:
        # Try to infer from parent folders
        for part in csv_path.parts:
            if "a" in part.lower() and "target" in part.lower():
                return "A"
            if "b" in part.lower() and "target" in part.lower():
                return "B"
        return "Unknown"


def load_nprw_csvs(root_dir: str) -> pd.DataFrame:
    """Load all NPRW peak timing CSVs into single DataFrame."""
    csvs = find_all_csvs(root_dir, "*nprw_peak_timings.csv")
    
    all_dfs = []
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
            df["source_file"] = str(csv_path)
            df["target"] = extract_target_from_path(csv_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {csv_path}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"Loaded {len(combined)} NPRW channel records from {len(csvs)} files")
        return combined
    return pd.DataFrame()


def load_ua_csvs(root_dir: str) -> pd.DataFrame:
    """Load all UA mean rate CSVs into single DataFrame."""
    csvs = find_all_csvs(root_dir, "*ua_mean_rates.csv")
    
    all_dfs = []
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
            df["source_file"] = str(csv_path)
            df["target"] = extract_target_from_path(csv_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {csv_path}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"Loaded {len(combined)} UA electrode records from {len(csvs)} files")
        return combined
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Between-Condition Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_condition_means_nprw(df: pd.DataFrame, high_activity_only: bool = True) -> Dict:
    """
    Compute mean peak timing and rate per condition for NPRW data.
    
    Returns dict with structure:
    {
        'A': {'stim': {'timing': mean, 'rate': mean, 'n': count}, 'ctrl': {...}, 'rest': {...}},
        'B': {...}
    }
    """
    results = {}
    
    for target in df["target"].unique():
        target_df = df[df["target"] == target]
        
        if high_activity_only and "is_high_activity" in target_df.columns:
            target_df = target_df[target_df["is_high_activity"] == True]
        
        results[target] = {}
        
        for cond in ["stim", "ctrl", "rest"]:
            timing_col = f"{cond}_peak_time_ms"
            rate_col = f"{cond}_peak_rate_hz"
            
            if timing_col in target_df.columns and rate_col in target_df.columns:
                valid = target_df[target_df[timing_col].notna() & target_df[rate_col].notna()]
                results[target][cond] = {
                    "timing_mean": valid[timing_col].mean(),
                    "timing_std": valid[timing_col].std(),
                    "rate_mean": valid[rate_col].mean(),
                    "rate_std": valid[rate_col].std(),
                    "n": len(valid)
                }
            else:
                results[target][cond] = {
                    "timing_mean": np.nan, "timing_std": np.nan,
                    "rate_mean": np.nan, "rate_std": np.nan, "n": 0
                }
    
    return results


def compute_condition_means_ua(df: pd.DataFrame) -> Dict:
    """Compute mean rates per condition for UA data."""
    results = {}
    
    for target in df["target"].unique():
        target_df = df[df["target"] == target]
        results[target] = {}
        
        for cond in ["stim", "ctrl", "rest"]:
            rate_col = f"{cond}_mean_rate_hz"
            
            if rate_col in target_df.columns:
                valid = target_df[target_df[rate_col].notna()]
                results[target][cond] = {
                    "rate_mean": valid[rate_col].mean(),
                    "rate_std": valid[rate_col].std(),
                    "n": len(valid)
                }
            else:
                results[target][cond] = {"rate_mean": np.nan, "rate_std": np.nan, "n": 0}
    
    return results


def plot_condition_comparison_bars(results: Dict, metric: str, title: str, ylabel: str, 
                                   output_path: str):
    """
    Plot bar chart comparing conditions (REST/CTRL/STIM) split by target.
    
    Args:
        results: Dict from compute_condition_means_*
        metric: 'timing_mean' or 'rate_mean'
        title: Plot title
        ylabel: Y-axis label
        output_path: Where to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    targets = sorted(results.keys())
    conditions = ["ctrl", "rest", "stim"]
    cond_labels = ["CTRL", "REST", "STIM"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]  # green, blue, red
    
    x = np.arange(len(targets))
    width = 0.25
    
    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, colors)):
        means = []
        stds = []
        for target in targets:
            if target in results and cond in results[target]:
                means.append(results[target][cond].get(metric, np.nan))
                stds.append(results[target][cond].get(metric.replace("mean", "std"), 0))
            else:
                means.append(np.nan)
                stds.append(0)
        
        bars = ax.bar(x + i * width, means, width, label=label, color=color, 
                      yerr=stds, capsize=3, alpha=0.8)
    
    ax.set_xlabel("Target", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Target {t}" for t in targets])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Channel Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_channels_nprw(df: pd.DataFrame, 
                           rate_threshold: float = RATE_CHANGE_THRESHOLD_HZ) -> pd.DataFrame:
    """
    Classify NPRW channels by response pattern (rate changes).
    
    Categories:
    - both_increase: Higher rate in both REST and STIM vs CTRL
    - stim_only_increase: Higher rate in STIM vs CTRL, not REST
    - rest_only_increase: Higher rate in REST vs CTRL, not STIM
    - both_decrease: Lower rate in both REST and STIM vs CTRL
    - stim_only_decrease: Lower rate in STIM vs CTRL, not REST
    - rest_only_decrease: Lower rate in REST vs CTRL, not STIM
    - mixed: Opposite directions for REST and STIM
    - no_change: No significant changes
    """
    df = df.copy()
    
    # Compute deltas
    df["delta_stim_vs_ctrl"] = df["stim_peak_rate_hz"] - df["ctrl_peak_rate_hz"]
    df["delta_rest_vs_ctrl"] = df["rest_peak_rate_hz"] - df["ctrl_peak_rate_hz"]
    
    # Classify
    def classify_row(row):
        d_stim = row["delta_stim_vs_ctrl"]
        d_rest = row["delta_rest_vs_ctrl"]
        
        if pd.isna(d_stim) or pd.isna(d_rest):
            return "insufficient_data"
        
        stim_inc = d_stim > rate_threshold
        stim_dec = d_stim < -rate_threshold
        rest_inc = d_rest > rate_threshold
        rest_dec = d_rest < -rate_threshold
        
        if stim_inc and rest_inc:
            return "both_increase"
        elif stim_inc and not rest_inc and not rest_dec:
            return "stim_only_increase"
        elif rest_inc and not stim_inc and not stim_dec:
            return "rest_only_increase"
        elif stim_dec and rest_dec:
            return "both_decrease"
        elif stim_dec and not rest_dec and not rest_inc:
            return "stim_only_decrease"
        elif rest_dec and not stim_dec and not stim_inc:
            return "rest_only_decrease"
        elif (stim_inc and rest_dec) or (stim_dec and rest_inc):
            return "mixed"
        else:
            return "no_change"
    
    df["classification"] = df.apply(classify_row, axis=1)
    return df


def classify_channels_ua(df: pd.DataFrame,
                         rate_threshold: float = RATE_CHANGE_THRESHOLD_HZ) -> pd.DataFrame:
    """Classify UA electrodes by response pattern."""
    df = df.copy()
    
    # Check if delta columns already exist, otherwise compute
    if "delta_stim_vs_ctrl" not in df.columns:
        df["delta_stim_vs_ctrl"] = df["stim_mean_rate_hz"] - df["ctrl_mean_rate_hz"]
    if "delta_rest_vs_ctrl" not in df.columns:
        df["delta_rest_vs_ctrl"] = df["rest_mean_rate_hz"] - df["ctrl_mean_rate_hz"]
    
    def classify_row(row):
        d_stim = row["delta_stim_vs_ctrl"]
        d_rest = row["delta_rest_vs_ctrl"]
        
        if pd.isna(d_stim) or pd.isna(d_rest):
            return "insufficient_data"
        
        stim_inc = d_stim > rate_threshold
        stim_dec = d_stim < -rate_threshold
        rest_inc = d_rest > rate_threshold
        rest_dec = d_rest < -rate_threshold
        
        if stim_inc and rest_inc:
            return "both_increase"
        elif stim_inc and not rest_inc and not rest_dec:
            return "stim_only_increase"
        elif rest_inc and not stim_inc and not stim_dec:
            return "rest_only_increase"
        elif stim_dec and rest_dec:
            return "both_decrease"
        elif stim_dec and not rest_dec and not rest_inc:
            return "stim_only_decrease"
        elif rest_dec and not stim_dec and not stim_inc:
            return "rest_only_decrease"
        elif (stim_inc and rest_dec) or (stim_dec and rest_inc):
            return "mixed"
        else:
            return "no_change"
    
    df["classification"] = df.apply(classify_row, axis=1)
    return df


def plot_classification_summary(df: pd.DataFrame, array_type: str, output_path: str):
    """Plot bar chart of classification counts."""
    if "classification" not in df.columns:
        print(f"  No classification column for {array_type}")
        return
    
    # Count by target and classification
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    targets = sorted(df["target"].unique())
    
    for ax, target in zip(axes, targets):
        target_df = df[df["target"] == target]
        counts = target_df["classification"].value_counts()
        
        # Order categories logically
        order = ["both_increase", "stim_only_increase", "rest_only_increase",
                 "no_change", "mixed",
                 "rest_only_decrease", "stim_only_decrease", "both_decrease",
                 "insufficient_data"]
        ordered_counts = [counts.get(cat, 0) for cat in order]
        
        colors = ["#27ae60", "#2ecc71", "#82e0aa",  # greens for increases
                  "#bdc3c7", "#f39c12",  # gray for no change, orange for mixed
                  "#f5b7b1", "#e74c3c", "#c0392b",  # reds for decreases
                  "#95a5a6"]  # gray for insufficient
        
        bars = ax.barh(order, ordered_counts, color=colors)
        ax.set_xlabel("Channel Count")
        ax.set_title(f"Target {target} - {array_type.upper()}")
        ax.invert_yaxis()
        
        # Add count labels
        for bar, count in zip(bars, ordered_counts):
            if count > 0:
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        str(count), va='center', fontsize=9)
    
    plt.suptitle(f"Channel Classification by Response Pattern\n({array_type.upper()}, threshold={RATE_CHANGE_THRESHOLD_HZ} Hz)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_classification_table(df: pd.DataFrame, array_type: str, output_path: str):
    """Generate CSV summary table of classifications."""
    if "classification" not in df.columns:
        return
    
    # Pivot table: rows=classification, columns=target
    pivot = df.groupby(["target", "classification"]).size().unstack(fill_value=0)
    
    # Add totals
    pivot["Total"] = pivot.sum(axis=1)
    pivot.loc["Total"] = pivot.sum()
    
    pivot.to_csv(output_path)
    print(f"  Saved: {output_path}")
    
    return pivot


# ─────────────────────────────────────────────────────────────────────────────
# Within-Condition Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_channel_scatter(df: pd.DataFrame, x_col: str, y_col: str, 
                         title: str, xlabel: str, ylabel: str,
                         output_path: str, hue_col: str = "target"):
    """Scatter plot comparing two metrics across channels."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    targets = df[hue_col].unique()
    colors = {"A": "#3498db", "B": "#e74c3c", "Unknown": "#95a5a6"}
    
    for target in targets:
        subset = df[df[hue_col] == target]
        valid = subset[subset[x_col].notna() & subset[y_col].notna()]
        ax.scatter(valid[x_col], valid[y_col], 
                   c=colors.get(target, "#95a5a6"),
                   label=f"Target {target} (n={len(valid)})",
                   alpha=0.6, s=30)
    
    # Add unity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Unity')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Peak CSV Summary Analysis")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n[1] Loading CSVs...")
    nprw_df = load_nprw_csvs(CSV_ROOT)
    ua_df = load_ua_csvs(CSV_ROOT)
    
    if nprw_df.empty and ua_df.empty:
        print("No data found. Exiting.")
        return
    
    # ─────────────────────────────────────────────────────────────────────────
    # NPRW Analysis
    # ─────────────────────────────────────────────────────────────────────────
    if not nprw_df.empty:
        print("\n[2] NPRW Analysis...")
        
        # Between-condition comparisons
        print("  Computing condition means (high-activity channels only)...")
        nprw_means = compute_condition_means_nprw(nprw_df, high_activity_only=True)
        
        print("  Plotting condition comparisons...")
        plot_condition_comparison_bars(
            nprw_means, "timing_mean",
            "NPRW Peak Timing by Condition (High-Activity Channels)",
            "Peak Time (ms)",
            os.path.join(OUTPUT_DIR, "nprw_condition_timing.png")
        )
        plot_condition_comparison_bars(
            nprw_means, "rate_mean",
            "NPRW Peak Rate by Condition (High-Activity Channels)",
            "Peak Rate (Hz)",
            os.path.join(OUTPUT_DIR, "nprw_condition_rate.png")
        )
        
        # Classification
        print("  Classifying channels...")
        nprw_classified = classify_channels_nprw(nprw_df)
        
        plot_classification_summary(
            nprw_classified, "nprw",
            os.path.join(OUTPUT_DIR, "nprw_classification_summary.png")
        )
        generate_classification_table(
            nprw_classified, "nprw",
            os.path.join(OUTPUT_DIR, "nprw_classification_counts.csv")
        )
        
        # Save classified data
        nprw_classified.to_csv(
            os.path.join(OUTPUT_DIR, "nprw_all_channels_classified.csv"),
            index=False
        )
        print(f"  Saved: nprw_all_channels_classified.csv")
        
        # Within-condition scatter plots
        print("  Plotting within-condition comparisons...")
        plot_channel_scatter(
            nprw_classified,
            "ctrl_peak_rate_hz", "stim_peak_rate_hz",
            "NPRW: CTRL vs STIM Peak Rate",
            "CTRL Peak Rate (Hz)", "STIM Peak Rate (Hz)",
            os.path.join(OUTPUT_DIR, "nprw_scatter_ctrl_vs_stim_rate.png")
        )
        plot_channel_scatter(
            nprw_classified,
            "ctrl_peak_rate_hz", "rest_peak_rate_hz",
            "NPRW: CTRL vs REST Peak Rate",
            "CTRL Peak Rate (Hz)", "REST Peak Rate (Hz)",
            os.path.join(OUTPUT_DIR, "nprw_scatter_ctrl_vs_rest_rate.png")
        )
        plot_channel_scatter(
            nprw_classified,
            "ctrl_peak_time_ms", "stim_peak_time_ms",
            "NPRW: CTRL vs STIM Peak Timing",
            "CTRL Peak Time (ms)", "STIM Peak Time (ms)",
            os.path.join(OUTPUT_DIR, "nprw_scatter_ctrl_vs_stim_timing.png")
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # UA Analysis
    # ─────────────────────────────────────────────────────────────────────────
    if not ua_df.empty:
        print("\n[3] UA Analysis...")
        
        # Between-condition comparisons
        print("  Computing condition means...")
        ua_means = compute_condition_means_ua(ua_df)
        
        print("  Plotting condition comparisons...")
        plot_condition_comparison_bars(
            ua_means, "rate_mean",
            "UA Mean Rate by Condition",
            "Mean Rate (Hz)",
            os.path.join(OUTPUT_DIR, "ua_condition_rate.png")
        )
        
        # Classification
        print("  Classifying electrodes...")
        ua_classified = classify_channels_ua(ua_df)
        
        plot_classification_summary(
            ua_classified, "ua",
            os.path.join(OUTPUT_DIR, "ua_classification_summary.png")
        )
        generate_classification_table(
            ua_classified, "ua",
            os.path.join(OUTPUT_DIR, "ua_classification_counts.csv")
        )
        
        # Save classified data
        ua_classified.to_csv(
            os.path.join(OUTPUT_DIR, "ua_all_electrodes_classified.csv"),
            index=False
        )
        print(f"  Saved: ua_all_electrodes_classified.csv")
        
        # Within-condition scatter plots
        print("  Plotting within-condition comparisons...")
        plot_channel_scatter(
            ua_classified,
            "ctrl_mean_rate_hz", "stim_mean_rate_hz",
            "UA: CTRL vs STIM Mean Rate",
            "CTRL Mean Rate (Hz)", "STIM Mean Rate (Hz)",
            os.path.join(OUTPUT_DIR, "ua_scatter_ctrl_vs_stim_rate.png")
        )
        plot_channel_scatter(
            ua_classified,
            "ctrl_mean_rate_hz", "rest_mean_rate_hz",
            "UA: CTRL vs REST Mean Rate",
            "CTRL Mean Rate (Hz)", "REST Mean Rate (Hz)",
            os.path.join(OUTPUT_DIR, "ua_scatter_ctrl_vs_rest_rate.png")
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary Statistics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4] Summary Statistics...")
    
    summary_lines = [
        "Peak CSV Summary Analysis Results",
        "=" * 50,
        f"Output directory: {OUTPUT_DIR}",
        "",
    ]
    
    if not nprw_df.empty:
        summary_lines.extend([
            "NPRW Summary:",
            f"  Total channels: {len(nprw_df)}",
            f"  Targets: {sorted(nprw_df['target'].unique())}",
            f"  Source files: {nprw_df['source_file'].nunique()}",
        ])
        if "classification" in nprw_classified.columns:
            class_counts = nprw_classified["classification"].value_counts()
            summary_lines.append("  Classification counts:")
            for cat, count in class_counts.items():
                summary_lines.append(f"    {cat}: {count}")
        summary_lines.append("")
    
    if not ua_df.empty:
        summary_lines.extend([
            "UA Summary:",
            f"  Total electrodes: {len(ua_df)}",
            f"  Targets: {sorted(ua_df['target'].unique())}",
            f"  Source files: {ua_df['source_file'].nunique()}",
        ])
        if "classification" in ua_classified.columns:
            class_counts = ua_classified["classification"].value_counts()
            summary_lines.append("  Classification counts:")
            for cat, count in class_counts.items():
                summary_lines.append(f"    {cat}: {count}")
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    with open(os.path.join(OUTPUT_DIR, "analysis_summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"\nSaved summary to: {os.path.join(OUTPUT_DIR, 'analysis_summary.txt')}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()