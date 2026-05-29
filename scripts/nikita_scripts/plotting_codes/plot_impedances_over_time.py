import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import csv
from matplotlib import cm
from matplotlib.colors import PowerNorm

# Local plotting settings
import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"

# Constants
METADATA_PATH = Path("Y:/Current Project Databases - NHP/2025 Cerebellum prosthesis/Nike/GENERAL_METADATA.xlsx")
DATABASE_DIR = METADATA_PATH.parent
OUTPUT_DIR = DATABASE_DIR

UTAH_REGIONS = [
    ("SMA", 1, 64),
    ("PMd", 65, 128),
    ("M1i", 129, 192),
    ("M1s", 193, 256)
]

def parse_intan(file_path):
    imp_dict = {}
    if not file_path.exists():
        return imp_dict
    with open(file_path, 'r', newline='') as f:
        rdr = csv.reader(f)
        try:
            header = next(rdr)
        except StopIteration:
            return imp_dict
        for row in rdr:
            if len(row) < 5:
                continue
            try:
                name = row[1]
                imp_str = row[4]
                digits = "".join([c for c in name if c.isdigit()])
                if not digits:
                    continue
                ch_id = int(digits[-3:])
                imp_val = float(imp_str) / 1000.0
                imp_dict[ch_id] = imp_val
            except Exception:
                continue
    return imp_dict

def parse_utah(file_path):
    imp_dict = {}
    if not file_path.exists():
        return imp_dict
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('Chan'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            chan_str = parts[0]
            val_str = parts[1]
            try:
                if '-' in chan_str:
                    ch_id = int(chan_str.split('-')[1])
                else:
                    continue
                
                if val_str == '<=':
                    imp_val = 15.0
                elif val_str == '>':
                    imp_val = 1500.0
                else:
                    val_num_str = val_str.replace('kOhm', '')
                    imp_val = float(val_num_str)
                imp_dict[ch_id] = imp_val
            except Exception:
                continue
    return imp_dict

def plot_utah_impedances(data_list, name, vmax, out_path):
    """Plot Utah array impedances over time with regional breakdown"""
    if not data_list:
        print(f"No data for {name}")
        return
        
    all_chs = set()
    for _, d in data_list:
        all_chs.update(d.keys())
    all_chs = sorted(list(all_chs))
    dates = [item[0] for item in data_list]
    
    cmap = cm.get_cmap('magma').copy()
    cmap.set_bad('0.25')
    cmap.set_under('cyan')     # Too low - stands out
    cmap.set_over('lime')      # Too high - stands out
    
    norm = PowerNorm(gamma=0.5, vmin=15, vmax=800)
    
    # Build blocks by region
    blocks = []
    block_names = []
    for region_name, lo, hi in UTAH_REGIONS:
        region_chs = [ch for ch in all_chs if lo <= ch <= hi]
        if not region_chs:
            continue
        mat = np.full((len(region_chs), len(dates)), np.nan)
        for j, (date, d) in enumerate(data_list):
            for i, ch in enumerate(region_chs):
                if ch in d:
                    mat[i, j] = d[ch]
        blocks.append((region_chs, mat))
        block_names.append(region_name)
        
    if not blocks:
        return
        
    total_chs = sum(len(b[0]) for b in blocks)
    height_ratios = [len(b[0]) / total_chs for b in blocks]
    
    fig = plt.figure(figsize=(0.4 * max(8, len(dates) * 0.35 + 3), max(6, total_chs * 0.04 + 2)))
    gs = gridspec.GridSpec(len(blocks), 2, width_ratios=[1, 0.05], 
                           height_ratios=height_ratios, hspace=0.08, wspace=0.05)
    
    for i, (region_chs, mat) in enumerate(blocks):
        ax = fig.add_subplot(gs[i, 0])
        im = ax.pcolormesh(np.arange(len(dates) + 1), np.arange(len(region_chs) + 1), mat, 
                           cmap=cmap, norm=norm, shading='auto', edgecolor='none')
        
        ax.set_yticks([])
        ax.set_ylabel(f"{block_names[i]}\n({len(region_chs)} chs)", 
                      rotation=0, labelpad=40, ha='center', va='center', fontsize=10)
        
        if i == 0:
            ax.set_title(f"{name} Impedances (kOhm)", fontsize=14, fontweight='bold', pad=15)
            
        if i == len(blocks) - 1:
            ax.set_xticks(np.arange(len(dates)) + 0.5)
            ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
        else:
            ax.set_xticks([])
            
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
    
    cax = fig.add_subplot(gs[:, 1])
    cbar = plt.colorbar(im, cax=cax, extend='both')
    cbar.set_label('Impedance (kOhm)', rotation=270, labelpad=15)
    
    # Add reference ticks at meaningful values
    cbar.set_ticks([15, 50, 100, 200, 400, 800])
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_region_trends(data_list, name, out_path):
    """Line plot showing median impedance and % good channels per region over time"""
    if not data_list:
        print(f"No data for {name}")
        return
        
    dates = [item[0] for item in data_list]
    
    fig, axes = plt.subplots(2, 1, figsize=(3, 6), sharex=True)
    colors = {'SMA': 'tab:blue', 'PMd': 'tab:orange', 'M1i': 'tab:green', 'M1s': 'tab:red'}
    
    for region_name, lo, hi in UTAH_REGIONS:
        medians = []
        pct_good = []
        
        for date, d in data_list:
            region_vals = [v for ch, v in d.items() if lo <= ch <= hi]
            if region_vals:
                medians.append(np.median(region_vals))
                pct_good.append(100 * sum(1 for v in region_vals if 15 <= v <= 800) / len(region_vals))
            else:
                medians.append(np.nan)
                pct_good.append(np.nan)
        
        axes[0].plot(dates, medians, 'o-', label=region_name, 
                     color=colors[region_name], linewidth=2, markersize=4)
        axes[1].plot(dates, pct_good, 'o-', label=region_name, 
                     color=colors[region_name], linewidth=2, markersize=4)
    
    # Power scale for y-axis (same gamma as heatmap)
    axes[0].set_yscale('function', functions=(lambda x: np.power(x, 0.5), 
                                               lambda x: np.power(x, 2)))
    axes[0].set_ylim(15, 800)
    axes[0].set_yticks([15, 50, 100, 200, 400, 800])
    axes[0].set_yticklabels(['15', '50', '100', '200', '400', '800'])
    
    axes[0].set_ylabel('Median Impedance (kOhm)', fontsize=11)
    axes[0].axhline(15, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(800, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhspan(15, 800, alpha=0.1, color='green')
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'{name} - Regional Trends', fontsize=12, fontweight='bold')
    
    axes[1].set_ylabel('% Good Channels\n(15-800 kOhm)', fontsize=11)
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc='lower right')
    
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.xlabel('Date', fontsize=11)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

def get_region_name(channel):
    """Get the region name for a given channel number"""
    for region_name, lo, hi in UTAH_REGIONS:
        if lo <= channel <= hi:
            return region_name
    return "Unknown"

def export_impedances_to_csv(data_list, name, out_path):
    """Export all impedance data to a CSV file with electrodes as rows and dates as columns"""
    if not data_list:
        print(f"No data to export for {name}")
        return
    
    # Get all unique channels across all dates
    all_chs = set()
    for _, d in data_list:
        all_chs.update(d.keys())
    all_chs = sorted(list(all_chs))
    
    # Get all dates
    dates = [item[0] for item in data_list]
    
    # Build the data for the CSV
    rows = []
    for ch in all_chs:
        row = {
            'Channel': ch,
            'Region': get_region_name(ch)
        }
        for date, d in data_list:
            row[date] = d.get(ch, np.nan)
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Reorder columns: Channel, Region, then dates in order
    cols = ['Channel', 'Region'] + dates
    df = df[cols]
    
    df.to_csv(out_path, index=False, float_format='%.2f')
    print(f"Saved CSV: {out_path}")
    
    # Also create a summary statistics CSV
    summary_path = out_path.parent / (out_path.stem + "_summary.csv")
    export_summary_stats(data_list, name, summary_path)

def export_summary_stats(data_list, name, out_path):
    """Export summary statistics (median, mean, % good) per region per date"""
    if not data_list:
        return
    
    summary_rows = []
    for date, d in data_list:
        for region_name, lo, hi in UTAH_REGIONS:
            region_vals = [v for ch, v in d.items() if lo <= ch <= hi]
            if region_vals:
                n_total = len(region_vals)
                n_good = sum(1 for v in region_vals if 15 <= v <= 800)
                n_low = sum(1 for v in region_vals if v < 15)
                n_high = sum(1 for v in region_vals if v > 800)
                
                summary_rows.append({
                    'Date': date,
                    'Region': region_name,
                    'N_Channels': n_total,
                    'Median_kOhm': np.median(region_vals),
                    'Mean_kOhm': np.mean(region_vals),
                    'Std_kOhm': np.std(region_vals),
                    'Min_kOhm': np.min(region_vals),
                    'Max_kOhm': np.max(region_vals),
                    'N_Good_15_800': n_good,
                    'Pct_Good': 100 * n_good / n_total,
                    'N_Low_lt15': n_low,
                    'N_High_gt800': n_high
                })
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_path, index=False, float_format='%.2f')
    print(f"Saved summary CSV: {out_path}")


def main():
    print("Loading metadata...")
    df = pd.read_excel(METADATA_PATH)
    
    utah_A_data = []
    utah_B_data = []
    
    for idx, row in df.iterrows():
        date_obj = row['Date']
        if pd.isna(date_obj):
            continue
        if hasattr(date_obj, 'strftime'):
            date_str = date_obj.strftime("%Y%m%d")
            date_disp = date_obj.strftime("%y-%m-%d")
        else:
            date_str = str(date_obj).replace("-", "")[:8]
            date_disp = str(date_obj)[2:10]
            
        session = row['Session']
        if pd.isna(session):
            continue
        
        sess_dir = DATABASE_DIR / f"{date_str}_{session}"
        imp_dir = sess_dir / "Impedances"
        
        if not imp_dir.exists():
            continue
            
        # Utah Port A
        if pd.notna(row.get('Utah_imp_end_PortA')) or pd.notna(row.get('Utah_imp_start_PortA')):
            for fname in ["Utah_imp_end_PortA.txt", "Utah_imp_start_PortA.txt", 
                          "Utah_imp_end_PortA", "Utah_imp_start_PortA"]:
                target = imp_dir / fname
                if target.exists():
                    utah_A_data.append((date_disp, parse_utah(target)))
                    break
                
        # Utah Port B
        if pd.notna(row.get('Utah_imp_end_PortB')) or pd.notna(row.get('Utah_imp_start_PortB')):
            for fname in ["Utah_imp_end_PortB.txt", "Utah_imp_start_PortB.txt", 
                          "Utah_imp_end_PortB", "Utah_imp_start_PortB"]:
                target = imp_dir / fname
                if target.exists():
                    utah_B_data.append((date_disp, parse_utah(target)))
                    break

    # Calculate vmax
    all_utah_vals = []
    for data_list in [utah_A_data, utah_B_data]:
        for _, d in data_list:
            all_utah_vals.extend(d.values())
            
    if all_utah_vals:
        utah_vmax = min(1000, np.nanpercentile(all_utah_vals, 95) * 1.2)
    else:
        utah_vmax = 800
        
    print(f"Utah vmax: {utah_vmax:.1f} kOhm")

    # Generate outputs
    for data, name in [(utah_A_data, "Utah_PortA"), (utah_B_data, "Utah_PortB")]:
        if data:
            plot_utah_impedances(data, name, utah_vmax, 
                                 OUTPUT_DIR / f"{name}_Impedances_Over_Time.png")
            plot_region_trends(data, name, 
                               OUTPUT_DIR / f"{name}_Regional_Trends.png")
            
            export_impedances_to_csv(data, name,
                                     OUTPUT_DIR / f"{name}_Impedances_All_Electrodes.csv")

    print("Done!")

if __name__ == "__main__":
    main()