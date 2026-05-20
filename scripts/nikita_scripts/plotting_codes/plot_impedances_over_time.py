import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from pathlib import Path
import csv
from matplotlib import cm
import re

# Local plotting settings
import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
# plt.style.use('dark_background') # Better aesthetics for heatmaps

# Constants
METADATA_PATH = Path("Y:/Current Project Databases - NHP/2025 Cerebellum prosthesis/Nike/GENERAL_METADATA.xlsx")
DATABASE_DIR = METADATA_PATH.parent
OUTPUT_DIR = DATABASE_DIR  # Save in the same folder as GENERAL_METADATA

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
            if len(row) < 5: continue
            try:
                name = row[1]
                imp_str = row[4]
                digits = "".join([c for c in name if c.isdigit()])
                if not digits: continue
                ch_id = int(digits[-3:])
                imp_val = float(imp_str) / 1000.0  # Convert from Ohm to kOhm
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
            if len(parts) < 2: continue
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
                    imp_val = 1500.0 # Just a high value for out of range
                else:
                    val_num_str = val_str.replace('kOhm', '')
                    imp_val = float(val_num_str)
                imp_dict[ch_id] = imp_val
            except Exception:
                continue
    return imp_dict

def plot_single_matrix(data_list, name, vmax, out_path, is_utah=False):
    if not data_list:
        print(f"No data for {name}")
        return
        
    all_chs = set()
    for _, d in data_list:
        all_chs.update(d.keys())
    all_chs = sorted(list(all_chs))
    
    dates = [item[0] for item in data_list]
    
    cmap = cm.get_cmap('viridis').copy() # viridis / RdYlGn_r
    cmap.set_bad('0.25') # Gray for NaNs
    
    # Intan Case
    if not is_utah:
        mat = np.full((len(all_chs), len(dates)), np.nan)
        for j, (date, d) in enumerate(data_list):
            for i, ch in enumerate(all_chs):
                if ch in d:
                    mat[i, j] = d[ch]
                    
        fig, ax = plt.subplots(figsize=(max(8, len(dates)*0.35 + 3), max(4, len(all_chs)*0.03 + 2)))
        
        im = ax.pcolormesh(np.arange(len(dates)+1), np.arange(len(all_chs)+1), mat, 
                           cmap=cmap, vmin=0, vmax=vmax, shading='auto', edgecolor='none')
        
        ax.set_xticks(np.arange(len(dates)) + 0.5)
        ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
        ax.set_yticks([0.5, len(all_chs)-0.5])
        ax.set_yticklabels([str(all_chs[0]), str(all_chs[-1])])
        ax.set_ylabel("Channel ID", fontsize=11)
        ax.set_title(f"{name} Impedances (kOhm)", fontsize=14, fontweight='bold', pad=15)
        
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Impedance (kOhm)', rotation=270, labelpad=15)
        
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")
        
    # Utah Case
    else:
        blocks = []
        block_names = []
        for region_name, lo, hi in UTAH_REGIONS:
            region_chs = [ch for ch in all_chs if lo <= ch <= hi]
            if not region_chs: continue
            mat = np.full((len(region_chs), len(dates)), np.nan)
            for j, (date, d) in enumerate(data_list):
                for i, ch in enumerate(region_chs):
                    if ch in d:
                        mat[i, j] = d[ch]
            blocks.append((region_chs, mat))
            block_names.append(region_name)
            
        if not blocks: return
        
        total_chs = sum(len(b[0]) for b in blocks)
        height_ratios = [len(b[0])/total_chs for b in blocks]
        
        fig = plt.figure(figsize=(max(8, len(dates)*0.35 + 3), max(6, total_chs*0.04 + 2)))
        gs = gridspec.GridSpec(len(blocks), 2, width_ratios=[1, 0.05], height_ratios=height_ratios, hspace=0.08, wspace=0.05)
        
        axes = []
        
        for i, (region_chs, mat) in enumerate(blocks):
            ax = fig.add_subplot(gs[i, 0])
            im = ax.pcolormesh(np.arange(len(dates)+1), np.arange(len(region_chs)+1), mat, 
                               cmap=cmap, vmin=0, vmax=vmax, shading='auto', edgecolor='none')
            
            # Simple horizontal line to separate blocks visually if desired
            ax.set_yticks([]) 
            ax.set_ylabel(f"{block_names[i]}\n({len(region_chs)} chs)", rotation=0, labelpad=40, ha='center', va='center', fontsize=10)
            
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
                
            axes.append(ax)
            
        cax = fig.add_subplot(gs[:, 1])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Impedance (kOhm)', rotation=270, labelpad=15)
        
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    print("Loading metadata...")
    df = pd.read_excel(METADATA_PATH)
    
    intan_data = [] 
    utah_A_data = []
    utah_B_data = []
    
    for idx, row in df.iterrows():
        date_obj = row['Date']
        if pd.isna(date_obj): continue
        if hasattr(date_obj, 'strftime'):
            date_str = date_obj.strftime("%Y%m%d")
            date_disp = date_obj.strftime("%y-%m-%d") # short format
        else:
            date_str = str(date_obj).replace("-", "")[:8]
            date_disp = str(date_obj)[2:10] # short format
            
        session = row['Session']
        if pd.isna(session): continue
        
        sess_dir = DATABASE_DIR / f"{date_str}_{session}"
        imp_dir = sess_dir / "Impedances"
        
        if not imp_dir.exists():
            continue
            
        # Intan
        if pd.notna(row.get('Intan_imp_end')) or pd.notna(row.get('Intan_imp_start')):
            target_intan = imp_dir / "Intan_imp_end.csv"
            if not target_intan.exists():
                target_intan = imp_dir / "Intan_imp_start.csv"
            if target_intan.exists():
                intan_data.append((date_disp, parse_intan(target_intan)))
                
        # Utah Port A
        if pd.notna(row.get('Utah_imp_end_PortA')) or pd.notna(row.get('Utah_imp_start_PortA')):
            target_ua = imp_dir / "Utah_imp_end_PortA.txt"
            if not target_ua.exists():
                target_ua = imp_dir / "Utah_imp_start_PortA.txt"
                if not target_ua.exists():
                    target_ua = imp_dir / "Utah_imp_end_PortA"
                    if not target_ua.exists():
                        target_ua = imp_dir / "Utah_imp_start_PortA"
            if target_ua.exists():
                utah_A_data.append((date_disp, parse_utah(target_ua)))
                
        # Utah Port B
        if pd.notna(row.get('Utah_imp_end_PortB')) or pd.notna(row.get('Utah_imp_start_PortB')):
            target_ub = imp_dir / "Utah_imp_end_PortB.txt"
            if not target_ub.exists():
                target_ub = imp_dir / "Utah_imp_start_PortB.txt"
                if not target_ub.exists():
                    target_ub = imp_dir / "Utah_imp_end_PortB"
                    if not target_ub.exists():
                        target_ub = imp_dir / "Utah_imp_start_PortB"
            if target_ub.exists():
                utah_B_data.append((date_disp, parse_utah(target_ub)))

    # Calculate global vmin/vmax using 95th percentile strategy 
    # to avoid a few blown-out channels ruining the visual scale
    
    all_intan_vals = []
    for _, d in intan_data:
        all_intan_vals.extend(d.values())
        
    if all_intan_vals:
        intan_vmax = np.nanpercentile(all_intan_vals, 95) * 1.2
        intan_vmax = max(100, intan_vmax) # Set a minimum scale
    else:
        intan_vmax = 1000  
        
    all_utah_vals = []
    for data_list in [utah_A_data, utah_B_data]:
        for _, d in data_list:
            all_utah_vals.extend(d.values())
            
    if all_utah_vals:
        utah_vmax = np.nanpercentile(all_utah_vals, 95) * 1.2
        utah_vmax = max(100, utah_vmax)
    else:
        utah_vmax = 800  
        
    print(f"Auto-scaling limits: Intan ~{intan_vmax:.1f} kOhm, Utah ~{utah_vmax:.1f} kOhm")

    plot_single_matrix(intan_data, "Intan", intan_vmax, OUTPUT_DIR / "Intan_Impedances_Over_Time.png", is_utah=False)
    plot_single_matrix(utah_A_data, "Utah_PortA", utah_vmax, OUTPUT_DIR / "Utah_PortA_Impedances_Over_Time.png", is_utah=True)
    plot_single_matrix(utah_B_data, "Utah_PortB", utah_vmax, OUTPUT_DIR / "Utah_PortB_Impedances_Over_Time.png", is_utah=True)
    
    print("Done!")

if __name__ == "__main__":
    main()
