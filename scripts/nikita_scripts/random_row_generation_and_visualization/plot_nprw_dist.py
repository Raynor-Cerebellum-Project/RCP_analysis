import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 1. Load the NPRW mapping
# Mapping from Channel ID (ElecID) to Row ID (0-63)
# Based on the probe: each row `k` has channels `2*k - 1` and `2*k` (1-indexed row). 
# Using 0-indexed row for plotting: Row 0 has channels 1, 2. Row 63 has 127, 128.
elec_to_row = {ch: (ch - 1) // 2 for ch in range(1, 129)}

# 2. Parse stimulation conditions
# input_file = r'e:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis\20260408_rand_sampling_seed_5'
# input_file = r'e:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis\testing_whatever_different_groups'
input_file = r'E:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis\20260430_rand_sampling_seed_6_3groups_24channels.json'
with open(input_file, 'r') as f:
    data = json.load(f)

# Sort conditions appropriately if they are numeric keys. "1", "2", ... "60"
try:
    conditions = sorted(data.items(), key=lambda x: int(x[0]))
except:
    conditions = list(data.items())

num_conditions = len(conditions)

row_counts = np.zeros(64)
# For the heatmap/raster: matrix of size [64 rows, num_conditions]
selection_matrix = np.zeros((64, num_conditions))

for trial_idx, (cond_key, cond_str) in enumerate(conditions):
    if not cond_str: continue
    # Format: "start-end start-end ..."
    ranges = cond_str.strip(';').split(' ')
    for r_str in ranges:
        if '-' in r_str:
            try:
                start, end = map(int, r_str.split('-'))
                for eid in range(start, end + 1):
                    if eid in elec_to_row:
                        r_idx = elec_to_row[eid]
                        row_counts[r_idx] += 1
                        selection_matrix[r_idx, trial_idx] = 1
            except:
                pass
        elif r_str:
            try:
                eid = int(r_str)
                if eid in elec_to_row:
                    r_idx = elec_to_row[eid]
                    row_counts[r_idx] += 1
                    selection_matrix[r_idx, trial_idx] = 1
            except:
                pass

# 3. Plot
fig, axs = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]})

# Top subplot: Bar chart
rows = np.arange(64)
axs[0].bar(rows, row_counts, color='teal', alpha=0.8, edgecolor='black', linewidth=0.5)

axs[0].set_title(f"Distribution of Selected Rows on NRPW Probe\nFile: {os.path.basename(input_file)} | Total Selections: {int(np.sum(row_counts))}", fontsize=14)
axs[0].set_ylabel("Activation Frequency", fontsize=12)
axs[0].grid(axis='y', linestyle='--', alpha=0.6)
axs[0].set_xticks(np.arange(0, 65, 4))
axs[0].set_xlim(-1, 64)

# Add a horizontal line for the average
avg_count = np.mean(row_counts)
axs[0].axhline(avg_count, color='red', linestyle='--', label=f'Average ({avg_count:.1f})')
axs[0].legend()

# Bottom subplot: Heatmap / Raster
cmap = plt.cm.get_cmap('Blues')
im = axs[1].imshow(selection_matrix, aspect='auto', origin='lower', cmap=cmap, interpolation='none')

axs[1].set_title("Row Selection Across Trials", fontsize=14)
axs[1].set_xlabel("Trial (Condition Index)", fontsize=12)
axs[1].set_ylabel("Physical Row Index (Depth, 0-63)", fontsize=12)
axs[1].set_yticks(np.arange(0, 65, 4))
axs[1].set_xlim(-0.5, num_conditions - 0.5)
axs[1].set_ylim(-0.5, 63.5)

plt.tight_layout()

output_plot = r'e:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis\scripts\nprw_row_distribution.png'
plt.savefig(output_plot, dpi=150)
print(f"Plot saved to {output_plot}")
