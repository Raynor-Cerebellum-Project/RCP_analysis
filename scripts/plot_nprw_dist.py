import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 1. Load the NPRW mapping
mapping_file = r'e:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis\scripts\nprw_row_mapping.csv'
df_map = pd.read_csv(mapping_file)
# Mapping from Channel ID (ElecID) to Row ID (0-63)
elec_to_row = dict(zip(df_map['chan'], df_map['row']))

# 2. Parse stimulation conditions
input_file = r'e:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis\20260408_rand_sampling_seed_5'
with open(input_file, 'r') as f:
    data = json.load(f)

row_counts = np.zeros(64)

for cond_str in data.values():
    if not cond_str: continue
    # Format: "start-end start-end ..."
    ranges = cond_str.strip(';').split(' ')
    for r_str in ranges:
        if '-' in r_str:
            try:
                start, end = map(int, r_str.split('-'))
                for eid in range(start, end + 1):
                    if eid in elec_to_row:
                        row_counts[elec_to_row[eid]] += 1
            except:
                pass
        elif r_str:
            try:
                eid = int(r_str)
                if eid in elec_to_row:
                    row_counts[elec_to_row[eid]] += 1
            except:
                pass

# 3. Plot
plt.figure(figsize=(12, 6))
rows = np.arange(64)
plt.bar(rows, row_counts, color='teal', alpha=0.8, edgecolor='black', linewidth=0.5)

plt.title(f"Distribution of Selected Rows on NRPW Probe\nFile: {os.path.basename(input_file)} | Total Selections: {int(np.sum(row_counts))}", fontsize=14)
plt.xlabel("Physical Row Index (Depth on Shank, 0-63)", fontsize=12)
plt.ylabel("Acitvation Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(np.arange(0, 65, 4))

# Add a horizontal line for the average
avg_count = np.mean(row_counts)
plt.axhline(avg_count, color='red', linestyle='--', label=f'Average ({avg_count:.1f})')
plt.legend()

plt.tight_layout()

output_plot = r'e:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis\scripts\nprw_row_distribution.png'
plt.savefig(output_plot, dpi=150)
print(f"Plot saved to {output_plot}")
