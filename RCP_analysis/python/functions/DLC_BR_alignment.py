import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.extractors as se

# =========================
# File Paths and Matching
# =========================
dlc_csv = Path(
    "/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/data/Bert_reaching_001/Video_DLC/videos/CamVideo_006_Cam-0DLC_Resnet50_Bert_reach_pilotAug6shuffle1_snapshot_010.csv"
)
blackrock_dir = Path(
    "/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/data/Bert_reaching_001/Blackrock"
)
SYNC_CHANNEL_ID = 142
SHOW_FIRST_SECONDS = 5

# Select hardcoded Blackrock file index based on DLC filename
if "CamVideo_006" in dlc_csv.name:
    blackrock_file_index = 2
elif "CamVideo_007" in dlc_csv.name:
    blackrock_file_index = 3
else:
    raise ValueError(
        "Unknown video number in filename; update logic to match the correct Blackrock file."
    )

# =========================
# Load DLC CSV
# =========================
# Load DLC CSV with 3 header rows
df_raw = pd.read_csv(dlc_csv, header=[0, 1, 2])

# Save frame index (0 to N-1, since DLC doesn't always include frame numbers)
frame_idx = np.arange(len(df_raw))

# Drop first column if it's unnamed or a leftover index
if df_raw.columns[0][0].lower() in ("scorer", "index", ""):
    df_raw = df_raw.drop(columns=[df_raw.columns[0]])

# Flatten multi-index column names for simplicity
df_raw.columns = ["_".join(col).strip() for col in df_raw.columns]

# Drop any unnecessary labels
df_raw = df_raw.loc[:, ~df_raw.columns.str.contains("bodyparts_coords")]


frame_idx = df_raw.index.to_numpy()  # use row index as frame numbers


def simple_label(colname: str) -> str:
    parts = colname.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else colname


# =========================
# Load Blackrock File
# =========================
ns5_files = sorted(blackrock_dir.glob("*.ns5"))
if len(ns5_files) <= blackrock_file_index:
    raise IndexError(
        f"Only {len(ns5_files)} .ns5 files found; index {blackrock_file_index} is out of range."
    )
file_to_load = ns5_files[blackrock_file_index]

rec = se.read_blackrock(str(file_to_load), all_annotations=True)
sf = rec.get_sampling_frequency()
ch_ids = rec.get_channel_ids()
SYNC_CHANNEL_ID = str(SYNC_CHANNEL_ID)  # convert to string for consistency

if SYNC_CHANNEL_ID not in ch_ids:
    raise ValueError(
        f"Sync channel {SYNC_CHANNEL_ID} not found in file. Available: {ch_ids}"
    )

sync = rec.get_traces(channel_ids=[SYNC_CHANNEL_ID]).astype(float).squeeze()


# =========================
# Detect Rising Edges
# =========================
def detect_rising_edges(sig):
    ds = sig[:: max(1, len(sig) // 20000 or 1)]
    lo = 0.5 * (ds.min() + ds.max())
    hi = 0.5 * (lo + ds.max())
    state, edges = False, []
    for i, v in enumerate(sig):
        if not state and v >= hi:
            edges.append(i)
            state = True
        elif state and v <= lo:
            state = False
    return np.array(edges, dtype=np.int64)


rising_edges = detect_rising_edges(sync)

# =========================
# Align Frames to Sync Pulses
# =========================
K = min(len(frame_idx), len(rising_edges))
aligned_times = rising_edges[:K] / sf
dlc_frames_used = frame_idx[:K]

mapping = pd.DataFrame(
    {
        "dlc_frame_index": dlc_frames_used,
        "ns5_sample": rising_edges[:K],
        "ns5_time_sec": aligned_times,
    }
)

out_map = dlc_csv.with_name(dlc_csv.stem + "_frame_to_ns5_mapping.csv")
mapping.to_csv(out_map, index=False)
print(f"✔ Saved mapping: {out_map}")

# =========================
# Plot All Aligned Behavior Traces
# =========================
plt.figure(figsize=(12, 6))
for col in df_raw.columns:
    label = simple_label(col)
    plt.plot(aligned_times, df_raw[col][:K], label=label)
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("All DLC Behaviors Aligned to Neural Time")
plt.grid(True)
plt.legend(fontsize=8, ncol=3)
plt.tight_layout()
plt.show()

# =========================
# Plot Sync Trace Preview
# =========================
sync_preview = (
    rec.get_traces(
        channel_ids=[SYNC_CHANNEL_ID],
        start_frame=0,
        end_frame=int(sf * SHOW_FIRST_SECONDS),
    )
    .astype(float)
    .squeeze()
)
t = np.arange(len(sync_preview)) / sf
preview_edges = rising_edges[rising_edges < len(sync_preview)]

plt.figure(figsize=(12, 4))
plt.plot(t, sync_preview, label=f"sync (channel {SYNC_CHANNEL_ID})")
plt.plot(preview_edges / sf, sync_preview[preview_edges], "r*", label="rising edges")
plt.xlabel("Time (s)")
plt.ylabel("Sync amplitude")
plt.title(f"Sync Preview – {file_to_load.name}")
plt.legend()
plt.tight_layout()
plt.show()
