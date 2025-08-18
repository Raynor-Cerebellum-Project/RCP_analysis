from pathlib import Path
import numpy as np
from scipy.io import loadmat

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.exporters as sexp
from spikeinterface.core import concatenate_recordings
from spikeinterface.widgets import plot_probe
import matplotlib.pyplot as plt

# Optional: probeinterface is only needed to build/attach geometry
from probeinterface import Probe

# ==============================
# Config
# ==============================
DATA_DIR = Path("/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/data")
GEOM_MAT = DATA_DIR / "BL_RW_003_Session_1/ImecPrimateStimRec128_042421.mat"
TARGET_STREAM = "RHS2000 amplifier channel"

OUT_BASE = DATA_DIR / "BL_RW_003_Session_1/Sorting Results/mountainSort5/combined_baseline"
OUT_BASE.mkdir(parents=True, exist_ok=True)


# Global SI job settings
si.set_global_job_kwargs(n_jobs=8, chunk_duration="1s", progress_bar=True)

# ==============================
# Helpers
# ==============================
def load_probe_from_mat(mat_path: Path) -> Probe:
    """Build a ProbeInterface Probe from a Kilosort-style .mat file."""
    mat = loadmat(mat_path)
    x = mat["xcoords"].astype(float).ravel()
    y = mat["ycoords"].astype(float).ravel()
    chanmap0 = mat["chanMap0ind"].astype(int).ravel()  # 0-based device indices
    kcoords = mat.get("kcoords", None)
    kcoords = None if kcoords is None else kcoords.ravel().astype(int)

    pr = Probe(ndim=2)
    pr.set_contacts(positions=np.c_[x, y], shapes="circle", shape_params={"radius": 5})
    pr.set_device_channel_indices(chanmap0)
    if kcoords is not None:
        try:
            pr.set_shank_ids(kcoords)
        except Exception:
            pass
    return pr


def ensure_signed(rec):
    """Convert unsigned dtype to signed (or float) before filtering."""
    if rec.get_dtype().kind == "u":
        rec = spre.unsigned_to_signed(rec)
        rec = spre.astype(rec, dtype="int16")
    return rec

# List the session folders in order
SESSION_FOLDERS = [
    DATA_DIR / "BL_RW_003_Session_1/Intan/BL_closed_loop_STIM_003_250528_140952",
    DATA_DIR / "BL_RW_003_Session_1/Intan/BL_closed_loop_STIM_003_250528_152812",
    DATA_DIR / "BL_RW_003_Session_1/Intan/BL_closed_loop_STIM_003_250528_160100",
]

recordings = [
    se.read_split_intan_files(
        folder_path=folder,
        mode="concatenate",
        stream_name=TARGET_STREAM,
        use_names_as_ids=True,
    )
    for folder in SESSION_FOLDERS
]

rec_concat = concatenate_recordings(recordings)

print(f"Loaded Intan: fs={rec_concat.get_sampling_frequency()} Hz, chans={rec_concat.get_num_channels()}")

rec = ensure_signed(rec_concat)
rec_pp = spre.highpass_filter(rec, freq_min=300)
rec_pp = spre.common_reference(rec_pp, reference="global", operator="median")

# Add 


# Add python version of artifact correction here if needed

rec_pp = rec_pp.save(folder=OUT_BASE / "preprocessed", overwrite=True)

# 3) Attach probe geometry
try:
    assert GEOM_MAT.exists(), f"Missing geometry file: {GEOM_MAT}"
    probe = load_probe_from_mat(GEOM_MAT)
    rec_pp = rec_pp.set_probe(probe, in_place=False)
    print("Attached probe geometry from .mat")
except Exception as e:
    print(f"Warning: failed to attach probe from {GEOM_MAT} ({e}). "
          "Proceeding without geometry; MS5 will use adjacency_radius=-1.")
    
probe = load_probe_from_mat(GEOM_MAT)
plot_probe(rec_pp)
plt.show()

# 4) Run MountainSort5
sorting = ss.run_sorter(
    "mountainsort5",
    rec_pp,
    folder=OUT_BASE / "ms5_output",
    remove_existing_folder=True,
    delete_output_folder=False,
    verbose=True,
    # adjacency_radius=-1,  # uncomment if you skip/aren't confident in geometry
    # detect_threshold=4,
)

# 5) Waveforms + Phy export via SortingAnalyzer
an_dir = OUT_BASE / "analyzer"
analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=rec_pp,
    format="binary_folder",
    folder=an_dir,
    sparse=False,
    overwrite=True,
)
analyzer.compute(
    ["random_spikes", "waveforms", "templates", "noise_levels",
     "spike_amplitudes", "principal_components"],
    extension_params={
        "random_spikes": {
            "method": "uniform",
            "max_spikes_per_unit": 500,   # <-- correct key for your build
            # "seed": 0,                   # optional for reproducibility
            # "margin_size": None,         # optional; keep None unless you need spacing
        },
        "waveforms": {"ms_before": 1.0, "ms_after": 2.0},  # dtype=None by default
        "spike_amplitudes": {"peak_sign": "neg"},
        "principal_components": {"n_components": 3, "mode": "by_channel_local"},
    },
    n_jobs=8, chunk_duration="1s", progress_bar=True, save=True
)


try:
    sexp.export_to_phy(analyzer, output_folder=OUT_BASE / "phy_output_ms5")
except TypeError:
    we = analyzer.get_extension("waveforms").get_waveform_extractor()
    sexp.export_to_phy(we, output_folder=OUT_BASE / "phy_output_ms5",
                       compute_pc_features=True, compute_amplitudes=True)


print("Done. Open with:")
print(f"  cd '{OUT_BASE / 'phy_output_ms5'}' && phy template-gui params.py")