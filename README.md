# Analysis Pipeline for the Raynor Project

## Overview
This repository contains the analysis pipeline for preprocessing, quantifying, and comparing stimulation trials from the Raynor experiments.  

![alt text](https://github.com/Raynor-Cerebellum-Project/RCP_analysis/blob/main/docs/pipeline_schematic.png "pipeline_schematic")

[Summary of recorded signals](https://docs.google.com/document/d/1C4-xSWL8n7P_mMrYqlUxQR6blz9WIHDU4B7O9bxhMnE/edit?tab=t.0)

[Example final figures: Nike's plots for reaching](https://docs.google.com/presentation/d/1L_EA5BOvmqIdInJ0WPRp5rtgeDXCOmu6qmt2yeqPC-s/edit?slide=id.g3afc5261748_0_102#slide=id.g3afc5261748_0_102)

The folder structure is as follows:
```
.
├── batch_scripts            # Steps of the pipeline
├── config
│   ├── params.yaml          # Params file indicating location of data and session
├── curation_scripts         # Curation includes visualizing kinematics and OCR frame correction
├── RCP_analysis
│   ├── MATLAB               # Unorganized MATLAB code
│   └── python
│       ├── functions           # Functions for scripts
│       └── plotting            # Plotting code
├── scripts                  # Scripts for running batch scripts and plotting
├── LICENSE
└── README.md
```
Outputs:
```
.
├── data                          # Experimental data files
│   ├── results                  # Output files
│   │   ├── aux_data             # Auxillary data (sync pulses etc.)
│   │   ├── checkpoints          # Intermediate files
│   │   └── figures
```
data and results (output) can be chosen to be other directories in config.


---

# Installing the pipeline
Use **Anaconda Prompt** on windows, make sure you have CUDA installed
```bash
*navigate to desired directory*
# 1) clone repo
git clone https://github.com/Raynor-Cerebellum-Project/RCP_analysis.git
cd RCP_analysis

# 2) create environment
conda env create -f environment.yml -n pipeline
conda activate pipeline
python -m pip install --upgrade pip

# 3) install easyocr without letting pip affect PyTorch
python -m pip install easyocr --no-deps

# 4) install RCP_analysis package in editable mode
pip install -e .
```

---
## 0: Frame correction
`curation_scripts/OCR_frame_mapping_BT_edit.py`

Run by specifying path to video folder:

```bash
python curation_scripts/OCR_frame_mapping_BT_edit.py PATH_TO_VIDEO_FOLDER
```
This creates the OCR files for the pipeline

NOTE: Make sure to use double quotes for windows paths.

# Details of the pipeline
Ideally, you should be able to run the pipeline by:
1. Set session info in `config/params.yaml`
2. Run `scripts/analyze_and_plot_FR_UA_NPRW.py`

As long as you have the inputs:
1. Raw neural signals (NPRW, UA)
2. Metadata from handwritten notes
3. Mapping and geometry if available
4. `config/params.yaml` indicating location of data and other params
5. Impedances for both NPRW and UA

## 1. Align kinematics from two cameras and VOG
`batch_scripts/align_dlc_two_cams_to_br.py`
`batch_scripts/align_VOG_to_br.py`

**Steps:**
1. Load OCR and DLC file for both cameras (or just VOG)
2. Find corresponding BR `.ns5` files to extract rising edges in `camera_sync_ch` (`VOG_sync_ch`) (corresponds to frames) Note: You can set these in `config/params.yaml`
3. Align both cameras to the BR frames
4. Output aligned .csv file

## 2. Intan / NPRW Neural data preprocessing
`batch_scripts/NPRW_Intan_analysis_threshold.py`

Preprocessing and spike sorting are handled in Python using [SpikeInterface](https://spikeinterface.readthedocs.io/)

**Steps:**
1. Load geometry and mapping (`.mat` file)
2. Extract stim data, ir crossings, and locations of stim pulses (individual pulses and blocks) - saves .npz file
3. Extract auxiliary data (sync pulses) - saves .npz file
4. Load Intan neural data, attach probe info, and reorder based on mapping
5. Preprocess Intan (`.rhs`) data (high-pass filter, common local median reference default radius: 30, 150 $\mu\text{m}$)
6. Remove artifacts (zero stim regions) with $\pm$ 20
7. Threshold and calculate MUA (peaks)
8. Saves `.npz` file per session

## 3. Compute time difference between BR and Intan recordings
`batch_scripts/compute_br_to_intan_shifts.py`

**Steps:**
1. Load metadata to match intan files to BR files
2. Load template sent from BR to Intan and Intan ADC file
3. Match the template to the BR template signal
4. Save the shifts and adjusted shifts and other information calculated from these two signals in `data_root/location/Metadata/br_to_intan_shifts.csv`

## 4. BR / UA Neural data preprocessing
`batch_scripts/UA_BR_analysis_threshold.py`

**Steps:**
1. Load geometry and mapping (`.xlsm` file)
2. Build `.npz` for auxiliary data (channels in .ns5 and .ns2 file, HR, touchscreen ... etc.)
3. Extract target labels from touchscreen signal
4. Load neural data (`.ns6`) and config (location to data, geometry, and mapping)
5. Preprocess UA data (high-pass filter)  
6. Use as sync template to align Intan with BR
7. Remove artifacts (zero stim regions) with $\pm$ 5
8. Threshold and calculate MUA (peaks)
9. Saves `.npz` file per session

## 5. Align data streams
`batch_scripts/make_aligned_npz_and_mat.py`

**Steps:**
1. Align Neural, aux, and behavioral data (NPRW, UA, kinematics, HR, target, and VOG)
2. Create peak dictionaries for both NPRW and UA
3. Permute UA channel labels and UA peak dictionary
2. Output `.npz` and `.mat` files in `~/results/checkpoints/Aligned`

## 6. Create peri-IR / peri-stim files
`batch_scripts/extract_peri_stim.py`

### There are three types of conditions that we need to separate. 1. Control reaches (No stim) 2. Stim condition reaches 3. At rest conditions (No reach) We separate these and process it separately

**Steps:**
1. Extract peri-event traces
2. Calculate velocity traces
3. Deduplicate MUA peaks, binning, and firing rate estimation using Gaussian Kernel
4. Reference to (substract mean of first 150 ms)
5. Calculate mean, variance, and median traces
6. PCA of the firing rate matrix (not used right now)
7. Separate left and right reaches
8. Output `.npz` and `.mat` files in `~results/checkpoints/PeriStim`
9. The process is repeated twice, once for control reaches (Use IR crossing as peri-event) and once for at rest conditions (No separation of left right reaches)

TODO: Concatenate control files by port and by depth
## 7. Make plots
`batch_scripts/plot_complete_shaded_BT.py`

**Steps:**
1. Plot aligned baseline traces
2. Plot aligned condition traces
3. Plot aligned at rest traces
4. These plots include (Median, variance, mean traces, median count traces, and chosen / best DLC coordinates)
4, Plot first 4 trials

## 8. LFP Band Analysis
`batch_scripts/analyze_lfp_bands.py`

Performs blanking, cleaning, and bandpass filtering of LFP data for both NPRW and Utah arrays, aligned to stimulation events.

**Steps:**
1.  **Alignment**: Loads stimulation events from Intan aux stream and aligns Utah data using calculated shifts.
2.  **Artifact Removal (Blanking)**: Applies "copy_baseline" blanking to stimulation artifacts (-5ms to +101ms).
3.  **Cleaning**:
    *   Reverse (Anti-Causal) Bandpass Filter (0.5 - 500 Hz).
    *   Resample to 1kHz.
    *   Baseline Correction (subtract pre-stim mean).
    *   Exponential Decay Removal (trial-by-trial fit & subtraction).
    *   Common Median Template Subtraction.
    *   Bad Channel Rejection (>700µV threshold).
4.  **Band Filtering**: Extracts power in Alpha (5-13Hz), Beta (13-30Hz), Low Gamma (30-60Hz), and High Gamma (60-120Hz).
5.  **Output**: Saves `.npz` files in `results/checkpoints/NPRW_LFP` and `results/checkpoints/UA_LFP`.

**Visualization:**
*   `scripts/plot_lfp_check.py`: Generates comprehensive plots (traces, heatmaps, PSDs) for all frequency bands.
*   `scripts/plot_lfp_artifact_comparison.py`: Validates cleaning by comparing Raw vs. Blanked vs. Cleaned signals.
*   `scripts/debug_lfp_pipeline_plots.py`: Generates step-by-step pipeline visualizations (Raw -> Blanked -> Filtered) for debugging.

## 9. Kinematics Quantification
`batch_scripts/quantify_stim_kinematics.py`

Analyzes reach kinematics (Duration, Peak Speed) comparing stimulation conditions to baseline.

**Features:**
- Speed-based reach endpoint detection.
- Statistical analysis (Welch's t-test vs Baseline) with significance clustering.
- Generates Violin/Box plots with transparent, jittered individual points.
- Parallel processing for fast dataset scanning.
- Output: Stats summary CSVs and Figures in `results/figures/quantify_kinematics`.

**Visualization:**
*   `scripts/plot_individual_kinematics.py`: Plots individual kinematic traces for every trial in a grid layout to allow visual inspection and outlier identification. Output: `results/figures/individual_trials`.

## Other plots
`RSA_poststim.ipynb`
- RSA plots
`plot_ns6_file_mean.ipynb`
- Walking analysis plots
`scripts/plot_HPF_traces_with_peaks.ipynb`
- Plot raw traces with peaks labeled
`scripts/PCA_analysis.ipynb`
- Visualize latent trajectories across conditions
`scripts/W_inference.ipynb`
- Extract low rank mapping from stim to latent responses from neural data (Using sessions 12-14 random stim trials)
---

## Intermediate outputs:
1. Stim stream npz files from Intan for stim timing and channels
2. Auxiliary npz files for Intan sync signals
3. Auxiliary npz files for BR sync signals
4. Metadata for alignment of BR and Intan
5. Checkpoint npz files for aligned DLC files  (Ex: Two camera to BR camera sync)
6. Preprocessed npy files in spikeinterface format (UA and NPRW)
7. Aligned kinematics, NPRW, and UA data

## Outputs
1. Plots of kinematics (position and velocity) and neural data (Firing rate plot)
2. RSA to evaluate consistency of stim response


See examples: [Nike's plots for reaching](https://docs.google.com/presentation/d/1L_EA5BOvmqIdInJ0WPRp5rtgeDXCOmu6qmt2yeqPC-s/edit?slide=id.g3afc5261748_0_102#slide=id.g3afc5261748_0_102)


# Headturn analysis in MATLAB
## May need to run batch_processing_neural (artifact correction) on the cluster

1. Change session name and input metadata info in each script (control, at rest, trials)
2. Run `~/RCP_analysis/MATLAB/batch_scripts/batch_analyze_behavior.m`
3. Run `~/RCP_analysis/MATLAB/batch_scripts/batch_processing_neural.m`
4. Run `~/RCP_analysis/MATLAB/batch_scripts/batch_analyze_with_neural.m`

See examples: [Bert's plots for headturn](https://docs.google.com/presentation/d/1z6fLBiO8Wbell_FSsJK0Mcj66stKMJZmZtJu6tcY7FA/edit?slide=id.g35fb40ee04d_0_42#slide=id.g35fb40ee04d_0_42)

## TODOs
- Switch to hdf5 / NWB format
- Refine alignment using the triangle sync pulse

---

## Evaluation

## Contact
For questions, contact Bryan Tseng btseng2@jh.edu.

## Acknowledgements
Data collected by Robyn Mildren, Nikita Lebedz, and Bryan Tseng.

## License
