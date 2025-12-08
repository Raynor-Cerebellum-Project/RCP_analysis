# Analysis Pipeline for the Raynor Project

## Overview
This repository contains the analysis pipeline for preprocessing, quantifying, and comparing stimulation trials from the Raynor experiments.  

![alt text](https://github.com/Raynor-Cerebellum-Project/RCP_analysis/blob/main/docs/pipeline_schematic.png "pipeline_schematic")

[Summary of recorded signals](https://docs.google.com/document/d/1C4-xSWL8n7P_mMrYqlUxQR6blz9WIHDU4B7O9bxhMnE/edit?tab=t.0)

TODO:
- Switch to hdf5 / NWB format
- Refine alignment using the triangle sync pulse
- Use Mixture of Gaussians to detect spikes instead of thresholding for UA
- Artifact correction using template subtraction of stimulation artifacts
- Spike sort using KS4 in Phy format (Including mean waveforms, spike times)
1. Concatenate sessions for sorting  
2. Run Kilosort4 (KS4)
3. Export results in Phy format
4. Optional now: SLAy
5. Separate by condition
6. FR estimation
7. Alignment
- Behavioral metrics:
1. Endpoint error, Absolute endpoint error, Variance (in velocity) after stim, Variance (in velocity) after endpoint, Max speed, Avg speed, Endpoint oscillation, FFTPower after endpoint
2. Calculate significance relative to baseline

[Example final figures: Bert's plots for headturn](https://docs.google.com/presentation/d/1z6fLBiO8Wbell_FSsJK0Mcj66stKMJZmZtJu6tcY7FA/edit?slide=id.g35fb40ee04d_0_42#slide=id.g35fb40ee04d_0_42)

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
├── data                         # Experimental data files
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

## 2. Neural data preprocessing (Intan / NPRW)
`batch_scripts/NPRW_Intan_analysis_threshold.py`

Preprocessing and spike sorting are handled in Python using [SpikeInterface](https://spikeinterface.readthedocs.io/)

**Steps:**
1. Load geometry and mapping (`.mat` file)
2. Extract stim data and locations of stim pulses (individual pulses and blocks) - saves .npz file
3. Extract auxiliary data (sync pulses) - saves .npz file
4. Load Intan neural data, attach probe info, and reorder based on mapping
5. Preprocess Intan (`.rhs`) data (high-pass filter, common local median reference default radius: 30, 150 $\mu\text{m}$)
6. Zero stim regions with $\pm$ 20
7. Threshold and calculate MUA (peaks), firing rate
8. PCA of the firing rate matrix (not used right now)
9. Saves `.npz` file per session

## 3. Compute time difference between BR and Intan recordings
`batch_scripts/compute_br_to_intan_shifts.py`

**Steps:**
1. Load metadata to match intan files to BR files
2. Load template sent from BR to Intan and Intan ADC file
3. Match the template to the BR template signal
4. Save the shifts and adjusted shifts calculated from these two signals

## 4. Neural data preprocessing (BR / UA)
`batch_scripts/UA_BR_analysis_threshold.py`

**Steps:**
1. Load geometry and mapping (`.xlsm` file)
2. Build `.npz` for auxiliary data (channels in .ns5 and .ns2 file)
3. Extract target labels
4. Load neural data (`.ns6`) and config (location to data, geometry, and mapping)
5. Preprocess UA data (high-pass filter)  
6. Use as sync template to align Intan with BR
7. Remove artifacts (zero stim regions) with $\pm$ 5
8. Saves `.npz` file per session
9. Threshold and calculate MUA (peaks), firing rate
10. Downsample target label timings to match FR bins

## 5. Behavioral Analysis and Stim timing Processing
`batch_scripts/make_aligned_npz_and_mat.py`

**Steps:**
1. Align NPRW, UA, and Behavior
2. Output `.npz` and `.mat` files in `~/results/checkpoints/Aligned`

## 6. Create aligned baseline
`batch_scripts/concat_and_extract_peri_IR_baseline.py`

**Steps:**
1. Concatenate baseline files by port and by depth
2. Extract peri IR-crossing traces
3. Export matrix in `~results/checkpoints/Behavior/baseline_concat`

## 7. Make baseline and peri-stim plots
`batch_scripts/NPRW_BEHV_UA_FR_plotting.py`

**Steps:**
1. Plot aligned baseline traces
2. Extract peri-stim traces TODO: Move this to its own code
3. Plot aligned condition traces

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

Use plot functions in `scripts` to plot raw traces with peaks labeled

See examples: [Bert's plots for headturn](https://docs.google.com/presentation/d/1z6fLBiO8Wbell_FSsJK0Mcj66stKMJZmZtJu6tcY7FA/edit?slide=id.g35fb40ee04d_0_42#slide=id.g35fb40ee04d_0_42)
/ [Nike's plots for reaching](https://docs.google.com/presentation/d/1IyVMR_M60ptEKUW1HeEVMPyRlBin64tVCVFx2P4NXo0/edit)

---

## Evaluation

## Contact
For questions, contact Bryan Tseng btseng2@jh.edu.

## Acknowledgements
Data collected by Robyn Mildren, Nikita Lebedz, and Bryan Tseng.
The artifact correction code for NPRW data was adapted from [this repository](https://github.com/RuihanQuan/Oculomotor_Pipeline) by Ruihan Quan.

## License
