# Analysis Codes: Behavioral and Neural Pipeline for the Raynor Project

## Overview
This repository contains the analysis pipeline for processing, quantifying, and comparing stimulation trials from the Raynor experiments.  

The workflow supports:
- Behavioral trial segmentation with and without stimulation  
- Calculation of behavioral metrics (endpoint error, velocity, variability, etc.) - TODO: Convert from MATLAB
- Visualization of stimulation on behavior
- Neural data preprocessing (artifact correction, template subtraction) - TODO: Convert from MATLAB

---

## Pipeline Overview

---

### 1. **TODO** Neural data processing (Intan / Neuropixel Read/Write (NPRW))
Neural spike sorting and preprocessing are handled in Python using [SpikeInterface](https://spikeinterface.readthedocs.io/)

**Steps:**
1. Load neural data and config (location to data, geometry and mapping)
2. Load geometry and mapping (.mat file)
3. Preprocess Intan (.rhs) data (high-pass filter, common local median reference default radius: 30, 150)  
4. Build bundles for data other then neural data (save as .npz file) (2 sync channels) (`intan_preproc.build_intan_bundle`) - TODO: Convert to NWB format
5. **TODO** Extract stim data (save as .npz file)
6. Save per-session preprocessed data
7. **TODO** Artifact correction via PCA fitting
8. Concatenate sessions for sorting  
9. Run Kilosort4 (KS4) with Intan geometry and mapping
10. Export results in Phy format
11. **TODO** Optional now: SLAy
12. **TODO** Separate by condition
13. **TODO** Firing rate (FR) estimation using Gaussian filter?
14. **TODO** Align with BR using two sync pulses (one from BR side)

### 2. Neural data processing (Blackrock (BR) / Utah Array (UA))
**Steps:**
1. Load neural data (.ns6) and config (location to data, geometry and mapping)
2. Load geometry and mapping (.xlsm file)
3. Preprocess UA data (high-pass filter, common global median reference)  
4. Build bundles for data other then neural data (save as .npz file) (2 sync digital channels in .ns5 and other .ns2 files) (`br_preproc.build_blackrock_bundle`) - TODO: Convert to NWB format
5. Save per-session preprocessed data
6. Concatenate sessions for sorting
7. Run Mountainsort5 (MS5) with UA mapping (per-channel sorting)
8. Export results in Phy format
9. Optional now: SLAy
10. Separate by condition
11. FR estimation?
12. Use as template to align Intan with BR using .ns5 sync pulses (two from BR side)
13. Use as template to align DLC kinematics file using the .ns5 sync pulses

### 3. Behavioral Analysis and Stim timing Processing
TODO: Need to convert from MATLAB
TODO: Need to create for DLC processed data

TODO:
1. DLC labeling and annotations
2. Identify movement segments from DLC data
3. Load stimulation timing from Intan
4. Plot movement traces (how to align? based on velocity or based on stim timing?, try to plot baseline aligned by velocity first)
5. Calculate movement metrics (Previously: Endpoint error, Absolute endpoint error, Variance (in velocity) after stim, Variance (in velocity) after endpoint, Max speed, Avg speed, Endpoint oscillation, FFTPower after endpoint)
6. Calculate significance relative to baseline
7. Plot bargraphs
See examples: https://docs.google.com/presentation/d/1z6fLBiO8Wbell_FSsJK0Mcj66stKMJZmZtJu6tcY7FA/edit?slide=id.g35fb40ee04d_0_42#slide=id.g35fb40ee04d_0_42

---

## Output
Intermediate files during pipeline:
1. Raw neural signals
2. Stim data for stim timing and channels
3. Preprocessed neural signals (UA and Intan) (After HPF and referencing)
4. Spike sorted data for MS5 and KS4 in Phy format (Including mean waveforms, spike times)
5. Curated and sorted data if exists
6. FR data after alignment with DLC and Intan

Final output
1. Metadata and other channel data in NWB format
2. Plots of velocity, position, and neural data (dim-reduced)

---

## Requirements
- Conda environment defined in `environment.yml` - NEED TO CUT DOWN

## Evaluation

## Contact
For questions, contact Bryan Tseng btseng2@jh.edu.

## Acknowledgements
Data collected by Robyn Mildren and Bryan Tseng.
The template subtraction code for neural data (under ./Oculomotor_Pipeline-main/) was adapted from [this repository](https://github.com/RuihanQuan/Oculomotor_Pipeline) by Ruihan Quan.

## Licence
