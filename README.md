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

### 1. Neural Preprocessing
Neural spike sorting and preprocessing are handled in Python using [SpikeInterface](https://spikeinterface.readthedocs.io/)

**Steps:**
Processing Intan data (NPRW)
1. Load neural data and config (location to data, geometry and mapping)
2. Load geometry and mapping (.mat file)
3. Preprocess Intan (.rhs) data (high-pass filter, common local median reference default radius: 30, 150)  
4. Build bundles for data other then neural data (save as .npz file) (2 sync channels) (`intan_preproc.build_intan_bundle`) - TODO: Convert to NWB format
5. **TODO** Extract stim data (save as .npz file)
6. Save per-session preprocessed data
5. Concatenate sessions for sorting  
6. Run Kilosort4 with Intan geometry and mapping
7. Export results in Phy format
8. **TODO** SLAy
9. **TODO** Separate by condition
10. **TODO** FR estimation?
11. **TODO** Align with BR using two sync pulses (one from BR side)

### 2. **TODO** Processing BR data (UA)
1. Load neural data (.ns6) and config (location to data, geometry and mapping)
2. Load geometry and mapping (.xlsm file)
3. Preprocess Intan (.rhs) data (high-pass filter, common global median reference)  
4. Build bundles for data other then neural data (save as .npz file) (2 sync digital channels in .ns5 and other .ns2 files) (`br_preproc.build_blackrock_bundle`) - TODO: Convert to NWB format
5. Save per-session preprocessed data
6. Concatenate sessions for sorting
7. Run Mountainsort5 with Utah-array mapping (per-channel sorting)
8. Export results in Phy format
9. SLAy
10. Separate by condition
11. FR estimation?
12. Use as template to align Intan with BR using .ns5 sync pulses (two from BR side)
13. Use as template to align DLC kinematics file using the .ns5 sync pulses


### 3. Behavioral Analysis and Stim Data Processing
TODO: Need to convert from MATLAB
TODO: Need to create for DLC processed data

ALL TODO:
1. Identify movement segments from DLC data
2. Extract stimulation timing from Intan
3. Plot movement traces (how to align? based on velocity or based on stim timing?, try to plot baseline aligned by velocity first)
4. Calculate movement metrics (Previously: Endpoint error, Absolute endpoint error, Variance (in velocity) after stim, Variance (in velocity) after endpoint, Max speed, Avg speed, Endpoint oscillation, FFTPower after endpoint)
5. Calculate significance relative to baseline
6. Plot bargraphs
See examples: https://docs.google.com/presentation/d/1z6fLBiO8Wbell_FSsJK0Mcj66stKMJZmZtJu6tcY7FA/edit?slide=id.g35fb40ee04d_0_42#slide=id.g35fb40ee04d_0_42

---

## Output
- Intermediate files during pipeline:
1. Raw neural signals
2. Preprocessed neural signals (UA and Intan) (After HPF and referencing)
3. Spike sorted data for MS5 and KS4 in Phy format (Including mean waveforms, spike times)
4. Curated and sorted data if exists
5. FR data after alignment with DLC and Intan

- Final output
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
