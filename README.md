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

### 1. Behavioral Analysis
TODO: Need to convert from MATLAB
TODO: Need to create for DLC processed data

---

### 2. Neural Preprocessing (Python)
Neural spike sorting and preprocessing are handled in Python using [SpikeInterface](https://spikeinterface.readthedocs.io/)

**Steps:**
1. Load Neural data and config
2. Preprocess NS6 data (high-pass filter, common global median reference for Utah, common local median reference for Intan)  
3. Build bundles for data other then neural data (`br_preproc.build_blackrock_bundle`) - TODO: Convert to NWB format
4. Save per-session preprocessed data  
5. Concatenate sessions for sorting  
6. Run Mountainsort5 with Utah-array mapping (per-channel sorting)
7. Run Kilosort4 with Intan geometry and mapping
7. Export results in Phy format
8. **TODO Later:** SLAy
9. Separate by condition?
10. FR estimation

---

## Output
- Plots of velocity, position, and neural data (dim-reduced)
- Intermediary: Preprocessed NS6 data, spike sorting results (MS5 and KS4) in Phy-ready folders  

---

## Requirements
- Conda environment defined in `environment.yml` - NEED TO CUT DOWN

## Evaluation

## Contact
For questions, contact Bryan Tseng btseng2@jh.edu.

## Acknowledgements
Data is collected by Robyn Mildren.
The template subtraction code for neural data (under ./Oculomotor_Pipeline-main/) was adapted from [this repository](https://github.com/RuihanQuan/Oculomotor_Pipeline) by Ruihan Quan.

## Licence
