# Analysis Codes: Behavioral and Neural pipline for the Raynor Project

## Overview
This repository contains the analysis pipeline for processing, quantifying, and comparing stimulation trials from the Raynor experiments. The workflow is designed to handle segmented movement trials with and without stimulation, calculate behavioral metrics, and visualize condition-specific effects. Neural data analysis will be added, currently artifact correction is already added.

## Folder Structure
- `functions/` — Modular helper functions used throughout the pipeline.
- `System_ID/` — Identification-related scripts. (PSID from Sani et al. 2021)
- `Oculomotor_Pipeline-main/` — Artifact correction helper functions.
- `README.md`

Outputs are in the `Data/Figures` folders.

## Pipeline Overview

### 1. **Start with `batch_analyze.m`**

This script performs the main behavioral metric extraction.

#### **Important Variables:**
- `session`: Name of the session to analyze (e.g., `'BL_RW_003_Session_1'`)
- `baseline_file_nums`: Index of files considered baseline
- `trial_indices`: Index of stimulation trials

#### **Steps:**
- Loads trial metadata and calibrated segment files.
- Calls `analyze_metrics.m` for each trial:
  - Extracts and aligns position/velocity traces.
  - Detects movement segments (e.g., `active_like_stim_*`).
  - Computes raw metrics: speed, error, endpoint position, etc.
- Combines all baseline trials into a merged summary.
- Optionally merges random trials with overall baseline metrics.
- Saves intermediate results as:
  - `[session]_raw_metrics_all.mat`
  - `[session]_merged_baseline.mat`
  - `[session]_summarized_metrics.mat`
- Saves figures:
  - `.fig`, `.png`, and `.svg` for all comparisons and individual traces into organized subfolders under `Figures/`

## Output figures
- `Figures/`: Contains visualization results, split into:
  - `vsBaselineTraces/` — baseline vs condition comparisons
  - `ComparisonTraces/` — intra-condition comparisons (e.g., across delays)

## Notes

- All paths are dynamically handled via `set_paths.m` to support running on different machines (Mac, workstation, or manual selection).
- `segment_fields_random` contains all condition labels used when `Stim_Delay == 'Random'`.
- Code assumes all data has been preprocessed and saved in the `Calibrated/` directory of the session.

### 2. **Follow up with `batch_correction.m`**

This script performans artifact correction for the Neural data.

#### **Steps:**
- Loads the summarized metrics.
- For each stimulation condition:
  - Compares velocity, endpoint, and trajectory metrics across:
    - Baseline (merged or individual)
    - Condition (stimulated)
  - Visualizes each comparison using `plot_traces.m` and `plot_rand_condition_traces.m`


## Requirements

## Execution

## Evaluation

## Contact
For questions, please contact Bryan Tseng btseng2@jh.edu.

## Acknowledgements
Data is collected by Robyn Mildren.
The template subtraction code for neural data (under ./Oculomotor_Pipeline-main/) wass adapted from [this repository](https://github.com/RuihanQuan/Oculomotor_Pipeline) by Ruihan Quan.

## Licence