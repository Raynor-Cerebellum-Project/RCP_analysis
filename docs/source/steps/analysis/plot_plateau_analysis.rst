Plateau Analysis
================

Script: ``analysis_scripts/plot_plateau_analysis.py``

Batch processing script for reach endpoint / plateau detection analysis.

Steps
-----

#. Load all peri-reach condition files for a session.
#. Smooth X/Y traces using Butterworth low-pass filtering.
#. Compute distance from start.
#. Detect plateau onset per trial using configurable multi-signal stability.
#. Generate per-condition summary plots.
#. Compare plateau onset times across conditions by target.
#. Run selected-condition statistical analysis.
