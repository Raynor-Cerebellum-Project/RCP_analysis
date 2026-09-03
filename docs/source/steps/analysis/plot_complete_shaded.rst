Peri-Stim Firing Rate Plots
============================

Script: ``analysis_scripts/plot_complete_shaded_BT.py``

Plots firing rate traces based on Gaussian smoothing parameters.

Steps
-----

#. Plot aligned baseline traces.
#. Plot aligned condition traces.
#. Plot aligned at-rest traces.

   These plots include median, variance, mean traces, median count traces, and chosen/best DLC coordinates.

#. Plot first 4 trials.

.. note::
   Gaussian smoother parameters can be set in ``config/params.yaml``.
