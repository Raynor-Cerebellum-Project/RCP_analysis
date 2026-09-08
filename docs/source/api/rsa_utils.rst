rsa_utils.py
============

``RCP_analysis/python/functions/rsa_utils.py``

The Representational Similarity Analysis pipeline. It reads peri-stim NPZs, cuts
one scalar feature per channel per trial out of a post-stim window, optionally
restricts the channel set by a functional criterion, correlates every trial
against every other, and scores how separable the conditions are.

Only :func:`run_rsa` is re-exported at the package top level; everything else is
a private stage of that pipeline, documented here because the stages are what
you tune.

Pipeline::

   NPZ files
     -> _trial_features_from_peristim_npz   (time window -> per-trial features)
     -> _collect_condition_blocks           (one Block per file, with filtering)
     -> _compute_movement_mask / _compute_stim_mask   (channel selection)
     -> global z-score across all trials
     -> _build_rsm_pairwise                 (trial x trial correlations)
     -> _pairwise_silhouette + _within_condition_consistency
     -> figures under results/figures/rsa_from_peristim/<target>/

.. py:currentmodule:: RCP_analysis.python.functions.rsa_utils

Module constants
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Name
     - Value
     - Meaning
   * - ``Z_SCORE_FEATURES``
     - True
     - Z-score channels across all trials before correlating.
   * - ``SAVE_SVG``
     - True
     - Write figures to ``FIG_DIR``.
   * - ``MIN_TRIALS_PER_COND``
     - 4
     - Drop non-baseline conditions with fewer trials.
   * - ``MIN_VALID_FEATURES``
     - 2
     - A trial needs at least this many finite channels.
   * - ``FIG_DIR``
     - —
     - ``OUT_BASE/figures/rsa_from_peristim`` (with ``target_A`` / ``target_B``
       subfolders), created on import.

The module also configures a ``logging`` logger named ``rsa`` at ``INFO``. Set
it to ``DEBUG`` for per-file detail.

Data container
--------------

.. py:class:: Block

   One condition's worth of trials, carried through every stage.

   :Fields:
      * ``X`` ``(n_trials, n_features)`` — raw per-channel features.
      * ``cond`` — the file's ``br_idx``, set for baseline files too so
        ``skip_conds`` can target a specific baseline by BR number.
      * ``target`` — ``"target_A"`` / ``"target_B"`` / ``"at_rest"``.
      * ``is_baseline`` (bool) — file lives under ``control_reaches/``; used for
        labeling and the relaxed trial-count floor.
      * ``path``, ``labels``.
      * ``stim_mask``, ``move_mask`` — channel masks; ``move_mask`` is the
        effective mask used when building the RSM, whatever criterion set it.
      * ``X_z`` — features after the global z-score.
      * ``n_trials`` (property).

Entry points
------------

.. py:function:: run_rsa(source="NPRW", target="target_A", poststim_win_ms=(0.0, 50.0), channel_criterion=None, cond_label_extras=None, vmin=-1.0, vmax=1.0, skip_conds=None, cond_order=None, move_alpha=0.05, stim_alpha=0.05, sil_vmax=None, sil_n_perm=1000, sil_tail="greater", sil_seed=0, debug_masks=False)

   Build a trial × trial RSM for one reach target from the peri-stim NPZs, plus
   the pairwise silhouette matrix, and save both figures.

   :param str source: ``"NPRW"`` or ``"UA"`` — selects the
      ``*_rates_zeroed`` / ``*_rel_t`` arrays read from each NPZ.
   :param str target: ``"target_A"`` / ``"target_B"``; picks the subfolder under
      ``stim_reaches/`` and ``control_reaches/``. Call once per target. NPZs
      under ``PERI_ROOT/at_rest`` have no target subfolder and are always
      included, labeled via ``cond_label_extras["at_rest"]``.
   :param tuple poststim_win_ms: window relative to stim **offset**, not onset —
      see the note below.
   :param channel_criterion: ``None``, one of ``"movement"``, ``"stim"``,
      ``"union"``, ``"intersection"``, or an iterable of those. An iterable
      loads, feature-extracts and z-scores the NPZs once, then builds one RSM
      and figure per criterion; the movement and stim masks are likewise
      computed once and reused.
   :param dict cond_label_extras: overrides for block labels, keyed by ``cond``,
      plus the special keys ``"Baseline"`` and ``"at_rest"``.
   :param vmin, vmax: colour limits for the RSM.
   :param skip_conds: BR numbers to exclude — applies to baseline files too.
   :param cond_order: explicit condition ordering; unlisted blocks sort last.
      Use ``"Baseline"`` for the baseline entry.
   :param move_alpha, stim_alpha: FDR-adjusted significance thresholds for the
      two channel tests.
   :param sil_vmax: colour limit for the silhouette heatmap; default is the
      largest finite ``|S|``.
   :param int sil_n_perm: label shuffles per pair for the permutation test;
      0 disables it.
   :param str sil_tail: ``"greater"``, ``"less"`` or two-sided.
   :param sil_seed: seed for the shuffle RNG.
   :param bool debug_masks: also render the mask diagnostic heatmaps.

   .. note::

      ``poststim_win_ms`` is relative to stim **offset**: the window actually
      used is ``poststim_win_ms + stim_dur``, where ``stim_dur`` is read from
      each file's own metadata. So ``(0.0, 50.0)`` against a 100 ms stim pulls
      ``rel_t`` in ``[100, 150]`` ms, and files with different stim durations
      get different absolute windows for the same argument. Files with no
      stim-duration metadata (baseline, at_rest) fall back to ``stim_dur = 0``,
      making the window absolute.

   Channel criteria:

   .. list-table::
      :header-rows: 1
      :widths: 22 78

      * - Criterion
        - Channels kept
      * - ``None``
        - All channels.
      * - ``"movement"``
        - Response window differs from that trial's own pre-movement baseline
          (paired Wilcoxon signed-rank, BH-corrected), OR'd across baseline files.
      * - ``"stim"``
        - Differs from the pooled control trials (Mann-Whitney U, BH-corrected),
          unioned across all conditions so every condition uses one fixed set.
      * - ``"union"`` / ``"intersection"``
        - The OR / AND of the two masks above.

.. py:function:: run_time_domain_corr(source="NPRW", target="target_A", poststim_win_ms=(-800, 600.0), channel_criterion=None, cond_label_extras=None, vmin=-1.0, vmax=1.0, skip_conds=None, cond_order=None, move_alpha=0.05, stim_alpha=0.05, debug_masks=False)

   .. warning::

      **Work in progress.** It runs the same loading, masking and z-scoring
      stages as :func:`run_rsa` but stops before building an RSM and returns
      ``1``. The intended output — a sliding-window (80 ms wide, 20 ms step)
      correlation of each condition against the control condition, plotted as
      condition × time bin × channels — is sketched in comments at the end of
      the criterion loop.

Feature extraction
------------------

.. py:function:: _trial_features_from_peristim_npz(npz_path, source="NPRW", poststim_win_ms=(0.0, 50.0))

   Build per-trial feature vectors from one peri-stim NPZ by averaging
   ``<source>_rates_zeroed`` over the post-stim window, giving one scalar per
   channel per trial. Trials with fewer than ``MIN_VALID_FEATURES`` finite
   channels are dropped.

   :returns: ``(X, labels, info)`` — ``X`` ``(n_trials, n_channels)``, one
      ``"cond<N>_<target>_t<i>"`` label per surviving trial, and an ``info``
      dict with ``cond``, ``target``, ``is_baseline``, ``n_trials``, ``n_ch``
      and ``file``.

   Returns an empty result — never raises — when the time axis or rate array is
   missing, the window contains no samples, or the arrays are empty or
   mismatched, so a corrupt file drops out of the analysis instead of stopping it.

.. py:function:: _collect_condition_blocks(npz_files, source, poststim_win_ms, skip_conds, min_trials)

   One :class:`Block` per NPZ, applying ``skip_conds`` by BR number and the
   ``min_trials`` floor (baseline blocks only need ≥1 trial).

   :returns: ``(blocks, skip_summary)``; the summary buckets dropped files into
      ``in_skip_conds``, ``too_few_trials`` and ``missing_or_empty`` for the
      one-line skip report.

.. py:function:: _load_rates(path, source)

   ``(rates (n_trials, n_ch, T), rel_t (T,))``, or ``(None, None)`` if the keys
   are missing or the time axis does not match.

.. py:function:: _get_region_arrays(npz_path, source)

   ``(ua_region_names, ua_region)`` for debug-plot labeling. Always
   ``(None, None)`` for NPRW — region grouping only exists for UA channels.

.. py:function:: _stim_dur_ms(peristim_npz)

   Median ``stim_dur`` from ``nprw_meta``, or ``0.0`` when absent.

Channel masks
-------------

.. py:function:: _compute_movement_mask(baseline_paths, source, base_win=(-800.0, -500.0), resp_win=(0.0, 300.0), alpha=0.05)

   Flag channels whose movement-response window differs from their own
   pre-movement baseline window. Paired Wilcoxon signed-rank per channel (the
   two windows come from the same trials), FDR-corrected per baseline file, then
   OR'd across files. Returns ``None`` if no baseline file was usable.

.. py:function:: _movement_pvalues(rates, mb, mr)

   The per-channel signed-rank test behind the mask. NaN where the test cannot
   run — fewer than 2 valid trials, or all-zero differences.

.. py:function:: _build_ctrl_pool(baseline_paths, source, stim_win=(105.0, 150.0))

   Per-trial mean rate in ``stim_win`` for every control trial, pooled across
   baseline files, as ``(n_ch, n_ctrl_trials)``. Columns are trials, so the
   rank-sum test treats trials — not trial × timepoint samples — as the
   independent unit.

.. py:function:: _compute_stim_mask(stim_path, ctrl_pool, source, stim_win=(105.0, 150.0), alpha=0.05)

   Flag channels in one stim file whose per-trial mean rate in ``stim_win``
   differs from the pooled control trials. Unpaired two-sided Mann-Whitney U per
   channel (control and stim are different trials), FDR-corrected across
   channels. Returns ``None`` on a channel-count mismatch.

.. py:function:: _fdr_pass_mask(pvals, alpha)

   Benjamini-Hochberg correction across channels. NaN p-values — a channel that
   could not be tested — never pass.

RSM and statistics
------------------

.. py:function:: _build_rsm_pairwise(blocks)

   Full trial × trial Pearson similarity matrix across blocks.

   Each block pair is correlated over the **union** of their two channel masks,
   falling back to all channels if that union is empty, so within-block and
   cross-block entries for a given pair always use a consistent channel set.
   This is well-defined because a Pearson correlation between two trials depends
   only on those two trials' features.

   :returns: ``(RSM, sizes)`` — ``(N, N)`` with NaN where a trial had
      non-finite features, and the per-block trial counts.

.. py:function:: _rsm_to_distance(RSM, kind="angular")

   Similarity → distance for the silhouette. ``"angular"`` gives
   ``sqrt(2(1 - r))``, ``"corr"`` gives ``1 - r`` clipped to ``[0, 2]``. The
   matrix is symmetrized and the diagonal zeroed.

.. py:function:: _two_group_silhouette(D, in_group_b)

   Mean silhouette for a 2-cluster split of a precomputed distance matrix.
   Equivalent to ``silhouette_score(D, labels, metric="precomputed")`` for
   K = 2 but roughly 100× faster, which matters when it runs ``n_perm`` times
   per pair. NaN if either group has fewer than 2 members.

.. py:function:: _pairwise_silhouette(RSM, block_sizes, kind="angular", n_perm=0, tail="greater", rng=None)

   K × K silhouette computed two conditions at a time, with a NaN diagonal.

   With ``n_perm > 0`` each pair also gets a null built by shuffling the two
   conditions' trial labels (group sizes preserved) and recomputing the
   silhouette on the same distance matrix.

   :returns: ``(S, P)`` — silhouettes and uncorrected empirical p-values; NaN in
      ``P`` means the pair was not tested.

   Warns when ``1/(n_perm + 1) × n_pairs > 0.05``, i.e. when the p-value floor
   makes it impossible for any pair to reach ``q < 0.05`` after correction.

.. py:function:: _null_pvalue(obs, null, tail)

   Empirical p-value with the ``+1`` correction: the fraction of the shuffle
   null at least as extreme as ``obs``. Resolution is floored at
   ``1/(n_perm + 1)``.

.. py:function:: _fdr_adjust_pairs(P)

   BH-adjust the ``K(K-1)/2`` unique pair p-values and mirror them back into a
   K × K matrix.

.. py:function:: _p_stars(q)

   ``"*"`` when ``q < 0.05``, else ``""``. Deliberately a single tier: with a
   ``1/(n_perm + 1)`` p-value floor, BH ties would make a pair's tier track *how
   many other pairs* are significant rather than its own separation.

.. py:function:: _within_condition_consistency(RSM, block_sizes)

   Per-condition mean within-block trial-to-trial correlation.

   :returns: ``(mean_r [K], n_pairs [K])``; NaN where a block has fewer than 2
      usable trials.

Plotting
--------

.. py:function:: _plot_rdm_with_block_ticks(RSM, block_sizes, block_labels, title, out_svg, ax=None, vmin=-1.0, vmax=1.0)

   Plot the similarity matrix with small unlabeled ticks at block boundaries and
   text labels centered on each block, mirrored on both axes. With ``ax=None``
   it owns the figure and returns the saved path relative to ``OUT_BASE``;
   given an ``ax`` it draws into it and leaves saving to the caller.

.. py:function:: _plot_pairwise_silhouette(S, block_labels, title, out_svg, vmax=None, annotate=True, diag_values=None, ax=None, qvals=None, footnote=None)

   Heatmap of the K × K silhouette matrix. The colour scale is symmetric about
   0 so "no separation" sits at the midpoint of the diverging map; undefined
   pairs and the diagonal render in a distinct grey.

   ``diag_values`` (typically the within-condition mean correlation) is printed
   on the uncoloured diagonal — a *different* quantity from the off-diagonal
   silhouettes, deliberately left off the colour scale, which is what the
   ``footnote`` is for. ``qvals`` adds a significance star to each annotated
   cell.

.. py:function:: _draw_channel_heatmap(ax, traces, rel_t, vext, title, region_arr=None, region_names=None, ch_subset=None, vlines=(), vspans=())

   Shared channel × time imshow used by both debug plots: draws the heatmap,
   any vertical lines and shaded spans, and — for UA data — horizontal rules at
   region boundaries with the region names as y tick labels.

.. py:function:: _plot_movement_mask_debug(baseline_paths, source, target, base_win=(-800.0, -500.0), resp_win=(0.0, 300.0), alpha=0.05)

   One figure per baseline file: median-trace heatmaps for all, failing and
   passing channels, with the baseline and response windows marked. Saved as
   ``debug_<source>_movement_criterion_*.png`` under ``FIG_DIR/<target>``.

.. py:function:: _plot_stim_mask_debug(stim_path, passes, source, target, stim_win=(105.0, 150.0), alpha=0.05, baseline_rates=None, baseline_rel_t=None)

   One figure for a single stim file: baseline (all channels), stim (all
   channels), failing, and passing, with the stim window marked and the artifact
   blanking span shaded. Saved as ``debug_<source>_stim_criterion_*.png``.
