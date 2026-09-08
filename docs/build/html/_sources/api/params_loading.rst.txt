params_loading.py
=================

``RCP_analysis/python/functions/params_loading.py``

Reads ``config/params.yaml`` and ``config/machines.yaml`` and turns them into a
single :class:`experimentParams` dataclass. This is the only module that knows
how the YAML is laid out; everything else consumes the dataclass.

Resolution order for the session to process:

#. ``RCP_LOCATION`` / ``RCP_SESSION`` environment variables (must be set together)
#. the first row of ``data_status_reaching.csv`` (or ``data_status_fastigial.csv``
   when ``paths.process_fastigial`` is true) whose ``Process Session?`` column is
   ``Yes``

.. py:currentmodule:: RCP_analysis.python.functions.params_loading

Dataclass
---------

.. py:class:: experimentParams

   Container for everything read out of ``params.yaml``.

   :Required fields:
      * ``data_root`` (str) — absolute data root, machine prefix already applied.
      * ``monkey`` (str) — ``"Nike"``, ``"Ada"`` or ``"Bert"``, taken from the
        last folder of ``data_root``.
      * ``location`` (str | None), ``session`` (str | None) — selected session.
      * ``geom_mat_rel`` (str | None) — repo-relative probe geometry ``.mat``.

   :Processing fields:
      * ``highpass_hz`` (default 300.0), ``lowpass_hz`` (default 10000.0)
      * ``probes``, ``sessions`` — per-probe / per-session config dicts.
      * ``parallel_jobs`` (8), ``threads_per_worker`` (1), ``chunk`` (``"1s"``)
      * ``preprocessing`` — includes ``process_only``, always coerced to
        ``list[int]``.
      * ``NPRW_rate_est``, ``UA_rate_est`` — rate-estimation settings.
      * ``Subspace_Params``, ``IPCA_Params`` — detector / artifact settings.
      * ``kinematics`` — includes ``num_camera`` and a ``keypoints`` tuple.
      * ``rsa_params`` — ROI windows and other RSA settings.

Functions
---------

.. py:function:: load_experiment_params(yaml_path, repo_root, machines_yaml_path=None, first_run=False)

   Load ``params.yaml`` and build an :class:`experimentParams`.

   ``{REPO_ROOT}`` placeholders anywhere in the YAML (strings, lists, nested
   dicts) are expanded to ``repo_root`` before parsing. The machine-specific
   ``data_root_prefix`` is looked up in ``machines.yaml`` by hostname and
   prepended to the lab-relative ``paths.data_root``.

   :param Path yaml_path: path to ``config/params.yaml``.
   :param Path repo_root: repository root, used for placeholder expansion.
   :param machines_yaml_path: defaults to ``repo_root/config/machines.yaml``.
   :param bool first_run: if True, skip session selection and leave
      ``location``/``session`` empty (used when bootstrapping a new machine).
   :returns: :class:`experimentParams`

   ``process_only`` priority: ``RCP_PROCESS_ONLY`` (JSON, e.g. ``'[1, 2, 5]'``)
   overrides ``preprocessing.process_only`` in the YAML.

.. py:function:: resolve_probe_geom_path(params, repo_root, session_key)

   Resolve the geometry / channel-mapping ``.mat`` path.

   Priority: the session's probe entry (``mapping_mat_rel``, else
   ``geom_mat_rel``), then the global ``params.geom_mat_rel``.

   :param params: an :class:`experimentParams`.
   :param Path repo_root: repository root the relative path is joined onto.
   :param session_key: key into ``params.sessions``, or None to skip the
      session-specific lookup.
   :returns: resolved absolute :class:`~pathlib.Path`.

Private helpers
---------------

.. py:function:: _resolve_data_root(machines_yaml_path, relative_data_root)

   Look up this machine's ``data_root_prefix`` in ``machines.yaml`` by
   ``socket.gethostname()`` and join it to the lab-relative data root. Raises
   ``FileNotFoundError`` if ``machines.yaml`` is missing — add an entry for the
   hostname when setting up a new machine.

.. py:function:: _get_monkey_from_data_root(data_root)

   Take the final folder of ``data_root`` as the monkey name. Raises
   ``ValueError`` unless it is ``Nike``, ``Ada`` or ``Bert``.

.. py:function:: _get_location_session_from_status_csv(data_root, process_fastigial)

   Return ``(location, session)`` from the first row of the data-status CSV whose
   ``Process Session?`` is ``Yes``. Returns ``("N/A", "N/A")`` and prints a
   message if no row is marked. Raises ``KeyError`` if the CSV is missing the
   ``Process Session?`` / ``Location`` / ``Session`` columns.
