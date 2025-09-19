# params
from .python.functions.params_loading import (
    load_experiment_params,
    resolve_data_root,
    resolve_output_root,
    resolve_session_intan_dir,
    resolve_probe_geom_path,
)

# blackrock / UA
from .python.functions.br_preproc import (
    list_br_sessions,
    ua_excel_path,
    load_ns6_spikes,
    load_ns5_aux,
    load_ns2_digi,
    load_UA_mapping_from_excel,
    apply_ua_mapping_properties,
    build_blackrock_bundle,
    save_UA_bundle_npz,
    threshold_mua_rates,
)

# intan
from .python.functions.intan_preproc import (
    read_intan_recording,
    load_stim_geometry,
    make_probe_from_geom,
    local_cm_reference,
    save_recording,
    list_intan_sessions,
    extract_and_save_stim_npz,
    extract_and_save_other_streams_npz,
    get_chanmap_perm_from_geom,
    make_identity_probe_from_geom,
    reorder_recording_to_geometry,
    load_stim_triggers_from_npz,
    StimTriggerConfig,
    StimTriggerResult,
    extract_stim_triggers_and_blocks,
)

# artifact correction (PCA template subtraction) — requires list_triggers now
from .python.functions.artifact_correction_template_matching import (
    remove_stim_pca_offline, cleaned_numpy_to_recording,
    PCAArtifactParams
)

# plotting
from .python.plotting.plotting import plot_all_quads_for_session


__all__ = [
    # params
    "load_experiment_params", "resolve_data_root", "resolve_output_root",
    "resolve_session_intan_dir", "resolve_probe_geom_path",

    # BR/UA
    "list_br_sessions", "ua_excel_path",
    "load_ns6_spikes", "load_ns5_aux", "load_ns2_digi",
    "load_UA_mapping_from_excel", "apply_ua_mapping_properties",
    "build_blackrock_bundle", "save_UA_bundle_npz",
    "threshold_mua_rates",

    # Intan
    "read_intan_recording", "load_stim_geometry", "make_probe_from_geom",
    "local_cm_reference", "save_recording", "list_intan_sessions",
    "extract_and_save_stim_npz", "extract_and_save_other_streams_npz",
    "get_chanmap_perm_from_geom", "make_identity_probe_from_geom",
    "reorder_recording_to_geometry", "load_stim_triggers_from_npz",

    # plotting
    "plot_all_quads_for_session",

    # artifact correction
    "ArtifactParams", "remove_stim_pca",

    # stim preprocessing
    "StimTriggerConfig", "StimTriggerResult", "extract_stim_triggers_and_blocks",
]
