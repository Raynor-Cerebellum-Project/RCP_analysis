# params
from .python.functions.params_loading import (
    load_experiment_params,
    resolve_data_root,
    resolve_output_root,
    resolve_session_intan_dir,
    resolve_probe_geom_path,
    resolve_intan_root,
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
    load_stim_geometry,
    make_probe_from_geom,
    local_cm_reference,
    extract_and_save_stim_npz,
    extract_and_save_other_streams_npz,
    get_chanmap_perm_from_geom,
    make_identity_probe_from_geom,
    reorder_recording_to_geometry,
    StimTriggerConfig,
    StimTriggerResult,
    extract_stim_triggers_and_blocks,
)

# artifact correction (PCA template subtraction)
from .python.functions.artifact_correction_template_matching import (
    remove_stim_pca_offline,
    cleaned_numpy_to_recording,
    PCAArtifactParams,
)

# utils
from .python.functions.utils import (
    load_rate_npz,
    median_across_trials,
    extract_peristim_segments,
    detect_stim_channels_from_npz,
    build_probe_and_locs_from_geom,
    baseline_zero_each_trial,
    load_combined_npz, 
    aligned_stim_ms, 
    ua_title_from_meta,
    load_intan_adc, 
    load_br_intan_sync_ns5, 
    read_intan_to_br_map, 
    parse_intan_session_dtkey, 
    build_session_index_map,
    list_intan_sessions,
    read_intan_recording, 
    load_stim_detection, 
    save_recording,
    load_behavior_npz,
)

# plotting
from .python.plotting.plotting import (
    # _find_interp_dir_for_session,
    # _load_cached_recording,
    # plot_all_quads_for_session,
    run_one_Intan_FR_heatmap,
    plot_single_channel_trial_quad_raw,
    plot_channel_heatmap,
    stacked_heatmaps_two_t,
    stacked_heatmaps_plus_behv,
    add_ua_region_bar,
)



__all__ = [
    # params
    "load_experiment_params",
    "resolve_data_root",
    "resolve_output_root",
    "resolve_session_intan_dir",
    "resolve_probe_geom_path",
    "resolve_intan_root",

    # BR/UA
    "list_br_sessions",
    "ua_excel_path",
    "load_ns6_spikes",
    "load_ns5_aux",
    "load_ns2_digi",
    "load_UA_mapping_from_excel",
    "apply_ua_mapping_properties",
    "build_blackrock_bundle",
    "save_UA_bundle_npz",
    "threshold_mua_rates",

    # Intan
    "load_stim_geometry",
    "make_probe_from_geom",
    "local_cm_reference",
    "extract_and_save_stim_npz",
    "extract_and_save_other_streams_npz",
    "get_chanmap_perm_from_geom",
    "make_identity_probe_from_geom",
    "reorder_recording_to_geometry",
    
    # artifact correction
    "PCAArtifactParams",
    "remove_stim_pca_offline",
    "cleaned_numpy_to_recording",

    # stim preprocessing
    "StimTriggerConfig",
    "StimTriggerResult",
    "extract_stim_triggers_and_blocks",
    
    # utils
    "load_rate_npz",
    "median_across_trials",
    "extract_peristim_segments",
    "detect_stim_channels_from_npz",
    "build_probe_and_locs_from_geom",
    "baseline_zero_each_trial",
    "load_combined_npz", 
    "aligned_stim_ms", 
    "ua_title_from_meta",
    "load_intan_adc", 
    "load_br_intan_sync_ns5", 
    "read_intan_to_br_map", 
    "parse_intan_session_dtkey", 
    "build_session_index_map",
    "list_intan_sessions",
    "read_intan_recording", 
    "load_stim_detection", 
    "save_recording",
    "load_behavior_npz",
    
    # plotting
    # "_find_interp_dir_for_session",
    # "_load_cached_recording",
    # "plot_all_quads_for_session",
    
    "stacked_heatmaps_plus_behv",
    "run_one_Intan_FR_heatmap",
    "plot_single_channel_trial_quad_raw",
    "plot_channel_heatmap",
    "stacked_heatmaps_two_t"
    "add_ua_region_bar",
]
