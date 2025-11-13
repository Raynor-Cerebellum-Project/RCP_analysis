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
    apply_ua_mapping_by_renaming,
    build_blackrock_bundle,
    save_UA_bundle_npz,
    threshold_mua_rates,
    ua_region_from_elec,
)

# intan
from .python.functions.intan_preproc import (
    load_stim_geometry,
    make_probe_from_geom,
    extract_stim_npz,
    extract_aux_streams_npz,
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
    find_per_cond_inputs,
    load_ocr_map,
    load_dlc, 
    align_dlc_to_corrected,
    find_nsx_by_br_index, 
    find_nsx_for_cond, 
    find_ns5_by_br_index, 
    find_ns5_for_cond,
    get_metadata_mapping,
    load_rate_npz,
    median_across_trials,
    variance_across_trials,
    extract_peristim_segments,
    detect_stim_channels_from_npz,
    build_probe_and_locs_from_geom,
    baseline_zero_each_trial,
    load_combined_npz, 
    aligned_stim_ms, 
    ua_title_from_meta,
    load_intan_adc,
    parse_intan_session_dtkey, 
    build_session_index_map,
    list_intan_sessions,
    read_intan_recording, 
    load_stim_detection, 
    save_recording,
    load_behavior_npz,
    frame2sample_br_ns5_sync
)

# plotting
from .python.plotting.plotting import (
    stacked_heatmaps_plus_behv,
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
    "apply_ua_mapping_by_renaming",
    "build_blackrock_bundle",
    "save_UA_bundle_npz",
    "threshold_mua_rates",
    "ua_region_from_elec",

    # Intan
    "load_stim_geometry",
    "make_probe_from_geom",
    "extract_stim_npz",
    "extract_aux_streams_npz",
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
    "get_metadata_mapping",
    "load_rate_npz",
    "load_combined_npz", 
    "load_intan_adc", 
    "load_br_intan_sync_ns5", 
    "load_behavior_npz",
    "load_stim_detection", 
    "load_dlc", 
    "load_ocr_map",
    
    "list_intan_sessions",
    
    "median_across_trials",
    "variance_across_trials",
    "extract_peristim_segments",
    
    "detect_stim_channels_from_npz",
    "build_probe_and_locs_from_geom",
    "baseline_zero_each_trial",
    "aligned_stim_ms", 
    "ua_title_from_meta",
    "parse_intan_session_dtkey", 
    "build_session_index_map",
    "read_intan_recording", 
    "save_recording",
    "frame2sample_br_ns5_sync",
    "align_dlc_to_corrected",
    
    "find_per_cond_inputs", 
    "find_nsx_by_br_index", 
    "find_nsx_for_cond", 
    "find_ns5_by_br_index", 
    "find_ns5_for_cond",
    
    # plotting    
    "stacked_heatmaps_plus_behv",
]
