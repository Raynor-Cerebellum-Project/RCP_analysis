# params
from .python.functions.params_loading import (
    load_experiment_params,
    resolve_probe_geom_path,
)
from .python.functions import config_loading
# utils
from .python.functions.utils import (
    find_per_cond_inputs,
    get_metadata_mapping,
    detect_IR_crossings,
    load_ocr_map,
    load_dlc, 
    align_dlc_to_corrected,
    frame2sample_br_ns5_sync,
    frame2sample_br_ns2_sync,
    
    list_intan_sessions,
    load_stim_detection, 
    save_recording,
    stim_npz_path_from_br_idx,
    butter_lowpass_pos_and_vel,
    butter_lowpass_pos_and_vel_3d,
    
    build_session_index_map,
    find_ns5_by_br_index, 
    find_ns2_by_br_index, 
    
    
    median_across_trials,
    variance_across_trials,
    extract_peristim_segments,
    detect_stim_channels_from_npz,
    baseline_zero_each_trial,
    aligned_stim_ms, 
    ua_title_from_meta,
    load_intan_aux,
    parse_intan_session_dtkey, 
    load_behavior_npz,
)
# intan
from .python.functions.intan_preproc import (
    reorder_recording_to_geometry,
    StimTriggerResult,
    extract_stim_npz,
    extract_intan_aux_streams_npz,
)    
# blackrock / UA
from .python.functions.br_preproc import (
    list_br_sessions,
    ua_excel_path,
    load_UA_mapping_from_excel,
    apply_ua_mapping_with_regions,
    extract_br_aux_streams_npz,
)

# artifact correction (PCA template subtraction)
from .python.functions.artifact_correction_template_matching import (
    remove_stim_pca_offline,
    cleaned_numpy_to_recording,
    PCAArtifactParams,
)

# plotting
from .python.plotting.plotting import (
    stacked_heatmaps_plus_behv,
)

__all__ = [
    # params
    "load_experiment_params",
    "resolve_probe_geom_path",
    "config_loading",

    # utils
    "find_per_cond_inputs", 
    "get_metadata_mapping",
    "detect_IR_crossings",
    "load_ocr_map",
    "load_dlc", 
    "align_dlc_to_corrected",
    "frame2sample_br_ns5_sync",
    "frame2sample_br_ns2_sync",
    
    "list_intan_sessions",
    "load_stim_detection", 
    "save_recording",
    
    "load_intan_aux", 
    "load_br_intan_sync_ns5", 
    "load_behavior_npz",
    
    
    "median_across_trials",
    "variance_across_trials",
    "extract_peristim_segments",
    
    
    "detect_stim_channels_from_npz",
    "baseline_zero_each_trial",
    "aligned_stim_ms", 
    "ua_title_from_meta",
    "parse_intan_session_dtkey", 
    "build_session_index_map",
    "find_ns5_by_br_index", 
    "find_ns2_by_br_index", 
    
    "butter_lowpass_pos_and_vel",
    "butter_lowpass_pos_and_vel_3d",
    
    # BR/UA
    "list_br_sessions",
    "ua_excel_path",
    "extract_br_aux_streams_npz",
    "apply_ua_mapping_with_regions",
    "load_UA_mapping_from_excel",
    
    "stim_npz_path_from_br_idx",

    # Intan
    "extract_stim_npz",
    "extract_intan_aux_streams_npz",
    "reorder_recording_to_geometry",
    "StimTriggerResult",
    
    # artifact correction
    "PCAArtifactParams",
    "remove_stim_pca_offline",
    "cleaned_numpy_to_recording",
    
    
    # plotting    
    "stacked_heatmaps_plus_behv",
]
