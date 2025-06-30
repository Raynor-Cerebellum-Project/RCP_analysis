clear; clc; close all;

addpath(genpath('C:\Users\bryan\Documents\workspace\Raynor\PSID test\'));

base_path = 'Z:\Current Project Databases - NHP\2025 Cerebellum prosthesis\Bryan\CRR_NPXL_011_Session_1\Seperate cells\Kilosort_all_cells\CRR_NPXL_011_Session_1\';
condition_files = {
    'BuH_yaw_active_like',
    'BuH_yaw_sine',
    'HoB_yaw_active',
    'HoB_yaw_active_like',
    'HoB_yaw_active_multiple_perturbations',
    'HoB_yaw_sine',
    'WB_yaw_active_like',
    'WB_yaw_active_like_multilevel',
    'WB_yaw_sine'
};
cluster_numbers = [213 205 219 45 67 214 212 217 142 151 158 165];
nx_size_chosen = 8;
n_ahead = 30;

for i = 1:length(condition_files)
    [spike_rates_mat_clean, bsv, hsv, t_vec, condition_name] = ...
        load_condition_data(base_path, condition_files, i, cluster_numbers);

    % --- BSV prediction ---
    [Z_preds_bsv, Z_true_bsv, rel_mse_bsv, corrs_bsv] = ...
        run_psid_prediction(spike_rates_mat_clean, bsv, nx_size_chosen, n_ahead);
    plot_psid_results(Z_preds_bsv, Z_true_bsv, rel_mse_bsv, corrs_bsv, condition_name, n_ahead, 'bsv', base_path);

    % --- HSV prediction ---
    [Z_preds_hsv, Z_true_hsv, rel_mse_hsv, corrs_hsv] = ...
        run_psid_prediction(spike_rates_mat_clean, hsv, nx_size_chosen, n_ahead);
    plot_psid_results(Z_preds_hsv, Z_true_hsv, rel_mse_hsv, corrs_hsv, condition_name, n_ahead, 'hsv', base_path);
end
