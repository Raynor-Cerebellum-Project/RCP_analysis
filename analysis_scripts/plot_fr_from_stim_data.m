%% Clearing workspace
clear; close all; clc;
addpath(genpath(fullfile('..', 'functions')));  % assumes functions is parallel to Neural Pipeline

%% Define base root
% Prompt or detect machine-specific root path
base_root = set_paths_cullen_lab();

%% Define folders
code_root = fullfile(base_root, 'Current Project Databases - NHP', '2025 Cerebellum prosthesis', 'Bryan', 'Analysis Codes');
session   = 'BL_RW_003_Session_1';
base_folder = fullfile(base_root, 'Current Project Databases - NHP', '2025 Cerebellum prosthesis', 'Bryan', 'Data', session);

% Paths
intan_folder = fullfile(base_folder, 'Intan');
br_folder = fullfile(base_folder, 'Calibrated');
fig_folder   = fullfile(base_folder, 'Figures');

% Add paths
addpath(genpath(fullfile(code_root, '..', 'functions')));
addpath(genpath(intan_folder));
addpath(genpath(br_folder));

% Create output folders
if ~exist(fig_folder, 'dir'), mkdir(fig_folder); end

%% Find valid trials intan 
trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
valid_trials = {};

for i = 1:length(trial_dirs)
    trial_name = trial_dirs(i).name;
    trial_path = fullfile(intan_folder, trial_name);
    
    has_fr_estimates = isfile(fullfile(trial_path, 'firing_rate_data.mat'));
    has_neural_corrected = isfile(fullfile(trial_path, 'neural_data_artifact_removed.mat'));
    has_neural = isfile(fullfile(trial_path, 'neural_data.mat'));
    has_stim   = isfile(fullfile(trial_path, 'stim_data.mat'));
    
    if has_fr_estimates
        valid_trials{end+1} = trial_name;
    end

    has_br_match = isfile(fullfile(trial_path, 'stim_data.mat'));
end

fprintf('Found %d valid Intan folders with required files.\n', numel(valid_trials));

%% Plot firing rate traces for a trial with stimulation trigger overlays
trial_idx = 1;  % Select trial to plot
trial_name = valid_trials{trial_idx};
trial_path = fullfile(intan_folder, trial_name);

% Load firing rates
load(fullfile(trial_path, 'firing_rate_data.mat'));  % smoothed_fr_all, fs
target_fs = 1000;  % Firing rate was estimated at 1 kHz
chan_to_plot = 32;  % Set channel number
fr_trace = smoothed_fr_all{chan_to_plot};

%% === Align firing rate to stim onset and plot average trace ===
pre_time = 0.8;   % seconds before stim
post_time = 1.2;  % seconds after stim
align_window = round((-pre_time * target_fs):(post_time * target_fs));  % index offsets

% Number of blocks (trials)
n_trials = numel(block_start_1kHz);

% Initialize matrix to hold aligned traces
aligned_traces = nan(n_trials, numel(align_window));

for i = 1:n_trials
    t0 = round(block_start_1kHz(i) * target_fs);  % stim start index
    idx_range = t0 + align_window;

    % Ensure index is within bounds
    valid_idx = idx_range > 0 & idx_range <= length(fr_trace);
    aligned_traces(i, valid_idx) = fr_trace(idx_range(valid_idx));
end

% Plot average trace with shaded error
time_axis = align_window / target_fs;  % in seconds

mean_trace = nanmean(aligned_traces, 1);
sem_trace  = nanstd(aligned_traces, 0, 1) / sqrt(n_trials);

% Plot all aligned traces and the mean
figure('Name', sprintf('Stim-Aligned FR: %s (Ch %d)', trial_name, chan_to_plot), ...
    'Position', [300 300 900 400]);

% Plot each trial trace in light gray
plot(time_axis, aligned_traces', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5); hold on;

% Plot the mean trace in bold blue
plot(time_axis, mean_trace, 'b', 'LineWidth', 2);

% Mark stim onset
xline(0, '--k', 'Stim Onset');

xlabel('Time from Stim Onset (s)');
ylabel('Firing Rate (Hz)');
title(sprintf('Stim-Aligned FR — Channel %d (%s)', chan_to_plot, trial_name), 'Interpreter', 'none');
