%% Clearing workspace
clear all; close all; clc;
addpath(genpath(fullfile('..', 'functions')));  % assumes functions is parallel to Neural Pipeline

%% Define base root
% Prompt or detect machine-specific root path
base_root = set_paths();

%% Define folders
code_root = fullfile(base_root, 'Current Project Databases - NHP', '2025 Cerebellum prosthesis', 'Bryan', 'Analysis Codes');
session   = 'BL_RW_003_Session_1';
base_folder = fullfile(base_root, 'Current Project Databases - NHP', '2025 Cerebellum prosthesis', 'Bryan', 'Data', session);

% Paths
intan_folder = fullfile(base_folder, 'Intan');
fig_folder   = fullfile(base_folder, 'Figures');

% Add paths
addpath(genpath(fullfile(code_root, '..', 'functions')));
addpath(genpath(intan_folder));

% Create output folders
if ~exist(fig_folder, 'dir'), mkdir(fig_folder); end

%% Find valid trials
trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
valid_trials = {};

for i = 1:length(trial_dirs)
    trial_name = trial_dirs(i).name;
    trial_path = fullfile(intan_folder, trial_name);
    
    has_neural_corrected = isfile(fullfile(trial_path, 'neural_data_artifact_removed.mat'));
    has_neural = isfile(fullfile(trial_path, 'neural_data.mat'));
    has_stim   = isfile(fullfile(trial_path, 'stim_data.mat'));
    
    if has_neural_corrected
        valid_trials{end+1} = trial_name;
    end
end

fprintf('Found %d valid Intan folders with required files.\n', numel(valid_trials));
%% Compute firing rates for all valid trials
fs = 30000;  % Sampling rate
filt_range = [300 6000];  % Spike band in Hz
bin_size = 0.050;  % Bin size for firing rate (s)
gauss_win_ms = 100;  % Smoothing window for spike train

[b, a] = butter(3, filt_range / (fs/2), 'bandpass');

for i = 2
    trial_name = valid_trials{i};
    trial_path = fullfile(intan_folder, trial_name);
    load(fullfile(trial_path, 'neural_data_artifact_removed.mat'));  % expects amplifier_data_cleaned
    [nChans, nSamples] = size(amplifier_data_cleaned);

    fprintf('Processing %s: %d channels, %.2f seconds...\n', trial_name, nChans, nSamples/fs);

    firing_rate_all = cell(nChans, 1);
    smoothed_fr_all = cell(nChans, 1);

    for ch = 1:nChans
        raw = amplifier_data_cleaned(ch, :);
        filtered = filtfilt(b, a, double(raw));

        % Spike detection: threshold at -3 std
        thresh = -3 * std(filtered);
        spike_idx = find(filtered < thresh);

        % Refractory period (1 ms)
        isi = diff(spike_idx);
        spike_idx = spike_idx([true, isi > fs*0.001]);

        % Histogram firing rate
        edges = 0:bin_size:(nSamples/fs);
        fr = histcounts(spike_idx/fs, edges) / bin_size;

        % Smooth spike train
        spike_train = zeros(1, nSamples);
        spike_train(spike_idx) = 1;
        gauss_len = round(fs * gauss_win_ms / 1000);
        gauss_win = gausswin(gauss_len);
        gauss_win = gauss_win / sum(gauss_win);
        smoothed_fr = conv(spike_train, gauss_win, 'same') * fs;

        % Save
        firing_rate_all{ch} = fr;
        smoothed_fr_all{ch} = smoothed_fr;
    end

    % Save to file
    save(fullfile(trial_path, 'firing_rate_data.mat'), ...
        'firing_rate_all', 'smoothed_fr_all', 'bin_size', 'fs');
end
