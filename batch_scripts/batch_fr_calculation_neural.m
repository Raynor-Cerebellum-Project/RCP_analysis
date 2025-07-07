%% Setup
clear all; close all; clc;
addpath(genpath(fullfile('..', 'functions')));

session = 'BL_RW_003_Session_1';
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);

intan_folder = fullfile(base_folder, 'Intan');

% Parameters
fs = 30000;
filt_range = [300 6000];
[b, a] = butter(3, filt_range / (fs/2), 'bandpass');
rate_mode = 'kaiser';
cutoff_freq = 5;
threshold_std = -3;
refractory_ms = 1;
target_fs = 1000;
ds_factor = round(fs / target_fs);

% Find valid trials
trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
fprintf('Found %d total folders.\n', numel(trial_dirs));

%% Loop through each trial
for i = 1:numel(trial_dirs)
    trial_name = trial_dirs(i).name;
    trial_path = fullfile(intan_folder, trial_name);

    % Check and load neural data
    artifact_path = fullfile(trial_path, 'neural_data_artifact_removed.mat');
    neural_path   = fullfile(trial_path, 'neural_data.mat');

    if isfile(artifact_path)
        load(artifact_path, 'artifact_removed_data');
        neural_data = artifact_removed_data;
        fprintf('\n[%d/%d] %s — Loaded artifact-corrected data.\n', i, numel(trial_dirs), trial_name);
    elseif isfile(neural_path)
        load(neural_path, 'neural_data');
        fprintf('\n[%d/%d] %s — Loaded raw data.\n', i, numel(trial_dirs), trial_name);
    else
        fprintf('\n[%d/%d] %s — No neural data found. Skipping.\n', i, numel(trial_dirs), trial_name);
        continue;
    end

    [nChans, nSamples] = size(neural_data);
    fprintf('  %d channels, %.1f seconds\n', nChans, nSamples/fs);

    % Preallocate
    spike_trains_all = cell(nChans, 1);
    smoothed_fr_all = cell(nChans, 1);

    % Channel-wise spike and FR computation
    for ch = 1:nChans
        raw = double(neural_data(ch, :));
        filtered = filtfilt(b, a, raw);

        thresh = threshold_std * std(filtered);
        spike_idx = find(filtered < thresh);
        isi = diff(spike_idx);
        spike_idx = spike_idx([true, isi > fs * refractory_ms / 1000]);

        spike_train = zeros(nSamples, 1);
        spike_train(spike_idx) = 1;

        fr_full = fr_estimate(spike_train, rate_mode, cutoff_freq, fs);
        fr = downsample(fr_full, ds_factor);

        spike_trains_all{ch} = spike_train;
        smoothed_fr_all{ch} = fr;

        clear raw filtered spike_idx spike_train fr fr_full
    end

    % Save firing rate data
    out_path = fullfile(trial_path, 'firing_rate_data.mat');
    if isfile(out_path)
        delete(out_path);  % remove old file if overwriting is intended
    end
    
    save(out_path, 'smoothed_fr_all', 'spike_trains_all', ...
         'fs', 'cutoff_freq', 'rate_mode', '-v7.3');

    clear smoothed_fr_all spike_trains_all
    fprintf('  Saved firing rates for %s\n', trial_name);
end
