%% Clearing workspace
clear all; close all; clc;
profile on -memory

%% Setup Paths
addpath(genpath(fullfile('..', 'functions')));
session = 'BL_RW_003_Session_1';
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);
intan_folder = fullfile(base_folder, 'Intan');
fig_folder   = fullfile(base_folder, 'Figures');
if ~exist(fig_folder, 'dir'), mkdir(fig_folder); end

%% Find valid trials
trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
valid_trials = {};
for i = 1:length(trial_dirs)
    trial_name = trial_dirs(i).name;
    trial_path = fullfile(intan_folder, trial_name);
    if isfile(fullfile(trial_path, 'neural_data.mat')) && ...
            isfile(fullfile(trial_path, 'stim_data.mat'))
        valid_trials{end+1} = trial_name;
    end
end
fprintf('Found %d valid trials with required files.\n', numel(valid_trials));

%% Parameters
fs = 30000;
fixed_params = struct( ...
    'NSTIM', 0, ...
    'buffer', 25, ...
    'template_leeway', 15, ...
    'stim_neural_delay', 13, ...
    'movmean_window', 3, ...
    'med_filt_range', 25, ...
    'gauss_filt_range', 25 ...
    );
template_modes = {'local_drift_corr'};

% Spike detection + FR params
filt_range = [300 6000];
[b, a] = butter(3, filt_range / (fs/2), 'bandpass');
rate_mode = 'kaiser';
cutoff_freq = 5;
threshold_std = -3;
refractory_ms = 1;
target_fs = 1000;
ds_factor = round(fs / target_fs);

if isempty(gcp('nocreate'))
    num_workers = min(30, feature('numcores')); % fallback
    parpool('local', num_workers);
end
%% Loop through each trial
for i = 16:numel(valid_trials)
    trial = valid_trials{i};
    trial_path = fullfile(intan_folder, trial);
    fprintf('\n[%d/%d] Processing: %s\n', i, numel(valid_trials), trial);

    % === Load stim data and extract triggers ===
    stim_struct = load(fullfile(trial_path, 'stim_data.mat'));
    if isfield(stim_struct, 'Stim_data')
        stim_data = stim_struct.Stim_data;
    elseif isfield(stim_struct, 'stim_data')
        stim_data = stim_struct.stim_data;
    else
        warning('No recognized stim_data field in %s. Skipping.', trial);
        continue;
    end
    [trigs, repeat_boundaries, STIM_CHANS, updated_params] = ...
        extract_triggers_and_repeats(stim_data, fs, fixed_params);
    if isempty(trigs)
        warning('No triggers found. Skipping.');
        continue;
    end
    clear stim_data

    % === Load neural data ===
    fprintf('  Loading data... ');
    tic;
    neural_struct = load(fullfile(trial_path, 'neural_data.mat'));
    elapsed_time = toc;
    fprintf('Done (%.2f sec).\n', elapsed_time);

    % === Artifact removal ===
    fprintf('  Running artifact removal... ');
    tic;
    neural_data = neural_struct.neural_data;
    artifact_removed_data = ...
        compare_template_modes(neural_data, updated_params, template_modes, ...
        trigs, repeat_boundaries, trial_path);
    elapsed_time = toc;
    fprintf('Done (%.2f sec).\n', elapsed_time);

    % Save artifact removed data
    save(fullfile(trial_path, 'trig_info.mat'), 'trigs', 'repeat_boundaries');
    save(fullfile(trial_path, 'neural_data_artifact_removed.mat'), 'artifact_removed_data', '-v7.3');
    fprintf('  Saved artifact-corrected data.\n');

    % === Spike detection + FR estimation ===
    fprintf('  Running FR estimation... ');
    tic;
    [nChans, nSamples] = size(artifact_removed_data);
    spike_trains_all = cell(nChans, 1);
    smoothed_fr_all  = cell(nChans, 1);

    parfor ch = 1:nChans
        raw = double(artifact_removed_data(ch, :));
        filtered = filtfilt(b, a, raw);

        % Robust threshold using MAD
        med = median(filtered);
        mad_val = median(abs(filtered - med));
        robust_std = 1.4826 * mad_val;
        thresh = threshold_std * robust_std;

        % Detect both positive and negative threshold crossings
        spike_idx = find(abs(filtered - med) > abs(thresh));

        % Enforce refractory period
        isi = diff(spike_idx);
        spike_idx = spike_idx([true, isi > fs * refractory_ms / 1000]);

        % Create binary spike train
        spike_train = zeros(nSamples, 1);
        spike_train(spike_idx) = 1;

        % Estimate firing rate
        fr_full = fr_estimate(spike_train, rate_mode, cutoff_freq, fs);
        smoothed_fr_all{ch} = downsample(fr_full, ds_factor);
        spike_trains_all{ch} = spike_train;
    end

    elapsed_time = toc;
    fprintf('Done (%.2f sec).\n', elapsed_time);

    % Save FR data
    fr_out_path = fullfile(trial_path, 'firing_rate_data.mat');
    if isfile(fr_out_path), delete(fr_out_path); end
    save(fr_out_path, 'smoothed_fr_all', 'spike_trains_all', ...
        'fs', 'cutoff_freq', 'rate_mode', '-v7.3');
    fprintf('  Saved firing rate data.\n');
end

%% Save profiler output
p = profile('info');
save('profiler_data.mat', 'p');
