%% Clearing workspace
clear all; close all; clc;
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

%% Find valid trials
trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
valid_trials = {};

for i = 1:length(trial_dirs)
    trial_name = trial_dirs(i).name;
    trial_path = fullfile(intan_folder, trial_name);

    % Check for neural data
    has_neural = isfile(fullfile(trial_path, 'neural_data.mat')) || ...
                 isfile(fullfile(trial_path, 'neural_data_artifact_removed.mat'));

    % Get BR number from trial_name if needed
    tokens = regexp(trial_name, 'STIM_\d+_(\d+)', 'tokens');
    if isempty(tokens)
        continue  % skip if BR number can't be parsed
    end
    br_num = str2double(tokens{1}{1});
    br_dir = fullfile(br_folder, sprintf('IntanFile_%d', br_num));

    % Check for Cal file
    cal_stim_path = fullfile(br_dir, [trial_name, '_Stim.mat']);
    cal_path      = fullfile(br_dir, [trial_name, '.mat']);

    if isfile(cal_stim_path)
        selected_cal_file = cal_stim_path;
    elseif isfile(cal_path)
        selected_cal_file = cal_path;
    else
        selected_cal_file = '';
    end

    % Only keep if both neural data and a calibration file exist
    if has_neural && ~isempty(selected_cal_file)
        valid_trials{end+1} = trial_name;
        cal_files{end+1} = selected_cal_file;
    end
end

fprintf('Found %d valid trials with neural + calibration.\n', numel(valid_trials));

%% Parameters
fs = 30000;                     % Sampling rate (Hz)
filt_range = [300 6000];        % Spike band for spike detection
rate_mode = 'kaiser';           % Options: 'kaiser', 'causal'
cutoff_freq = 5;                % Firing rate filter cutoff (Hz)
threshold_std = -3;             % Spike threshold (× std)
refractory_ms = 1;              % Minimum ISI in ms
target_fs = 1000;               % Downsampled rate
ds_factor = fs / target_fs;

[b, a] = butter(3, filt_range / (fs/2), 'bandpass');

%% Process each trial individually
for i = 1:numel(valid_trials)
    trial_name = valid_trials{i};
    trial_path = fullfile(intan_folder, trial_name);
    fprintf('\n=== Processing trial: %s ===\n', trial_name);

    % Load neural data
    artifact_path = fullfile(trial_path, 'neural_data_artifact_removed.mat');
    neural_path   = fullfile(trial_path, 'neural_data.mat');

    if isfile(artifact_path)
        load(artifact_path, 'artifact_removed_data');
        neural_data = artifact_removed_data;
        disp('Loaded artifact-corrected data');
    elseif isfile(neural_path)
        load(neural_path, 'amplifier_data');
        neural_data = amplifier_data;
        disp('Loaded raw data');
    else
        warning('No neural data found for %s. Skipping.', trial_name);
        continue
    end

    [nChans, nSamples] = size(neural_data);
    fprintf('  %d channels, %.1f seconds\n', nChans, nSamples/fs);

    % Allocate outputs
    spike_trains_all = cell(nChans, 1);
    smoothed_fr_all = cell(nChans, 1);

    for ch = 1:nChans
        raw = neural_data(ch, :);
        filtered = filtfilt(b, a, double(raw));

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
    end

    clear neural_data filtered raw spike_train fr_full

    % Load stim_data and extract trigger blocks
    stim_file = fullfile(trial_path, 'stim_data.mat');
    block_start_1kHz = []; block_end_1kHz = [];

    if isfile(stim_file)
        load(stim_file, 'Stim_data');

        fs_stim = 30000;
        ds_factor_stim = fs_stim / 1000;

        STIM_CHANS = find(any(Stim_data ~= 0, 2));
        TRIGDAT = Stim_data(STIM_CHANS(1), :)';

        trigs1 = find(diff(TRIGDAT) < 0);  % falling edges
        trigs_rz = arrayfun(@(idx) ...
            idx + find(TRIGDAT(idx+1:end) == 0, 1, 'first'), ...
            trigs1, 'UniformOutput', false);
        trigs_rz = cell2mat(trigs_rz(~cellfun('isempty', trigs_rz)));

        trigs_beg = trigs1;
        trigs_end = trigs_rz;

        n_trigs = min(length(trigs_beg), length(trigs_end));
        trigs = [trigs_beg(1:n_trigs), trigs_end(1:n_trigs)];

        buffer = 60;
        time_diffs = diff(trigs_beg);
        repeat_gap_threshold = 2 * (2 * buffer + 1);
        repeat_boundaries = [0; find(time_diffs > repeat_gap_threshold); numel(trigs_beg)];
        num_repeats = numel(repeat_boundaries) - 1;

        block_start_samples = trigs_beg(repeat_boundaries(1:end-1) + 1);
        block_end_samples   = trigs_end(repeat_boundaries(2:end)) + buffer;

        block_start_1kHz = round(block_start_samples / ds_factor_stim) / 1000;
        block_end_1kHz   = round(block_end_samples   / ds_factor_stim) / 1000;
    else
        warning('  Stim file not found.');
    end

    % Save all trial-level results
    save(fullfile(trial_path, 'firing_rate_data.mat'), ...
        'smoothed_fr_all', 'spike_trains_all', ...
        'fs', 'cutoff_freq', 'rate_mode', ...
        'block_start_1kHz', 'block_end_1kHz', '-v7.3');

    fprintf('  Saved data for %s\n', trial_name);
end
