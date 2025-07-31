%% Clearing workspace
clear all; close all; clc;
%%
profile on -memory
%% Setup Paths
addpath(genpath(fullfile('..', 'functions')));
session = 'BL_RW_003_Session_1';
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);
intan_folder = fullfile(base_folder, 'Intan');
fig_folder   = fullfile(base_folder, 'Figures');
metadata_csv     = fullfile(base_folder, 'Metadata', [session, '_metadata.csv']);
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
    'gauss_filt_range', 25, ...
    'pca_k', 3 ...
    );
template_modes = {'pca'};

% Spike detection + FR params
filt_range = [300 6000];
[b, a] = butter(3, filt_range / (fs/2), 'bandpass');
rate_mode = 'kaiser';
cutoff_freq = 5;
threshold_std = -3;
refractory_ms = 1;
target_fs = 1000;
ds_factor = round(fs / target_fs);

%% --- Load Metadata ---
T = readtable(metadata_csv);
has_stim_all = false(height(T), 1);  % Preallocate logical array

%% Start parallel pool before trial loop
if isempty(gcp('nocreate'))
    myCluster = parcluster('local');
    myCluster.IdleTimeout = 60;  % Keep pool alive for up to 60 minutes idle
    parpool(myCluster, min(30, feature('numcores')));
end

%% Loop through each trial
for i = 1:numel(valid_trials)
    trial = valid_trials{i};
    trial_path = fullfile(intan_folder, trial);
    logmsg(sprintf('[%d/%d] Processing: %s', i, numel(valid_trials), trial));

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
    has_trigs = ~isempty(trigs);
    if ~has_trigs
        warning('No triggers found. Skipping artifact removal.');
    end
    % Match current trial name to metadata row
    if i <= height(T)
        has_stim_all(i) = has_trigs;
    end
    clear stim_data
    % Save artifact removed data and trig info with method suffix
    trig_info_file = fullfile(trial_path, sprintf('trig_info.mat'));
    save(trig_info_file, 'trigs', 'repeat_boundaries');

    % === Load neural data ===
    logmsg('  Loading data...');

    tic;
    neural_struct = load(fullfile(trial_path, 'neural_data.mat'));
    elapsed_time = toc;
    logmsg(sprintf('Done (%.2f sec).', elapsed_time));

    for m = 1:numel(template_modes)
        method = template_modes{m};
        if has_trigs
            % === Artifact removal ===
            fprintf('  Running artifact removal (%s)... ', method);
            tic;
            neural_data = neural_struct.neural_data;
            artifact_removed_data = ...
                template_subtraction_all_parallel(neural_data, updated_params, {method}, ...
                trigs, repeat_boundaries, trial_path);
            elapsed_time = toc;
            logmsg(sprintf('Done (%.2f sec).', elapsed_time));

            artifact_file  = fullfile(trial_path, sprintf('neural_data_artifact_removed_%s.mat', method));
            save(artifact_file, 'artifact_removed_data', '-v7.3');
            fprintf('  Saved artifact-corrected data (%s).\n', method);
        else
            % Use raw data if no artifact correction
            artifact_removed_data = neural_struct.neural_data;
        end

        % === Spike detection + FR estimation ===
        fprintf('  Running FR estimation (%s)... ', method);
        tic;
        [nChans, nSamples] = size(artifact_removed_data);
        spike_trains_all = cell(nChans, 1);
        smoothed_fr_all  = cell(nChans, 1);

        parfor ch = 1:nChans
            raw = double(artifact_removed_data(ch, :));
            filtered = filtfilt(b, a, raw);

            med = median(filtered);
            mad_val = median(abs(filtered - med));
            robust_std = 1.4826 * mad_val;
            thresh = threshold_std * robust_std;

            spike_idx = find(abs(filtered - med) > abs(thresh));
            spike_idx = sort(spike_idx);
            refractory_samples = round(fs * refractory_ms / 1000);

            isi = diff([-Inf; spike_idx(:)]);
            spike_idx = spike_idx(isi >= refractory_samples);

            spike_train = zeros(nSamples, 1);
            spike_train(spike_idx) = 1;

            fr_full = fr_estimate(spike_train, rate_mode, cutoff_freq, fs);
            smoothed_fr_all{ch} = downsample(fr_full, ds_factor);
            spike_trains_all{ch} = spike_train;
        end

        elapsed_time = toc;
        logmsg(sprintf('Done (%.2f sec).', elapsed_time));

        % Save FR data
        if has_trigs
            fr_out_path = fullfile(trial_path, sprintf('firing_rate_data_%s.mat', method));
        else
            fr_out_path = fullfile(trial_path, 'firing_rate_data.mat');
        end
        if isfile(fr_out_path), delete(fr_out_path); end
        save(fr_out_path, 'smoothed_fr_all', 'spike_trains_all', ...
            'fs', 'cutoff_freq', 'rate_mode', '-v7.3');
        fprintf('  Saved firing rate data (%s).\n', method);
    end
end

% Add stim flag to metadata
T.Has_Stim = has_stim_all;
writetable(T, metadata_csv);  % Overwrite CSV with new column
save(fullfile(base_folder, 'Metadata', [session '_metadata_with_stim.mat']), 'T');

%% Save profiler output
p = profile('info');
save('profiler_data.mat', 'p');

function logmsg(msg)
    fprintf('[%s] %s\n', datestr(datetime('now'), 'yyyy-mm-dd HH:MM:SS'), msg);
end
