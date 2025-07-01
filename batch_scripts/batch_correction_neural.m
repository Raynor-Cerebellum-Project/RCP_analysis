%% Clearing workspace
clear all; close all; clc;
profile on -memory
addpath(genpath(fullfile('..', 'functions')));  % assumes functions is parallel to Neural Pipeline
%% Define base root
% Prompt or detect machine-specific root path
base_root = set_paths();  % e.g., '/Volumes/cullenlab_server'

%% Define code and data folders
code_root = fullfile(base_root, 'Current Project Databases - NHP', '2025 Cerebellum prosthesis', 'Bryan', 'Analysis Codes');
session   = 'BL_RW_003_Session_1';
base_folder = fullfile(base_root, 'Current Project Databases - NHP', '2025 Cerebellum prosthesis', 'Bryan', 'Data', session);

% Analysis subfolders
intan_folder = fullfile(base_folder, 'Intan');
outputfolder = fullfile(base_folder, 'Artifact_Corrected');
fig_folder   = fullfile(base_folder, 'Figures');

%% Add relevant analysis paths
addpath(genpath(fullfile(code_root, '..', 'functions')));  % assumes functions is parallel to Neural Pipeline
addpath(genpath(intan_folder));
%% Create output folders if needed
if ~exist(fig_folder, 'dir'), mkdir(fig_folder); end
if ~exist(outputfolder, 'dir'), mkdir(outputfolder); end

%% Find valid trials
trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
valid_trials = {};

for i = 1:length(trial_dirs)
    trial_name = trial_dirs(i).name;
    trial_path = fullfile(intan_folder, trial_name);
    
    has_neural = isfile(fullfile(trial_path, 'neural_data.mat'));
    has_stim   = isfile(fullfile(trial_path, 'stim_data.mat'));
    
    if has_neural && has_stim
        valid_trials{end+1} = trial_name;
    end
end

fprintf('Found %d valid Intan folders with required files.\n', numel(valid_trials));

%% Parameters
fs = 30000;  % Hz
fixed_params = struct( ...
    'NSTIM', 0, ...
    'buffer', 25, ... % 10, 20, 30, or 40 are all fine
    'template_leeway', 15, ... % 10, 20, 25
    'stim_neural_delay', 13, ... 
    'period_avg', 25, ... % 20, 30, or 40
    'movmean_window', 3, ... % greater than 3
    'pca_components', 3, ...
    'minus_decay', false, ...
    'med_filt_range', 25, ...
    'gauss_filt_range', 25 ...
);

template_modes = {'local_drift_corr'};

%% Loop through trials
skip_ids = [];  % Large file size; skip for now

 %[3, 10, 11, 12]
for i = [3, 10, 11, 12]%1:length(valid_trials)
    if ismember(i, skip_ids)
        fprintf('Skipping trial %d: %s\n', i, valid_trials{i});
        continue;
    end
    trial = valid_trials{i};
    trial_path = fullfile(intan_folder, trial);
    
    fprintf('\nProcessing trial: %s\n', trial);

    % Load stim_data robustly
    stim_struct = load(fullfile(trial_path, 'stim_data.mat'));
    if isfield(stim_struct, 'Stim_data')
        Stim_data = stim_struct.Stim_data;
    elseif isfield(stim_struct, 'stim_data')
        Stim_data = stim_struct.stim_data;
    else
        warning('No recognized stim_data field in %s. Skipping.', trial);
        continue;
    end

    % Detect stim channels
    STIM_CHANS = find(any(Stim_data ~= 0, 2));
    if isempty(STIM_CHANS)
        warning('No stim signal detected in %s. Skipping.', trial);
        continue;
    end

    % Artifact removal
    compare_plot = false;
    plot_chan = false;
    load(fullfile(trial_path, 'neural_data.mat'));
    [cleaned_all, trigs] = compare_template_modes(neural_data, Stim_data, fixed_params, plot_chan, template_modes, compare_plot);
    clear neural_data Stim_data;
    artifact_removed_data = squeeze(cleaned_all(:, 1, :));  % [nChans x time]

    % Save cleaned data
    save_path = fullfile(outputfolder, [trial '_artifact_removed.mat']);
    save(save_path, 'artifact_removed_data', 'trigs', '-v7.3');
    clear artifact_removed_data;
    fprintf('Saved cleaned data for %s\n', trial);
end
% % Filtering
% spike_data = zeros(size(neural_data));
% lfp_data = zeros(size(neural_data));
% for chan = 1:size(neural_data, 1)
%     temp = artifact_removed_data(chan, :);
% 
%     % LFP < 250 Hz
%     [b_lfp, a_lfp] = butter(4, 250/(fs/2), 'low');
%     lfp_data(chan, :) = filtfilt(b_lfp, a_lfp, temp);
% 
%     % Spikes > 250 Hz
%     [b_spk, a_spk] = butter(4, 250/(fs/2), 'high');
%     spike_data(chan, :) = filtfilt(b_spk, a_spk, temp);
% end
% 
% % Plotting
% figure;
% hold on; box off;
% N = size(neural_data, 2);
% time_ms = (0:N-1) / fs * 1000;
% 
% plot(time_ms, stim_data_scaled(STIM_CHANS(1), :), 'k', 'LineWidth', 1.5);
% plot(time_ms, neural_data(100, :), 'r', 'LineWidth', 1.5);
% plot(time_ms, artifact_removed_data(100, :)', 'b', 'LineWidth', 1.5);
% xlabel('Time (ms)'); ylabel('Amplitude (µV)');
% legend({'Stim', 'Raw', 'Cleaned'}, 'Location', 'best');
% title(sprintf('Artifact Removal: %s', trial), 'Interpreter', 'none');
% set(gca, 'TickDir', 'out');
% 
% % Save outputs
% data_filename = fullfile(outputfolder, ['artifact_removed_' trial '.mat']);
% save(data_filename, 'artifact_removed_data', 'lfp_data', 'spike_data', '-v7.3');
% 
% fig_base = fullfile(fig_folder, ['artifact_removal_plot_' trial '_full']);
% saveas(gcf, [fig_base, '.png']);
% savefig([fig_base, '.fig']);
% close(gcf);
p = profile('info');
save('profiler_data.mat', 'p');