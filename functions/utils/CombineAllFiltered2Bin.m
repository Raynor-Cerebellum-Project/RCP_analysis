clear all; close all; clc;

%% === Setup root path using machine-aware logic ===
addpath(genpath(fullfile('..', '..', 'functions')));
session = 'BL_RW_003_Session_1';
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);

%% === Define artifact root and output folder ===
artifact_root = fullfile(base_folder, 'Intan');
if ~exist(artifact_root, 'dir')
    error('Artifact folder does not exist: %s', artifact_root);
end

output_folder = fullfile(base_folder, 'CombinedBinFile');
if ~exist(output_folder, 'dir'); mkdir(output_folder); end

% === Manually declare which trial folders to include ===
selected_trials = {
    'BL_closed_loop_STIM_003_250528_140952';
    'BL_closed_loop_STIM_003_250528_152812';
    'BL_closed_loop_STIM_003_250528_160100'

    % 'BL_closed_loop_STIM_003_250528_143802';
    % BL_closed_loop_STIM_003_250528_144603
    % BL_closed_loop_STIM_003_250528_145904
    % BL_closed_loop_STIM_003_250528_152226
    % BL_closed_loop_STIM_003_250528_153443
    % BL_closed_loop_STIM_003_250528_153849
    % BL_closed_loop_STIM_003_250528_154801
    % BL_closed_loop_STIM_003_250528_160557
    % BL_closed_loop_STIM_003_250528_160920
};

%% === Find all trial subfolders ===
% Replace this line:
% trial_dirs = dir(fullfile(artifact_root, 'BL_closed_loop_STIM_*'));

% With this:
trial_dirs = cellfun(@(name) struct( ...
    'name', name, ...
    'folder', artifact_root, ...
    'isdir', true), ...
    selected_trials, 'UniformOutput', false);
trial_dirs = [trial_dirs{:}];

fprintf('Found %d trial folders.\n', numel(trial_dirs));
%% === Accumulate all data for concatenation ===
combined_data = [];

for i = 1:length(trial_dirs)
    trial_path = fullfile(trial_dirs(i).folder, trial_dirs(i).name);

    % Preferred file
    preferred_file = fullfile(trial_path, 'neural_data_artifact_removed_pca.mat');
    fallback_file  = fullfile(trial_path, 'neural_data.mat');

    if exist(preferred_file, 'file')
        mat_path = preferred_file;
    elseif exist(fallback_file, 'file')
        mat_path = fallback_file;
    else
        fprintf('No neural data found in %s. Skipping.\n', trial_dirs(i).name);
        continue;
    end

    fprintf('Loading %s\n', mat_path);
    S = load(mat_path);

    % === Support both variable names ===
    if isfield(S, 'artifact_removed_data')
        data = S.artifact_removed_data;
    elseif isfield(S, 'neural_data')
        data = S.neural_data;
    else
        warning('No valid data field found in %s. Skipping.', mat_path);
        continue;
    end

    if isempty(data)
        warning('Empty data in %s. Skipping.', mat_path);
        continue;
    end

    if size(data,1) < 128
        warning('File %s has fewer than 128 channels. Skipping.', mat_path);
        continue;
    end

    % Append [128 x time] to combined data
    combined_data = [combined_data, data(1:128, :)];
    clear data S;
end

if isempty(combined_data)
    error('No valid data found to concatenate.');
end

% Transpose to [time x channels] and convert to int16
data_to_write = int16(combined_data');  % [total_time x 128]

%% === Save combined .bin to output folder ===
bin_path = fullfile(output_folder, 'combined_baseline_data.bin');
fileID = fopen(bin_path, 'w');
fwrite(fileID, data_to_write, 'int16');
fclose(fileID);

fprintf('Saved combined .bin file: %s (%d bytes)\n', bin_path, dir(bin_path).bytes);
