clear all; close all; clc;

%% === Setup root path using machine-aware logic ===
addpath(genpath('functions'));
base_root = set_paths_cullen_lab();  % Customize to your setup

%% === Define artifact root and output folder ===
artifact_root = fullfile(base_root, 'Current Project Databases - NHP', ...
    '2025 Cerebellum prosthesis', 'Bryan', 'Data');

% Set selection point for GUI
original_dir = pwd;
cd(artifact_root);
artifact_folder = uigetdir(pwd, 'Select Artifact_Corrected folder');
cd(original_dir);

if isequal(artifact_folder, 0)
    error('No folder selected. Exiting.');
end

bin_output_dir = fullfile(artifact_folder, 'binFiles');
if ~exist(bin_output_dir, 'dir'); mkdir(bin_output_dir); end

%% === Find all artifact-corrected .mat files ===
mat_files = dir(fullfile(artifact_folder, '*_artifact_removed.mat'));
fprintf('Found %d artifact-corrected MAT files.\n', numel(mat_files));

%% === Combine all valid artifact_removed_data ===
combined_data = [];

for i = 1:length(mat_files)
    mat_path = fullfile(artifact_folder, mat_files(i).name);
    fprintf('Loading %s...\n', mat_files(i).name);

    S = load(mat_path);
    if ~isfield(S, 'artifact_removed_data')
        warning('Missing field "artifact_removed_data" in %s. Skipping.', mat_files(i).name);
        continue;
    end

    data = S.artifact_removed_data;
    if isempty(data)
        warning('Empty data in %s. Skipping.', mat_files(i).name);
        continue;
    end

    if size(data, 1) < 128
        warning('Fewer than 128 channels in %s. Skipping.', mat_files(i).name);
        continue;
    end

    % Truncate to first 128 channels and concatenate along time axis
    combined_data = [combined_data, data(1:128, :)];  % [128 x total_time]
    clear data;
end

% Transpose to [time x channels]
data_to_write = int16(combined_data');  % [total_time x 128]

%% === Write combined data to .bin ===
bin_path = fullfile(bin_output_dir, 'combined_artifact_removed.bin');
fileID = fopen(bin_path, 'w');
fwrite(fileID, data_to_write, 'int16');
fclose(fileID);

fprintf('Saved combined .bin file: %s (%d bytes)\n', bin_path, dir(bin_path).bytes);
