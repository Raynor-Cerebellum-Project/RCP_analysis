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

%% === Recursively find all *_artifact_removed_pca.mat files ===
mat_files = dir(fullfile(artifact_root, '**', '*_artifact_removed_pca.mat'));
fprintf('Found %d PCA artifact-corrected MAT files.\n', numel(mat_files));

%% === Accumulate all data for concatenation ===
combined_data = [];

for i = 1:length(mat_files)
    mat_path = fullfile(mat_files(i).folder, mat_files(i).name);
    fprintf('Loading %s\n', mat_path);

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

    if size(data,1) < 128
        warning('File %s has fewer than 128 channels. Skipping.', mat_files(i).name);
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
bin_path = fullfile(output_folder, 'combined_artifact_removed_pca.bin');
fileID = fopen(bin_path, 'w');
fwrite(fileID, data_to_write, 'int16');
fclose(fileID);

fprintf('Saved combined .bin file: %s (%d bytes)\n', bin_path, dir(bin_path).bytes);
