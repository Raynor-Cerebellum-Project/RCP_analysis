clear all; close all; clc;

%% === Setup root path using machine-aware logic ===
addpath(genpath(fullfile('..', '..', 'functions')));
session = 'BL_RW_003_Session_1';
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);

%% === Define artifact root ===
artifact_root = fullfile(base_folder, 'Intan');
if ~exist(artifact_root, 'dir')
    error('Artifact folder does not exist: %s', artifact_root);
end

%% === Recursively find all *_artifact_removed_pca.mat files ===
mat_files = dir(fullfile(artifact_root, '**', '*_artifact_removed_pca.mat'));
fprintf('Found %d PCA artifact-corrected MAT files.\n', numel(mat_files));

%% === Convert each .mat to .bin in-place ===
for i = 1:length(mat_files)
    mat_path = fullfile(mat_files(i).folder, mat_files(i).name);
    fprintf('Converting %s\n', mat_path);

    % Load data
    S = load(mat_path);
    if isfield(S, 'artifact_removed_data')
        data = S.artifact_removed_data;
    else
        warning('Missing field "artifact_removed_data" in %s. Skipping.', mat_files(i).name);
        continue;
    end

    if isempty(data)
        warning('Empty data in %s. Skipping.', mat_files(i).name);
        continue;
    end

    % Transpose to [time x channels], only first 128 channels
    data_to_write = int16(data(1:128, :)');  % Transpose to [time x channels]

    % Save .bin file to same folder with same base name
    [~, base_name, ~] = fileparts(mat_files(i).name);
    bin_path = fullfile(mat_files(i).folder, [base_name '.bin']);

    fileID = fopen(bin_path, 'w');
    fwrite(fileID, data_to_write, 'int16');
    fclose(fileID);

    fprintf('Saved: %s (%d bytes)\n', bin_path, dir(bin_path).bytes);
    clear data data_to_write;
end
