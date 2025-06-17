%% === Setup root path using machine-aware logic ===
base_root = set_paths();  % e.g., returns '/Volumes/CullenLab_Server/...' or prompts via uigetdir

%% === Define artifact root and output folder ===
artifact_root = fullfile(base_root, 'Current Project Databases - NHP', ...
    '2025 Cerebellum prosthesis', 'Bryan', 'Data');

% Change to artifact root temporarily to set the uigetdir start point
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

%% === Convert each .mat to .bin ===
for i = 1:length(mat_files)
    mat_path = fullfile(artifact_folder, mat_files(i).name);
    fprintf('Processing %s\n', mat_files(i).name);

    % Load MAT file
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

    % Write to .bin file
    [~, base_name, ~] = fileparts(mat_files(i).name);
    bin_path = fullfile(bin_output_dir, [base_name '.bin']);
    fileID = fopen(bin_path, 'w');
    fwrite(fileID, data_to_write, 'int16');
    fclose(fileID);

    fprintf('Saved: %s (%d bytes)\n', bin_path, dir(bin_path).bytes);
end
