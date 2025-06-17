%% Clearing workspace
clear all; close all; clc;

%% Loading data
session = 'BL_RW_003_Session_1';
base_folder = fullfile('/Volumes/CullenLab_Server/Current Project Databases - NHP', ...
    '2025 Cerebellum prosthesis/Bryan/Data', session);
intan_folder = fullfile(base_folder, 'Intan');

addpath(genpath(fullfile(base_folder, 'Intan')));
addpath('/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Analysis Codes/Oculomotor_Pipeline-main/Neural Pipeline');

outputfolder = fullfile(base_folder, 'Artifact_Corrected');
fig_folder = fullfile(base_folder, 'Figures');
if ~exist(fig_folder, 'dir'), mkdir(fig_folder); end
if ~exist(outputfolder, 'dir'), mkdir(outputfolder); end

%% Find valid trials
trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
valid_trials = {};  % Store names of valid trials

for i = 1:length(trial_dirs)
    trial_name = trial_dirs(i).name;
    trial_path = fullfile(intan_folder, trial_name);
    
    has_neural = isfile(fullfile(trial_path, 'neural_data.mat'));
    has_stim   = isfile(fullfile(trial_path, 'stim_data.mat'));

    if has_neural && has_stim
        valid_trials{end+1} = trial_name;
    end
end

fprintf('Found %d valid trials with all required files.\n', numel(valid_trials));

%% Load one of them
if ~isempty(valid_trials)
    trial = valid_trials{5};  % Or loop through them
    trial_path = fullfile(intan_folder, trial);

    load(fullfile(trial_path, 'neural_data.mat'));
    load(fullfile(trial_path, 'stim_data.mat'));

    fprintf('Loaded data from: %s\n', trial);
end

%% Artifact removal
NA_data = neural_data;
stim_data_scaled = stim_data .* 5000;

fixed_params = struct( ...
    'NSTIM', 0, ...
    'isstim', true, ...
    'start', 1, ... % 1, 10, 20, 30, 40 all seem fine
    'buffer', 30, ... % 10, 20, 30, or 40 are all fine
    'period_avg', 30, ... % 20, 30, or 40
    'skip_n', 2, ... % Should be fine
    'movmean_window', 3, ... % greater than 3
    'pca_components', 3 ...
);

template_modes = {'local', 'carryover'};
artifact_removed_data = sweep_template_param(neural_data, stim_data, 16, 'start', [1], fixed_params, template_modes);
