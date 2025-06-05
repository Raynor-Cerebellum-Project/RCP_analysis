%% Plotting individual traces
clear all; close all; clc;

%% --- Setup Paths ---
% addpath(genpath('/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/'));
addpath(genpath('functions'));
session = 'BL_RW_001_Session_1';
base_folder = fullfile(['/Volumes/CullenLab_Server/Current Project Databases - NHP' ...
    '/2025 Cerebellum prosthesis/Bryan/Data'], session);

% Load checkpoint with metadata and error traces
load(fullfile(base_folder, [session, '_endpoints_checkpoint.mat']));
%% Read in the meta file
T = readtable(fullfile(base_folder, [session, '_metadata_with_metrics.csv']));
%% --- Set Plotting Options ---
offset = false;  % vertically offset traces
use_ci = true;
conditions = [14, 15, 21, 22, 23, 24, 25, 26]; %
% Trial numbers of condition files
sides = {'active_like_stim_pos', 'active_like_stim_neg'};  % Both sides
%% --- Session-specific Endpoint Targets ---
switch session
    case 'BL_RW_001_Session_1'
        EndPoint_pos = 32;
        EndPoint_neg = -26;
        subfolder = 'Renamed2';
        file_pattern = '*_Cal.mat';
        baseline_file_num = 26;
        conditions = [19, 20, 27, 21, 3, 22, 23, 24, 5, 28, 4, 6, 7, 9, 12, 13, 14, 15, 16];
        
    case 'BL_RW_002_Session_1'
        EndPoint_pos = 49.5857;
        EndPoint_neg = -34.4605;
        subfolder = 'Calibrated';
        file_pattern = fullfile('IntanFile_*', '*_Cal.mat');
        baseline_file_num = 27;
        
    otherwise
        EndPoint_pos = 30;
        EndPoint_neg = -30;
        subfolder = 'Calibrated';
        file_pattern = fullfile('IntanFile_*', '*_Cal.mat');
        baseline_file_num = 4;
        condition_file_num = 5;
end
%% 
% Common logic
search_folder = fullfile(base_folder, subfolder);
trial_mat_files = dir(fullfile(search_folder, file_pattern));

for cond_idx = 1:length(conditions)
    condition_file_num = conditions(cond_idx);

     % --- Find matching trial_mat_file by parsing BR_File number ---
    match_str = sprintf('_%03d_Cal.mat$', condition_file_num);  % use trailing anchor
    match_idx = find(~cellfun(@isempty, regexp({trial_mat_files.name}, match_str)));

    if isempty(match_idx)
        warning('Condition trial %d not found in file list.', condition_file_num);
        continue;
    end

    % --- Load condition trial ---
    condition_filename = trial_mat_files(match_idx).name;
    condition_path     = trial_mat_files(match_idx).folder;
    condition_fullpath = fullfile(condition_path, condition_filename);
    tmp = load(condition_fullpath, 'Data');
    condition_data = tmp.Data;

    % --- Load metadata row for this trial ---
    row_idx_condition = find(T.BR_File == condition_file_num);
    if isempty(row_idx_condition)
        warning('Condition trial %d not found in T.BR_File.', condition_file_num);
        continue;
    end
    metadata_cond = T(row_idx_condition, :);

    % --- Load baseline once per loop (unless you vary it) ---
    baseline_match_str = sprintf('_%03d_Cal.mat$', baseline_file_num);
    baseline_match_idx = find(~cellfun(@isempty, regexp({trial_mat_files.name}, baseline_match_str)));

    if isempty(baseline_match_idx)
        warning('Baseline trial %d not found.', baseline_file_num);
        continue;
    end

    baseline_filename = trial_mat_files(baseline_match_idx).name;
    baseline_path     = trial_mat_files(baseline_match_idx).folder;
    baseline_fullpath = fullfile(baseline_path, baseline_filename);
    tmp = load(baseline_fullpath, 'Data');
    baseline_data = tmp.Data;
    
    row_idx_baseline = find(T.BR_File == baseline_file_num);
    if isempty(row_idx_baseline)
        warning('Baseline trial %d not found in T.BR_File.', baseline_file_num);
        continue;
    end
    metadata_baseline = T(row_idx_baseline, :);

    for s = 1:length(sides)
        side = sides{s};

        % === Plot ===
        save_dir = fullfile(base_folder, 'Figures', 'ComparisonTraces');
        fig = plot_traces(baseline_data, condition_data, side, offset, ...
                  metadata_baseline, metadata_cond, save_dir, ...
                  baseline_file_num, condition_file_num, Summary_all, use_ci);

        % === Save ===

        if ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
        side_short = strrep(side, 'active_like_stim_', '');  % e.g., 'pos' or 'neg'
        stim_str_for_file = sprintf('%dCh_%dHz_%duA_%dmsdelay_%s', ...
            metadata_cond.Channels, ...
            metadata_cond.Stim_Frequency_Hz, ...
            metadata_cond.Current_uA, ...
            metadata_cond.Stim_Delay, ...
            metadata_cond.Movement_Trigger{1});
        
        save_filename = sprintf('%s_Condition_%03d_%s_ComparisonTraces.png', ...
            stim_str_for_file, ...
            condition_file_num, ...
            side);
        
        save_path = fullfile(save_dir, save_filename);

        print(fig, save_path, '-dpng', '-r300');
        close(fig);
        fprintf('Saved: %s\n', save_filename);
    end
end