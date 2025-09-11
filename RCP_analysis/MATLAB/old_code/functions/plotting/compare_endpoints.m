clear all; close all; clc;

%% --- Setup Paths ---
% addpath(genpath('/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/'));
addpath(genpath('functions'));
%% Session 1
% Plot all trials individually
% plot_metric_by_condition(T, 'Channels', [], base_folder);

% Plot a specific list of trials
session = 'BL_RW_001_Session_1';
base_folder = fullfile(['/Volumes/CullenLab_Server/Current Project Databases - NHP' ...
    '/2025 Cerebellum prosthesis/Bryan/Data'], session);

T_path = fullfile(base_folder, [session, '_metadata_with_metrics.csv']);
T = readtable(T_path);

plot_metric_by_condition(T, 'Channels', [26, 19, 20, 21, 22, 23, 24, 5, 6, 7, 9, 12, 13, 14, 15, 16], base_folder);

% plot_metric_by_condition(T, 'ChoiceOf16', [26, 6, 7, 9, 12, 13, 14, 15, 16], base_folder);

plot_metric_by_condition(T, 'Stim_Frequency_Hz', [26, 5, 19, 20, 27, 28], base_folder);
%% Session 2
session = 'BL_RW_002_Session_1';
base_folder = fullfile(['/Volumes/CullenLab_Server/Current Project Databases - NHP' ...
    '/2025 Cerebellum prosthesis/Bryan/Data'], session);

T_path = fullfile(base_folder, [session, '_metadata_with_metrics.csv']);
T = readtable(T_path);

% Load checkpoint with metadata and error traces
plot_metric_by_condition(T, 'Stim_Delay', [4, 21:27], base_folder);