%% Clearing workspace
close all;
clear all;
clc;

%% Loading data
% Define base project path
base_folder = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_001_Session_1';

% Define subfolders based on base path
session_folder = fullfile(base_folder);  % Could also just use base_folder directly
outputfolder = fullfile(base_folder, 'Artifact_Corrected');
fig_folder = fullfile(base_folder, 'Figures');
addpath('//Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_001_Session_1/Intan');
addpath('//Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Analysis Codes/Oculomotor_Pipeline-main/Neural Pipeline');

% Ensure figure folder exists
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end
if ~exist(outputfolder, 'dir')
    mkdir(outputfolder);
end

trial = 'BL_closed_loop_STIM_001_250501_133137';

load([session_folder, '/Intan/', trial, '/neural_data.mat']);
load([session_folder, '/Intan/', trial, '/stim_data.mat']);
load([session_folder, '/Intan/', trial, '/session_triger.mat']);
%% Artifact removal
NA_data = neural_data;
stim_data_scaled = stim_data .* 5000;

template_params = struct( 'NSTIM', 0, ...  % number of stim pulses
    'isstim', true, ... % true if the data is from a stim channel
    'period_avg', 30, ... % number of points to average for the template
    'start', 40, ... % skip the first number of pulses when calculating the template
    'buffer', 0, ... % the number of points before each oulse to be considered in calculating the template
    'skip_n', 0 ...% number of initial pulses to skip to calculate the template...
    );

% Call artifact_removal 1. template subtraction (Neural Pipeline) 2. LS method (filter test)
artifact_removed_data = artifact_Removal(neural_data, stim_data, template_params);

STIM_CHANS = find(any(stim_data_scaled~=0, 2));
%% Filtering for spikes
fs = 30000;  % Sampling rate in Hz
spike_data = zeros(128, size(NA_data, 2));
lfp_data = zeros(128, size(NA_data, 2));

for chan = 1:128
    temp = artifact_removed_data(chan, :);
    
    % Low-pass filter for LFP (< 250 Hz)
    [b_lfp, a_lfp] = butter(4, 250/(fs/2), 'low');
    lfp_data(chan, :) = filtfilt(b_lfp, a_lfp, temp);

    % High-pass filter for spikes (> 250 Hz, for example)
    [b_spk, a_spk] = butter(4, 250/(fs/2), 'high');
    spike_data(chan, :) = filtfilt(b_spk, a_spk, temp);
end
%% Plotting
figure;
hold on;
box off;

N = size(neural_data, 2);
time_ms = (0:N-1) / fs * 1000;  % convert to milliseconds

plot(time_ms, stim_data_scaled(STIM_CHANS(1), :), 'k', 'LineWidth', 1.5);  % Stim channel
plot(time_ms, neural_data(100, :), 'r', 'LineWidth', 1.5);                  % Raw neural
plot(time_ms, artifact_removed_data(100, :)', 'b', 'LineWidth', 1.5);      % Cleaned neural
% plot(time_ms, spike_data(100, :)', 'g', 'LineWidth', 1.5);               % Optional spike band

xlabel('Time (ms)');
ylabel('Signal Amplitude (µV)');
legend({'Stim Channel', 'Raw Neural', 'Cleaned Neural'}, 'Location', 'best');
title('Artifact Removal Comparison');
set(gca, 'TickDir', 'out');

%% Save figure and run for each session 
% Save artifact_removed_data to outputfolder (data)
data_filename = fullfile(outputfolder, ['artifact_removed_' trial '.mat']);
save(data_filename, 'artifact_removed_data', 'lfp_data', 'spike_data', '-v7.3');

% Save figure to figure folder
fig_base = fullfile(fig_folder, ['artifact_removal_plot_' trial '_full']);
saveas(gcf, [fig_base, '.png']);     % PNG for quick viewing
savefig([fig_base, '.fig']);         % Editable MATLAB .fig file
