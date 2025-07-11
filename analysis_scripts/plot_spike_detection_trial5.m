clear all; close all; clc;
addpath(genpath(fullfile('..', 'functions')));

%% Parameters
fs = 30000;
filt_range = [300 6000];
[b, a] = butter(3, filt_range / (fs/2), 'bandpass');
threshold_std = -3;
refractory_ms = 1;

%% Locate Trial 5
session = 'BL_RW_003_Session_1';
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);
intan_folder = fullfile(base_folder, 'Intan');

trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
if numel(trial_dirs) < 5
    error('Less than 5 trials found.');
end

trial_path = fullfile(intan_folder, trial_dirs(5).name);
artifact_file = fullfile(trial_path, 'neural_data_artifact_removed.mat');
firing_file   = fullfile(trial_path, 'firing_rate_data.mat');

load(artifact_file, 'artifact_removed_data', 'trigs');
load(firing_file, 'spike_trains_all');

neural_data = artifact_removed_data;
[nChans, nSamples] = size(neural_data);
t = (0:nSamples-1) / fs;
%% Identify stim boundaries
stim_neural_delay = 0;  % in samples, e.g. fs*0.001 if 1 ms delay
buffer = round(fs * 0.01);  % 10 ms buffer

trigs_beg = trigs(:, 1);
trigs_end = trigs(:, 2) + stim_neural_delay;

% Convert to seconds
trigs_beg_sec = trigs_beg / fs;
trigs_end_sec = trigs_end / fs;

% Identify block boundaries
time_diffs = diff(trigs_beg);
repeat_gap_threshold = 2 * (2 * buffer + 1);  % this should be in samples
repeat_boundaries = [0; find(time_diffs > repeat_gap_threshold); numel(trigs_beg)];
num_repeats = numel(repeat_boundaries) - 1;

%% Spike Raster
figure('Name', 'Spike Detection: Trial 5 — Zoom on 2nd Spike', 'Position', [100 100 1200 800]);
zoom_win = 20;  % 0.5s window around 2nd spike

for ch = 1:5
    raw = double(neural_data(ch, :));
    filtered = filtfilt(b, a, raw);

    thresh = threshold_std * std(filtered);
    spike_idx = find(filtered < thresh);
    isi = diff(spike_idx);
    spike_idx = spike_idx([true, isi > fs * refractory_ms / 1000]);

    subplot(5, 1, mod(ch-1, 5)+1);
    plot(t, filtered, 'k'); hold on;
    scatter(t(spike_idx), filtered(spike_idx), 10, 'r', 'filled');
    yline(thresh, '--b', sprintf('Thresh = %.2f', thresh));

    if numel(spike_idx) >= 2
        t_center = t(spike_idx(2));
        % Shade stim blocks
        y_limits = ylim;
        shade_stim_blocks(repeat_boundaries, trigs_beg_sec, trigs_end_sec, t_center, zoom_win, y_limits);

        xlim([t_center - zoom_win/2, t_center + zoom_win/2]);
        title(sprintf('Channel %d — Zoom on 2nd Spike (%.3f s)', ch, t_center));
    else
        xlim([0, 1]);  % fallback if <2 spikes
        title(sprintf('Channel %d — <2 spikes detected', ch));
    end

    xlabel('Time (s)');
    ylabel('Filtered');
end

sgtitle(sprintf('Spike Detection on Trial 5 — %s', trial_dirs(5).name));
% === Raster Plot Around 2nd Spike in Channel 1 ===
figure('Name', 'Spike Raster Around 2nd Spike', 'Position', [100 100 1400 900]); hold on;

% Plot spike rasters
for ch = 1:nChans
    raw = double(neural_data(ch, :));
    filtered = filtfilt(b, a, raw);

    thresh = threshold_std * std(filtered);
    spike_idx = find(filtered < thresh);
    isi = diff(spike_idx);
    spike_idx = spike_idx([true, isi > fs * refractory_ms / 1000]);

    t_spikes = t(spike_idx);
    in_window = t_spikes >= (t_center - zoom_win) & t_spikes <= (t_center + zoom_win);
    t_spikes_window = t_spikes(in_window);
    y_ch = ch * ones(size(t_spikes_window));

    scatter(t_spikes_window, y_ch, 3, 'r', 'filled');
end

% Set axis limits first
xlim([t_center - zoom_win, t_center + zoom_win]);
ylim([1, nChans]);
y_limits = ylim;

% Shade stim blocks using actual y-axis limits
shade_stim_blocks(repeat_boundaries, trigs_beg_sec, trigs_end_sec, t_center, zoom_win, y_limits);

xlabel('Time (s)');
ylabel('Channel');
title(sprintf('Spike Raster — ±%.0f ms Around 2nd Spike (%.3f s) on Ch %d', ...
    zoom_win * 1000, t_center, ref_ch));
set(gca, 'YDir', 'reverse');  % optional flip


