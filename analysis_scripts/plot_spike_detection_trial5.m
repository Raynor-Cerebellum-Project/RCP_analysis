clear all; close all; clc;
addpath(genpath(fullfile('..', 'functions')));

%% Parameters
fs = 30000;
fs_ds = 1000;  % Downsampled FR sampling rate

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
trig_file     = fullfile(trial_path, 'trig_info.mat');
raw_file      = fullfile(trial_path, 'neural_data.mat');

load(artifact_file, 'artifact_removed_data');
load(firing_file, 'spike_trains_all', 'smoothed_fr_all');
load(trig_file, 'trigs', 'repeat_boundaries');
load(raw_file, 'neural_data');

[nChans, nSamples] = size(artifact_removed_data);
t = (0:nSamples-1) / fs;
t_fr = (0:length(smoothed_fr_all{1}) - 1) / fs_ds;

%% Identify stim boundaries
stim_neural_delay = 0;
buffer = round(fs * 0.01);
trigs_beg = trigs(:, 1);
trigs_end = trigs(:, 2) + stim_neural_delay;
trigs_beg_sec = trigs_beg / fs;
trigs_end_sec = trigs_end / fs;

%% === Figure 1: Per-channel raw/corrected + smoothed FR ===
ref_ch = 1;
t_stim_start = trigs_beg_sec(repeat_boundaries(1) + 1);
t_window = [-0.8, 1.2];
t_start = t_stim_start + t_window(1);
t_end   = t_stim_start + t_window(2);

figure('Name', 'Spike Detection: Trial 5 — Raw vs Corrected + Smoothed FR', ...
       'Position', [100 100 1800 1000]);

tiledlayout(5, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
ax = gobjects(5, 2);

for ch = 1:5
    % --- Raw and Corrected Plot ---
    ax(ch, 1) = nexttile((ch-1)*2 + 1);
    raw_ch = double(neural_data(ch, :));
    corrected_ch = double(artifact_removed_data(ch, :));

    plot(t, raw_ch, 'Color', [0.6 0.6 0.6]); hold on;
    plot(t, corrected_ch, 'k');

    spike_idx = find(spike_trains_all{ch});
    scatter(t(spike_idx), corrected_ch(spike_idx), 10, 'r', 'filled');

    ylim([-200 200]);
    shade_stim_blocks(repeat_boundaries, trigs_beg_sec, trigs_end_sec, ...
                      t_stim_start, diff(t_window), [-200 200]);
    xlim([t_start, t_end]);

    ylabel('Voltage');
    title(sprintf('Ch %d — Raw vs Corrected', ch));
    if ch == 5
        xlabel('Time (s)');
    end

    % --- Smoothed FR Plot ---
    ax(ch, 2) = nexttile((ch-1)*2 + 2);
    t_idx = (t_fr >= t_start) & (t_fr <= t_end);
    t_plot = t_fr(t_idx);
    fr_plot = smoothed_fr_all{ch}(t_idx);

    plot(t_plot, fr_plot, 'b', 'LineWidth', 1.2); hold on;
    yline(mean(smoothed_fr_all{ch}), '--', 'Mean', 'Color', [0.4 0.4 1]);

    ylim([0, max(smoothed_fr_all{ch}) * 1.1]);
    xlim([t_start, t_end]);
    shade_stim_blocks(repeat_boundaries, trigs_beg_sec, trigs_end_sec, ...
                      t_stim_start, diff(t_window), ylim);

    ylabel('FR (Hz)');
    title(sprintf('Ch %d — Smoothed FR', ch));
    if ch == 5
        xlabel('Time (s)');
    end
end

linkaxes(ax(:, 1), 'x');
linkaxes(ax(:, 2), 'x');
sgtitle(sprintf('Raw/Corrected Traces and Smoothed FR — Trial 5 (%s)', trial_dirs(5).name));

%% === Figure 2: Full Spike Raster and FR Heatmap (Aligned to Stim Block) ===
t_lim = [t_start, t_end];  % Match window from Figure 1
fr_idx = (t_fr >= t_lim(1)) & (t_fr <= t_lim(2));
t_fr_window = t_fr(fr_idx);
fr_mat = nan(nChans, numel(t_fr_window));

for ch = 1:nChans
    fr_mat(ch, :) = smoothed_fr_all{ch}(fr_idx);
end

figure('Name', 'Full Spike Raster and FR Heatmap', 'Position', [100 100 1600 900]);
tiledlayout(2,1, 'Padding', 'compact', 'TileSpacing', 'compact');

% --- Raster Plot ---
nexttile(1); hold on;
for ch = 1:nChans
    spike_idx = find(spike_trains_all{ch});
    t_spikes = spike_idx / fs;
    in_window = (t_spikes >= t_lim(1)) & (t_spikes <= t_lim(2));
    scatter(t_spikes(in_window), ch * ones(nnz(in_window), 1), 3, 'r', 'filled');
end

xlim(t_lim);
ylim([1, nChans]);
shade_stim_blocks(repeat_boundaries, trigs_beg_sec, trigs_end_sec, ...
                  t_stim_start, diff(t_window), ylim);
xlabel('Time (s)');
ylabel('Channel');
title(sprintf('Spike Raster — Stim Aligned Window (%.3f to %.3f s)', t_lim(1), t_lim(2)));
set(gca, 'YDir', 'reverse');

% --- Heatmap Plot ---
nexttile(2);
imagesc(t_fr_window, 1:nChans, fr_mat);
set(gca, 'YDir', 'reverse');
xlabel('Time (s)');
ylabel('Channel');
title('Smoothed Firing Rate (Hz)');
colormap jet;
colorbar;
xlim(t_lim);
