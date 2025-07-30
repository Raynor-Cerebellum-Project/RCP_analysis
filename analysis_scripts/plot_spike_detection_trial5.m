clear all; close all; clc;
addpath(genpath(fullfile('..', 'functions')));

%% Parameters
fs = 30000;
fs_ds = 1000;  % Downsampled FR sampling rate

%% Locate Trial
session = 'BL_RW_003_Session_1';
template_mode = 'pca';
trial_num = 3;
pulse_num = 2;
ref_ch = 21;
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);
intan_folder = fullfile(base_folder, 'Intan');

trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
if numel(trial_dirs) < 3
    error('Less than 3 trials found.');
end

valid_modes = {'local', 'pca'};
if ~ismember(template_mode, valid_modes)
    error('Invalid template mode. Choose either ''local'' or ''pca''.');
end

trial_path = fullfile(intan_folder, trial_dirs(trial_num).name);
artifact_file = fullfile(trial_path, sprintf('neural_data_artifact_removed_%s.mat', template_mode));
firing_file   = fullfile(trial_path, 'firing_rate_data.mat');
trig_file     = fullfile(trial_path, 'trig_info.mat');
raw_file      = fullfile(trial_path, 'neural_data.mat');
stim_drift_file = fullfile(trial_path, 'stim_drift_all.mat');

fig_folder = fullfile(base_folder, 'Figures/spikeDetection/Neural');
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end

load(artifact_file, 'artifact_removed_data');
load(firing_file, 'spike_trains_all', 'smoothed_fr_all');
load(trig_file, 'trigs', 'repeat_boundaries');
load(raw_file, 'neural_data');

[nChans, nSamples] = size(artifact_removed_data);
t = (0:nSamples-1) / fs;
t_fr = (0:length(smoothed_fr_all{1}) - 1) / fs_ds;

if isfile(stim_drift_file)
    load(stim_drift_file, 'stim_drift_all');
    use_drift = true;

    % Preallocate full drift matrix
    drift_full = nan(nChans, nSamples);
    for ch = 1:min(nChans, numel(stim_drift_all))
        drift_ch_full = zeros(1, nSamples);
        drift_structs = stim_drift_all{ch};
        for k = 1:numel(drift_structs)
            idx_range = round(drift_structs(k).range);  % [start, end]
            vals = double(drift_structs(k).values(:)');
            if ~isempty(vals) && all(idx_range > 0) && idx_range(2) <= nSamples
                drift_ch_full(idx_range(1):idx_range(2)) = vals;
            end
        end
        drift_full(ch, :) = drift_ch_full;
    end
    fprintf('  Reconstructed drift signals for overlay.\n');
else
    warning('  stim_drift_all.mat not found. Skipping template subtraction overlay.');
    use_drift = false;
    drift_full = [];
end
%% Identify stim boundaries
stim_neural_delay = 0;
buffer = round(fs * 0.01);
trigs_beg = trigs(:, 1);
trigs_end = trigs(:, 2) + stim_neural_delay;
trigs_beg_sec = trigs_beg / fs;
trigs_end_sec = trigs_end / fs;

%% === Figure 1: Per-channel raw/corrected + smoothed FR ===
t_stim_start = trigs_beg_sec(repeat_boundaries(pulse_num) + 1);
t_window = [-0.8, 1.2];
t_start = t_stim_start + t_window(1);
t_end   = t_stim_start + t_window(2);

figure('Name', 'Spike Detection: Trial 5 — Raw vs Corrected + Smoothed FR', ...
    'Position', [100 100 1800 1000]);
tiledlayout(5, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
ax = gobjects(5, 2);

for ch = 1:5
    row_idx = ch;

    % === Left Plot (Span columns 1–2 of this row) ===
    ax(ch, 1) = nexttile((row_idx - 1) * 3 + 1, [1 2]);
    % Downsampled or trimmed window (optional)
    t_idx = (t >= t_start) & (t <= t_end);
    t_trim = t(t_idx);
    raw_trim = double(neural_data(ch, t_idx));
    corrected_trim = double(artifact_removed_data(ch, t_idx));

    h1 = plot(t_trim, raw_trim, 'Color', [0.6 0.6 0.6]); hold on;
    h2 = plot(t_trim, corrected_trim, 'k');


    if use_drift && ~isempty(drift_full)
        drift_trim = drift_full(ch, t_idx);
        template_trim = corrected_trim - (raw_trim - drift_trim);

        h3 = plot(t_trim, drift_trim, 'Color', [0 0.4470 0.7410], 'LineWidth', 0.5);
        h4 = plot(t_trim, template_trim, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 0.5);
    end

    spike_locs = find(spike_trains_all{ch} > 0);
    spike_times = t(spike_locs);
    spike_vals = artifact_removed_data(ch, spike_locs);

    in_range = spike_times >= t_start & spike_times <= t_end;
    h5 = scatter(spike_times(in_range), spike_vals(in_range), 10, 'r', 'filled');

    ylim([-200 200]);
    xlim([t_start, t_end]);
    shade_stim_blocks(repeat_boundaries, trigs_beg_sec, trigs_end_sec, ...
        t_stim_start, diff(t_window), ylim);

    ylabel('Voltage');
    title(sprintf('Ch %d — Raw, Corrected, Drift, Template', ch));
    set(ax(ch, 1), 'Box', 'off');

    if ch == 5
        xlabel('Time (s)');
        legend([h1, h2, h3, h4, h5], ...
            {'Raw', 'Corrected', 'Drift', 'Template', 'Spikes'}, ...
            'Orientation', 'horizontal', 'Location', 'southoutside', ...
            'Box', 'off', 'FontSize', 8);
    end

    % === Right Plot (Column 3 of this row) ===F
    ax(ch, 2) = nexttile((row_idx - 1) * 3 + 3);
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
    set(ax(ch, 2), 'Box', 'off');
    if ch == 5, xlabel('Time (s)'); end
end

linkaxes(ax(:, 1), 'x');
linkaxes(ax(:, 2), 'x');
sgtitle(sprintf('Raw/Corrected Traces and Smoothed FR — Trial %d (%s)', ...
    trial_num, trial_dirs(trial_num).name));
% === Save Figure 1 ===
fig1 = figure(1);  % assuming this is Figure 1
fig1_name = sprintf('%s_Trial%d_SpikeOverlay_%s', session, trial_num, template_mode);
saveas(fig1, fullfile(fig_folder, 'png', [fig1_name '.png']));
%savefig(fig1, fullfile(fig_folder, 'fig', [fig1_name '.fig']), 'compact');
%% === Figure 2: Full Spike Raster and FR Heatmap (Aligned to Stim Block) ===
t_lim = [t_start, t_end];  % Match window from Figure 1
fr_idx = (t_fr >= t_lim(1)) & (t_fr <= t_lim(2));
t_fr_window = t_fr(fr_idx);
fr_mat = nan(nChans, numel(t_fr_window));
% Define baseline window in seconds
baseline_window = [t_stim_start - 0.5, t_stim_start];  % e.g., 0.5 sec before stim

% Get indices for baseline window
baseline_idx = (t_fr >= baseline_window(1)) & (t_fr <= baseline_window(2));

% Normalize each channel's FR by its baseline mean
fr_mat_norm = fr_mat;
for ch = 1:nChans
    baseline_mean = mean(smoothed_fr_all{ch}(baseline_idx));
    if baseline_mean > 0
        fr_mat_norm(ch, :) = fr_mat(ch, :) / baseline_mean;
    end
end
% 
% for ch = 1:nChans
%     fr_mat(ch, :) = smoothed_fr_all{ch}(fr_idx);
% end

figure('Name', 'Full Spike Raster and FR Heatmap', 'Position', [100 100 1600 900]);
tiledlayout(2,1, 'Padding', 'compact', 'TileSpacing', 'compact');

% --- Raster Plot ---
nexttile(1); hold on; box off;
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
title(sprintf('Spike Raster — Stim Aligned Window (%.3f to %.3f s) — Trial %d', t_lim(1), t_lim(2), trial_num));
set(gca, 'YDir', 'reverse');

% --- Heatmap Plot ---
nexttile(2); box off;
imagesc(t_fr_window, 1:nChans, fr_mat_norm);
set(gca, 'YDir', 'reverse');
xlabel('Time (s)');
ylabel('Channel');
title('Normalized Smoothed Firing Rate (Baseline = 1)');
colormap jet;
colorbar;
xlim(t_lim);

% === Save Figure 2 ===
fig2 = figure(2);  % assuming this is Figure 2
fig2_name = sprintf('%s_Trial%d_SpikeRaster_FR_%s', session, trial_num, template_mode);
saveas(fig2, fullfile(fig_folder, 'png', [fig2_name '.png']));
savefig(fig2, fullfile(fig_folder, 'fig', [fig2_name '.fig']));

fprintf('Saved .png and .fig versions to: %s\n', fig_folder);