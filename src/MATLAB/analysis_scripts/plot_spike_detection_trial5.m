clear all; close all; clc;
addpath(genpath(fullfile('..', 'functions')));

%% Parameters
fs = 30000;
fs_ds = 1000;  % Downsampled FR sampling rate

%% Locate Trial
session = 'BL_RW_003_Session_1';
template_modes = {'local', 'pca'};  % or just {'local'}

trial_num = 3;
pulse_num = 2;
ref_ch = 1;
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);
intan_folder = fullfile(base_folder, 'Intan');
raw_metrics_path   = fullfile(base_folder, 'Checkpoints', [session, '_raw_metrics_all.mat']);
%% Load data
artifact_data = struct();
fr_data = struct();

trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
if numel(trial_dirs) < 3
    error('Less than 3 trials found.');
end

valid_modes = {'local', 'pca'};
if any(~ismember(template_modes, valid_modes))
    error('Invalid template_mode. Choose from: ''local'', ''pca''.');
end

trial_path = fullfile(intan_folder, trial_dirs(trial_num).name);
for m = 1:numel(template_modes)
    mode = template_modes{m};
    art_file = fullfile(trial_path, sprintf('neural_data_artifact_removed_%s.mat', mode));
    fr_file  = fullfile(trial_path, sprintf('firing_rate_data_%s.mat', mode));
    tmp_art = load(art_file, 'artifact_removed_data');
    tmp_fr  = load(fr_file, 'smoothed_fr_all', 'spike_trains_all');
    artifact_data.(mode) = tmp_art.artifact_removed_data;
    fr_data.(mode) = tmp_fr.smoothed_fr_all;
    spike_data.(mode) = tmp_fr.spike_trains_all;
end

trig_file     = fullfile(trial_path, 'trig_info.mat');
raw_file      = fullfile(trial_path, 'neural_data.mat');
stim_drift_file = fullfile(trial_path, 'stim_drift_all.mat');

fig_folder = fullfile(base_folder, 'Figures/Neural/spikeDetection');
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end

load(trig_file, 'trigs', 'repeat_boundaries');
load(raw_file, 'neural_data');
behv_file = load(raw_metrics_path, 'raw_metrics_all');
raw_metrics_all = behv_file.raw_metrics_all;

ref_mode = template_modes{1};
[nChans, nSamples] = size(artifact_data.(ref_mode));
t_fr = (0:length(fr_data.(ref_mode){1}) - 1) / fs_ds;
t = (0:nSamples-1) / fs;

% === Use full drift trace directly ===
if isfile(stim_drift_file)
    load(stim_drift_file, 'stim_drift_all');
    use_drift = true;

    nChans = min(nChans, numel(stim_drift_all));
    drift_full = nan(nChans, nSamples);

    for ch = 1:nChans
        if ~isempty(stim_drift_all{ch}) && isfield(stim_drift_all{ch}, 'values')
            vals = stim_drift_all{ch}.values;
            if numel(vals) == nSamples
                drift_full(ch, :) = double(vals(:)');
            else
                warning('Drift vector for ch %d does not match signal length. Skipping.', ch);
            end
        end
    end
    fprintf('  Loaded full-length drift traces for overlay.\n');
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
    h1 = plot(t_trim, raw_trim, 'Color', [0.6 0.6 0.6]); hold on;

    color_map = lines(numel(template_modes));
    % --- Plot corrected traces and compute template/drift per mode ---
    h_corrected = gobjects(numel(template_modes), 1);
    h_template = gobjects(numel(template_modes), 1);

    drift_trim = [];
    if use_drift && ~isempty(drift_full)
        drift_trim = drift_full(ch, t_idx);
        h_drift = plot(t_trim, drift_trim, '--', 'Color', [0.2 0.2 0.8], 'LineWidth', 0.8);
    else
        h_drift = gobjects(1);
    end

    for m = 1:numel(template_modes)
        mode = template_modes{m};
        corrected_trim = double(artifact_data.(mode)(ch, t_idx));
        h_corrected(m) = plot(t_trim, corrected_trim, ...
            'Color', color_map(m, :), 'LineWidth', 1.2);

        % if ~isempty(drift_trim)
        %     template_trim = corrected_trim - (raw_trim - drift_trim);
        %     h_template(m) = plot(t_trim, template_trim, ':', 'Color', color_map(m, :), 'LineWidth', 1.0);
        % else
            h_template(m) = gobjects(1);
        % end
    end


    h_spikes = gobjects(numel(template_modes), 1);
    spike_colors = [0.9 0.1 0.1; 0.1 0.5 0.1];  % red, green, etc.

    for m = 1:numel(template_modes)
        mode = template_modes{m};
        spike_locs = find(spike_data.(mode){ch} > 0);
        spike_times = t(spike_locs);
        spike_vals = artifact_data.(mode)(ch, spike_locs);
        in_range = spike_times >= t_start & spike_times <= t_end;

        h_spikes(m) = scatter(spike_times(in_range), spike_vals(in_range), ...
            10, spike_colors(m, :), 'filled');
    end

    ylim([-200 200]);
    xlim([t_start, t_end]);
    shade_stim_blocks(repeat_boundaries, trigs_beg_sec, trigs_end_sec, ...
        t_stim_start, diff(t_window), ylim);

    ylabel('Voltage (mV)');
    title(sprintf('Ch %d — Raw, Corrected, Drift, Template', ch));
    set(ax(ch, 1), 'Box', 'off');

    if ch == 5
        xlabel('Time (s)');

        % === Construct and clean legend entries ===
        legend_handles = [h1; h_corrected(:)];
        legend_labels  = [{'Raw'}, strcat('Corrected (', template_modes, ')')];
        
        if use_drift && isgraphics(h_drift)
            legend_handles = [legend_handles; h_drift];
            legend_labels  = [legend_labels, {'Drift'}];
        end
        
        % Only include spike handles that were actually plotted
        for m = 1:numel(h_spikes)
            if isgraphics(h_spikes(m))
                legend_handles(end+1) = h_spikes(m);
                legend_labels{end+1} = sprintf('Spikes (%s)', template_modes{m});
            end
        end
        
        % Plot legend
        legend(legend_handles, legend_labels, ...
            'Orientation', 'horizontal', 'Location', 'southoutside', ...
            'Box', 'off', 'FontSize', 8);

    end


    % legend([h1, h2, h3, h4, h5], ...
    %     {'Raw', 'Corrected', 'Drift', 'Template', 'Spikes'}, ...
    %     'Orientation', 'horizontal', 'Location', 'southoutside', ...
    %     'Box', 'off', 'FontSize', 8); end

    % === Right Plot (Column 3 of this row) ===
    ax(ch, 2) = nexttile((row_idx - 1) * 3 + 3);
    t_idx = (t_fr >= t_start) & (t_fr <= t_end);
    t_plot = t_fr(t_idx);
    for m = 1:numel(template_modes)
        mode = template_modes{m};
        fr_plot = fr_data.(mode){ch}(t_idx);
        plot(t_plot, fr_plot, 'Color', color_map(m, :), 'LineWidth', 1.2); hold on;
    end
    ylim([-100, 500]);
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
mode_str = strjoin(template_modes, '_vs_');
sgtitle(sprintf('Raw/Corrected Traces and Smoothed FR — Trial %d (%s)', ...
    trial_num, mode_str));
fig1_name = sprintf('%s_Trial%d_SpikeOverlay_%s', session, trial_num, mode_str);

saveas(fig1, fullfile(fig_folder, 'png', [fig1_name '.png']));
%savefig(fig1, fullfile(fig_folder, 'fig', [fig1_name '.fig']), 'compact');
%% === Figure 2: Full Spike Raster and FR Heatmap (Aligned to Stim Block) ===
t_lim = [t_start, t_end];  % Match window from Figure 1
fr_idx = (t_fr >= t_lim(1)) & (t_fr <= t_lim(2));
t_fr_window = t_fr(fr_idx);
% Define baseline window in seconds
baseline_window = [t_stim_start - 0.5, t_stim_start];  % e.g., 0.5 sec before stim
baseline_idx = (t_fr >= baseline_window(1)) & (t_fr <= baseline_window(2));

% Normalize
fr_mat_norm = nan(nChans, numel(t_fr_window));
for ch = 1:nChans
    baseline_mean = mean(fr_data.(ref_mode){ch}(baseline_idx));
    if baseline_mean > 0
        fr_mat_norm(ch, :) = fr_data.(ref_mode){ch}(fr_idx) / baseline_mean;
    end
end
%
% for ch = 1:nChans
%     fr_mat(ch, :) = smoothed_fr_all{ch}(fr_idx);
% end

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
title(sprintf('Spike Raster — Stim Aligned Window (%.3f to %.3f s) — Trial %d', t_lim(1), t_lim(2), trial_num));
set(gca, 'YDir', 'reverse');
box off;

% --- Heatmap Plot ---
nexttile(2);
imagesc(t_fr_window, 1:nChans, fr_mat_norm);
set(gca, 'YDir', 'reverse');
xlabel('Time (s)');
ylabel('Channel');
title('Normalized Smoothed Firing Rate (Baseline = 1)');
colormap jet;
colorbar;
xlim(t_lim);
xline(t_stim_start, 'r--', 'LineWidth', 1.2);
box off;

% === Save Figure 2 ===
fig2 = figure(2);  % assuming this is Figure 2
fig2_name = sprintf('%s_Trial%d_SpikeRaster_FR_%s', session, trial_num, mode_str);
saveas(fig2, fullfile(fig_folder, 'png', [fig2_name '.png']));
savefig(fig2, fullfile(fig_folder, 'fig', [fig2_name '.fig']));

fprintf('Saved .png and .fig versions to: %s\n', fig_folder);