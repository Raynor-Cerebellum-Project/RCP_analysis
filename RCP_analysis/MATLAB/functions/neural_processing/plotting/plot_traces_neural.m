function fig = plot_traces_neural(base_data, cond_data, side_label, offset, meta_cond, use_ci, show_figs)
if nargin < 8
    use_ci = true;
end

trace_spacing = 20 * offset;
t = linspace(-800, 1200, 2001);

is_ipsi = contains(side_label, 'ipsi');
suffix = 'ipsi';
if ~is_ipsi
    suffix = 'contra';
end

% === Extract delay info and determine baseline label ===
raw_delay = string(meta_cond.Stim_Delay);
use_random_baseline = false;

if strcmpi(raw_delay, "Random")
    % parse delay from side_label suffix e.g. ipsi_100, contra_nan, ipsi, ipsi_0
    tokens = regexp(side_label, '^(ipsi|contra)(?:_(\d+|nan))?$', 'tokens', 'ignorecase');
    if ~isempty(tokens)
        suffix_str = '';
        if numel(tokens{1}) >= 2
            suffix_str = tokens{1}{2};
        end
        if isempty(suffix_str) || strcmp(suffix_str, '0')
            % bare 'ipsi'/'contra' or explicit '_0' — no delay
            delay = 0;
            delay_str = '0ms';
        elseif strcmpi(suffix_str, 'nan')
            delay = NaN;
            delay_str = 'NaNms';
        else
            delay = str2double(suffix_str);
            delay_str = sprintf('%dms', delay);
        end
    else
        delay = NaN;
        delay_str = 'NaNms';
    end
    % use random baseline if available
    if contains(side_label, 'ipsi') && isfield(base_data, 'ipsi_nan')
        base_side_label = 'ipsi_nan';
        use_random_baseline = true;
    elseif contains(side_label, 'contra') && isfield(base_data, 'contra_nan')
        base_side_label = 'contra_nan';
        use_random_baseline = true;
    end
else
    delay = str2double(raw_delay);
    if isnan(delay)
        delay = 0;
        delay_str = 'NaNms';
        warning('Unrecognized Stim_Delay value: %s', raw_delay);
    else
        delay_str = sprintf('%dms', delay);
    end
end

% Fallback
if ~use_random_baseline
    base_side_label = side_label;
end

base_summary_field = [base_side_label '_summary'];
cond_summary_field = [side_label '_summary'];

% Baseline
baseline_segment = base_data.(base_side_label).segments3;

% Condition
if isfield(cond_data.(side_label), 'segments3_from_stim') && ...
        ~all(isnan(cond_data.(side_label).segments3_from_stim), 'all')
    condition_segment = cond_data.(side_label).segments3_from_stim;
else
    condition_segment = cond_data.(side_label).segments3;
end

if strcmp(meta_cond.Movement_Trigger{1}, 'End')
    baseline_window = [baseline_segment(:,2), baseline_segment(:,2)];
    condition_window = [condition_segment(:,2), condition_segment(:,2)];
else
    baseline_window = [baseline_segment(:,1), baseline_segment(:,1)];
    condition_window = [condition_segment(:,1), condition_segment(:,1)];
end

% Setup figure
fig = figure('Visible', ternary(show_figs, 'on', 'off'), 'Position', [100, 100, 1200, 800]);
layout = tiledlayout(4, 6, 'TileSpacing', 'compact', 'Padding', 'tight');

%% --- Baseline Velocity ---
ax_vel_baseline = nexttile(1, [1 2]); hold on;
title('Baseline Velocity'); ylabel('Velocity (deg/s)');
box off; set(gca, 'TickDir', 'out');
max_v = -inf; min_v = inf;

for i = 1:size(baseline_window,1)
    vel = base_data.(base_side_label).velocity_traces(i, :);
    stacked = vel + i * trace_spacing;
    plot(t, stacked, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.75);
    max_v = max(max_v, max(stacked));
    min_v = min(min_v, min(stacked));
end
xlim([-800 1200]); ylim([min_v-10 max_v+10]);

vel_mat = nan(size(baseline_window,1), 2001);
for i = 1:size(baseline_window,1)
    vel_mat(i, :) = base_data.(base_side_label).velocity_traces(i, :);
end
mean_trace = nanmean(vel_mat, 1);
plot(t, mean_trace + (size(baseline_window,1)+1) * trace_spacing, 'k-', 'LineWidth', 2);

%% --- Condition Velocity ---
ax_vel_condition = nexttile(3, [1 2]); hold on;
title('Condition Velocity');
box off; set(gca, 'TickDir', 'out');
max_v = -inf; min_v = inf;

for i = 1:size(condition_window,1)
    vel = cond_data.(side_label).velocity_traces(i, :);
    stacked = vel + i * trace_spacing;
    plot(t, stacked, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.75);
    max_v = max(max_v, max(stacked));
    min_v = min(min_v, min(stacked));
end
xlim([-800 1200]); ylim([min_v-10 max_v+10]);

vel_mat = nan(size(condition_window,1), 2001);
for i = 1:size(condition_window,1)
    vel_mat(i, :) = cond_data.(side_label).velocity_traces(i, :);
end
mean_trace = nanmean(vel_mat, 1);
plot(t, mean_trace + (size(condition_window,1)+1) * trace_spacing, 'k-', 'LineWidth', 2);

%% --- Overlay: Baseline vs Condition Velocity ---
ax_vel_overlay = nexttile(5, [1 2]); hold on;
title('Overlay Velocity'); ylabel('Velocity (deg/s)');
box off; set(gca, 'TickDir', 'out');

baseline_vel_mat = nan(size(baseline_window,1), 2001);
for i = 1:size(baseline_window,1)
    baseline_vel_mat(i,:) = base_data.(base_side_label).velocity_traces(i, :);
end

if use_ci
    STDplot(t, baseline_vel_mat, [0 0 0]);
    STDplot(t, vel_mat, [0 0.2 1]);
else
    for i = 1:size(baseline_vel_mat,1)
        plot(t, baseline_vel_mat(i,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.75);
    end
    for i = 1:size(vel_mat,1)
        plot(t, vel_mat(i,:), 'Color', [0.4 0.6 1], 'LineWidth', 0.75);
    end
    plot(t, nanmean(baseline_vel_mat,1), 'k-', 'LineWidth', 2);
    plot(t, nanmean(vel_mat, 1), 'Color', [0 0.2 1], 'LineWidth', 2);
end

xlim([-800 1200]);

duration = meta_cond.Stim_Duration_ms;
linkaxes([ax_vel_baseline, ax_vel_condition, ax_vel_overlay], 'y');
shared_yl = ylim(ax_vel_overlay);

% Baseline: dotted black line only
xline(ax_vel_baseline, delay, 'k--', 'LineWidth', 1.2);

% Condition and overlay: red rectangle + dotted black line
for ax = [ax_vel_condition, ax_vel_overlay]
    hold(ax, 'on');
    fill(ax, ...
        [delay delay+duration delay+duration delay], ...
        [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)], ...
        [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    xline(ax, delay, 'k--', 'LineWidth', 1.2);
end

%% --- Baseline Position ---
ax_pos_baseline = nexttile(7, [1 2]); hold on;
title('Baseline Position'); xlabel('Time (ms)'); ylabel('Position (deg)');
box off; set(gca, 'TickDir', 'out');
max_p = -inf; min_p = inf;

for i = 1:size(baseline_window,1)
    pos = base_data.(base_side_label).position_traces(i, :);
    stacked = pos + i * trace_spacing;
    plot(t, stacked, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.75);
    max_p = max(max_p, max(stacked));
    min_p = min(min_p, min(stacked));
end
xlim([-800 1200]); ylim([min_p-10 max_p+10]);

pos_mat = nan(size(baseline_window,1), 2001);
for i = 1:size(baseline_window,1)
    pos_mat(i, :) = base_data.(base_side_label).position_traces(i, :);
end
mean_trace = nanmean(pos_mat, 1);
plot(t, mean_trace + (size(baseline_window,1)+1) * trace_spacing, 'k-', 'LineWidth', 2);

%% --- Condition Position ---
ax_pos_condition = nexttile(9, [1 2]); hold on;
title('Condition Position'); xlabel('Time (ms)');
box off; set(gca, 'TickDir', 'out');
max_p = -inf; min_p = inf;

for i = 1:size(condition_window,1)
    pos = cond_data.(side_label).position_traces(i, :);
    stacked = pos + i * trace_spacing;
    plot(t, stacked, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.75);
    max_p = max(max_p, max(stacked));
    min_p = min(min_p, min(stacked));
end
xlim([-800 1200]); ylim([min_p-10 max_p+10]);

pos_mat = nan(size(condition_window,1), 2001);
for i = 1:size(condition_window,1)
    pos_mat(i, :) = cond_data.(side_label).position_traces(i, :);
end
mean_trace = nanmean(pos_mat, 1);
plot(t, mean_trace + (size(condition_window,1)+1) * trace_spacing, 'k-', 'LineWidth', 2);

%% --- Overlay: Baseline vs Condition Position ---
ax_pos_overlay = nexttile(11, [1 2]); hold on;
title('Overlay Position'); xlabel('Time (ms)'); ylabel('Position (deg)');
box off; set(gca, 'TickDir', 'out');

baseline_pos_mat = nan(size(baseline_window,1), 2001);
for i = 1:size(baseline_window,1)
    baseline_pos_mat(i,:) = base_data.(base_side_label).position_traces(i, :);
end

if use_ci
    STDplot(t, baseline_pos_mat, [0 0 0]);
    STDplot(t, pos_mat, [0 0.2 1]);
else
    for i = 1:size(baseline_pos_mat,1)
        plot(t, baseline_pos_mat(i,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.75);
    end
    for i = 1:size(pos_mat,1)
        plot(t, pos_mat(i,:), 'Color', [0.4 0.6 1], 'LineWidth', 0.75);
    end
    plot(t, nanmean(baseline_pos_mat,1), 'k-', 'LineWidth', 2);
    plot(t, nanmean(pos_mat, 1), 'Color', [0 0.2 1], 'LineWidth', 2);
end

xlim([-800 1200]);
linkaxes([ax_pos_baseline, ax_pos_condition, ax_pos_overlay], 'y');
shared_yl = ylim(ax_pos_overlay);

% Baseline: dotted black line only
xline(ax_pos_baseline, delay, 'k--', 'LineWidth', 1.2);

% Condition and overlay: red rectangle + dotted black line
for ax = [ax_pos_condition, ax_pos_overlay]
    hold(ax, 'on');
    fill(ax, ...
         [delay delay+duration delay+duration delay], ...
         [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)], ...
         [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    xline(ax, delay, 'k--', 'LineWidth', 1.2);
end

% Title
n_base = size(baseline_segment, 1);
n_cond = size(condition_segment, 1);

stim_str = sprintf('Ch: %g, Freq: %gHz, Curr: %gμA, Dur: %gms, Delay: %s, Depth: %gmm', ...
    meta_cond.Channels, meta_cond.Stim_Frequency_Hz, meta_cond.Current_uA, ...
    meta_cond.Stim_Duration_ms, delay_str, meta_cond.Depth_mm);

trial_num = meta_cond.BR_File;
sgtitle({sprintf('Condition %03d - %s  |  Base n=%d, Cond n=%d', trial_num, suffix, n_base, n_cond), stim_str}, ...
    'FontWeight', 'bold');

%% === Compute Z-Scored Rasters ===
if ~isfield(base_data.(base_summary_field), 'fr_mean') || ...
   ~isfield(cond_data.(cond_summary_field), 'fr_mean')
    return;
end
baseline_mean = mean(base_data.(base_summary_field).fr_mean, 2);
baseline_std  = std(base_data.(base_summary_field).fr_mean, 0, 2);

cond_mean = mean(cond_data.(cond_summary_field).fr_mean, 2);
cond_std  = std(cond_data.(cond_summary_field).fr_mean, 0, 2);

baseline_std(baseline_std == 0) = 1e-6;
cond_std(cond_std == 0) = 1e-6;

fr_base_zscore = (base_data.(base_summary_field).fr_mean - baseline_mean) ./ baseline_std;
fr_cond_zscore = (cond_data.(cond_summary_field).fr_mean - cond_mean) ./ cond_std;
fr_zdiff = fr_cond_zscore - fr_base_zscore;

%% === Z-Scored Baseline Raster ===
ax_fr_baseline = nexttile(13, [2 2]); hold on;
imagesc(t, 1:128, fr_base_zscore);
title('Baseline Firing Rate (Z-scored)');
xlabel('Time (ms)'); ylabel('Channel');
colormap(ax_fr_baseline, 'parula');
colorbar; caxis([-3 3]);
box off; set(gca, 'YDir', 'normal', 'TickDir', 'out');
xlim([-800 1200]); ylim([0 128]);

%% === Z-Scored Condition Raster ===
ax_fr_condition = nexttile(15, [2 2]); hold on;
imagesc(t, 1:128, fr_cond_zscore);
title('Condition Firing Rate (Z-scored)');
xlabel('Time (ms)'); ylabel('Channel');
colormap(ax_fr_condition, 'parula');
colorbar; caxis([-3 3]);
box off; set(gca, 'YDir', 'normal', 'TickDir', 'out');
xlim([-800 1200]); ylim([0 128]);

%% === Z-Score Difference Raster ===
ax_fr_diff = nexttile(17, [2 2]); hold on;
imagesc(t, 1:128, fr_zdiff);
title('Firing Rate Z-Diff (Cond − Base)');
xlabel('Time (ms)'); ylabel('Channel');
colormap(ax_fr_diff, 'redbluecmap');
colorbar; caxis([-3 3]);
box off; set(gca, 'YDir', 'normal', 'TickDir', 'out');
xlim([-800 1200]); ylim([0 128]);

end

function out = ternary(cond, valTrue, valFalse)
    if cond
        out = valTrue;
    else
        out = valFalse;
    end
end