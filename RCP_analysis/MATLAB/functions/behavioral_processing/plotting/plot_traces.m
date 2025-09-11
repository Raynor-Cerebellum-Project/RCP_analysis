function fig = plot_traces(base_data, cond_data, side_label, offset, meta_cond, use_ci, show_figs)
if nargin < 8
    use_ci = true;  % default: show individual traces
end

trace_spacing = 20 * offset;
t = linspace(-800, 1200, 2001);

is_ipsi = contains(side_label, 'pos');
suffix = 'ipsi';
if ~is_ipsi
    suffix = 'contra';
end

% === Extract delay info and determine baseline label ===
if iscell(meta_cond.Stim_Delay)
    raw_delay = meta_cond.Stim_Delay{1};
else
    raw_delay = meta_cond.Stim_Delay;
end

use_random_baseline = false;

if isnumeric(raw_delay)
    delay = raw_delay;
    delay_str = sprintf('%dms', delay);
elseif ischar(raw_delay)
    if strcmpi(raw_delay, 'Random')
        % --- Try to parse delay info from side_label ---
        tokens = regexp(side_label, 'active_like_stim_(pos|neg)_(\d+)', 'tokens');
        if ~isempty(tokens)
            polarity = tokens{1}{1};
            delay_str_part = tokens{1}{2};
            delay = str2double(delay_str_part);
            delay_str = sprintf('%dms', delay);
        else
            delay = 0;
            delay_str = 'NaNms';
            warning('Could not parse delay from side_label: %s', side_label);
        end
        % --- Determine if Random baseline exists ---
        if contains(side_label, 'pos') && isfield(base_data, 'active_like_stim_pos_nan')
            base_side_label = 'active_like_stim_pos_nan';
            use_random_baseline = true;
        elseif contains(side_label, 'neg') && isfield(base_data, 'active_like_stim_neg_nan')
            base_side_label = 'active_like_stim_neg_nan';
            use_random_baseline = true;
        end
    else
        delay = str2double(raw_delay);
        if isnan(delay)
            delay = 0;
            delay_str = 'NaNms';
        else
            delay_str = sprintf('%dms', delay);
        end
    end
else
    delay = 0;
    delay_str = 'NaNms';
    warning('Unrecognized format for Stim_Delay: %s', string(raw_delay));
end

% --- Fallback: use original side_label if no Random override ---
if ~use_random_baseline
    base_side_label = side_label;
end

base_summary_field = [suffix '_' strrep(base_side_label, 'active_like_stim_', '') '_summary'];
cond_summary_field = [suffix '_' strrep(side_label, 'active_like_stim_', '') '_summary'];

% === Baseline ===
baseline_segment = base_data.(base_side_label).segments3;

% === Condition ===
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
layout = tiledlayout(3, 6, 'TileSpacing', 'compact', 'Padding', 'tight');

%% --- Baseline Velocity ---
ax_vel_baseline = nexttile(1, [1 2]); hold on;
title('Baseline Velocity'); ylabel('Velocity (deg/s)');
box off; set(gca, 'TickDir', 'out');
max_v = -inf; min_v = inf;

for i = 1:size(baseline_window,1)
    vel = base_data.(base_side_label).velocity_traces(i, :);
    vel = vel;% - mean(vel(1:10));  % normalize as originally
    stacked = vel + i * trace_spacing;
    plot(t, stacked, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.75);
    max_v = max(max_v, max(stacked));
    min_v = min(min_v, min(stacked));
end
xlim([-800 1200]); ylim([min_v-10 max_v+10]);

% --- Compute and plot average trace ---
vel_mat = nan(size(baseline_window,1), 2001);
for i = 1:size(baseline_window,1)
    vel = base_data.(base_side_label).velocity_traces(i, :);
    vel = vel;% - mean(vel(1:10));
    vel_mat(i, :) = vel;
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
    vel = vel;% - mean(vel(1:10));  % normalize as originally
    stacked = vel + i * trace_spacing;
    plot(t, stacked, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.75);
    max_v = max(max_v, max(stacked));
    min_v = min(min_v, min(stacked));
end
xlim([-800 1200]); ylim([min_v-10 max_v+10]);

vel_mat = nan(size(condition_window,1), 2001);
for i = 1:size(condition_window,1)
    vel = cond_data.(side_label).velocity_traces(i, :);
    vel = vel;% - mean(vel(1:10));
    vel_mat(i, :) = vel;
end
mean_trace = nanmean(vel_mat, 1);
plot(t, mean_trace + (size(condition_window,1)+1) * trace_spacing, 'k-', 'LineWidth', 2);

%% --- Overlay: Baseline vs Condition Velocity ---
ax_vel_overlay = nexttile(5, [1 2]); hold on;
title('Overlay Velocity'); ylabel('Velocity (deg/s)');
box off; set(gca, 'TickDir', 'out');

% Baseline average in black
baseline_vel_mat = nan(size(baseline_window,1), 2001);
for i = 1:size(baseline_window,1)
    vel = base_data.(base_side_label).velocity_traces(i, :);
    vel = vel;% - mean(vel(1:10));
    baseline_vel_mat(i,:) = vel;
end

if use_ci
    STDplot(t, baseline_vel_mat, [0 0 0]);         % black
    STDplot(t, vel_mat, [0 0.2 1]);                % blue
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

% Draw stim box with correct y-limits
duration = meta_cond.Stim_Duration_ms;
linkaxes([ax_vel_baseline, ax_vel_condition, ax_vel_overlay], 'y');
shared_yl = ylim(ax_vel_overlay);

for ax = [ax_vel_baseline, ax_vel_condition, ax_vel_overlay]
    hold(ax, 'on');
    fill(ax, ...
        [delay delay+duration delay+duration delay], ...
        [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)], ...
        [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end

%
% for ax = [ax_vel_baseline, ax_vel_condition, ax_vel_overlay]
%     fill(ax, ...
%         [delay delay+duration delay+duration delay], ...
%         [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)], ...
%         [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
% end

%% --- Baseline Position ---
ax_pos_baseline = nexttile(7, [1 2]); hold on;
title('Baseline Position'); xlabel('Time (ms)'); ylabel('Position (deg)');
box off; set(gca, 'TickDir', 'out');
max_p = -inf; min_p = inf;

for i = 1:size(baseline_window,1)
    pos = base_data.(base_side_label).position_traces(i, :);
    pos = pos;% - mean(pos(1:10));  % normalize as originally
    stacked = pos + i * trace_spacing;
    plot(t, stacked, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.75);
    max_p = max(max_p, max(stacked));
    min_p = min(min_p, min(stacked));
end
xlim([-800 1200]); ylim([min_p-10 max_p+10]);

pos_mat = nan(size(baseline_window,1), 2001);
for i = 1:size(baseline_window,1)
    pos = base_data.(base_side_label).position_traces(i, :);
    pos = pos;% - mean(pos(1:10));
    pos_mat(i, :) = pos;
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
    pos = pos;% - mean(pos(1:10));  % normalize as originally
    stacked = pos + i * trace_spacing;
    plot(t, stacked, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.75);
    max_p = max(max_p, max(stacked));
    min_p = min(min_p, min(stacked));
end
xlim([-800 1200]); ylim([min_p-10 max_p+10]);
% Draw stim box with correct y-limits
pos_mat = nan(size(condition_window,1), 2001);
for i = 1:size(condition_window,1)
    pos = cond_data.(side_label).position_traces(i, :);
    pos = pos;% - mean(pos(1:10));
    pos_mat(i, :) = pos;
end
mean_trace = nanmean(pos_mat, 1);
plot(t, mean_trace + (size(condition_window,1)+1) * trace_spacing, 'k-', 'LineWidth', 2);

%% --- Overlay: Baseline vs Condition Position ---
ax_pos_overlay = nexttile(11, [1 2]); hold on;
title('Overlay Position'); xlabel('Time (ms)'); ylabel('Position (deg)');
box off; set(gca, 'TickDir', 'out');

baseline_pos_mat = nan(size(baseline_window,1), 2001);
for i = 1:size(baseline_window,1)
    pos = base_data.(base_side_label).position_traces(i, :);
    pos = pos;% - mean(pos(1:10));
    baseline_pos_mat(i,:) = pos;
end

if use_ci
    STDplot(t, baseline_pos_mat, [0 0 0]);         % black
    STDplot(t, pos_mat, [0 0.2 1]);                % blue
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

for ax = [ax_pos_baseline, ax_pos_condition, ax_pos_overlay]
    hold(ax, 'on');
    fill(ax, ...
         [delay delay+duration delay+duration delay], ...
         [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)], ...
         [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end

% === Title using Metadata ===
stim_str = sprintf('Ch: %g | Freq: %gHz | Curr: %gμA | Dur: %gms | Delay: %s | Depth: %gmm | Trig: %s', ...
    meta_cond.Channels, meta_cond.Stim_Frequency_Hz, meta_cond.Current_uA, ...
    meta_cond.Stim_Duration_ms, delay_str, meta_cond.Depth_mm, ...
    meta_cond.Movement_Trigger{1});

% === Save figure to ComparisonTraces directory ===
% Get trial number from metadata (assumes BR_File holds the numeric ID)
trial_num = meta_cond.BR_File;
side_short = strrep(side_label, 'active_like_stim_', '');
title(layout, sprintf('Condition %03d - %s', trial_num, strrep(side_label, '_', ' ')), ...
    'FontWeight', 'bold', 'FontSize', 14, 'HorizontalAlignment', 'center');

title_str = sprintf('Condition %03d - %s', trial_num, suffix);
sgtitle({title_str, stim_str}, 'FontWeight', 'bold');
%% --- Endpoint Error Bar Plot ---
nexttile(layout, 13); hold on;
box off; set(gca, 'TickDir', 'out');

% Get data
baseline_trials = base_data.(base_side_label).all_err;
condition_trials = cond_data.(side_label).all_err;
n_base = sum(~isnan(baseline_trials));
n_cond = sum(~isnan(condition_trials));
baseline_trials = baseline_trials(~isnan(baseline_trials));
condition_trials = condition_trials(~isnan(condition_trials));

err_data = [
    base_data.(base_summary_field).all_err_mean, ...
    cond_data.(cond_summary_field).all_err_mean
    ];
err_sem = [
    sqrt(base_data.(base_summary_field).all_err_var), ...
    sqrt(cond_data.(cond_summary_field).all_err_var)
    ];

% Bar plot
bar_err = bar(1:2, err_data, 'FaceColor', 'flat');
bar_err.CData = [0.6 0.6 0.6; 0 0.4 1];
errorbar(1:2, err_data, err_sem, 'k', 'linestyle', 'none', 'LineWidth', 1);

% Significance
[~, p] = ttest2(baseline_trials, condition_trials);
if p < 0.001, sig_label = '***';
elseif p < 0.01, sig_label = '**';
elseif p < 0.05, sig_label = '*';
else, sig_label = 'n.s.';
end
% Determine y-axis limits
bar_top = max(err_data + err_sem);
yl = ylim;
y_sig = max(bar_top, yl(2)) + 0.05 * range(yl);

text(1.5, y_sig, sig_label, ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', ...
    'FontSize', 10);

% Axis
ylabel('Error (deg)');
ylow = min(err_data - err_sem); yhigh = max(err_data + err_sem);
ylim([ylow - 0.1*abs(ylow), yhigh + 0.1*abs(yhigh)]);
% Ensure the axis can accommodate the significance label
ylim([yl(1), max(y_sig, yl(2)) * 1.05]);
xticks(1.5); xticklabels({'EndpointError'});
xtickangle(45);
set(gca, 'FontSize', 10);
%% --- Other Metrics Bar Plot Group 1 ---
nexttile(layout, 14, [1 2]); hold on;
box off; set(gca, 'TickDir', 'out');

group1_metrics = {'all_err_abs_mean', 'var_500ms_mean', 'all_var_mean'};
group1_labels  = {'AbsEndpointError', 'VarAfterStim', 'VarAfterEndpoint'};

bar_data1 = zeros(2, numel(group1_metrics));
bar_std1  = zeros(2, numel(group1_metrics));
sig_labels1 = strings(1, numel(group1_metrics));

for i = 1:numel(group1_metrics)
    raw_field = erase(group1_metrics{i}, '_mean');
    
    % Extract per-trial data
    if strcmp(raw_field, 'all_err_abs')
        base_vals = abs(base_data.(base_side_label).all_err);
        cond_vals = abs(cond_data.(side_label).all_err);
    else
        base_vals = base_data.(base_side_label).(raw_field);
        cond_vals = cond_data.(side_label).(raw_field);
    end

    base_vals = base_vals(~isnan(base_vals));
    cond_vals = cond_vals(~isnan(cond_vals));

    % Mean
    bar_data1(1, i) = mean(base_vals);
    bar_data1(2, i) = mean(cond_vals);

    % Plot SD (not SEM)
    bar_std1(1, i) = std(base_vals);
    bar_std1(2, i) = std(cond_vals);

    % T-test using raw values (which implicitly uses SEM)
    [~, p] = ttest2(base_vals, cond_vals);
    if p < 0.001
        sig_labels1(i) = '***';
    elseif p < 0.01
        sig_labels1(i) = '**';
    elseif p < 0.05
        sig_labels1(i) = '*';
    else
        sig_labels1(i) = 'n.s.';
    end
end

bar_handle1 = bar(bar_data1', 'grouped');
bar_handle1(1).FaceColor = [0.6 0.6 0.6];
bar_handle1(2).FaceColor = [0 0.4 1];

ng1 = size(bar_data1, 2);
nb1 = size(bar_data1, 1);
gw1 = min(0.8, nb1/(nb1 + 1.5));
for i = 1:nb1
    x = (1:ng1) - gw1/2 + (2*i-1)*gw1/(2*nb1);
    errorbar(x, bar_data1(i,:), bar_std1(i,:), 'k', 'linestyle', 'none', 'LineWidth', 1);
end

yl1_max = max(bar_data1(:) + bar_std1(:));
yl1_min = min(bar_data1(:) - bar_std1(:));
yl1_range = yl1_max - yl1_min;
ylim([yl1_min - 0.1*abs(yl1_min), yl1_max + 0.1*yl1_range]);

for i = 1:ng1
    bar_top = max(bar_data1(:, i) + bar_std1(:, i));
    y_sig = bar_top + 0.05 * yl1_range;
    text(i, y_sig, sig_labels1(i), 'HorizontalAlignment', 'center', 'FontSize', 10);
end

xticks(1:ng1);
xticklabels(group1_labels);
xtickangle(45);
ylabel('deg/s');
set(gca, 'FontSize', 10);

%% --- Other Metrics Bar Plot Group 2 ---
nexttile(layout, 16); hold on;
box off; set(gca, 'TickDir', 'out');

group2_metrics = {'max_speed_mean', 'avg_speed_mean'};
group2_labels  = {'MaxSpeed', 'AvgSpeed'};

bar_data2 = zeros(2, numel(group2_metrics));
bar_std2  = zeros(2, numel(group2_metrics));
sig_labels2 = strings(1, numel(group2_metrics));

for i = 1:numel(group2_metrics)
    raw_field = erase(group2_metrics{i}, '_mean');

    base_vals = base_data.(base_side_label).(raw_field);
    cond_vals = cond_data.(side_label).(raw_field);

    base_vals = base_vals(~isnan(base_vals));
    cond_vals = cond_vals(~isnan(cond_vals));

    bar_data2(1, i) = mean(base_vals);
    bar_data2(2, i) = mean(cond_vals);

    % Plot SD
    bar_std2(1, i) = std(base_vals);
    bar_std2(2, i) = std(cond_vals);

    [~, p] = ttest2(base_vals, cond_vals);
    if p < 0.001
        sig_labels2(i) = '***';
    elseif p < 0.01
        sig_labels2(i) = '**';
    elseif p < 0.05
        sig_labels2(i) = '*';
    else
        sig_labels2(i) = 'n.s.';
    end
end

bar_handle2 = bar(bar_data2', 'grouped');
bar_handle2(1).FaceColor = [0.6 0.6 0.6];
bar_handle2(2).FaceColor = [0 0.4 1];

ng2 = size(bar_data2, 2);
nb2 = size(bar_data2, 1);
gw2 = min(0.8, nb2/(nb2 + 1.5));
for i = 1:nb2
    x = (1:ng2) - gw2/2 + (2*i-1)*gw2/(2*nb2);
    errorbar(x, bar_data2(i,:), bar_std2(i,:), 'k', 'linestyle', 'none', 'LineWidth', 1);
end

yl2_max = max(bar_data2(:) + bar_std2(:));
yl2_min = min(bar_data2(:) - bar_std2(:));
yl2_range = yl2_max - yl2_min;
ylim([yl2_min - 0.1*abs(yl2_min), yl2_max + 0.1*yl2_range]);

for i = 1:ng2
    bar_top = max(bar_data2(:, i) + bar_std2(:, i));
    y_sig = bar_top + 0.05 * yl2_range;
    text(i, y_sig, sig_labels2(i), 'HorizontalAlignment', 'center', 'FontSize', 10);
end

legend_labels = {
    sprintf('Baseline (n = %d)', n_base), ...
    sprintf('Condition (n = %d)', n_cond)
    };

legend(bar_handle2, legend_labels, ...
    'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off');

xticks(1:ng2);
xticklabels(group2_labels);
xtickangle(45);
ylabel('deg/s');
set(gca, 'FontSize', 10);

%% --- Endpoint Oscillation Bar Plot ---
nexttile(layout, 17); hold on;
box off; set(gca, 'TickDir', 'out');

osc_data = [
    base_data.(base_summary_field).oscillations_mean, ...
    cond_data.(cond_summary_field).oscillations_mean
    ];
osc_std = [
    sqrt(base_data.(base_summary_field).oscillations_var), ...
    sqrt(cond_data.(cond_summary_field).oscillations_var)
    ];

bar_osc = bar(1:2, osc_data, 'FaceColor', 'flat');
bar_osc.CData = [0.6 0.6 0.6; 0 0.4 1];
errorbar(1:2, osc_data, osc_std, 'k', 'linestyle', 'none', 'LineWidth', 1);

xticks(1.5);
xticklabels({'EndpointOscillation'});

% Perform t-test on raw per-trial data
baseline_osc = base_data.(base_side_label).oscillations;
condition_osc = cond_data.(side_label).oscillations;

baseline_osc = baseline_osc(~isnan(baseline_osc));
condition_osc = condition_osc(~isnan(condition_osc));

[~, p] = ttest2(baseline_osc, condition_osc);

% Significance stars
if p < 0.001, sig_label = '***';
elseif p < 0.01, sig_label = '**';
elseif p < 0.05, sig_label = '*';
else, sig_label = 'n.s.';
end
bar_top = max(osc_data + osc_std);
yl = ylim;
y_sig = max(bar_top, yl(2)) + 0.05 * range(yl);

text(1.5, y_sig, sig_label, 'HorizontalAlignment', 'center', 'FontSize', 10);

xtickangle(45);
ylabel('Count');
ylim([0 1.2 * max(osc_data + osc_std)]);
set(gca, 'FontSize', 10);

%% --- FFT Power Bar Plot ---
nexttile(layout, 18); hold on;
box off; set(gca, 'TickDir', 'out');

fft_power_data = [
    base_data.(base_summary_field).fft_power_mean, ...
    cond_data.(cond_summary_field).fft_power_mean
    ];
fft_power_std = [
    sqrt(base_data.(base_summary_field).fft_power_var), ...
    sqrt(cond_data.(cond_summary_field).fft_power_var)
    ];

% Plot each bar separately so we get two handles
hold on;
bar1 = bar(1, fft_power_data(1), 'FaceColor', [0.6 0.6 0.6], 'BarWidth', 0.4);
bar2 = bar(2, fft_power_data(2), 'FaceColor', [0 0.4 1], 'BarWidth', 0.4);

% Add error bars
errorbar(1:2, fft_power_data, fft_power_std, 'k', ...
    'linestyle', 'none', 'LineWidth', 1);

xticks(1.5);
xticklabels({'FFTPowerAfterEnd'});

% Perform t-test on raw per-trial data
baseline_fft = base_data.(base_side_label).fft_power;
condition_fft = cond_data.(side_label).fft_power;

baseline_fft = baseline_fft(~isnan(baseline_fft));
condition_fft = condition_fft(~isnan(condition_fft));

[~, p] = ttest2(baseline_fft, condition_fft);

if p < 0.001, sig_label = '***';
elseif p < 0.01, sig_label = '**';
elseif p < 0.05, sig_label = '*';
else, sig_label = 'n.s.';
end
bar_top = max(fft_power_data + fft_power_std);
yl = ylim;
y_sig = max(bar_top, yl(2)) + 0.05 * range(yl);

text(1.5, y_sig, sig_label, 'HorizontalAlignment', 'center', 'FontSize', 10);

xtickangle(45);
ylabel('Power');
ylim([0 1.2 * max(fft_power_data + fft_power_std)]);
set(gca, 'FontSize', 10);
% Adjust layout to make room for annotation
outerpos = get(layout, 'OuterPosition');
outerpos(2) = outerpos(2) + 0.08; % Shift layout up a bit
outerpos(4) = outerpos(4) - 0.08; % Shrink height to make room below
set(layout, 'OuterPosition', outerpos);
% === Add boxed significance legend at bottom center ===
annotation('textbox', [0.35, 0.01, 0.3, 0.08], ...
    'String', {'\bf{Significance Legend}', ...
    '*   p < 0.05    **   p < 0.01    ***   p < 0.001    n.s. = not significant'}, ...
    'EdgeColor', 'k', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'FontSize', 9, ...
    'FitBoxToText', 'on', ...
    'BackgroundColor', 'white');
end
function out = ternary(cond, valTrue, valFalse)
    if cond
        out = valTrue;
    else
        out = valFalse;
    end
end
