function fig = plot_rand_condition_traces_neural(base_data, cond_data_list, side_label, offset, meta_cond, use_ci, show_figs)
if nargin < 6
    use_ci = true;
end

%% === Setup ===
% session = 'BL_RW_003_Session_1';
% base_folder = fullfile('/Volumes/CullenLab_Server/Current Project Databases - NHP', ...
%     '2025 Cerebellum prosthesis/Bryan/Data', session);
% summary_path = fullfile(base_folder, [session '_summarized_metrics.mat']);
% meta_path = fullfile(base_folder, [session '_metadata_with_metrics.csv']);
% side_label = 'active_like_stim_pos';
% condition_num = 6;
% 
% load(summary_path, 'summary_struct');
% offset = false;
% cond_data_list = summary_struct(condition_num).merged_with_summary;
% base_data = cond_data_list;
% T = readtable(meta_path);
% meta_cond = T(T.BR_File == condition_num, :);
%% === Initialize figure and layout ===
fig = figure('Visible', ternary(show_figs, 'on', 'off'), 'Position', [100, 100, 1400, 800]);
layout = tiledlayout(3, 6, 'TileSpacing', 'compact', 'Padding', 'tight');

%% === Trace alignment & setup ===
t = linspace(-800, 1200, 2001);
trace_spacing = 20 * offset;
delay_suffixes = {'nan', '0', '100', '200'};
x_labels = {'Baseline', '0 ms', '100 ms', '200 ms'};
n_groups = numel(delay_suffixes);
err_means = nan(1, n_groups);
err_sems  = nan(1, n_groups);
sig_labels = strings(1, n_groups - 1);

%% === Metadata ===
is_ipsi = contains(side_label, 'pos');
suffix = ternary(is_ipsi, 'ipsi', 'contra');

tokens = regexp(side_label, 'active_like_stim_(pos|neg)', 'tokens');
assert(~isempty(tokens), 'Invalid side_label: %s', side_label);
polarity = tokens{1}{1};

base_side_label = sprintf('active_like_stim_%s_nan', polarity);
if ~isfield(base_data, base_side_label)
    warning('Missing baseline: using original side_label');
    base_side_label = side_label;
end

% Create delay-specific condition labels
delay_labels = cellfun(@(s) sprintf('active_like_stim_%s_%s', polarity, s), ...
    delay_suffixes, 'UniformOutput', false);

% Colors for each condition
trace_colors = [
    0.3 0.3 0.3;   % Baseline - dark gray
    0.0 0.45 0.7;  % 0 ms     - blue
    0.8 0.4 0.0;   % 100 ms   - orange
    0.2 0.7 0.3    % 200 ms   - green
];
%% === Velocity and Position Traces ===
% === Compute global y-limits for velocity and position with padding ===
vel_all = [];
pos_all = [];
for i = 1:4
    vel_all = [vel_all; cond_data_list.(delay_labels{i}).velocity_traces];
    pos_all = [pos_all; cond_data_list.(delay_labels{i}).position_traces];
end

vel_min = min(vel_all(:));
vel_max = max(vel_all(:));
vel_range = vel_max - vel_min;
yl_vel = [vel_min - 0.05 * vel_range, vel_max + 0.05 * vel_range];

pos_min = min(pos_all(:));
pos_max = max(pos_all(:));
pos_range = pos_max - pos_min;
yl_pos = [pos_min - 0.05 * pos_range, pos_max + 0.05 * pos_range];

for i = 1:4
    field = delay_labels{i};

    % --- Velocity ---
    ax_vel = nexttile(layout, i); hold on;
    vel = cond_data_list.(field).velocity_traces;
    trace_fade = 0.5;
    light_color = [0.6, 0.6, 0.6] * trace_fade + [1 1 1] * (1 - trace_fade);

    for tr = 1:size(vel, 1)
        plot(t, vel(tr,:) + tr * trace_spacing, 'Color', light_color, 'LineWidth', 0.75);
    end
    plot(t, nanmean(vel, 1) + (size(vel,1)+1) * trace_spacing, 'Color', trace_colors(i,:), 'LineWidth', 2.0);

    xline(0, '--k');
    ylim(yl_vel);

    if i > 1
        delay = str2double(delay_suffixes{i});
        duration = meta_cond.Stim_Duration_ms;
        fill([delay delay+duration delay+duration delay], ...
             [yl_vel(1) yl_vel(1) yl_vel(2) yl_vel(2)], ...
             [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
    title(['Vel ' x_labels{i}]); ylabel('Velocity (deg/s)');
    xlim([-800 1200]); box off; set(gca, 'TickDir', 'out');

    % --- Position ---
    ax_pos = nexttile(layout, i+6); hold on;
    pos = cond_data_list.(field).position_traces;

    for tr = 1:size(pos, 1)
        plot(t, pos(tr,:) + tr * trace_spacing, 'Color', light_color, 'LineWidth', 0.75);
    end
    plot(t, nanmean(pos, 1) + (size(pos,1)+1) * trace_spacing, 'Color', trace_colors(i,:), 'LineWidth', 2.0);

    xline(0, '--k');
    ylim(yl_pos);

    if i > 1
        fill([delay delay+duration delay+duration delay], ...
             [yl_pos(1) yl_pos(1) yl_pos(2) yl_pos(2)], ...
             [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
    title(['Pos ' x_labels{i}]); ylabel('Position (deg)');
    xlim([-800 1200]); box off; set(gca, 'TickDir', 'out');
end
% --- Overlay Velocity ---
nexttile(layout, 5, [1, 2]); hold on;
title('Overlay Velocity'); ylabel('Velocity (deg/s)'); box off; set(gca, 'TickDir', 'out');
for i = 1:4
    field = delay_labels{i};
    STDplot(t, cond_data_list.(field).velocity_traces, trace_colors(i,:));
end
xline(0, '--k');
ylim(yl_vel);
xlim([-800 1200]);

% --- Overlay Position ---
nexttile(layout, 11, [1, 2]); hold on;
title('Overlay Position'); ylabel('Position (deg)'); box off; set(gca, 'TickDir', 'out');
for i = 1:4
    field = delay_labels{i};
    STDplot(t, cond_data_list.(field).position_traces, trace_colors(i,:));
end
xline(0, '--k');
ylim(yl_pos);
xlim([-800 1200]);

%% === Endpoint Error Bar Plot (Tile 13) ===
nexttile(layout, 13); hold on;
box off;
set(gca, 'TickDir', 'out', 'FontSize', 12);

for i = 1:n_groups
    delay_suffix = delay_suffixes{i};
    cond_label = delay_labels{i};
    summary_label = sprintf('%s_%s_%s_summary', suffix, polarity, delay_suffix);

    cond_data = cond_data_list.(cond_label);
    cond_summary = cond_data_list.(summary_label);

    trials = cond_data.all_err;
    trials = trials(~isnan(trials));
    err_means(i) = cond_summary.all_err_mean;
    err_sems(i)  = sqrt(cond_summary.all_err_var);

    if i > 1
        base_trials = cond_data_list.(delay_labels{1}).all_err;
        base_trials = base_trials(~isnan(base_trials));
        [~, p] = ttest2(base_trials, trials);
        sig_labels(i-1) = significance_label(p);
    end
end

x_single = 1:n_groups;

bar_err = bar(x_single, err_means, 'FaceColor', 'flat');
bar_err.CData = trace_colors;
errorbar(x_single, err_means, err_sems, 'k', 'linestyle', 'none', 'LineWidth', 1.5);

yrange = range([err_means - err_sems, err_means + err_sems]);
ylim([min(err_means - err_sems) - 0.15 * yrange, max(err_means + err_sems) + 0.15 * yrange]);
xticks(mean(x_single)); xticklabels({'EndpointError'});
xtickangle(45);
ylabel('Endpoint Error (deg)');
yline(0, '--k');

for i = 2:n_groups
    y_sig = err_means(i) + err_sems(i) + 0.05 * yrange;
    text(i, y_sig, sig_labels(i-1), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end
%% --- Other Metrics Bar Plot Group 1 (Match Endpoint Error Layout) ---
nexttile(layout, 14, [1 2]); hold on;
box off; set(gca, 'TickDir', 'out');

group1_metrics = {'all_err_abs_mean', 'var_500ms_mean', 'all_var_mean'};
group1_labels  = {'AbsEndpointError', 'VarAfterStim', 'VarAfterEndpoint'};
n_metrics = numel(group1_metrics);
bar_data1 = nan(n_groups, n_metrics);
bar_std1  = nan(n_groups, n_metrics);
sig_labels1 = strings(n_metrics, n_groups-1);

for m = 1:n_metrics
    metric_field = erase(group1_metrics{m}, '_mean');
    base_label = sprintf('%s_nan', side_label);

    % === Get baseline values ===
    if strcmp(metric_field, 'all_err_abs')
        base_vals = abs(cond_data_list.(base_label).all_err);
    else
        base_vals = cond_data_list.(base_label).(metric_field);
    end
    base_vals = base_vals(~isnan(base_vals));

    for i = 1:n_groups
        cond_label = sprintf('%s_%s', side_label, delay_suffixes{i});

        % === Get condition values ===
        if strcmp(metric_field, 'all_err_abs')
            vals = abs(cond_data_list.(cond_label).all_err);
        else
            vals = cond_data_list.(cond_label).(metric_field);
        end
        vals = vals(~isnan(vals));

        % === Store stats ===
        bar_data1(i, m) = mean(vals);
        bar_std1(i, m)  = std(vals);

        % === Significance vs. baseline (skip i==1 which is baseline) ===
        if i > 1
            [~, p] = ttest2(base_vals, vals);
            sig_labels1(m, i-1) = significance_label(p);
        end
    end
end


% Plot: for each delay, one color, across metrics
x = 1:n_metrics;
bar_width = 0.15;
spacing_factor = 2;  % try values between 1.4 and 2.0 for more spacing
offsets = linspace(-1, 1, n_groups) * bar_width * spacing_factor;

for i = 1:n_groups
    bar(x + offsets(i), bar_data1(i,:), bar_width, ...
        'FaceColor', trace_colors(i,:));
    errorbar(x + offsets(i), bar_data1(i,:), bar_std1(i,:), ...
        'k', 'linestyle', 'none', 'LineWidth', 1);
end

% Significance stars
yl = ylim;
for m = 1:n_metrics
    for i = 2:n_groups
        xpos = x(m) + offsets(i);
        ypos = bar_data1(i,m) + bar_std1(i,m) + 0.05 * range(yl);
        text(xpos, ypos, sig_labels1(m,i-1), ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 12, 'FontWeight', 'bold');
    end
end

xticks(x);
xticklabels(group1_labels);
xtickangle(45);
ylabel('deg/s');
set(gca, 'FontSize', 12);
yline(0, '--k');

%% === Group 2 Metrics (Tile 16, Match Layout) ===
nexttile(layout, 16); hold on;
box off; set(gca, 'TickDir', 'out');

group2_metrics = {'max_speed_mean', 'avg_speed_mean'};
group2_labels  = {'MaxSpeed', 'AvgSpeed'};
n_metrics2 = numel(group2_metrics);

bar_data2 = nan(n_groups, n_metrics2);
bar_std2  = nan(n_groups, n_metrics2);
sig_labels2 = strings(n_metrics2, n_groups - 1);

sample_sizes = zeros(n_groups, 1);  % [baseline, delay1, delay2, delay3]

% Get baseline size
sample_sizes(1) = numel(base_vals);  % base_vals already defined outside the loop

% Get condition sizes
for i = 2:n_groups
    cond_label = sprintf('%s_%s', side_label, delay_suffixes{i});
    vals = cond_data_list.(cond_label).(metric_field);
    vals = vals(~isnan(vals));
    sample_sizes(i) = numel(vals);  % because baseline is at index 1
end

for m = 1:n_metrics2
    metric_field = erase(group2_metrics{m}, '_mean');
    base_vals = base_data.(base_side_label).(metric_field);
    base_vals = base_vals(~isnan(base_vals));

    for i = 1:n_groups
        cond_label = sprintf('%s_%s', side_label, delay_suffixes{i});
        vals = cond_data_list.(cond_label).(metric_field);
        vals = vals(~isnan(vals));

        bar_data2(i, m) = mean(vals);
        bar_std2(i, m)  = std(vals);

        if i > 1
            [~, p] = ttest2(base_vals, vals);
            sig_labels2(m, i-1) = significance_label(p);
        end
    end
end

% Manual bar plotting (same layout as Group 1)
x2 = 1:n_metrics2;
bar_width = 0.15;
spacing_factor = 2.2;
offsets = linspace(-1, 1, n_groups) * bar_width * spacing_factor;

for i = 1:n_groups
    bar(x2 + offsets(i), bar_data2(i,:), bar_width, ...
        'FaceColor', trace_colors(i,:), 'HandleVisibility', 'off');
    errorbar(x2 + offsets(i), bar_data2(i,:), bar_std2(i,:), ...
        'k', 'linestyle', 'none', 'LineWidth', 1, 'HandleVisibility', 'off');
end

% Significance annotations
yl = ylim;
for m = 1:n_metrics2
    for i = 2:n_groups
        xpos = x2(m) + offsets(i);
        ypos = bar_data2(i, m) + bar_std2(i, m) + 0.05 * range(yl);
        text(xpos, ypos, sig_labels2(m, i-1), ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 12, 'FontWeight', 'bold');
    end
end

xticks(x2);
xticklabels(group2_labels);
xtickangle(45);
ylabel('deg/s');
set(gca, 'FontSize', 12);
yline(0, '--k');

% Add color-coded legend for delay conditions
legend_labels = {
    sprintf('Baseline (n=%d)', sample_sizes(1)),
    sprintf('0 ms (n=%d)',  sample_sizes(2)),
    sprintf('100 ms (n=%d)', sample_sizes(3)),
    sprintf('200 ms (n=%d)', sample_sizes(4))
};

h_legend = gobjects(1, n_groups);  % preallocate

for i = 1:n_groups
    h_legend(i) = bar(nan, nan, 'FaceColor', trace_colors(i,:), ...
        'HandleVisibility', 'on');
end

% Now display clean legend
legend(h_legend, legend_labels, ...
    'Orientation', 'horizontal', ...
    'Location', 'northoutside', ...
    'Box', 'off', ...
    'FontSize', 10);
%% --- Endpoint Oscillation Bar Plot ---
nexttile(layout, 17); hold on;
box off; set(gca, 'TickDir', 'out', 'FontSize', 12);

osc_means = nan(1, n_groups);
osc_sems  = nan(1, n_groups);
osc_sig_labels = strings(1, n_groups - 1);

for i = 1:n_groups
    summary_field = sprintf('%s_%s_%s_summary', suffix, polarity, delay_suffixes{i});
    osc_means(i) = cond_data_list.(summary_field).oscillations_mean;
    osc_sems(i)  = sqrt(cond_data_list.(summary_field).oscillations_var);

    if i > 1
        base_vals = base_data.(base_side_label).oscillations;
        cond_vals = cond_data_list.(delay_labels{i}).oscillations;
        base_vals = base_vals(~isnan(base_vals));
        cond_vals = cond_vals(~isnan(cond_vals));
        [~, p] = ttest2(base_vals, cond_vals);
        osc_sig_labels(i-1) = significance_label(p);
    end
end

bar_osc = bar(x_single, osc_means, 'FaceColor', 'flat');
bar_osc.CData = trace_colors;
errorbar(x_single, osc_means, osc_sems, 'k', 'linestyle', 'none', 'LineWidth', 1);

yrange = range([osc_means - osc_sems, osc_means + osc_sems]);
ylim([min(osc_means - osc_sems) - 0.15 * yrange, max(osc_means + osc_sems) + 0.15 * yrange]);
xticks(mean(x_single)); xticklabels({'EndpointOscillation'});
xtickangle(45);
ylabel('Count');
yline(0, '--k');

for i = 2:n_groups
    y_sig = osc_means(i) + osc_sems(i) + 0.05 * range(ylim);
    text(i, y_sig, osc_sig_labels(i-1), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end

%% --- FFT Power Bar Plot ---
nexttile(layout, 18); hold on;
box off; set(gca, 'TickDir', 'out', 'FontSize', 12);

fft_means = nan(1, n_groups);
fft_sems  = nan(1, n_groups);
fft_sig_labels = strings(1, n_groups - 1);

for i = 1:n_groups
    summary_field = sprintf('%s_%s_%s_summary', suffix, polarity, delay_suffixes{i});
    fft_means(i) = cond_data_list.(summary_field).fft_power_mean;
    fft_sems(i)  = sqrt(cond_data_list.(summary_field).fft_power_var);

    if i > 1
        base_vals = base_data.(base_side_label).fft_power;
        cond_vals = cond_data_list.(delay_labels{i}).fft_power;
        base_vals = base_vals(~isnan(base_vals));
        cond_vals = cond_vals(~isnan(cond_vals));
        [~, p] = ttest2(base_vals, cond_vals);
        fft_sig_labels(i-1) = significance_label(p);
    end
end

bar_fft = bar(x_single, fft_means, 'FaceColor', 'flat');
bar_fft.CData = trace_colors;
errorbar(x_single, fft_means, fft_sems, 'k', 'linestyle', 'none', 'LineWidth', 1);

yrange = range([fft_means - fft_sems, fft_means + fft_sems]);
ylim([min(fft_means - fft_sems) - 0.15 * yrange, max(fft_means + fft_sems) + 0.15 * yrange]);
xticks(mean(x_single)); xticklabels({'FFTPowerAfterEnd'});
xtickangle(45);
ylabel('Power');
yline(0, '--k');

for i = 2:n_groups
    y_sig = fft_means(i) + fft_sems(i) + 0.05 * range(ylim);
    text(i, y_sig, fft_sig_labels(i-1), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end

% === Title using Metadata ===
stim_str = sprintf('Ch: %g | Freq: %gHz | Curr: %gμA | Dur: %gms | Depth: %gmm | Trig: %s', ...
    meta_cond.Channels, meta_cond.Stim_Frequency_Hz, meta_cond.Current_uA, ...
    meta_cond.Stim_Duration_ms, meta_cond.Depth_mm, ...
    meta_cond.Movement_Trigger{1});

trial_num = meta_cond.BR_File;
title_str = sprintf('Delay Comparison: Condition %03d - %s', trial_num, suffix);
sgtitle({title_str, stim_str}, 'FontWeight', 'bold');

% === Adjust layout if needed ===
outerpos = get(layout, 'OuterPosition');
outerpos(2) = outerpos(2) + 0.08;
outerpos(4) = outerpos(4) - 0.08;
set(layout, 'OuterPosition', outerpos);

% === Add boxed significance legend ===
annotation('textbox', [0.35, 0.01, 0.3, 0.08], ...
    'String', {'\bf{Significance Legend}', ...
    '*   p < 0.05    **   p < 0.01    ***   p < 0.001    n.s. = not significant'}, ...
    'EdgeColor', 'k', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 12, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white');
%% === Significance helper ===
function label = significance_label(p)
    if p < 0.001, label = '***';
    elseif p < 0.01, label = '**';
    elseif p < 0.05, label = '*';
    else, label = 'n.s.';
    end
end

%% === Ternary helper ===
function val = ternary(cond, a, b)
    if cond
        val = a;
    else
        val = b;
    end
end

end
function out = ternary(cond, valTrue, valFalse)
    if cond
        out = valTrue;
    else
        out = valFalse;
    end
end
