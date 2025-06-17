function fig = plot_adaptation_traces(base_data, cond_data_list, side_label, offset, meta_cond, use_ci)
if nargin < 6
    use_ci = true;
end

%% === Setup ===
clear all;
session = 'BL_RW_003_Session_1';
base_folder = fullfile('/Volumes/CullenLab_Server/Current Project Databases - NHP', ...
    '2025 Cerebellum prosthesis/Bryan/Data', session);
summary_path = fullfile(base_folder, [session '_summarized_metrics.mat']);
meta_path = fullfile(base_folder, [session '_metadata_with_metrics.csv']);
side_label = 'active_like_stim_pos';
condition_num = 9;

load(summary_path, 'summary_struct', 'merged_baseline_summary');
offset = false;
cond_data_list = summary_struct(condition_num).merged_with_summary;
base_data = cond_data_list;
T = readtable(meta_path);
meta_cond = T(T.BR_File == condition_num, :);


%% === Trace alignment & setup ===
t = linspace(-800, 1200, 2001);
trace_spacing = 20 * offset;
trial_sets = {'first10', 'last10', 'catch'};
x_labels = {'Baseline', 'First 10', 'Last 10', 'Catch'};
n_sets = numel(x_labels);

% === Extract first and last 10 trials from both _pos and _neg fields ===
all_fields = fieldnames(cond_data_list);
sides = all_fields(contains(all_fields, 'active_like_stim'));

first10_data = struct();
last10_data  = struct();

for i = 1:numel(sides)
    field = sides{i};
    data = cond_data_list.(field);

    if isfield(data, 'n_trials')
        n_trials = data.n_trials;
    else
        n_trials = size(data.all_err, 1); % fallback to array size
    end

    inner_fields = fieldnames(data);
    for j = 1:numel(inner_fields)
        inner_field = inner_fields{j};
        val = data.(inner_field);

        if isnumeric(val) && size(val, 1) >= 10 && ~strcmp(inner_field, 'n_trials')
            first10_data.(field).(inner_field) = val(1:10, :);
            last10_data.(field).(inner_field)  = val(end-9:end, :);
        elseif isnumeric(val) && ~strcmp(inner_field, 'n_trials')
            warning('Field %s.%s has fewer than 10 rows. Using available %d rows.', ...
                field, inner_field, size(val, 1));
            first10_data.(field).(inner_field) = val;
            last10_data.(field).(inner_field)  = val;
        else
            first10_data.(field).(inner_field) = val; % non-numeric
            last10_data.(field).(inner_field)  = val;
        end
    end
    % Set n_trials field separately (just for reference)
    first10_data.(field).n_trials = min(10, n_trials);
    last10_data.(field).n_trials  = min(10, n_trials);
    % Rename for first10
    new_field_first = sprintf('%s_first10', field);
    merged_data.(new_field_first) = first10_data.(field);

    % Rename for last10
    new_field_last = sprintf('%s_last10', field);
    merged_data.(new_field_last) = last10_data.(field);
end

% === Add catch trials directly from original Data ===
if isfield(cond_data_list, 'catch_pos')
    merged_data.catch_pos = cond_data_list.catch_pos;
end
if isfield(cond_data_list, 'catch_neg')
    merged_data.catch_neg = cond_data_list.catch_neg;
end


all_fields = fieldnames(merged_data);
ipsi_fields = all_fields(contains(all_fields, '_pos'));
contra_fields = all_fields(contains(all_fields, '_neg'));

summary_ipsi = calculate_mean_metrics(merged_data, ipsi_fields, 'ipsi');
summary_contra = calculate_mean_metrics(merged_data, contra_fields, 'contra');

merged_with_summary = merged_data;
for s = fieldnames(summary_ipsi)'
    merged_with_summary.(s{1}) = summary_ipsi.(s{1});
end
for s = fieldnames(summary_contra)'
    merged_with_summary.(s{1}) = summary_contra.(s{1});
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

%% === Metadata ===
% Set baseline and condition containers
base_data = merged_baseline_summary;
cond_data_list = merged_with_summary;

% Parse polarity and direction
is_ipsi = contains(side_label, 'pos');
suffix = ternary(is_ipsi, 'ipsi', 'contra');
%%
tokens = regexp(side_label, 'active_like_stim_(pos|neg)', 'tokens');
assert(~isempty(tokens), 'Invalid side_label: %s', side_label);
polarity = tokens{1}{1};

% Baseline label (same format as before)
base_side_label = sprintf('active_like_stim_%s', polarity);
if ~isfield(base_data, base_side_label)
    warning('Missing baseline: using original side_label');
    base_side_label = side_label;
end

% Condition labels
n_sets = numel(trial_sets) + 1;

cond_labels = cell(1, n_sets);      % Add one extra slot
summary_labels = cell(1, n_sets);   % Same here

cond_labels{1} = base_side_label;
summary_labels{1} = sprintf('%s_%s_summary', suffix, polarity);

for i = 2:numel(trial_sets)
    cond_labels{i} = sprintf('active_like_stim_%s_%s', polarity, trial_sets{i-1});
    summary_labels{i} = sprintf('%s_%s_%s_summary', suffix, polarity, trial_sets{i-1});
end

% === Add catch trial ===
catch_field = sprintf('catch_%s', polarity);  % 'catch_pos' or 'catch_neg'
cond_labels{end} = catch_field;
summary_labels{end} = sprintf('%s_%s_summary', suffix, 'catch');

% Colors per delay (same as Endpoint Error panel)
condition_colors = [
    0.6 0.6 0.6;  % Baseline
    0.0 0.4 1.0;  % First 10 (blue)
    1.0 0.8 0.0;   % Last 10 (yellow)
    0.8 0.2 0.2
];
legend_labels = {'Baseline', 'First 10', 'Last 10', 'Catch'};
%% === Initialize figure and layout ===
fig = figure('Visible', 'on', 'Position', [100, 100, 1400, 800]);
layout = tiledlayout(3, 6, 'TileSpacing', 'compact', 'Padding', 'tight');

%% === Placeholder for velocity and position subplots ===
trace_tiles = [1 3 5 7 9 11];
titles = {'First 10 Vel', 'Last 10 Vel', 'Overlay Vel', 'First 10 Pos', 'Last 10 Pos', 'Overlay Pos'};
trace_fields = {'first10', 'last10'};
trace_colors = {[0.6 0.6 0.6], [0.6 0.6 0.6]};

% Velocity plots
ax_vel = gobjects(1,3);
for k = 1:2
    ax_vel(k) = nexttile(layout, trace_tiles(k), [1 2]); hold on;
    title(titles{k}); ylabel('Velocity (deg/s)'); box off; set(gca, 'TickDir', 'out');
    fieldname = sprintf('active_like_stim_%s_%s', polarity, trace_fields{k});
    vel_mat = cond_data_list.(fieldname).velocity_traces;
    for i = 1:size(vel_mat,1)
        stacked = vel_mat(i,:) + i * trace_spacing;
        plot(t, stacked, 'Color', trace_colors{k}, 'LineWidth', 0.75);
    end
    plot(t, nanmean(vel_mat, 1) + (size(vel_mat,1)+1)*trace_spacing, 'Color', condition_colors(k+1, :), 'LineWidth', 2);
    xlim([-800 1200]);
end

% Velocity overlay
ax_vel(3) = nexttile(layout, trace_tiles(3), [1 2]); hold on;
title(titles{3}); ylabel('Velocity (deg/s)'); box off; set(gca, 'TickDir', 'out');
baseline_vel = base_data.(base_side_label).velocity_traces;
vel1 = cond_data_list.(sprintf('active_like_stim_%s_first10', polarity)).velocity_traces;
vel2 = cond_data_list.(sprintf('active_like_stim_%s_last10', polarity)).velocity_traces;
STDplot(t, baseline_vel, [0 0 0]);
STDplot(t, vel1, condition_colors(2, :));
STDplot(t, vel2, condition_colors(3, :));
if isfield(cond_data_list, sprintf('catch_%s', polarity))
    pos_catch = cond_data_list.(sprintf('catch_%s', polarity)).velocity_traces;
    STDplot(t, pos_catch, condition_colors(4, :));
end

xlim([-800 1200]);

linkaxes(ax_vel, 'y');
shared_yl = ylim(ax_vel(3));
duration = meta_cond.Stim_Duration_ms;
fill([delay delay+duration delay+duration delay], [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)], [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
for k = 1:2
    ax = ax_vel(k);
    axes(ax);
    fill(ax, [delay delay+duration delay+duration delay], ...
             [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)], ...
             [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end

% Position plots
ax_pos = gobjects(1,3);
for k = 1:2
    ax_pos(k) = nexttile(layout, trace_tiles(k+3), [1 2]); hold on;
    title(titles{k+3}); ylabel('Position (deg)'); box off; set(gca, 'TickDir', 'out');
    fieldname = sprintf('active_like_stim_%s_%s', polarity, trace_fields{k});
    pos_mat = cond_data_list.(fieldname).position_traces;
    for i = 1:size(pos_mat,1)
        stacked = pos_mat(i,:) + i * trace_spacing;
        plot(t, stacked, 'Color', trace_colors{k}, 'LineWidth', 0.75);
    end
    plot(t, nanmean(pos_mat, 1) + (size(pos_mat,1)+1)*trace_spacing, 'Color', condition_colors(k+1, :), 'LineWidth', 2);
    xlim([-800 1200]);
end

% Position overlay
ax_pos(3) = nexttile(layout, trace_tiles(6), [1 2]); hold on;
title(titles{6}); ylabel('Position (deg)'); box off; set(gca, 'TickDir', 'out');
baseline_pos = base_data.(base_side_label).position_traces;
pos1 = cond_data_list.(sprintf('active_like_stim_%s_first10', polarity)).position_traces;
pos2 = cond_data_list.(sprintf('active_like_stim_%s_last10', polarity)).position_traces;
STDplot(t, baseline_pos, [0 0 0]);
STDplot(t, pos1, condition_colors(2, :));
STDplot(t, pos2, condition_colors(3, :));
if isfield(cond_data_list, sprintf('catch_%s', polarity))
    pos_catch = cond_data_list.(sprintf('catch_%s', polarity)).position_traces;
    STDplot(t, pos_catch, condition_colors(4, :));
end
xlim([-800 1200]);

linkaxes(ax_pos, 'y');
shared_yl = ylim(ax_pos(3));
fill([delay delay+duration delay+duration delay], [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)], [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
for k = 1:2
    ax = ax_pos(k);
    axes(ax);
    fill(ax, [delay delay+duration delay+duration delay], ...
             [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)], ...
             [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end


%% === Baseline Endpoint Error Bar Plot ===
nexttile(layout, 13); hold on;
box off;
set(gca, 'TickDir', 'out', 'FontSize', 12);

err_means = nan(1, n_sets);
err_sems  = nan(1, n_sets);
sig_labels = strings(1, n_sets - 1);

% Baseline values
baseline_trials = base_data.(base_side_label).all_err;
baseline_trials = baseline_trials(~isnan(baseline_trials));
err_means(1) = mean(baseline_trials);
err_sems(1)  = std(baseline_trials) / sqrt(numel(baseline_trials));

% Loop through conditions
for i = 2:n_sets
    cond_field = cond_labels{i};
    cond_data = cond_data_list.(cond_field);

    trial_vals = cond_data.all_err;
    trial_vals = trial_vals(~isnan(trial_vals));

    err_means(i) = mean(trial_vals);
    err_sems(i)  = std(trial_vals) / sqrt(numel(trial_vals));

    [~, p] = ttest2(baseline_trials, trial_vals);
    sig_labels(i-1) = significance_label(p);
end

% Plotting
x_single = 1:n_sets;
bar_err = bar(x_single, err_means, 'FaceColor', 'flat');
bar_err.CData = condition_colors;
errorbar(x_single, err_means, err_sems, 'k', 'linestyle', 'none', 'LineWidth', 1.5);

yrange = range([err_means - err_sems, err_means + err_sems]);
ylim([min(err_means - err_sems) - 0.15 * yrange, max(err_means + err_sems) + 0.15 * yrange]);
xticks(mean(1:n_sets)); xticklabels({'EndpointError'});
xtickangle(45);
ylabel('Endpoint Error (deg)');
yline(0, '--k');

for i = 2:n_sets
    y_sig = err_means(i) + err_sems(i) + 0.05 * yrange;
    text(i, y_sig, sig_labels(i-1), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end

%% Group 1
nexttile(layout, 14, [1 2]); hold on;
box off; set(gca, 'TickDir', 'out');

group1_metrics = {'all_err_abs_mean', 'var_500ms_mean', 'all_var_mean'};
group1_labels  = {'AbsEndpointError', 'VarAfterStim', 'VarAfterEndpoint'};
n_metrics = numel(group1_metrics);
bar_data1 = nan(n_sets, n_metrics);
bar_std1  = nan(n_sets, n_metrics);
sig_labels1 = strings(n_metrics, n_sets - 1);

for m = 1:n_metrics
    metric_field = erase(group1_metrics{m}, '_mean');

    % Baseline values
    if strcmp(metric_field, 'all_err_abs')
        base_vals = abs(base_data.(base_side_label).all_err);
    else
        base_vals = base_data.(base_side_label).(metric_field);
    end
    base_vals = base_vals(~isnan(base_vals));

    bar_data1(1, m) = mean(base_vals);
    bar_std1(1, m)  = std(base_vals);

    for i = 2:n_sets
        cond_data = cond_data_list.(cond_labels{i});
        if strcmp(metric_field, 'all_err_abs')
            vals = abs(cond_data.all_err);
        else
            vals = cond_data.(metric_field);
        end
        vals = vals(~isnan(vals));

        bar_data1(i, m) = mean(vals);
        bar_std1(i, m)  = std(vals);

        [~, p] = ttest2(base_vals, vals);
        sig_labels1(m, i-1) = significance_label(p);
    end
end

x = 1:n_metrics;
bar_width = 0.15;
spacing_factor = 1.6;
offsets = linspace(-1, 1, n_sets) * bar_width * spacing_factor;

for i = 1:n_sets
    bar(x + offsets(i), bar_data1(i,:), bar_width, 'FaceColor', condition_colors(i,:));
    errorbar(x + offsets(i), bar_data1(i,:), bar_std1(i,:), 'k', 'linestyle', 'none', 'LineWidth', 1);
end

yl = ylim;
for m = 1:n_metrics
    for i = 2:n_sets
        xpos = x(m) + offsets(i);
        ypos = bar_data1(i,m) + bar_std1(i,m) + 0.05 * range(yl);
        text(xpos, ypos, sig_labels1(m,i-1), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

xticks(x);
xticklabels(group1_labels);
xtickangle(45);
ylabel('deg/s');
set(gca, 'FontSize', 12);
yline(0, '--k');
%% Group 2
nexttile(layout, 16); hold on;
box off; set(gca, 'TickDir', 'out');

group2_metrics = {'max_speed_mean', 'avg_speed_mean'};
group2_labels  = {'MaxSpeed', 'AvgSpeed'};
n_metrics2 = numel(group2_metrics);

bar_data2 = nan(n_sets, n_metrics2);
bar_std2  = nan(n_sets, n_metrics2);
sig_labels2 = strings(n_metrics2, n_sets - 1);
sample_sizes = zeros(n_sets, 1);

for m = 1:n_metrics2
    metric_field = erase(group2_metrics{m}, '_mean');
    base_vals = base_data.(base_side_label).(metric_field);
    base_vals = base_vals(~isnan(base_vals));

    sample_sizes(1) = numel(base_vals);
    bar_data2(1, m) = mean(base_vals);
    bar_std2(1, m)  = std(base_vals);

    for i = 2:n_sets
        cond_data = cond_data_list.(cond_labels{i});
        vals = cond_data.(metric_field);
        vals = vals(~isnan(vals));

        sample_sizes(i) = numel(vals);
        bar_data2(i, m) = mean(vals);
        bar_std2(i, m)  = std(vals);

        [~, p] = ttest2(base_vals, vals);
        sig_labels2(m, i-1) = significance_label(p);
    end
end

x2 = 1:n_metrics2;
bar_width = 0.15;
spacing_factor = 2.2;
offsets = linspace(-1, 1, n_sets) * bar_width * spacing_factor;

for i = 1:n_sets
    bar(x2 + offsets(i), bar_data2(i,:), bar_width, 'FaceColor', condition_colors(i,:), 'HandleVisibility', 'off');
    errorbar(x2 + offsets(i), bar_data2(i,:), bar_std2(i,:), 'k', 'linestyle', 'none', 'LineWidth', 1, 'HandleVisibility', 'off');
end

yl = ylim;
for m = 1:n_metrics2
    for i = 2:n_sets
        xpos = x2(m) + offsets(i);
        ypos = bar_data2(i, m) + bar_std2(i, m) + 0.05 * range(yl);
        text(xpos, ypos, sig_labels2(m, i-1), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

xticks(x2);
xticklabels(group2_labels);
xtickangle(45);
ylabel('deg/s');
set(gca, 'FontSize', 12);
yline(0, '--k');

h_legend = gobjects(1, n_sets);
for i = 1:n_sets
    h_legend(i) = bar(nan, nan, 'FaceColor', condition_colors(i,:), 'HandleVisibility', 'on');
end
legend(h_legend, arrayfun(@(i) sprintf('%s (n=%d)', legend_labels{i}, sample_sizes(i)), 1:n_sets, 'UniformOutput', false), 'Orientation', 'horizontal', 'Location', 'northoutside', 'Box', 'off', 'FontSize', 10);

%% Oscillations
nexttile(layout, 17); hold on;
box off; set(gca, 'TickDir', 'out', 'FontSize', 12);

osc_means = nan(1, n_sets);
osc_sems  = nan(1, n_sets);
osc_sig_labels = strings(1, n_sets - 1);

base_vals = base_data.(base_side_label).oscillations;
base_vals = base_vals(~isnan(base_vals));
osc_means(1) = mean(base_vals);
osc_sems(1)  = std(base_vals) / sqrt(numel(base_vals));

for i = 2:n_sets
    cond_data = cond_data_list.(cond_labels{i});
    vals = cond_data.oscillations;
    vals = vals(~isnan(vals));

    osc_means(i) = mean(vals);
    osc_sems(i)  = std(vals) / sqrt(numel(vals));

    [~, p] = ttest2(base_vals, vals);
    osc_sig_labels(i-1) = significance_label(p);
end

bar_osc = bar(1:n_sets, osc_means, 'FaceColor', 'flat');
bar_osc.CData = condition_colors;
errorbar(1:n_sets, osc_means, osc_sems, 'k', 'linestyle', 'none', 'LineWidth', 1);

yrange = range([osc_means - osc_sems, osc_means + osc_sems]);
ylim([min(osc_means - osc_sems) - 0.15 * yrange, max(osc_means + osc_sems) + 0.15 * yrange]);
xticks(mean(1:n_sets));
xticklabels({'EndpointOscillation'});
xtickangle(45);
ylabel('Count');
yline(0, '--k');

for i = 2:n_sets
    y_sig = osc_means(i) + osc_sems(i) + 0.05 * range(ylim);
    text(i, y_sig, osc_sig_labels(i-1), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end


%% FFT Power
nexttile(layout, 18); hold on;
box off; set(gca, 'TickDir', 'out', 'FontSize', 12);

fft_means = nan(1, n_sets);
fft_sems  = nan(1, n_sets);
fft_sig_labels = strings(1, n_sets - 1);

base_vals = base_data.(base_side_label).fft_power;
base_vals = base_vals(~isnan(base_vals));
fft_means(1) = mean(base_vals);
fft_sems(1)  = std(base_vals) / sqrt(numel(base_vals));

for i = 2:n_sets
    cond_data = cond_data_list.(cond_labels{i});
    vals = cond_data.fft_power;
    vals = vals(~isnan(vals));

    fft_means(i) = mean(vals);
    fft_sems(i)  = std(vals) / sqrt(numel(vals));

    [~, p] = ttest2(base_vals, vals);
    fft_sig_labels(i-1) = significance_label(p);
end

bar_fft = bar(1:n_sets, fft_means, 'FaceColor', 'flat');
bar_fft.CData = condition_colors;
errorbar(1:n_sets, fft_means, fft_sems, 'k', 'linestyle', 'none', 'LineWidth', 1);

yrange = range([fft_means - fft_sems, fft_means + fft_sems]);
ylim([min(fft_means - fft_sems) - 0.15 * yrange, max(fft_means + fft_sems) + 0.15 * yrange]);
xticks(mean(1:n_sets));
xticklabels({'FFTPowerAfterEnd'});
xtickangle(45);
ylabel('Power');
yline(0, '--k');

for i = 2:n_sets
    y_sig = fft_means(i) + fft_sems(i) + 0.05 * range(ylim);
    text(i, y_sig, fft_sig_labels(i-1), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end


%% === Metadata title ===
stim_str = sprintf('Ch: %g | Freq: %gHz | Curr: %gμA | Dur: %gms | Delay: %s | Depth: %gmm | Trig: %s', ...
    meta_cond.Channels, meta_cond.Stim_Frequency_Hz, meta_cond.Current_uA, ...
    meta_cond.Stim_Duration_ms, delay_str, meta_cond.Depth_mm, ...
    meta_cond.Movement_Trigger{1});

trial_num = meta_cond.BR_File;
title_str = sprintf('Adaptation condition %03d - %s', trial_num, suffix);
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
%% === Save figure in multiple formats ===
save_base = sprintf('%dCh_Condition_%03d_%s_AdaptationTraces', ...
    meta_cond.Channels, trial_num, polarity);
save_root = fullfile(base_folder, 'Figures', 'AdaptationTraces');

% Ensure subfolders exist
subfolders = {'pngFigs', 'svgFigs', 'figFigs'};
for k = 1:numel(subfolders)
    subfolder_path = fullfile(save_root, subfolders{k});
    if ~exist(subfolder_path, 'dir')
        mkdir(subfolder_path);
    end
end

% Filepaths
png_path = fullfile(save_root, 'pngFigs', [save_base '.png']);
svg_path = fullfile(save_root, 'svgFigs', [save_base '.svg']);
fig_path = fullfile(save_root, 'figFigs', [save_base '.fig']);

% Save
print(fig, png_path, '-dpng', '-r300');
set(fig, 'Renderer', 'painters');  % Ensure vector rendering
print(fig, svg_path, '-dsvg', '-painters');
savefig(fig, fig_path);
close(fig);
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