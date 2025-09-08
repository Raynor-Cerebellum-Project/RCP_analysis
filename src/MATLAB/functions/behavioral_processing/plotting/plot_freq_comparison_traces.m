function fig = plot_freq_comparison_traces(cond_data1, cond_data2, side_label, offset, meta_cond1, meta_cond2, use_ci)
if nargin < 8
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
cond_nums = [18, 8];  % specify conditions to compare

load(summary_path, 'summary_struct', 'merged_baseline_summary');
offset = false;
base_data = merged_baseline_summary;
T = readtable(meta_path);

meta_cond1 = T(T.BR_File == cond_nums(1), :);
meta_cond2 = T(T.BR_File == cond_nums(2), :);
cond_data1 = summary_struct(cond_nums(1)).merged_with_summary;
cond_data2 = summary_struct(cond_nums(2)).merged_with_summary;

%% === Trace alignment & setup ===
t = linspace(-800, 1200, 2001);
trace_spacing = 20 * offset;

% Parse polarity and side
tokens = regexp(side_label, 'active_like_stim_(pos|neg)', 'tokens');
polarity = tokens{1}{1};
is_ipsi = contains(side_label, 'pos');
suffix = ternary(is_ipsi, 'ipsi', 'contra');

% Extract condition frequencies from metadata
freq1 = meta_cond1.Stim_Frequency_Hz;
freq2 = meta_cond2.Stim_Frequency_Hz;

% Construct trial_sets and x_labels
trial_sets = {num2str(freq1), num2str(freq2)};
x_labels = {'Baseline', sprintf('%d Hz', freq1), sprintf('%d Hz', freq2)};
n_sets = numel(x_labels);

% Construct condition and summary labels
cond_labels = cell(1, n_sets);
summary_labels = cell(1, n_sets);

% 1. Baseline (from base_data)
cond_labels{1} = struct('data', base_data, 'label', sprintf('active_like_stim_%s', polarity));
summary_labels{1} = struct('data', base_data, 'label', sprintf('%s_%s_summary', suffix, polarity));

% 2. Condition 1 (from cond_data1)
cond_labels{2} = struct('data', cond_data1, 'label', sprintf('active_like_stim_%s', polarity));
summary_labels{2} = struct('data', cond_data1, 'label', sprintf('%s_%s_summary', suffix, polarity));

% 3. Condition 2 (from cond_data2)
cond_labels{3} = struct('data', cond_data2, 'label', sprintf('active_like_stim_%s', polarity));
summary_labels{3} = struct('data', cond_data2, 'label', sprintf('%s_%s_summary', suffix, polarity));

% Colors and legend labels
condition_colors = [
    0.6 0.6 0.6;  % Baseline
    0.0 0.4 1.0;  % Condition 1
    1.0 0.8 0.0;  % Condition 2
    ];
legend_labels = x_labels;

%% === Initialize figure and layout ===
fig = figure('Visible', 'on', 'Position', [100, 100, 1400, 800]);
layout = tiledlayout(3, 6, 'TileSpacing', 'compact', 'Padding', 'tight');

%% === Velocity Plots ===
trace_tiles = [1 3 5 7 9 11];
titles = {sprintf('%d Hz Vel', freq1), sprintf('%d Hz Vel', freq2), 'Overlay Vel', ...
    sprintf('%d Hz Pos', freq1), sprintf('%d Hz Pos', freq2), 'Overlay Pos'};
trace_colors = {[0.6 0.6 0.6], [0.6 0.6 0.6]};

% 1. Trial-wise velocity traces
ax_vel = gobjects(1, 3);
for k = 1:2
    ax_vel(k) = nexttile(layout, trace_tiles(k), [1 2]); hold on;
    title(titles{k}); ylabel('Velocity (deg/s)'); box off; set(gca, 'TickDir', 'out');

    cond_data = ternary(k == 1, cond_data1, cond_data2);
    fieldname = sprintf('active_like_stim_%s', polarity);
    vel_mat = cond_data.(fieldname).velocity_traces;

    for i = 1:size(vel_mat,1)
        stacked = vel_mat(i,:) + i * trace_spacing;
        plot(t, stacked, 'Color', trace_colors{k}, 'LineWidth', 0.75);
    end

    plot(t, nanmean(vel_mat, 1) + (size(vel_mat,1)+1)*trace_spacing, ...
        'Color', condition_colors(k+1,:), 'LineWidth', 2);
    xlim([-800 1200]);
end

% 2. Overlay velocity plot
ax_vel(3) = nexttile(layout, trace_tiles(3), [1 2]); hold on;
title(titles{3}); ylabel('Velocity (deg/s)'); box off; set(gca, 'TickDir', 'out');

base_field = sprintf('active_like_stim_%s', polarity);
baseline_vel = base_data.(base_field).velocity_traces;
vel1 = cond_data1.(base_field).velocity_traces;
vel2 = cond_data2.(base_field).velocity_traces;

STDplot(t, baseline_vel, [0 0 0]);
STDplot(t, vel1, condition_colors(2,:));
STDplot(t, vel2, condition_colors(3,:));
xlim([-800 1200]);

% 3. Stim bar overlay
if iscell(meta_cond1.Stim_Delay)
    delay = str2double(meta_cond1.Stim_Delay{1});
else
    delay = str2double(meta_cond1.Stim_Delay);
end
delay_str = sprintf('%dms', delay);
duration = meta_cond1.Stim_Duration_ms;

if ~isnan(delay) && ~isnan(duration)
    linkaxes(ax_vel, 'y');
    drawnow;
    shared_yl = ylim(ax_vel(3));
    stim_patch_x = [delay delay+duration delay+duration delay];
    stim_patch_y = [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)];

    for k = 1:3
        fill(ax_vel(k), stim_patch_x, stim_patch_y, ...
            [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end

% === Position Plots ===
ax_pos = gobjects(1, 3);
for k = 1:2
    ax_pos(k) = nexttile(layout, trace_tiles(k+3), [1 2]); hold on;
    title(titles{k+3}); ylabel('Position (deg)'); box off; set(gca, 'TickDir', 'out');

    cond_data = ternary(k == 1, cond_data1, cond_data2);
    fieldname = sprintf('active_like_stim_%s', polarity);
    pos_mat = cond_data.(fieldname).position_traces;

    for i = 1:size(pos_mat, 1)
        stacked = pos_mat(i,:) + i * trace_spacing;
        plot(t, stacked, 'Color', trace_colors{k}, 'LineWidth', 0.75);
    end

    plot(t, nanmean(pos_mat, 1) + (size(pos_mat,1)+1)*trace_spacing, ...
        'Color', condition_colors(k+1,:), 'LineWidth', 2);
    xlim([-800 1200]);
end

% Overlay position plot
ax_pos(3) = nexttile(layout, trace_tiles(6), [1 2]); hold on;
title(titles{6}); ylabel('Position (deg)'); box off; set(gca, 'TickDir', 'out');

base_field = sprintf('active_like_stim_%s', polarity);
baseline_pos = base_data.(base_field).position_traces;
pos1 = cond_data1.(base_field).position_traces;
pos2 = cond_data2.(base_field).position_traces;

STDplot(t, baseline_pos, [0 0 0]);
STDplot(t, pos1, condition_colors(2,:));
STDplot(t, pos2, condition_colors(3,:));
xlim([-800 1200]);

% Stim bar overlay for position
if ~isnan(delay) && ~isnan(duration)
    linkaxes(ax_pos, 'y');
    drawnow;
    shared_yl = ylim(ax_pos(3));
    stim_patch_x = [delay delay+duration delay+duration delay];
    stim_patch_y = [shared_yl(1) shared_yl(1) shared_yl(2) shared_yl(2)];

    for k = 1:3
        fill(ax_pos(k), stim_patch_x, stim_patch_y, ...
            [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end


%% === Endpoint Error Bar Plot (Group 1) ===
nexttile(layout, 13); hold on;
box off;
set(gca, 'TickDir', 'out', 'FontSize', 12);

err_means = nan(1, n_sets);
err_sems  = nan(1, n_sets);
sig_labels = strings(1, n_sets - 1);

% Baseline
base_field = sprintf('active_like_stim_%s', polarity);
baseline_trials = base_data.(base_field).all_err;
baseline_trials = baseline_trials(~isnan(baseline_trials));
err_means(1) = mean(baseline_trials);
err_sems(1)  = std(baseline_trials) / sqrt(numel(baseline_trials));

% Conditions
for i = 2:n_sets
    cond_data = ternary(i == 2, cond_data1, cond_data2);
    trial_vals = cond_data.(base_field).all_err;
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
xticks(x_single); xticklabels({'EndpointError'});
xtickangle(45);
ylabel('Endpoint Error (deg)');
yline(0, '--k');

for i = 2:n_sets
    y_sig = err_means(i) + err_sems(i) + 0.05 * yrange;
    text(i, y_sig, sig_labels(i-1), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end
%% === Group 1: Additional Kinematic Metrics ===
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

    % --- Baseline ---
    if strcmp(metric_field, 'all_err_abs')
        base_vals = abs(base_data.(base_field).all_err);
    else
        base_vals = base_data.(base_field).(metric_field);
    end
    base_vals = base_vals(~isnan(base_vals));

    bar_data1(1, m) = mean(base_vals);
    bar_std1(1, m)  = std(base_vals);

    % --- Conditions ---
    for i = 2:n_sets
        cond_data = ternary(i == 2, cond_data1, cond_data2);
        if strcmp(metric_field, 'all_err_abs')
            vals = abs(cond_data.(base_field).all_err);
        else
            vals = cond_data.(base_field).(metric_field);
        end
        vals = vals(~isnan(vals));

        bar_data1(i, m) = mean(vals);
        bar_std1(i, m)  = std(vals);

        [~, p] = ttest2(base_vals, vals);
        sig_labels1(m, i-1) = significance_label(p);
    end
end

% --- Plotting ---
x = 1:n_metrics;
bar_width = 0.15;
spacing_factor = 1.6;
offsets = linspace(-1, 1, n_sets) * bar_width * spacing_factor;

for i = 1:n_sets
    bar(x + offsets(i), bar_data1(i,:), bar_width, 'FaceColor', condition_colors(i,:));
    errorbar(x + offsets(i), bar_data1(i,:), bar_std1(i,:), ...
        'k', 'linestyle', 'none', 'LineWidth', 1);
end

yl = ylim;
for m = 1:n_metrics
    for i = 2:n_sets
        xpos = x(m) + offsets(i);
        ypos = bar_data1(i,m) + bar_std1(i,m) + 0.05 * range(yl);
        text(xpos, ypos, sig_labels1(m,i-1), ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

xticks(x);
xticklabels(group1_labels);
xtickangle(45);
ylabel('deg/s');
set(gca, 'FontSize', 12);
yline(0, '--k');

%% === Group 2: Speed Metrics ===
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

    % --- Baseline ---
    base_vals = base_data.(base_field).(metric_field);
    base_vals = base_vals(~isnan(base_vals));
    sample_sizes(1) = numel(base_vals);

    bar_data2(1, m) = mean(base_vals);
    bar_std2(1, m)  = std(base_vals);

    % --- Conditions ---
    for i = 2:n_sets
        cond_data = ternary(i == 2, cond_data1, cond_data2);
        vals = cond_data.(base_field).(metric_field);
        vals = vals(~isnan(vals));
        sample_sizes(i) = numel(vals);

        bar_data2(i, m) = mean(vals);
        bar_std2(i, m)  = std(vals);

        [~, p] = ttest2(base_vals, vals);
        sig_labels2(m, i-1) = significance_label(p);
    end
end

% Plotting
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

% Legend
h_legend = gobjects(1, n_sets);
for i = 1:n_sets
    h_legend(i) = bar(nan, nan, 'FaceColor', condition_colors(i,:), 'HandleVisibility', 'on');
end
legend(h_legend, arrayfun(@(i) sprintf('%s (n=%d)', legend_labels{i}, sample_sizes(i)), 1:n_sets, 'UniformOutput', false), ...
    'Orientation', 'horizontal', 'Location', 'northoutside', 'Box', 'off', 'FontSize', 10);

%% === Oscillations After Endpoint ===
nexttile(layout, 17); hold on;
box off; set(gca, 'TickDir', 'out', 'FontSize', 12);

osc_means = nan(1, n_sets);
osc_sems  = nan(1, n_sets);
osc_sig_labels = strings(1, n_sets - 1);

base_vals = base_data.(base_field).oscillations;
base_vals = base_vals(~isnan(base_vals));
osc_means(1) = mean(base_vals);
osc_sems(1)  = std(base_vals) / sqrt(numel(base_vals));

for i = 2:n_sets
    cond_data = ternary(i == 2, cond_data1, cond_data2);
    vals = cond_data.(base_field).oscillations;
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

%% === FFT Power After Endpoint ===
nexttile(layout, 18); hold on;
box off; set(gca, 'TickDir', 'out', 'FontSize', 12);

fft_means = nan(1, n_sets);
fft_sems  = nan(1, n_sets);
fft_sig_labels = strings(1, n_sets - 1);

base_vals = base_data.(base_field).fft_power;
base_vals = base_vals(~isnan(base_vals));
fft_means(1) = mean(base_vals);
fft_sems(1)  = std(base_vals) / sqrt(numel(base_vals));

for i = 2:n_sets
    cond_data = ternary(i == 2, cond_data1, cond_data2);
    vals = cond_data.(base_field).fft_power;
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



%% === Metadata Title and Layout ===
trial_num1 = meta_cond1.BR_File;
trial_num2 = meta_cond2.BR_File;
freq1 = meta_cond1.Stim_Frequency_Hz;
freq2 = meta_cond2.Stim_Frequency_Hz;

title_str = sprintf('Conditions %03d (%d Hz) vs %03d (%d Hz) - %s', ...
    trial_num1, freq1, trial_num2, freq2, suffix);

stim_str = sprintf('Ch: %g | Curr: %g μA | Dur: %g ms | Delay: %s | Depth: %g mm | Trig: %s', ...
    meta_cond1.Channels, meta_cond1.Current_uA, ...
    meta_cond1.Stim_Duration_ms, delay_str, meta_cond1.Depth_mm, ...
    meta_cond1.Movement_Trigger{1});

sgtitle({title_str, stim_str}, 'FontWeight', 'bold', 'FontSize', 14);

% Adjust layout vertically to fit sgtitle
outerpos = get(layout, 'OuterPosition');
outerpos(2) = outerpos(2) + 0.08;
outerpos(4) = outerpos(4) - 0.08;
set(layout, 'OuterPosition', outerpos);

% Add boxed significance legend
annotation('textbox', [0.35, 0.01, 0.3, 0.08], ...
    'String', {'\bf{Significance Legend}', ...
    '*   p < 0.05    **   p < 0.01    ***   p < 0.001    n.s. = not significant'}, ...
    'EdgeColor', 'k', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 11, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white');
%% === Save Figure in Multiple Formats ===
% Construct save directory and filename
save_base = sprintf('%dCh_Condition_%03d_%03d_%s_FrequencyComparison', ...
    meta_cond1.Channels, trial_num1, trial_num2, polarity);
save_root = fullfile(base_folder, 'Figures', 'FrequencyComparison');

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