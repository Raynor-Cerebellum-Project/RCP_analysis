function plot_metric_by_condition(T, condition_field, include_trials, session_folder)
% Plots metrics per trial (no grouping). X-axis is condition_field. Includes trial IDs and counts.

% T_path = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_002_Session_1/BL_RW_002_Session_1_metadata_with_metrics.csv';
% T = readtable(T_path);
% 
% session = 'BL_RW_002_Session_1';
% session_folder = fullfile(['/Volumes/CullenLab_Server/Current Project Databases - NHP' ...
%     '/2025 Cerebellum prosthesis/Bryan/Data'], session);
% condition_field = 'Stim_Delay';
% include_trials = [4, 21:27];

if nargin < 3 || isempty(include_trials)
    include_trials = unique(T.BR_File);
end
if nargin < 4
    session_folder = pwd;
end

% Step 1: Load only selected trials
Tsub = T(ismember(T.BR_File, include_trials), :);

% Step 2: Remove trials with zero counts
nonzero_trials = ~(Tsub.n_trials_ipsi == 0 & Tsub.n_trials_contra == 0);
Tsub = Tsub(nonzero_trials, :);

% Step 3: Subtract baseline for error only
is_baseline = strcmp(Tsub.Movement_Trigger, 'Baseline Active HoB');

% Separate baseline and non-baseline trials
Tbase = Tsub(is_baseline, :);
Tnon  = Tsub(~is_baseline, :);

% Metadata fields for commonness check
meta_fields = {'Stim_Frequency_Hz', 'Stim_Delay', 'Current_uA', 'Depth_mm', 'Stim_Duration_ms'};
all_meta_constant = true;
for f = 1:numel(meta_fields)
    if ismember(meta_fields{f}, Tnon.Properties.VariableNames)
        col = Tnon.(meta_fields{f});
        if any(col ~= col(1))
            all_meta_constant = false;
            break;
        end
    end
end

% Auto-switch to channel labeling if all other metadata are constant
if ~strcmp(condition_field, 'Channels') && all_meta_constant && ismember('Channels', T.Properties.VariableNames)
    condition_field = 'Channels';
end

% Sort only the non-baseline trials by condition_field if it exists
if ismember(condition_field, T.Properties.VariableNames)
    [~, sort_idx] = sort(Tnon.(condition_field));
    Tnon = Tnon(sort_idx, :);
end


% Combine back
Tsub = [Tbase; Tnon];

% Recompute baseline values using baseline rows
baseline_ipsi = mean(Tbase.all_err_mean_ipsi(~isnan(Tbase.all_err_mean_ipsi)));
baseline_contra = mean(Tbase.all_err_mean_contra(~isnan(Tbase.all_err_mean_contra)));

metrics = struct( ...
    'Error_ipsi',     Tsub.all_err_mean_ipsi - baseline_ipsi, ...
    'Error_contra',   Tsub.all_err_mean_contra - baseline_contra, ...
    'Var_ipsi',       Tsub.all_var_mean_ipsi, ...
    'Var_contra',     Tsub.all_var_mean_contra ...
);

n_trials = struct( ...
    'ipsi',   Tsub.n_trials_ipsi, ...
    'contra', Tsub.n_trials_contra ...
);

titles = {
    'Δ Error (vs. Baseline) - Ipsi', ...
    'Δ Error (vs. Baseline) - Contra', ...
    'Variability - Ipsi', ...
    'Variability - Contra' ...
};
ylabels = {
    'Δ Signed Endpoint Error (deg)', ...
    'Δ Signed Endpoint Error (deg)', ...
    'Endpoint Variability (deg/s)', ...
    'Endpoint Variability (deg/s)' ...
};
colors = {[0.2 0.4 0.8], [0.9 0.4 0.1], [0.2 0.6 1.0], [1.0 0.5 0.3]};
data_fields = {'Error_ipsi', 'Error_contra', 'Var_ipsi', 'Var_contra'};
n_fields = {'ipsi', 'contra', 'ipsi', 'contra'};

x = 1:height(Tsub);
if strcmp(condition_field, 'Channels')
    x_labels = string(Tsub.Channels);
elseif ismember(condition_field, T.Properties.VariableNames)
    x_labels = string(Tsub.(condition_field));
else
    x_labels = repmat("Trial", height(Tsub), 1);  % Fallback label
end

fig = figure('Visible', 'off', 'Position', [100, 100, 1100, 500]);
layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Adjust layout to make room for annotation at the bottom
outerpos = get(layout, 'OuterPosition');
outerpos(2) = outerpos(2) + 0.12;
outerpos(4) = outerpos(4) - 0.12;
set(layout, 'OuterPosition', outerpos);

% Build summary strings with baseline label
trial_ids = string(Tsub.BR_File);
is_base   = ismember(Tsub.BR_File, Tbase.BR_File);
trial_labels = trial_ids;
x_labels(is_base) = "Baseline";

% Collect common metadata for sgtitle
meta_fields = {'Channels', 'Stim_Frequency_Hz', 'Stim_Delay', 'Current_uA', 'Depth_mm', 'Stim_Duration_ms'};
field_labels = {'Channels', 'Freq', 'Delay', 'Current', 'Depth', 'Duration'};
field_units  = {'', 'Hz', 'ms', 'uA', 'mm', 'ms'};
param_strs = {};
for f = 1:numel(meta_fields)
    if ismember(meta_fields{f}, T.Properties.VariableNames)
        col_non = Tnon.(meta_fields{f});
        if all(col_non == col_non(1))
            param_strs{end+1} = sprintf('%s: %g %s', field_labels{f}, col_non(1), field_units{f}); %#ok<AGROW>
        end
    end
end
if ~isempty(param_strs)
    sgtitle(strjoin(param_strs, ', '), 'FontWeight', 'bold');
end

% Split by ipsi / contra
cond_summary.ipsi = strjoin(trial_labels, ', ');
cond_summary.contra = cond_summary.ipsi;
n_summary.ipsi   = strjoin(string(n_trials.ipsi), ', ');
n_summary.contra = strjoin(string(n_trials.contra), ', ');

for i = 1:4
    nexttile(i);
    Y = metrics.(data_fields{i});
    n_val = n_trials.(n_fields{i});
    valid_idx = ~isnan(Y) & n_val > 0;

    bar(x(valid_idx), Y(valid_idx), 0.6, 'FaceColor', colors{i});
    set(gca, 'XTick', x(valid_idx), 'XTickLabel', x_labels(valid_idx), 'XTickLabelRotation', 45);
    ylabel(ylabels{i});
    title(titles{i});
    if i > 2
        xlabel(strrep(condition_field, '_', ' '));
    end
    grid on;
end

% Add summary text per column
for col = 1:2
    if col == 1
        side = 'ipsi'; xpos = 0.25;
    else
        side = 'contra'; xpos = 0.75;
    end

    ypos = 0.005;
    fontsize = 10;

    annotation('textbox', [xpos-0.2, ypos, 0.4, 0.08], ...
        'String', {
            sprintf('Cond #: [%s]', cond_summary.(side)), ...
            sprintf('n = [%s]', n_summary.(side))
        }, ...
        'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', ...
        'FontSize', fontsize);
end

% Save with condition field in filename
out_dir = fullfile(session_folder, 'Figures', 'GroupedMetrics');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
fname = sprintf('group_by_%s_derror_variability.png', lower(condition_field));
saveas(gcf, fullfile(out_dir, fname));
close(fig);
fprintf('Saved: %s\n', fullfile(out_dir, fname));
end
