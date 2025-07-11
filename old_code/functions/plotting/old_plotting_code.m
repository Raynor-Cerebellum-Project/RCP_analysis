%% Plot error
x = categorical(sort_key.Label, sort_key.Label, 'Ordinal', true);

bar_data = [sort_key.MeanIpsi, sort_key.MeanContra];
bar_errs = [sort_key.StdIpsi, sort_key.StdContra];

figure('Position', [100 100 1200 500]);
b = bar(x, bar_data, 'grouped');
hold on;

% Error bars
ngroups = size(bar_data, 1);
nbars   = size(bar_data, 2);
xvals   = nan(nbars, ngroups);
for i = 1:nbars
    xvals(i,:) = b(i).XEndPoints;
end
for i = 1:nbars
    errorbar(xvals(i,:), bar_data(:,i), bar_errs(:,i), '.k', 'LineWidth', 1.5);
end

% Format
ylabel('Absolute Endpoint Error (deg)');
xlabel('Stimulation Condition');
legend(b, {'Ipsi', 'Contra'});

combined_title = sprintf('Endpoint Errors: %s\n%s', eval_condition, plot_title);
sgtitle(combined_title, ...
    'FontWeight', 'bold', ...
    'FontSize', 14);

xtickangle(45);
box off; set(gca, 'TickDir', 'out');

%% Add lines
% Get trigger labels in the same order as the x-axis
trigger_group = sort_key.Trigger;

% Convert to string for comparison
trigger_str = string(trigger_group);

% Find group boundaries
begin_end_idx = find(trigger_str == "Beginning" | trigger_str == "End");
other_idx     = find(~(trigger_str == "Beginning" | trigger_str == "End"));

% Find last index of each section
last_other     = max(other_idx);
last_beginning = max(find(trigger_str == "Beginning"));
last_end       = max(find(trigger_str == "End"));

% Get x-axis tick positions
for i = 1:nbars
    xvals(i,:) = b(i).XEndPoints;
end

% Choose one row of xvals for positioning (all rows same x)
xpos = xvals(1,:);

% Draw lines between the three sections
if ~isempty(last_other)
    xline(xpos(last_other) + 0.5, '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end
if ~isempty(last_beginning)
    xline(xpos(last_beginning) + 0.5, '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end

% Add text labels above each section
% Expand y-limit manually to leave headroom for section labels
yl = ylim;
ylim([yl(1), yl(2) * 1.15]);  % Increase top y-limit

% Get the new expanded top limit
yl = ylim;
label_y = yl(2) * 0.98;
group1_idx = 1:last_other;
group2_idx = (last_other+1):last_beginning;
group3_idx = (last_beginning+1):length(xpos);

text(mean(xpos(group1_idx)), label_y, 'Baselines', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
text(mean(xpos(group2_idx)), label_y, 'Beginning', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
text(mean(xpos(group3_idx)), label_y, 'End', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);

%% Save
filename = fullfile(base_folder, 'Figures', [eval_condition, '_', session, '_', 'Mean_Error_Comparison.jpg']);
mkdir(fileparts(filename));  % Ensure folder exists
print(gcf, filename, '-djpeg', '-r300');
%% Plot variability
x = categorical(sort_key.Label, sort_key.Label, 'Ordinal', true);

bar_data = [sort_key.VarIpsi, sort_key.VarContra];
bar_errs = [sort_key.VarStdIpsi, sort_key.VarStdContra];

figure('Position', [100 100 1200 500]);
b = bar(x, bar_data, 'grouped');
hold on;

% Error bars
ngroups = size(bar_data, 1);
nbars   = size(bar_data, 2);
xvals   = nan(nbars, ngroups);
for i = 1:nbars
    xvals(i,:) = b(i).XEndPoints;
end
for i = 1:nbars
    errorbar(xvals(i,:), bar_data(:,i), bar_errs(:,i), '.k', 'LineWidth', 1.5);
end

% Format
ylabel('Absolute Endpoint Variability (deg)');
xlabel('Stimulation Condition');
legend(b, {'Ipsi', 'Contra'});

combined_title = sprintf('Endpoint Variability: %s\n%s', eval_condition, plot_title);
sgtitle(combined_title, ...
    'FontWeight', 'bold', ...
    'FontSize', 14);

xtickangle(45);
box off; set(gca, 'TickDir', 'out');

%% Add lines
% Get trigger labels in the same order as the x-axis
trigger_group = sort_key.Trigger;

% Convert to string for comparison
trigger_str = string(trigger_group);

% Find group boundaries
begin_end_idx = find(trigger_str == "Beginning" | trigger_str == "End");
other_idx     = find(~(trigger_str == "Beginning" | trigger_str == "End"));

% Find last index of each section
last_other     = max(other_idx);
last_beginning = max(find(trigger_str == "Beginning"));
last_end       = max(find(trigger_str == "End"));

% Get x-axis tick positions
for i = 1:nbars
    xvals(i,:) = b(i).XEndPoints;
end

% Choose one row of xvals for positioning (all rows same x)
xpos = xvals(1,:);

% Draw lines between the three sections
if ~isempty(last_other)
    xline(xpos(last_other) + 0.5, '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end
if ~isempty(last_beginning)
    xline(xpos(last_beginning) + 0.5, '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end

% Add text labels above each section
% Expand y-limit manually to leave headroom for section labels
yl = ylim;
ylim([yl(1), yl(2) * 1.15]);  % Increase top y-limit

% Get the new expanded top limit
yl = ylim;
label_y = yl(2) * 0.98;
group1_idx = 1:last_other;
group2_idx = (last_other+1):last_beginning;
group3_idx = (last_beginning+1):length(xpos);

text(mean(xpos(group1_idx)), label_y, 'Baselines', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
text(mean(xpos(group2_idx)), label_y, 'Beginning', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
text(mean(xpos(group3_idx)), label_y, 'End', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
%% Save
filename = fullfile(base_folder, 'Figures', [eval_condition, '_', session, '_', 'End_point_Variability_Comparison.jpg']);
print(gcf, filename, '-djpeg', '-r300');

%% Line plot
% Ensure T_filtered is in same order as sort_key
[~, match_idx] = ismember(sort_key.TrialID, T_filtered.BR_File);
T_filtered_sorted = T_filtered(match_idx, :);
y_err = T_filtered_sorted.ErrMean_combined;
y_var = T_filtered_sorted.varMean_combined;
err_std = T_filtered_sorted.ErrStd_combined;
var_std = T_filtered_sorted.varStd_combined;

% x-axis: trial index or condition index
x = categorical(sort_key.Label, sort_key.Label, 'Ordinal', true);
x_vals = 1:length(x);

% Create figure
figure('Position', [100 100 1000 400]);

% Left axis: Endpoint Error
yyaxis left
plot(x_vals, y_err, '-o', 'LineWidth', 2, 'Color', [0.2 0.2 0.8], ...
    'DisplayName', 'Combined Endpoint Error');
ylabel('Endpoint Error (deg)');

% Right axis: Variability
yyaxis right
plot(x_vals, y_var, '-s', 'LineWidth', 2, 'Color', [0.85 0.33 0.1], ...
    'DisplayName', 'Endpoint Variability');
ylabel('Endpoint Variability (deg)');

% Shared x-axis
xticks(x_vals);
xticklabels(cellstr(x));
xtickangle(45);
xlabel('Stimulation Condition');
xlim([0.2, length(x) + 0.5]);  % Add left-side space
% Move title higher and prevent overlap
% Combine both messages into one sgtitle with line break
combined_title = sprintf('Endpoint Errors and Variability: %s\n%s', eval_condition, plot_title);


title(combined_title, 'Units', 'normalized', ...
    'Position', [0.5, 1.00, 0], ...
    'FontWeight', 'normal', 'Interpreter', 'none');
legend('Location', 'northeast');
box off;
set(gca, 'TickDir', 'out');
%% Add lines
% Get trigger labels in same order as x
trigger_group = sort_key.Trigger;
trigger_str   = string(trigger_group);

% Indices of groups
begin_end_idx = find(trigger_str == "Beginning" | trigger_str == "End");
other_idx     = find(~(trigger_str == "Beginning" | trigger_str == "End"));

% Section boundaries
last_other     = max(other_idx);
last_beginning = max(find(trigger_str == "Beginning"));

% Get x-axis positions (numeric)
xpos = 1:length(x); % same as x_vals

% Draw vertical lines between sections
if ~isempty(last_other)
    xline(xpos(last_other) + 0.5, '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end
if ~isempty(last_beginning)
    xline(xpos(last_beginning) + 0.5, '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end

% Add text labels above each section
% Expand y-limit manually to leave headroom for section labels
yl = ylim;
ylim([yl(1), yl(2) * 1.15]);  % Increase top y-limit

% Get the new expanded top limit
yl = ylim;
label_y = yl(2) * 0.98;
group1_idx = 1:last_other;
group2_idx = (last_other+1):last_beginning;
group3_idx = (last_beginning+1):length(xpos);

text(mean(xpos(group1_idx)), label_y, 'Baselines', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
text(mean(xpos(group2_idx)), label_y, 'Beginning', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
text(mean(xpos(group3_idx)), label_y, 'End', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);

%% Save
filename = fullfile(base_folder, 'Figures', [eval_condition, '_', session, '_', 'Combined_Line_Graph_Comparison.jpg']);
print(gcf, filename, '-djpeg', '-r300');