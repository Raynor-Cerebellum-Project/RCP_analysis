clear all; close all; clc;
addpath(genpath('functions'));

session = 'BL_RW_003_Session_1';
base_folder = fullfile('/Volumes/CullenLab_Server/Current Project Databases - NHP', ...
    '2025 Cerebellum prosthesis/Bryan/Data', session);
search_folder = fullfile(base_folder, 'Calibrated');

% === Identify File ===
br_num = 10;  % example: change as needed
% === File Info ===
filepath = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_003_Session_1/Calibrated/IntanFile_8/BL_closed_loop_STIM_003_010_Cal_stim_curated.mat';
% === Load metadata ===
meta_path = fullfile(base_folder, [session '_metadata_with_metrics.csv']);
T = readtable(meta_path);

% Extract metadata for current trial
meta_cond = T(T.BR_File == br_num, :);

% Load data
load(filepath, 'Data');
zerovel = 49.5531;

fs = 1000;  % Hz
t = linspace(-800, 1200, 2001);

% === Detect stim rising from Neural (Channel 1) ===
stim_signal = Data.Neural(:, 1);
stim_thresh = 2;
stim_binary = stim_signal > stim_thresh;
rising_edges = find(diff(stim_binary) == 1) + 1;

% === Filter rising events: keep those separated by >150ms ===
neural_fs = 30000;
min_gap = round(0.15 * neural_fs);
burst_starts = rising_edges([true; diff(rising_edges) > min_gap]);

% Convert to 1kHz index
stim_times_1khz = round(burst_starts / (neural_fs / fs));

% === Manually exclude selected stim trials ===
remove_idx = [1, 4, 6];  % indices of stim_times_1khz to exclude
stim_times_1khz(remove_idx) = [];

% === Extract segments aligned to stim ===
N = numel(stim_times_1khz);
velocity_traces = nan(N, 2001);
position_traces = nan(N, 2001);

for i = 1:N
    idx0 = stim_times_1khz(i);
    win = idx0 - 800 : idx0 + 1200;
    if win(1) < 1 || win(end) > length(Data.headYawVel), continue; end
    velocity_traces(i, :) = Data.headYawVel(win)+zerovel;
    position_traces(i, :) = Data.headYawPos(win);
end

%% === Plot ===
figure('Position', [100, 100, 1000, 600]);
layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

stim_window = [0, 100];  % ms
stim_color = [1 0 0];    % red
ax_handles = gobjects(4,1);  % to store axes handles

% --- Row 1, Col 1: Velocity individual traces ---
ax_handles(1) = nexttile(1); hold on;
plot(t, velocity_traces', 'Color', [0.6 0.6 0.6 0.3]);
mean_vel = nanmean(velocity_traces,1);
h_mean_vel = plot(t, mean_vel, 'm', 'LineWidth', 2);
yline(0, 'k--', 'LineWidth', 1);
xlabel('Time (ms)'); ylabel('Velocity (deg/s)');
title('Velocity (all traces)');
legend(h_mean_vel, sprintf('Mean (n = %d)', N), 'Location', 'northeast');

% --- Row 1, Col 2: Velocity STDplot ---
ax_handles(2) = nexttile(2); hold on;
STDplot(t, velocity_traces, [1 0 1]);  % magenta
yline(0, 'k--', 'LineWidth', 1);
xlabel('Time (ms)'); ylabel('Velocity (deg/s)');
title('Velocity (mean ± SD)');
legend(sprintf('n = %d', N), 'Location', 'northeast');

% --- Row 2: Demeaned Position Traces ---
demeaned_pos = position_traces - mean(position_traces, 2, 'omitnan');  % Demean per trial

% --- Row 2, Col 1: Demeaned individual position traces ---
ax_handles(3) = nexttile(3); hold on;
plot(t, demeaned_pos', 'Color', [0.3 0.3 1 0.3]);
mean_pos = nanmean(demeaned_pos, 1);
h_mean_pos = plot(t, mean_pos, 'b', 'LineWidth', 2);
xlabel('Time (ms)'); ylabel('\Delta position (deg)');
title('Position (demeaned traces)');
legend(h_mean_pos, sprintf('Mean (n = %d)', N), 'Location', 'northeast');

% --- Row 2, Col 2: STD of demeaned position traces ---
ax_handles(4) = nexttile(4); hold on;
STDplot(t, demeaned_pos, [0 0 1]);  % blue
xlabel('Time (ms)'); ylabel('\Delta Position (deg)');
title('Position (demeaned, mean ± SD)');
legend(sprintf('n = %d', N), 'Location', 'northeast');

% === Align axes vertically (rows) ===
linkaxes(ax_handles(1:2), 'y');  % velocity row
linkaxes(ax_handles(3:4), 'y');  % position row

% === Add stim bar AFTER aligning y-limits ===
for i = 1:4
    axes(ax_handles(i));  % make current
    yL = ylim();
    fill([stim_window, fliplr(stim_window)], ...
         [yL(1) yL(1) yL(2) yL(2)], ...
         stim_color, 'FaceAlpha', 0.1, 'EdgeColor', 'none', ...
     'HandleVisibility', 'off');
end

% === Title using Metadata ===
stim_str = sprintf('Ch: %g | Freq: %gHz | Curr: %gμA | Dur: %gms | Depth: %gmm', ...
    meta_cond.Channels, meta_cond.Stim_Frequency_Hz, meta_cond.Current_uA, ...
    meta_cond.Stim_Duration_ms, meta_cond.Depth_mm);

% Get trial number from metadata
trial_num = meta_cond.BR_File;

% Cleaned side label for title
if exist('side_label', 'var')
    label_str = strrep(side_label, '_', ' ');
else
    label_str = 'NoMovement';
end

title_str = sprintf('Stim-aligned: Condition %03d — %s', trial_num, label_str);

% Add multi-line super title
sgtitle({title_str, stim_str}, 'FontWeight', 'bold', 'FontSize', 14, 'Interpreter', 'none');

% === Save figure as SVG ===
save_base = sprintf('%dCh_Condition_%03d_NoMovementTraces', ...
    meta_cond.Channels, br_num);
save_root = fullfile(base_folder, 'Figures', 'NoMovementTraces');

% Ensure subfolders exist
mkdir(fullfile(save_root, 'pngFigs'));
mkdir(fullfile(save_root, 'svgFigs'));
mkdir(fullfile(save_root, 'figFigs'));

% Filepaths
png_path = fullfile(save_root, 'pngFigs', [save_base '.png']);
svg_path = fullfile(save_root, 'svgFigs', [save_base '.svg']);
fig_path = fullfile(save_root, 'figFigs', [save_base '.fig']);

% Save
set(gcf, 'Renderer', 'painters');  % Best for vector output
print(gcf, png_path, '-dpng', '-r300');
set(gcf, 'Renderer', 'painters');  % Ensure vector rendering
print(gcf, svg_path, '-dsvg', '-painters');
savefig(gcf, fig_path);
close(gcf);