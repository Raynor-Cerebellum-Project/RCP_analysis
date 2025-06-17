clear all; close all; clc;

% File paths
filename_old = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_003_Session_1/Calibrated/IntanFile_3/BL_closed_loop_STIM_003_006_Cal_stim.mat';
filename_new = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_003_Session_1/Calibrated/IntanFile_3/BL_closed_loop_STIM_003_006_Cal_stim2.mat';

% Load data
load(filename_old);  % loads variable `Data`

%%
close all;
% Settings
fieldname = 'active_like_stim_pos_nan';
highlight_idx = [5];  % Trials to highlight and remove
%
%
segment = Data.segments.(fieldname);
n_trials = size(segment, 1);

% === Create subplot figure ===
figure('Name', 'Trial Removal Comparison', 'Position', [100, 100, 1400, 800]);

% --- Velocity BEFORE ---
subplot(2, 2, 1); hold on;
title(sprintf('Velocity Before Removal\n(%s)', fieldname), 'Interpreter', 'none');
for i = 1:n_trials
    idx1 = segment(i, 1) - 800;
    idx2 = segment(i, 2) + 1200;
    if idx1 < 1 || idx2 > length(Data.headYawVel), continue; end
    trace = Data.headYawVel(idx1:idx2);
    t = linspace(-800, 1200, length(trace));
    
    if ismember(i, highlight_idx)
        h = plot(t, trace, 'r', 'LineWidth', 2);
    else
        h = plot(t, trace, 'Color', [0.7 0.7 0.7]);
    end
    set(h, 'ButtonDownFcn', @(src, event) disp(['Clicked on trial #' num2str(i)]));
end
xlabel('Time (ms)'); ylabel('Head Yaw Velocity (deg/s)');
box off; set(gca, 'TickDir', 'out');

% --- Position BEFORE ---
subplot(2, 2, 3); hold on;
title(sprintf('Position Before Removal\n(%s)', fieldname), 'Interpreter', 'none');
for i = 1:n_trials
    idx1 = segment(i, 1) - 800;
    idx2 = segment(i, 2) + 1200;
    if idx1 < 1 || idx2 > length(Data.headYawPos), continue; end
    trace = Data.headYawPos(idx1:idx2);
    t = linspace(-800, 1200, length(trace));
    
    if ismember(i, highlight_idx)
        h = plot(t, trace, 'r', 'LineWidth', 2);
    else
        h = plot(t, trace, 'Color', [0.7 0.7 0.7]);
    end
    % Add interactive callback
    set(h, 'ButtonDownFcn', @(src, event) disp(['Clicked on trial #' num2str(i)]));
end
xlabel('Time (ms)'); ylabel('Head Yaw Position (deg)');
box off; set(gca, 'TickDir', 'out');
%% === Remove specified trials ===
segment(highlight_idx, :) = [];
Data.segments.(fieldname) = segment;

%% === Save updated data ===
save(filename_new, 'Data');
fprintf('Saved updated file with trials %s removed.\n', mat2str(highlight_idx));

% --- Velocity AFTER ---
subplot(2, 2, 2); hold on;
title(sprintf('Velocity After Removal\n(%s)', fieldname), 'Interpreter', 'none');
n_trials = size(segment, 1);
for i = 1:n_trials
    idx1 = segment(i, 1) - 800;
    idx2 = segment(i, 2) + 1200;
    if idx1 < 1 || idx2 > length(Data.headYawVel), continue; end
    trace = Data.headYawVel(idx1:idx2);
    % trace = trace - mean(trace(1:10));  % normalize
    t = linspace(-800, 1200, length(trace));
    plot(t, trace, 'Color', [0.3 0.7 1]);  % Blue after cleanup
end
xlabel('Time (ms)'); ylabel('Head Yaw Velocity (deg/s)');
box off; set(gca, 'TickDir', 'out');

% --- Position AFTER ---
subplot(2, 2, 4); hold on;
title(sprintf('Position After Removal\n(%s)', fieldname), 'Interpreter', 'none');
for i = 1:n_trials
    idx1 = segment(i, 1) - 800;
    idx2 = segment(i, 2) + 1200;
    if idx1 < 1 || idx2 > length(Data.headYawPos), continue; end
    trace = Data.headYawPos(idx1:idx2);
    % trace = trace - mean(trace(1:10));  % normalize
    t = linspace(-800, 1200, length(trace));
    plot(t, trace, 'Color', [0.3 0.7 1]);  % Blue after cleanup
end
xlabel('Time (ms)'); ylabel('Head Yaw Position (deg)');
box off; set(gca, 'TickDir', 'out');
