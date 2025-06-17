clear all; close all; clc;

% File paths
filename_old = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_003_Session_1/Calibrated/IntanFile_8/BL_closed_loop_STIM_003_010_Cal_stim.mat';
filename_new = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_003_Session_1/Calibrated/IntanFile_8/BL_closed_loop_STIM_003_010_Cal_stim_curated.mat';

load(filename_old);  % loads `Data`

% === Parameters ===
fs = 1000;
neural_fs = 30000;
t = linspace(-800, 1200, 2001);
stim_thresh = 2;
min_gap = round(0.15 * neural_fs);  % 150 ms

% === Detect stim rising edges ===
stim_signal = Data.Neural(:,1);
stim_binary = stim_signal > stim_thresh;
rising_edges = find(diff(stim_binary) == 1) + 1;
burst_starts = rising_edges([true; diff(rising_edges) > min_gap]);
stim_times_1khz = round(burst_starts / (neural_fs / fs));

%% === Trial to curate ===
trial_num = 10;
highlight_idx = [4, 6];  % update manually if needed

% === Extract traces aligned to stim ===
N = numel(stim_times_1khz);
velocity_traces = nan(N, 2001);
position_traces = nan(N, 2001);

for i = 1:N
    idx0 = stim_times_1khz(i);
    win = idx0 - 800 : idx0 + 1200;
    if win(1) < 1 || win(end) > length(Data.headYawVel), continue; end
    velocity_traces(i,:) = Data.headYawVel(win);
    position_traces(i,:) = Data.headYawPos(win);
end

% === Plot BEFORE removal ===
figure('Name', sprintf('Curate Trial %d (Stim-Aligned)', trial_num), 'Position', [100, 100, 1400, 800]);

subplot(2,2,1); hold on;
title('Velocity (Before)');
for i = 1:N
    if ismember(i, highlight_idx)
        h = plot(t, velocity_traces(i,:), 'r', 'LineWidth', 2);
    else
        h = plot(t, velocity_traces(i,:), 'Color', [0.7 0.7 0.7]);
    end
    set(h, 'ButtonDownFcn', @(~,~) fprintf('Clicked on stim trial %d\n', i));
end
xlabel('Time (ms)'); ylabel('Velocity (deg/s)');
box off; set(gca, 'TickDir', 'out');

subplot(2,2,3); hold on;
title('Position (Before)');
for i = 1:N
    if ismember(i, highlight_idx)
        h = plot(t, position_traces(i,:), 'r', 'LineWidth', 2);
    else
        h = plot(t, position_traces(i,:), 'Color', [0.7 0.7 0.7]);
    end
    set(h, 'ButtonDownFcn', @(~,~) fprintf('Clicked on stim trial %d\n', i));
end
xlabel('Time (ms)'); ylabel('Position (deg)');
box off; set(gca, 'TickDir', 'out');
%%
% === Remove matching rows from Data.segments ===
if isfield(Data, 'segments')
    segment_fields = fieldnames(Data.segments);
    for f = 1:numel(segment_fields)
        seg_field = segment_fields{f};
        seg = Data.segments.(seg_field);
        if size(seg,2) ~= 2, continue; end

        % Build a logical mask to remove rows with stim-aligned start
        remove_mask = false(size(seg,1),1);
        for j = 1:size(seg,1)
            for i = highlight_idx
                stim_start = stim_times_1khz(i);
                if abs(seg(j,1) - stim_start) <= 10  % allow small tolerance
                    remove_mask(j) = true;
                end
            end
        end

        % Apply removal
        Data.segments.(seg_field)(remove_mask, :) = [];
    end
end

stim_times_1khz_clean = stim_times_1khz;
stim_times_1khz_clean(highlight_idx) = [];

% Re-extract traces
N_clean = numel(stim_times_1khz_clean);
velocity_clean = nan(N_clean, 2001);
position_clean = nan(N_clean, 2001);

for i = 1:N_clean
    idx0 = stim_times_1khz_clean(i);
    win = idx0 - 800 : idx0 + 1200;
    if win(1) < 1 || win(end) > length(Data.headYawVel), continue; end
    velocity_clean(i,:) = Data.headYawVel(win);
    position_clean(i,:) = Data.headYawPos(win);
end

% === Plot AFTER removal ===
subplot(2,2,2); hold on;
title('Velocity (After)');
for i = 1:N_clean
    plot(t, velocity_clean(i,:), 'Color', [0.3 0.7 1]);
end
xlabel('Time (ms)'); ylabel('Velocity (deg/s)');
box off; set(gca, 'TickDir', 'out');

subplot(2,2,4); hold on;
title('Position (After)');
for i = 1:N_clean
    plot(t, position_clean(i,:), 'Color', [0.3 0.7 1]);
end
xlabel('Time (ms)'); ylabel('Position (deg)');
box off; set(gca, 'TickDir', 'out');

% === Optional: update stim index list in Data (if you store them)
% You could save the cleaned stim_times_1khz_clean if needed
% Or remove related trials from Data.segments if desired

% === Save cleaned file ===
save(filename_new, 'Data');
fprintf('Saved cleaned file to: %s\nRemoved stim trials: %s\n', filename_new, mat2str(highlight_idx));
