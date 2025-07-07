clear all; clc;
%% === Setup Paths and Trial Info ===
fs = 30000;  % Sampling frequency (Hz)

base_folder = '/Volumes/cullenlab_server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_003_Session_1';
manual_trial = 'BL_closed_loop_STIM_003_250528_152812';
manual_path = fullfile(base_folder, 'Intan', manual_trial);
calib_file = fullfile(base_folder, 'Calibrated', 'IntanFile_9', 'BL_closed_loop_STIM_003_011_Cal.mat');
fig_folder = fullfile(base_folder, 'Figures');
if ~exist(fig_folder, 'dir'); mkdir(fig_folder); end

%% === Load Behavioral Timing from Calibrated File ===
if ~isfile(calib_file)
    error('Missing behavioral file: %s', calib_file);
end
load(calib_file, 'Data');  % Contains Data.velocity and Data.segment

%% === Load Neural Data ===
neural_data_file = fullfile(manual_path, 'neural_data.mat');
if ~isfile(neural_data_file)
    error('File not found: %s', neural_data_file);
end

fprintf('\n--- Inspecting trial: %s ---\n', manual_trial);
ndata = load(neural_data_file);

if isfield(ndata, 'neural_data')
    neural_data = ndata.neural_data;
    [nChans, nSamples] = size(neural_data);
    fprintf('Loaded neural_data: [%d channels × %d timepoints]\n', nChans, nSamples);
else
    error('Variable ''neural_data'' not found.');
end
%% === Aligned Preview: Single Trial from -800 to +1200 ms around movement onset ===
win_before = 800;  % ms
win_after = 1200;  % ms
ds_factor = fs / 1000;  % Downsampling to 1 kHz
trace_spacing = 200;
ch_per_bank = 32;
n_banks = 4;
time_axis = -win_before:win_after;  % in ms
segment_fields = {'active_like_stim_pos', 'active_like_stim_neg'};

segment_label = 'active_like_stim_pos';
trial_to_plot = 1;
if ~isfield(Data.segments, segment_label)
    error('Segment label not found: %s', segment_label);
end
segs = Data.segments.(segment_label);
if trial_to_plot > size(segs, 1)
    error('Trial index exceeds available segments.');
end

% Get onset time and compute index range
onset_ms = segs(trial_to_plot, 1);
center_idx = round(onset_ms);
idx_start = center_idx - win_before;
idx_end = center_idx + win_after;

% Downsample and extract window
neural_ds = downsample(neural_data', ds_factor)';  % [128 × T]
if idx_start <= 0 || idx_end > size(neural_ds, 2)
    error('Segment window exceeds neural signal length.');
end
aligned_data = neural_ds(:, idx_start:idx_end);  % [128 × window]

% Plot 128 channels across 4 banks
figure('Name', '128-Channel Aligned Neural Trace', 'Position', [100 100 1600 800]);
for b = 1:n_banks
    subplot(1, n_banks, b); hold on;
    ch_start = (b-1) * ch_per_bank + 1;
    ch_end = b * ch_per_bank;
    cmap = lines(ch_per_bank);

    for row = 1:ch_per_bank
        ch = ch_start + row - 1;
        plot(time_axis, aligned_data(ch, :) + trace_spacing * (row - 1), ...
            'Color', cmap(row, :), 'LineWidth', 0.8);
    end

    xline(0, '--k', 'LineWidth', 1.0);
    ylim([-trace_spacing, trace_spacing * ch_per_bank]);
    yticks(trace_spacing * (0:ch_per_bank - 1));
    yticklabels(string(ch_start:ch_end));
    title(sprintf('Bank %d: Channels %d–%d', b, ch_start, ch_end));
    xlabel('Time (ms from onset)');
    if b == 1, ylabel('Channel (offset)'); end
end

%% === Downsample Neural and Velocity to 1 kHz ===
neural_ds = downsample(neural_data', ds_factor)';  % [nChans x T]
velocity_ds = Data.headYawVel;
fprintf('Downsampled to %d samples.\n', size(neural_ds, 2));

%% === Define 'Still' Periods (|velocity| ≤ 10) ===
is_still = abs(velocity_ds) <= 40;
idx_still = find(is_still);

%% === Extract 'Movement' Segments Using Data.segment ===
win_before = 800; win_after = 1200;
T_ds = size(neural_ds, 2);
segments = Data.segments(:,1);  % Use first column for movement onset

move_windows = [];
X_move_ipsi = [];
X_move_contra = [];

for f = 1:numel(segment_fields)
    label = segment_fields{f};
    if ~isfield(Data.segments, label)
        continue;
    end

    segs = Data.segments.(label);  % [nSegments x 2]
    for s = 1:size(segs,1)
        onset = segs(s,1);  % Assume first column is onset
        idx_start = round((onset - win_before) * 1e-3 * fs / ds_factor);
        idx_end   = round((onset + win_after)  * 1e-3 * fs / ds_factor);

        if idx_start > 0 && idx_end <= T_ds
            window = neural_ds(:, idx_start:idx_end)';  % [T x nChans]
            if strcmp(label, 'active_like_stim_pos')
                X_move_ipsi = [X_move_ipsi; window];
            else
                X_move_contra = [X_move_contra; window];
            end
        end
    end
end


%% === Match Sample Count and Build 'Still' Matrix ===
n_ipsi   = size(X_move_ipsi, 1);
n_contra = size(X_move_contra, 1);
n_still  = length(idx_still);
n = min([n_ipsi, n_contra, n_still]);
% === Smoothed velocity and relaxed window ===
velocity_ds_smoothed = movmean(velocity_ds, 50);
is_still = abs(velocity_ds_smoothed) <= 30;

min_still_duration = 1800;  % relaxed from 2001
win_still = win_before + win_after + 1;
stride = 500;  % allows overlap

% Detect blocks
still_start = find(diff([0; is_still]) == 1);
still_end = find(diff([is_still; 0]) == -1);

% Filter blocks with enough length
block_lengths = still_end - still_start + 1;
valid_blocks = find(block_lengths >= min_still_duration);

% Extract segments
X_still = [];
for i = 1:numel(valid_blocks)
    start_idx = still_start(valid_blocks(i));
    len = block_lengths(valid_blocks(i));

    for j = 0:stride:(len - win_still)
        idx_range = start_idx + j : start_idx + j + win_still - 1;
        if idx_range(end) <= size(neural_ds, 2)
            seg = neural_ds(:, idx_range)';
            X_still = [X_still; seg];
        end
    end
end

% Match to number of movement segments
n = min([size(X_move_ipsi,1), size(X_move_contra,1), size(X_still,1)]);
X_move_ipsi  = X_move_ipsi(1:n, :);
X_move_contra = X_move_contra(1:n, :);
X_still      = X_still(1:n, :);
%% === Run Factor Analysis ===
k_factors = 2;
[Lambda_ipsi, ~]   = factoran(X_move_ipsi, k_factors);
[Lambda_contra, ~] = factoran(X_move_contra, k_factors);
[Lambda_still, ~]  = factoran(X_still, k_factors);

%% === Project to Latent Factors ===
proj_ipsi   = X_move_ipsi * Lambda_ipsi;
proj_contra = X_move_contra * Lambda_contra;
proj_still  = X_still * Lambda_still;

%% === Define Segment Info ===
n_per_segment = win_before + win_after + 1;
n_segments = size(X_move_ipsi, 1) / n_per_segment;
if rem(n_segments, 1) ~= 0
    error('Mismatch in sample count and segment length.');
end

% Time-relative color map (same for all conditions)
colors = jet(n_per_segment);  % Each row = RGB for a timepoint

%% === Plot Factor Trajectories with Time Gradient ===
figure('Name', manual_trial, 'Position', [100 100 1200 400]);

% --- Ipsi ---
subplot(1,3,1); hold on;
for i = 1:n_segments
    idx = (i-1)*n_per_segment + (1:n_per_segment);  % segment indices
    scatter(proj_ipsi(idx,1), proj_ipsi(idx,2), 10, colors, 'filled');
end
title('Ipsi Movement'); xlabel('Factor 1'); ylabel('Factor 2');
axis equal; grid on;
colormap(jet);
colorbar('Ticks', [0 1], 'TickLabels', {sprintf('%d ms', -win_before), sprintf('%d ms', win_after)});
xlim padded; ylim padded;

% --- Contra ---
subplot(1,3,2); hold on;
for i = 1:n_segments
    idx = (i-1)*n_per_segment + (1:n_per_segment);
    scatter(proj_contra(idx,1), proj_contra(idx,2), 10, colors, 'filled');
end
title('Contra Movement'); xlabel('Factor 1'); ylabel('Factor 2');
axis equal; grid on;
xlim padded; ylim padded;

% --- Still ---
subplot(1,3,3); hold on;
for i = 1:n_segments
    idx = (i-1)*n_per_segment + (1:n_per_segment);
    scatter(proj_still(idx,1), proj_still(idx,2), 10, colors, 'filled');
end
title('Still'); xlabel('Factor 1'); ylabel('Factor 2');
axis equal; grid on;
xlim padded; ylim padded;

%% === Save Figure ===
save_base = [manual_trial '_factor_analysis_gradient'];
savefig(fullfile(fig_folder, [save_base '.fig']));
print(fullfile(fig_folder, [save_base '.png']), '-dpng', '-r300');
%% === Plot 6 Segments × 3 Conditions with Temporal Gradient ===
max_show = 4;
max_segments = floor(size(proj_ipsi,1) / n_per_segment);
n_show = min(max_show, max_segments);
n_per_segment = win_before + win_after + 1;
colors = jet(n_per_segment);

figure('Name', [manual_trial ' – Segments by Condition'], ...
       'Position', [100, 100, 900, 1000]);

t = tiledlayout(n_show, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% === Compute shared axis limits from Ipsi and Contra ===
all_proj = [proj_ipsi; proj_contra];
xlim_shared = [min(all_proj(:,1)), max(all_proj(:,1))];
ylim_shared = [min(all_proj(:,2)), max(all_proj(:,2))];

for i = 1:n_show
    idx = (i-1)*n_per_segment + (1:n_per_segment);

    % --- Ipsi ---
    subplot(n_show, 3, (i-1)*3 + 1); hold on;
    traj = proj_ipsi(idx, :);
    scatter(traj(:,1), traj(:,2), 15, colors, 'filled');
    scatter(traj(1,1), traj(1,2), 30, 'k', 'filled');
    scatter(traj(end,1), traj(end,2), 30, 'r', 'filled');
    title(sprintf('Segment %d – Ipsi', i));
    axis equal; grid on;
    xlim(xlim_shared); ylim(ylim_shared);
    if i == n_show, xlabel('Factor 1'); end
    ylabel('Factor 2');

    % --- Contra ---
    subplot(n_show, 3, (i-1)*3 + 2); hold on;
    traj = proj_contra(idx, :);
    scatter(traj(:,1), traj(:,2), 15, colors, 'filled');
    scatter(traj(1,1), traj(1,2), 30, 'k', 'filled');
    scatter(traj(end,1), traj(end,2), 30, 'r', 'filled');
    title(sprintf('Segment %d – Contra', i));
    axis equal; grid on;
    xlim(xlim_shared); ylim(ylim_shared);
    if i == n_show, xlabel('Factor 1'); end

    % --- Still ---
    subplot(n_show, 3, (i-1)*3 + 3); hold on;
    traj = proj_still(idx, :);
    scatter(traj(:,1), traj(:,2), 15, colors, 'filled');
    scatter(traj(1,1), traj(1,2), 30, 'k', 'filled');
    scatter(traj(end,1), traj(end,2), 30, 'r', 'filled');
    title(sprintf('Segment %d – Still', i));
    axis equal; grid on;
    xlim(xlim_shared); ylim(ylim_shared);
    if i == n_show, xlabel('Factor 1'); end
end

% === Shared colorbar on right ===
colormap(jet(n_per_segment));  % force consistent length
cb = colorbar('Position', [0.93 0.1 0.015 0.8]);
cb.Ticks = [0 1];
cb.TickLabels = {sprintf('%d ms', -win_before), sprintf('%d ms', win_after)};
cb.Label.String = 'Time within segment';
