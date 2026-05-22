function [MetricStruct] = analyze_metrics(filename, EndPoint_pos, EndPoint_neg, metadata_row, segment_fields, do_plot)
if nargin < 6, do_plot = false; end
% ANALYZE_METRICS: Computes trace-level metrics and aligned segments per field.
%
% INPUTS:
%   - filename: path to .mat file (Blackrock 1 kHz)
%   - EndPoint_pos, EndPoint_neg: reference for ipsi/contra movements
%   - metadata_row: table row with stim timing info
%   - segment_fields: cell array of segment field names
%   - do_plot: optional, default false
%
% OUTPUT:
%   - MetricStruct: structure with one entry per segment field

% Load trial data
load(filename, 'Data');

% Constants
fs    = 1000;   % Blackrock behavioral data sample rate (Hz)
fs_30 = 30000;  % Intan / neural sample rate (Hz)

[b, a] = butter(4, 2 / (fs/2), 'high');

% Field name compatibility: Bert uses different naming conventions
if ~isfield(Data, 'yaw_vel') && isfield(Data, 'headYawVel')
    Data.yaw_vel = Data.headYawVel;
end
if ~isfield(Data, 'head_pos') && isfield(Data, 'headYawPos')
    Data.head_pos = Data.headYawPos;
end

Data.headYawVel_filtered = filtfilt(b, a, Data.yaw_vel);
accel = diff(Data.yaw_vel) / 0.001;

% Resolve stim_data.mat path from filename
% filename: <session_root>/Calibrated/IntanFile_N/<name>_Cal.mat
% stim_data: <session_root>/Intan/<Nth sorted Intan folder>/stim_data.mat
burst_starts_bk = [];
cal_dir      = fileparts(filename);        % .../IntanFile_N
calibrated   = fileparts(cal_dir);         % .../Calibrated
session_root = fileparts(calibrated);      % .../20260326_NRR_RW027_fastig

[~, intanfile_name] = fileparts(cal_dir);
tok = regexp(intanfile_name, 'IntanFile_(\d+)', 'tokens');
stim_mat_path = '';

if ~isempty(tok)
    intan_idx  = str2double(tok{1}{1});
    intan_root = fullfile(session_root, 'Intan');
    intan_folders = dir(intan_root);
    intan_folders = intan_folders([intan_folders.isdir] & ~startsWith({intan_folders.name}, '.'));
    [~, sort_order] = sort({intan_folders.name});
    intan_folders   = intan_folders(sort_order);
    if intan_idx <= length(intan_folders)
        stim_mat_path = fullfile(intan_root, intan_folders(intan_idx).name, 'stim_data.mat');
    else
        warning('analyze_metrics: IntanFile_%d exceeds number of Intan folders (%d)', ...
            intan_idx, length(intan_folders));
    end
else
    warning('analyze_metrics: could not parse IntanFile index from "%s"', intanfile_name);
end

% Load stim burst times and convert to Blackrock 1 kHz indices
if ~isempty(stim_mat_path) && exist(stim_mat_path, 'file')
    S = load(stim_mat_path);

    % Normalize field name: accept 'stim_data' or 'Stim_data'
    if ~isfield(S, 'Stim_data') && isfield(S, 'stim_data')
        S.Stim_data = S.stim_data;
    end

    if isfield(S, 'Stim_data') && isfield(Data, 'Intan_idx')
        % S.Stim_data: 128 x N_intan at 30 kHz (Intan clock)
        % Collapse across channels: any nonzero sample = stim active
        stim_signal = any(S.Stim_data ~= 0, 1);  % 1 x N_intan logical

        % Rising edges = burst onsets in Intan sample indices (30 kHz)
        rising_intan = find(diff([0, stim_signal]) == 1);

        % Deduplicate bursts separated by less than 150 ms at 30 kHz
        min_gap_30k = round(0.150 * fs_30);
        if isempty(rising_intan)
            burst_intan = [];
        else
            keep = [true, diff(rising_intan) > min_gap_30k];
            burst_intan = rising_intan(keep);
        end

        % Convert Intan-sample burst times to Blackrock 1 kHz sample indices
        % Data.Intan_idx(1, bk) = Intan sample for Blackrock sample bk
        % Invert via nearest-neighbour lookup in the monotonic Intan_idx vector
        intan_map = double(Data.Intan_idx(1,:));
        burst_starts_bk = zeros(length(burst_intan), 1);
        for bi = 1:length(burst_intan)
            [~, bk_idx] = min(abs(intan_map - burst_intan(bi)));
            burst_starts_bk(bi) = round(bk_idx / 30);
        end

    else
        if ~isfield(S, 'Stim_data')
            warning('analyze_metrics: stim_data.mat has no Stim_data field');
        end
        if ~isfield(Data, 'Intan_idx')
            warning('analyze_metrics: Data has no Intan_idx; cannot map stim to Blackrock time');
        end
    end
end

has_stim_file = ~isempty(burst_starts_bk);

% Initialize output
MetricStruct = struct();

% Loop over segment fields
for i = 1:length(segment_fields)
    field = segment_fields{i};
    if ~isfield(Data.segments, field)
        continue
    end

    Segment = Data.segments.(field);
    N = size(Segment, 1);
    segments3           = zeros(N, 2);
    vel_thresh_idx      = nan(N, 1);
    stim_idx_relative   = nan(N, 2);
    segments3_from_stim = nan(N, 2);

    % Resolve stim_delay from metadata once per field
    stim_delay = 0;
    if ismember('Stim_Delay', metadata_row.Properties.VariableNames) && ...
            ~any(ismissing(metadata_row.Stim_Delay))
        sdv = metadata_row.Stim_Delay;
        if iscell(sdv), sdv = sdv{1}; end
        if strcmpi(sdv, 'Random')
            % Delay encoded in field name suffix e.g. 'active_like_stim_pos_100'
            tokens = regexp(field, '_(\d+)$', 'tokens');
            if ~isempty(tokens)
                stim_delay = str2double(tokens{1}{1});
            end
        else
            val = str2double(sdv);  % handles '0', numeric strings; NaN for '-' or 'NaN'
            if ~isnan(val)
                stim_delay = val;
            end
        end
    end

    % Per-segment alignment
    for s = 1:N
        % Segment(s,:) defines the search region only
        if isfield(metadata_row, 'Movement_Trigger') && strcmpi(metadata_row.Movement_Trigger{1}, 'End')
            search_center = Segment(s,2);
        else
            search_center = Segment(s,1);
        end
        search_start = search_center - 20;
        search_end   = search_center + 200;

        % Bounds check on search window
        if search_start <= 0 || search_end > length(Data.yaw_vel)
            continue
        end

        % Step 1: velocity crossing >85 deg/s → segments3
        vel_search   = Data.yaw_vel(search_start:search_end);
        thresh_cross = find(abs(vel_search) > 85, 1);
        if ~isempty(thresh_cross)
            onset_global   = search_start + thresh_cross - 1;
            segments3(s,:) = [onset_global - 800, onset_global + 1200];
        else
            segments3(s,:) = [search_center - 800, search_center + 1200];
        end

        % Step 2: if stim file present, re-center on burst → segments3_from_stim
        if has_stim_file
            in_window    = burst_starts_bk >= search_start & burst_starts_bk <= search_end;
            valid_bursts = burst_starts_bk(in_window);

            if ~isempty(valid_bursts)
                chosen_burst = valid_bursts(1);
                win_start = chosen_burst - (800 + stim_delay);
                win_end   = chosen_burst + (1200 - stim_delay);
                segments3_from_stim(s,:) = [win_start, win_end];

                stim_start_rel = 800 + stim_delay + 1;
                stim_end_rel   = stim_start_rel + round(0.150 * fs) - 1;
                stim_idx_relative(s,:) = [stim_start_rel, stim_end_rel];
            end
            % If no burst found, segments3_from_stim stays nan; metrics loop uses segments3
        end
    end

    % Preallocate metrics and trace containers
    EndPoint     = nan(N,1);
    EndPointVar  = nan(N,1);
    max_speed    = nan(N,1);
    avg_speed    = nan(N,1);
    var_500ms    = nan(N,1);
    oscillations = nan(N,1);
    fft_power    = nan(N,1);

    velocity_traces          = nan(N, 2001);
    position_traces          = nan(N, 2001);
    acceleration_traces      = nan(N, 2001);
    velocity_filtered_traces = nan(N, 2001);

    for s = 1:N
        % Choose final window: prefer stim-aligned, fall back to vel-threshold
        if ~any(isnan(segments3_from_stim(s,:)))
            idx = segments3_from_stim(s,:);
        else
            idx = segments3(s,:);
        end

        if any(idx <= 0) || idx(2) > length(Data.yaw_vel), continue; end

        vel_seg   = Data.yaw_vel(idx(1):idx(2));
        pos_seg   = Data.head_pos(idx(1):idx(2));
        accel_seg = accel(idx(1):min(idx(2), length(accel)));
        vel_filt  = Data.headYawVel_filtered(idx(1):idx(2));

        velocity_traces(s,:)          = vel_seg;
        position_traces(s,:)          = pos_seg;
        acceleration_traces(s,:)      = accel_seg;
        velocity_filtered_traces(s,:) = vel_filt;

        % Velocity threshold index (>85 deg/s) within extracted window
        thresh_cross = find(abs(vel_seg) > 85, 1);
        if ~isempty(thresh_cross)
            vel_thresh_idx(s) = thresh_cross;
        end

        % Endpoint detection via acceleration zero-crossing
        if length(accel_seg) >= 1500
            a_slice = accel_seg(1050:1500);
            v_slice = vel_seg(1050:1500);
            p_slice = pos_seg(1050:1500);

            zc_idx = find(diff(sign(a_slice)) ~= 0, 1);
            if isempty(zc_idx)
                zc_idx = find(diff(sign(v_slice)) ~= 0, 1);
            end

            if ~isempty(zc_idx)
                EndPoint(s) = p_slice(zc_idx);
                win_idx = 800 + zc_idx;
                if win_idx + 250 <= length(vel_seg)
                    EndPointVar(s) = std(vel_seg(win_idx:win_idx+250));
                end
            end
        end

        max_speed(s)    = max(abs(vel_seg(800:1300)));
        avg_speed(s)    = mean(abs(vel_seg(800:1300)));
        var_500ms(s)    = std(vel_seg(800:1300));
        oscillations(s) = sum(diff(sign(accel_seg(1050:end))) ~= 0);
        fft_power(s)    = mean(vel_filt(800:1300).^2);
    end

    if do_plot
        t = linspace(-800, 1200, 2001);
        fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 500]);
        subplot(1,2,1); hold on;
        for s = 1:N
            plot(t, velocity_traces(s,:), 'Color', [0.5 0.5 0.5 0.3]);
            if ~isnan(vel_thresh_idx(s))
                plot(t(vel_thresh_idx(s)), velocity_traces(s, vel_thresh_idx(s)), ...
                    'go', 'MarkerSize', 5, 'LineWidth', 1.2);
            end
            if length(acceleration_traces(s,:)) >= 1400
                a_slice = acceleration_traces(s, 1050:1400);
                zc_idx  = find(diff(sign(a_slice)) ~= 0, 1);
                if ~isempty(zc_idx)
                    plot(t(1050 + zc_idx - 1), velocity_traces(s, 1050 + zc_idx - 1), ...
                        'ro', 'MarkerSize', 5, 'LineWidth', 1.2);
                end
            end
            if ~any(isnan(stim_idx_relative(s,:)))
                stim_range = stim_idx_relative(s,:);
                stim_times = (stim_range - 1) / fs * 1000 + t(1);
                y_limits = ylim;
                fill([stim_times(1) stim_times(2) stim_times(2) stim_times(1)], ...
                    [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
                    [1 0 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            end
        end
        plot(t, nanmean(velocity_traces, 1), 'm', 'LineWidth', 2);
        xlabel('Time (ms)'); ylabel('Velocity (deg/s)');
        title([field ' velocity']);

        subplot(1,2,2); hold on;
        for s = 1:N
            plot(t, position_traces(s,:), 'Color', [0.3 0.3 1 0.3]);
            if length(acceleration_traces(s,:)) >= 1500
                a_slice = acceleration_traces(s, 1050:1500);
                zc_idx  = find(diff(sign(a_slice)) ~= 0, 1);
                if ~isempty(zc_idx)
                    plot(t(1050 + zc_idx - 1), position_traces(s, 1050 + zc_idx - 1), ...
                        'ro', 'MarkerSize', 5, 'LineWidth', 1.2);
                end
            end
            if ~any(isnan(stim_idx_relative(s,:)))
                stim_range = stim_idx_relative(s,:);
                stim_times = (stim_range - 1) / fs * 1000 + t(1);
                y_limits = ylim;
                fill([stim_times(1) stim_times(2) stim_times(2) stim_times(1)], ...
                    [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
                    [1 0 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            end
        end
        plot(t, nanmean(position_traces, 1), 'b', 'LineWidth', 2);
        xlabel('Time (ms)'); ylabel('Position (deg)');
        title([field ' position']);

        sgtitle(sprintf('All Trials — File: %s | Segment: %s', ...
            strrep(filename, '_', '\_'), strrep(field, '_', ' ')));

        save_dir = fullfile('Figures', 'allTrials');
        if ~exist(save_dir, 'dir'), mkdir(save_dir); end
        [~, fname, ~] = fileparts(filename);
        save_path = fullfile(save_dir, sprintf('%s_%s_allTrials.png', fname, field));
        print(fig, save_path, '-dpng', '-r300');
        close(fig);
    end

    % Sign correction for endpoint error
    if contains(field, '_pos')
        ref_EP = EndPoint_pos;
        sign_factor = 1;
    else
        ref_EP = EndPoint_neg;
        sign_factor = -1;
    end

    MetricStruct.(field) = struct( ...
        'all_err',                  sign_factor * (EndPoint - ref_EP), ...
        'all_var',                  EndPointVar, ...
        'max_speed',                max_speed, ...
        'avg_speed',                avg_speed, ...
        'var_500ms',                var_500ms, ...
        'oscillations',             oscillations, ...
        'fft_power',                fft_power, ...
        'n_trials',                 N, ...
        'velocity_traces',          velocity_traces, ...
        'position_traces',          position_traces, ...
        'acceleration_traces',      acceleration_traces, ...
        'velocity_filtered_traces', velocity_filtered_traces, ...
        'segments3',                segments3, ...
        'vel_thresh_idx',           vel_thresh_idx, ...
        'stim_idx_relative',        stim_idx_relative, ...
        'segments3_from_stim',      segments3_from_stim ...
        );
end
end
