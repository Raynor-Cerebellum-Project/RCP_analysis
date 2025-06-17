function [MetricStruct] = analyze_metrics(filename, EndPoint_pos, EndPoint_neg, metadata_row, segment_fields, do_plot)
if nargin < 6, do_plot = false; end
% ANALYZE_METRICS: Computes trace-level metrics and aligned segments per field.
%
% INPUTS:
%   - filename: path to .mat file
%   - EndPoint_pos, EndPoint_neg: reference for ipsi/contra movements
%   - metadata_row: table row with stim timing info
%   - segment_fields: cell array of segment field names
%
% OUTPUT:
%   - MetricStruct: structure with one entry per segment field containing all metrics + aligned segments + raw traces

% Load trial data
load(filename, 'Data');
% Constants
fs = 1000;
[b, a] = butter(4, 2 / (fs/2), 'high');
Data.headYawVel_filtered = filtfilt(b, a, Data.headYawVel);
accel = diff(Data.headYawVel) / 0.001;

% Loop over segment fields
for i = 1:length(segment_fields)
    field = segment_fields{i};
    if ~isfield(Data.segments, field), continue; end

    Segment = Data.segments.(field);
    N = size(Segment, 1);
    segments3 = zeros(N, 2);
    vel_thresh_idx = nan(N, 1);     % Relative index within segment
    stim_idx_relative   = nan(N, 2);
    segments3_from_stim = nan(N, 2);

    % Align each segment
    for s = 1:N
        % === Skip stim-based alignment for 'nan' fields ===
        if strcmp(field, 'active_like_stim_pos_nan') || strcmp(field, 'active_like_stim_neg_nan')
            segments3(s,:) = [Segment(s,1) - 800, Segment(s,1) + 1200];
            continue;
        end

        if isfield(metadata_row, 'Stim_Delay') && isnumeric(metadata_row.Stim_Delay) ...
                && ~any(ismissing(metadata_row.Stim_Delay)) && metadata_row.Stim_Delay > 0
            abs_loc = Segment(s,1) + metadata_row.Stim_Delay;
        elseif isfield(metadata_row, 'Movement_Trigger') && strcmpi(metadata_row.Movement_Trigger{1}, 'End')
            abs_loc = Segment(s,2);
        else
            abs_loc = Segment(s,1);
        end
        segments3(s,:) = [abs_loc - 800, abs_loc + 1200];

        % === Stim alignment logic (as before) ===
        if endsWith(filename, '_stim.mat') && isfield(Data, 'Neural')

            neural_fs = 30000;  % Hz
            stim_signal = Data.Neural(:,1);
            threshold = 2;
            stim_binary = stim_signal > threshold;
            rising_edges = find(diff(stim_binary) == 1) + 1;
            min_inter_burst_gap = round(0.150 * neural_fs);  % 150 ms gap

            if ~isempty(rising_edges)
                burst_starts = rising_edges([true; diff(rising_edges) > min_inter_burst_gap]);

                % Convert segment bounds to 30kHz
                neural_segment = round(segments3(s,:) * neural_fs / fs);  % fs = 1000

                % Create burst intervals
                stim_bursts = [burst_starts, burst_starts + round(0.150 * neural_fs)];

                % Keep bursts that overlap this extended window
                is_within = stim_bursts(:,2) >= neural_segment(1) & stim_bursts(:,1) <= neural_segment(2);
                valid_bursts = stim_bursts(is_within, :);

                % Convert burst indices back to 1kHz and segment-relative
                valid_bursts_1khz = round(valid_bursts / (neural_fs / fs));  % from 30kHz to 1kHz
                relative_starts = valid_bursts_1khz(:,1) - segments3(s,1) + 1;
                rel_idx_after_750 = find(relative_starts > 750, 1);

                stim_delay = 0;  % default fallback

                if ismember('Stim_Delay', metadata_row.Properties.VariableNames) && ...
                        ~any(ismissing(metadata_row.Stim_Delay))

                    stim_delay_val = metadata_row.Stim_Delay;

                    if iscell(stim_delay_val)
                        stim_delay_val = stim_delay_val{1};  % extract string
                    end

                    if ischar(stim_delay_val) && strcmpi(stim_delay_val, 'Random')
                        % === Parse delay from field name like 'active_like_stim_pos_100' ===
                        tokens = regexp(field, '_(\d+)$', 'tokens');
                        if ~isempty(tokens)
                            stim_delay = str2double(tokens{1}{1});
                        end
                    elseif isnumeric(stim_delay_val)
                        stim_delay = stim_delay_val;
                    end
                end

                if ~isempty(valid_bursts_1khz)
                    first_rel = valid_bursts_1khz(1,:) - segments3(s,1) + 1;
                    stim_idx_relative(s,:) = first_rel;
                end

                if ~isempty(rel_idx_after_750)
                    stim_start_global = valid_bursts_1khz(rel_idx_after_750, 1);
                    segments3_from_stim(s,:) = [stim_start_global - (800 + stim_delay), stim_start_global + (1200 - stim_delay)];
                end
            end
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

    velocity_traces       = nan(N, 2001);
    position_traces       = nan(N, 2001);
    acceleration_traces   = nan(N, 2001);
    velocity_filtered_traces = nan(N, 2001);

    for s = 1:N
        idx = segments3(s,:);
        if any(idx <= 0) || idx(2) > length(Data.headYawVel), continue; end
        % Choose which alignment to use
        % Choose which alignment to use
        if strcmp(field, 'active_like_stim_pos_nan') || strcmp(field, 'active_like_stim_neg_nan')
            idx = segments3(s,:);  % Always use original if '_nan' condition
        elseif ~any(isnan(segments3_from_stim(s,:)))
            idx = segments3_from_stim(s,:);
        else
            idx = segments3(s,:);
        end

        vel_seg   = Data.headYawVel(idx(1):idx(2));
        pos_seg   = Data.headYawPos(idx(1):idx(2));
        accel_seg = accel(idx(1):idx(2));
        vel_filt  = Data.headYawVel_filtered(idx(1):idx(2));

        % Save raw traces
        velocity_traces(s, :)         = vel_seg;
        position_traces(s, :)         = pos_seg;
        acceleration_traces(s, :)     = accel_seg;
        velocity_filtered_traces(s,:) = vel_filt;

        % Endpoint detection and metrics
        if length(accel_seg) >= 1500
            a_slice = accel_seg(1050:1500);
            v_slice = vel_seg(1050:1500);
            p_slice = pos_seg(1050:1500);

            zc_idx = find(diff(sign(a_slice)) ~= 0, 1);

            if isempty(zc_idx)
                zc_idx = find(diff(sign(v_slice)) ~= 0, 1);  % fallback: velocity zero crossing
            end

            if ~isempty(zc_idx)
                ep = p_slice(zc_idx);
                EndPoint(s) = ep;

                win_idx = 800 + zc_idx;
                if win_idx + 250 <= length(vel_seg)
                    EndPointVar(s) = std(vel_seg(win_idx:win_idx+250));
                end
            end
        end

        % Velocity threshold detection (>85 deg/s)
        thresh_cross = find(abs(vel_seg) > 85, 1);
        if ~isempty(thresh_cross)
            vel_thresh_idx(s) = thresh_cross;
        else
            vel_thresh_idx(s) = NaN;  % Explicit fallback
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

        % === Velocity subplot ===
        subplot(1,2,1); hold on;
        for s = 1:N
            plot(t, velocity_traces(s,:), 'Color', [0.5 0.5 0.5 0.3]);

            % Add green marker for velocity threshold
            if ~isnan(vel_thresh_idx(s))
                plot(t(vel_thresh_idx(s)), velocity_traces(s, vel_thresh_idx(s)), 'go', 'MarkerSize', 5, 'LineWidth', 1.2);
            end

            % Add red marker for endpoint if detectable
            if length(acceleration_traces(s,:)) >= 1400
                a_slice = acceleration_traces(s, 1050:1400);
                p_slice = position_traces(s, 1050:1400);
                zc_idx  = find(diff(sign(a_slice)) ~= 0, 1);
                if ~isempty(zc_idx)
                    plot(t(1050 + zc_idx - 1), velocity_traces(s, 1050 + zc_idx - 1), 'ro', 'MarkerSize', 5, 'LineWidth', 1.2);
                end
            end

            % Add stim region if applicable
            if ~any(isnan(stim_idx_relative(s,:)))
                stim_range = stim_idx_relative(s,:);  % Now a 1x2 double
                stim_times = (stim_range - 1) / fs * 1000 + t(1);  % convert to ms
                y_limits = ylim;
                fill([stim_times(1) stim_times(2) stim_times(2) stim_times(1)], ...
                    [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
                    [1 0 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            end

        end
        plot(t, nanmean(velocity_traces, 1), 'm', 'LineWidth', 2);
        xlabel('Time (ms)'); ylabel('Velocity (deg/s)');
        title([field ' velocity']);

        % === Position subplot ===
        subplot(1,2,2); hold on;
        for s = 1:N
            plot(t, position_traces(s,:), 'Color', [0.3 0.3 1 0.3]);

            % Red marker for endpoint (position)
            if length(acceleration_traces(s,:)) >= 1500
                a_slice = acceleration_traces(s, 1050:1500);
                p_slice = position_traces(s, 1050:1500);
                zc_idx  = find(diff(sign(a_slice)) ~= 0, 1);
                if ~isempty(zc_idx)
                    plot(t(1050 + zc_idx - 1), position_traces(s, 1050 + zc_idx - 1), 'ro', 'MarkerSize', 5, 'LineWidth', 1.2);
                end
            end

            % Add stim region if applicable
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

        sgtitle(sprintf('All Trials — File: %s | Segment: %s', strrep(filename, '_', '\_'), strrep(field, '_', ' ')));

        % Save
        save_dir = fullfile('Figures', 'allTrials');
        if ~exist(save_dir, 'dir'), mkdir(save_dir); end
        [~, fname, ~] = fileparts(filename);
        save_name = sprintf('%s_%s_allTrials.png', fname, field);
        save_path = fullfile(save_dir, save_name);
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

    % Store all metrics + aligned segments + traces
    MetricStruct.(field) = struct( ...
        'all_err',        sign_factor * (EndPoint - ref_EP), ...
        'all_var',        EndPointVar, ...
        'max_speed',      max_speed, ...
        'avg_speed',      avg_speed, ...
        'var_500ms',      var_500ms, ...
        'oscillations',   oscillations, ...
        'fft_power',      fft_power, ...
        'n_trials',       N, ...
        'velocity_traces',         velocity_traces, ...
        'position_traces',         position_traces, ...
        'acceleration_traces',     acceleration_traces, ...
        'velocity_filtered_traces', velocity_filtered_traces, ...
        'segments3',      segments3, ...
        'vel_thresh_idx',        vel_thresh_idx, ...
        'stim_idx_relative', stim_idx_relative, ...
        'segments3_from_stim', segments3_from_stim ...
        );
end
end
