function [MetricStruct] = analyze_metrics(filename, EndPoint_pos, EndPoint_neg, metadata_row, segment_fields)
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

    % Align each segment
    for s = 1:N
        if isfield(metadata_row, 'Stim_Delay') && isnumeric(metadata_row.Stim_Delay) ...
                && ~any(ismissing(metadata_row.Stim_Delay)) && metadata_row.Stim_Delay > 0
            abs_loc = Segment(s,1) + metadata_row.Stim_Delay;
        elseif isfield(metadata_row, 'Movement_Trigger') && strcmpi(metadata_row.Movement_Trigger{1}, 'End')
            abs_loc = Segment(s,2);
        else
            abs_loc = Segment(s,1);
        end
        segments3(s,:) = [abs_loc - 800, abs_loc + 1200];
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
        if length(accel_seg) >= 1400
            a_slice = accel_seg(1050:1400);
            p_slice = pos_seg(1050:1400);
            zc_idx  = find(diff(sign(a_slice)) ~= 0, 1);
            if ~isempty(zc_idx)
                ep = p_slice(zc_idx);
                EndPoint(s) = ep;
                win_idx = 800 + zc_idx;
                if win_idx + 250 <= length(vel_seg)
                    EndPointVar(s) = std(vel_seg(win_idx:win_idx+250));
                end
            end
        end

        max_speed(s)    = max(abs(vel_seg(800:1300)));
        avg_speed(s)    = mean(abs(vel_seg(800:1300)));
        var_500ms(s)    = std(vel_seg(800:1300));
        oscillations(s) = sum(diff(sign(vel_seg(1050:end))) ~= 0);
        fft_power(s)    = mean(vel_filt(800:1300).^2);
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
        'segments3',      segments3, ...
        'n_trials',       N, ...
        'velocity_traces',         velocity_traces, ...
        'position_traces',         position_traces, ...
        'acceleration_traces',     acceleration_traces, ...
        'velocity_filtered_traces', velocity_filtered_traces ...
    );
end
end
