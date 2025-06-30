function [SummaryMetrics] = analysis_metrics(filename, EndPoint_pos, EndPoint_neg, isBaseline, metadata_row, segment_fields)
% ANALYSIS_METRICS: Computes metrics for each segment field.
%
% INPUTS:
%   - filename: path to .mat file
%   - EndPoint_pos: reference for ipsilateral movements
%   - EndPoint_neg: reference for contralateral movements
%   - isBaseline: flag if trial is baseline (not used internally)
%   - metadata_row: table row with delay/trigger info
%   - segment_fields: segment names to process
%
% OUTPUT:
%   - SummaryMetrics: flat struct with metrics per segment field

% === Load and preprocess data ===
load(filename, 'Data');
fs = 1000;
[b, a] = butter(4, 2 / (fs/2), 'high');
Data.headYawVel_filtered = filtfilt(b, a, Data.headYawVel);
accel = diff(Data.headYawVel) / 0.001;
t = linspace(-800, 1200, 2001);

% === Process each segment field ===
for i = 1:length(segment_fields)
    field = segment_fields{i};
    if ~isfield(Data.segments, field), continue; end
    Segment = Data.segments.(field);

    polarity = 'neg'; if contains(field, '_pos'), polarity = 'pos'; end
    stim_val = regexp(field, 'stim_(\w+)_', 'tokens', 'once');
    if isempty(stim_val), stim_val = {'x'}; end
    stim_label = stim_val{1};
    side_label = ternary(strcmp(polarity, 'pos'), 'ipsi', 'contra');

    % === Align segments ===
    for s = 1:size(Segment,1)
        if isfield(metadata_row, 'Stim_Delay') && isnumeric(metadata_row.Stim_Delay) && ~any(ismissing(metadata_row.Stim_Delay)) ...
                && metadata_row.Stim_Delay > 0
            abs_loc = Segment(s,1) + metadata_row.Stim_Delay;
        elseif isfield(metadata_row, 'Movement_Trigger') && strcmpi(metadata_row.Movement_Trigger{1}, 'End')
            abs_loc = Segment(s,2);
        else
            abs_loc = Segment(s,1);
        end
        Data.segments3.(field)(s,1) = abs_loc - 800;
        Data.segments3.(field)(s,2) = abs_loc + 1200;
    end

    % === Initialize and compute metrics ===
    N = size(Segment,1);
    EndPoint     = nan(N,1);
    EndPointVar  = nan(N,1);
    max_speed    = nan(N,1);
    avg_speed    = nan(N,1);
    var_500ms    = nan(N,1);
    oscillations = nan(N,1);
    fft_power    = nan(N,1);

    for s = 1:N
        idx1 = Data.segments3.(field)(s,1);
        idx2 = Data.segments3.(field)(s,2);
        if any([idx1, idx2] <= 0) || idx2 > length(Data.headYawVel), continue; end

        vel_seg   = Data.headYawVel(idx1:idx2);
        pos_seg   = Data.headYawPos(idx1:idx2);
        accel_seg = accel(idx1:idx2);
        vel_filt  = Data.headYawVel_filtered(idx1:idx2);

        if length(accel_seg) >= 1300
            a_slice = accel_seg(1050:1300);
            p_slice = pos_seg(1050:1300);
            zc_idx  = find(diff(sign(a_slice)) ~= 0, 1);
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
        oscillations(s) = sum(diff(sign(vel_seg(1050:end))) ~= 0);
        fft_power(s)    = mean(vel_filt(800:1300).^2);
    end

    % === Signed endpoint error ===
    ref_EP  = ternary(strcmp(polarity, 'pos'), EndPoint_pos, EndPoint_neg);
    err_sign = ternary(strcmp(polarity, 'pos'), 1, -1);
    all_err = err_sign * (EndPoint - ref_EP);
    all_var = EndPointVar;

    % === Store raw metrics ===
    metrics = struct();
    metrics.all_err      = all_err;
    metrics.all_var      = all_var;
    metrics.max_speed    = max_speed;
    metrics.avg_speed    = avg_speed;
    metrics.var_500ms    = var_500ms;
    metrics.oscillations = oscillations;
    metrics.fft_power    = fft_power;

    SummaryMetrics.(field) = metrics;

    % === Store summary stats ===
    summary = struct();
    summary.n_trials = sum(~isnan(all_err));
    for f = fieldnames(metrics)'
        v = metrics.(f{1});
        summary.([f{1} '_mean'])    = mean(v, 'omitnan');
        summary.([f{1} '_var'])     = var(v,  'omitnan');
        if strcmp(f{1}, 'all_err')
            summary.([f{1} '_abs_mean']) = mean(abs(v), 'omitnan');
            summary.([f{1} '_abs_var'])  = var(abs(v),  'omitnan');
        end
    end

    summary_field = sprintf('%s_%s_summary', side_label, stim_label);
    SummaryMetrics.(summary_field) = summary;
end

% Optionally store list of fields by side
SummaryMetrics.side_labels = struct( ...
    'ipsi_fields',   segment_fields(contains(segment_fields, '_pos')), ...
    'contra_fields', segment_fields(contains(segment_fields, '_neg')) ...
);
end

function out = ternary(cond, val_true, val_false)
if cond
    out = val_true;
else
    out = val_false;
end
end
