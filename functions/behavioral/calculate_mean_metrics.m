function Summary = calculate_mean_metrics(MetricStruct, segment_fields, side)
% CALCULATE_MEAN_METRICS: Averages metrics across trials for each segment field.
%
% INPUTS:
%   - MetricStruct: struct of raw trial metrics
%   - segment_fields: cell array of fields to include
%   - side: 'ipsi' or 'contra'
%
% OUTPUT:
%   - Summary: struct with means/vars of key fields per segment

Summary = struct();
for i = 1:length(segment_fields)
    field = segment_fields{i};
    if ~isfield(MetricStruct, field), continue; end
    M = MetricStruct.(field);

    S = struct();
    if isfield(M, 'all_err')
        S.n_trials = sum(~isnan(M.all_err));
    end

    for f = fieldnames(M)'
        name = f{1};
        % Skip non-scalar fields
        skip_fields = {'n_trials', 'segments3', 'velocity_traces', ...
            'position_traces', 'acceleration_traces', ...
            'velocity_filtered_traces', 'stim_idx', ...
            'vel_thresh_idx', 'segments3_from_stim', 'fr_traces'};

        if any(strcmp(name, skip_fields))
            continue;
        end

        v = M.(name);
        S.([name '_mean']) = mean(v, 'omitnan');
        S.([name '_var'])  = var(v,  'omitnan');

        if strcmp(name, 'all_err')
            S.([name '_abs_mean']) = mean(abs(v), 'omitnan');
            S.([name '_abs_var'])  = var(abs(v),  'omitnan');
        end
    end

    if startsWith(field, 'catch_')
        % Field like 'catch_pos' or 'catch_neg' → becomes 'ipsi_catch_summary'
        Summary.([side '_catch_summary']) = S;
    else
        % General case: extract after 'stim_' (e.g., 'pos_first10', etc.)
        suffix = extractAfter(field, 'stim_');
        Summary.([side '_' suffix '_summary']) = S;
    end

end
end
