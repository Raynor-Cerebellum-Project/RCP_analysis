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
        skip_fields = {'n_trials', 'segments3', 'velocity_traces', ...
            'position_traces', 'acceleration_traces', ...
            'velocity_filtered_traces', 'stim_idx', ...
            'vel_thresh_idx', 'segments3_from_stim'};

        if any(strcmp(name, skip_fields))
            continue;
        end
        
        if strcmp(name, 'fr_traces') && iscell(M.fr_traces)
            nChans = length(M.fr_traces);
            fr_mean = [];  % true empty so isempty() works
            for ch = 1:nChans
                ch_data = M.fr_traces{ch};  % [nTrials x nTimepoints]
                if isempty(ch_data) || all(isnan(ch_data), 'all')
                    continue;
                end
                if isempty(fr_mean)
                    fr_mean = nan(nChans, size(ch_data, 2));  % now correctly initialized
                end
                fr_mean(ch, :) = mean(ch_data, 1, 'omitnan');
            end
            if ~isempty(fr_mean)
                S.fr_mean = fr_mean;
            end
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

    % Naming output summary field
    if startsWith(field, 'catch_')
        Summary.([side '_catch_summary']) = S;
    else
        Summary.([field '_summary']) = S;
    end
end
end
