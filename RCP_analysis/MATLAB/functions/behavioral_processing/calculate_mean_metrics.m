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
            fr_mean = nan(nChans, 0);  % Will set cols after seeing first non-empty

            for ch = 1:nChans
                ch_data = M.fr_traces{ch};  % [nTrials x nTimepoints]
                if isempty(ch_data) || all(isnan(ch_data), 'all')
                    continue;
                end

                if isempty(fr_mean)
                    fr_mean = nan(nChans, size(ch_data, 2));
                end

                % Average across trials (rows)
                fr_mean(ch, :) = mean(ch_data, 1, 'omitnan');  % [1 x T]
            end

            S.fr_mean = fr_mean;  % [nChans x nTimepoints]
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
        suffix = extractAfter(field, 'stim_');
        Summary.([side '_' suffix '_summary']) = S;
    end
end
end
