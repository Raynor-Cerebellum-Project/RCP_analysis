function combined = combine_baseline_with_rand(base_metrics, cond_metrics, combine_map)
% combine_baseline_with_rand: Merge specified fields between baseline and condition trials.
%
% INPUTS:
%   - base_metrics: struct from baseline trial (e.g. trial 4)
%   - cond_metrics: struct from condition trial (e.g. trial 12)
%   - combine_map: struct specifying source-to-merged field names
%
% OUTPUT:
%   - combined: struct with merged raw traces

combined = cond_metrics;  % Start from condition

map_fields = fieldnames(combine_map);
for i = 1:numel(map_fields)
    combined_field = map_fields{i};
    sources = combine_map.(combined_field);

    if numel(sources) ~= 2
        warning("Skipping field %s: must specify two source fields.", combined_field);
        continue;
    end

    base_field = sources{1};
    cond_field = sources{2};

    if ~isfield(base_metrics, base_field) || ~isfield(cond_metrics, cond_field)
        warning("Missing fields in base or cond: %s / %s", base_field, cond_field);
        continue;
    end

    base_data = base_metrics.(base_field);
    cond_data = cond_metrics.(cond_field);

    merged_data = struct();
    subfields = union(fieldnames(base_data), fieldnames(cond_data));

    for j = 1:numel(subfields)
        sf = subfields{j};

        bval = getfield_safe(base_data, sf);
        cval = getfield_safe(cond_data, sf);

        try
            % Special handling for fr_traces (cell array of channels)
            if strcmp(sf, 'fr_traces') && iscell(bval) && iscell(cval)
                ch_count = min(length(bval), length(cval));
                traces_merged = cell(ch_count, 1);
                for ch = 1:ch_count
                    bch = get_cell_safe(bval, ch);
                    cch = get_cell_safe(cval, ch);
                    try
                        traces_merged{ch} = [bch; cch];  % concat by row (segment)
                    catch
                        warning('Failed to concatenate fr_traces channel %d', ch);
                        traces_merged{ch} = cch;
                    end
                end
                merged_data.fr_traces = traces_merged;
            else
                merged_data.(sf) = [bval; cval];
            end
        catch
            warning("Failed to concatenate %s.%s", combined_field, sf);
            merged_data.(sf) = cval;  % fallback to condition
        end
    end

    % Add total trial count
    if isfield(merged_data, 'all_err')
        merged_data.n_trials = size(merged_data.all_err, 1);
    end

    % Save back into new combined field
    combined.(combined_field) = merged_data;
end
end

function val = getfield_safe(S, f)
if isfield(S, f)
    val = S.(f);
else
    val = [];
end
end

function val = get_cell_safe(C, i)
if iscell(C) && i <= length(C)
    val = C{i};
else
    val = [];
end
end
