addpath(genpath('functions'));
session = 'BL_RW_003_Session_1';

%% --- Setup Paths ---
base_folder = fullfile(['/Volumes/CullenLab_Server/Current Project Databases - NHP' ...
    '/2025 Cerebellum prosthesis/Bryan/Data'], session);

search_folder = fullfile(base_folder, 'Calibrated');
trial_mat_files = dir(fullfile(search_folder, 'IntanFile_*', '*_Cal.mat'));
% --- Sort trial_mat_files by BR_File number extracted from filename ---
br_nums = zeros(length(trial_mat_files), 1);
for i = 1:length(trial_mat_files)
    tokens = regexp(trial_mat_files(i).name, 'STIM_\d+_(\d+)_Cal\.mat', 'tokens');
    if ~isempty(tokens)
        br_nums(i) = str2double(tokens{1}{1});
    else
        br_nums(i) = NaN;
    end
end

[~, sort_idx] = sort(br_nums);
trial_mat_files = trial_mat_files(sort_idx);
%% --- Session-specific Endpoint Targets ---
switch session
    case 'BL_RW_001_Session_1'
        EndPoint_pos = 32; EndPoint_neg = -26;
        search_folder = fullfile(base_folder, 'Renamed2');
        trial_mat_files = dir(fullfile(search_folder, '*_Cal.mat'));
    case 'BL_RW_002_Session_1'
        EndPoint_pos = 49.5857; EndPoint_neg = -34.4605;
    case 'BL_RW_003_Session_1'
        EndPoint_pos = 30; EndPoint_neg = -30;
        segment_fields_random = {'active_like_stim_nan_pos', 'active_like_stim_0_pos', ...
            'active_like_stim_100_pos', 'active_like_stim_200_pos', ...
            'active_like_stim_nan_neg', 'active_like_stim_0_neg', ...
            'active_like_stim_100_neg', 'active_like_stim_200_neg'};
    otherwise
        EndPoint_pos = 30; EndPoint_neg = -30;
end


%% --- Load Metadata ---
T = readtable(fullfile(base_folder, [session, '_metadata.csv']));
%% --- Analyze Each Trial ---
Summary_all = cell(height(T), 1);
endPointErr_all = cell(height(T), 1);
endPointVar_all = cell(height(T), 1);

opts = struct('save_stacked', true, 'offset', false);

trial_indices = [4 8 9 12]; % 1:length(trial_mat_files) after Robyn is done segmenting
for i = trial_indices
    fname = trial_mat_files(i).name;
    tokens = regexp(fname, 'STIM_\d+_(\d+)_Cal\.mat', 'tokens');
    if isempty(tokens), warning("Couldn't parse file: %s", fname); continue; end

    br_id = str2double(tokens{1}{1});
    row_idx = find(T.BR_File == br_id);
    if isempty(row_idx), warning("No metadata match: %d", br_id); continue; end

    isBaseline = startsWith(string(T.Movement_Trigger(row_idx)), "Baseline");
    metadata_row = T(row_idx, :);
    filename = fullfile(trial_mat_files(i).folder, fname);
    % --- Determine which segment fields to use ---
    % --- Determine which segment fields to use ---
    segment_fields = {'active_like_stim_pos', 'active_like_stim_neg'};  % Default fallback

    if ismember('Stim_Delay', metadata_row.Properties.VariableNames)
        stim_delay_val = metadata_row.Stim_Delay;

        % --- Clean and convert to string ---
        if iscell(stim_delay_val)
            stim_delay_val = string(stim_delay_val{1});
        elseif isnumeric(stim_delay_val)
            if isnan(stim_delay_val)
                stim_delay_val = "NaN";
            else
                stim_delay_val = string(num2str(stim_delay_val));
            end
        elseif ismissing(stim_delay_val)
            stim_delay_val = "NaN";
        else
            stim_delay_val = string(stim_delay_val);
        end

        % --- Choose randomized segment fields if needed ---
        if strcmpi(stim_delay_val, "Random")
            segment_fields = segment_fields_random;
        end
    else
        if ~isBaseline
            warning("Missing Stim_Delay field in metadata_row for trial %d", br_id);
        end
    end



    try
        % === Call new version of analysis_metrics with dynamic segment fields ===
        SummaryMetrics = analysis_metrics( ...
            filename, EndPoint_pos, EndPoint_neg, isBaseline, metadata_row, segment_fields);

        if ~isempty(Summary_all{row_idx})
            Summary_all{row_idx} = [];  % reset if needed
        end
        % Clear previously stored struct (MATLAB caches structure layout across rows)
        Summary_all{row_idx} = [];

        % Force consistent shape using struct2table and back (resolves layout mismatches)
        Summary_all{row_idx} = cell2struct( ...
            table2cell(struct2table(SummaryMetrics)), ...
            fieldnames(SummaryMetrics), 2);


        % === Extract ipsi/contra trial-wise error & variance from all segments ===
        all_err_pos = NaN; all_var_pos = NaN;
        all_err_neg = NaN; all_var_neg = NaN;

        if length(SummaryMetrics) >= 2  % ipsi = 1, contra = 2
            if isfield(SummaryMetrics(1), 'side') && strcmp(SummaryMetrics(1).side, 'ipsi')
                % concatenate all *_pos segment errors
                seg_fields = segment_fields(contains(segment_fields, '_pos'));
                err_list = []; var_list = [];
                for sf = seg_fields
                    if isfield(SummaryMetrics(1), sf{1})
                        err_list = [err_list; SummaryMetrics(1).(sf{1}).all_err(:)];
                        var_list = [var_list; SummaryMetrics(1).(sf{1}).all_var(:)];
                    end
                end
                all_err_pos = err_list;
                all_var_pos = var_list;
            end

            if isfield(SummaryMetrics(2), 'side') && strcmp(SummaryMetrics(2).side, 'contra')
                seg_fields = segment_fields(contains(segment_fields, '_neg'));
                err_list = []; var_list = [];
                for sf = seg_fields
                    if isfield(SummaryMetrics(2), sf{1})
                        err_list = [err_list; SummaryMetrics(2).(sf{1}).all_err(:)];
                        var_list = [var_list; SummaryMetrics(2).(sf{1}).all_var(:)];
                    end
                end
                all_err_neg = err_list;
                all_var_neg = var_list;
            end
        end

        endPointErr_all{row_idx} = [all_err_pos(:), all_err_neg(:)];
        endPointVar_all{row_idx} = [all_var_pos(:), all_var_neg(:)];

        % === Transfer summary metrics ===
        for k = 1:numel(SummaryMetrics)
            side_struct = SummaryMetrics(k);
            summary_fields = fieldnames(side_struct);
            for f = 1:length(summary_fields)
                sf = summary_fields{f};
                if endsWith(sf, '_summary')
                    summary_data = side_struct.(sf);
                    fields = fieldnames(summary_data);
                    for fld = 1:length(fields)
                        colname = sprintf('%s_%s', fields{fld}, sf);  % fully labeled
                        T.(colname)(row_idx) = summary_data.(fields{fld});
                    end
                end
            end
        end

    catch ME
        warning("Error processing %s: %s", fname, ME.message);
    end
end

writetable(T, fullfile(base_folder, [session '_metadata_with_metrics.csv']));

%% Save all relevant variables
save(fullfile(base_folder, [session, '_endpoints_checkpoint.mat']), ...
    'endPointErr_all', 'endPointVar_all', 'T', 'Summary_all');
%% Read in the checkpoint file
T = readtable(fullfile(base_folder, [session, '_metadata_with_metrics.csv']));