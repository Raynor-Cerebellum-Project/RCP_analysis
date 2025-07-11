clear; close all; clc;
addpath(genpath('functions'));

%% --- Setup Session and Paths ---
session = 'BL_RW_003_Session_1';
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);

intan_folder     = fullfile(base_folder, 'Intan');
cal_folder       = fullfile(base_folder, 'Calibrated');
metadata_csv     = fullfile(base_folder, 'Metadata', [session, '_metadata.csv']);
raw_metrics_path   = fullfile(base_folder, 'Checkpoints', [session, '_raw_metrics_all.mat']);

summary_with_fr_path       = fullfile(base_folder, 'Checkpoints', [session, '_summarized_metrics.mat']);
merged_baseline_with_fr_path = fullfile(base_folder, 'Checkpoints', [session, '_merged_baseline.mat']);
raw_metrics_with_fr_path        = fullfile(base_folder, 'Checkpoints', [session, '_raw_metrics_all_with_fr.mat']);

window_ms = [-800, 1200];
window_samples = round(window_ms * 30 / 1000);  % Convert ms to 1 kHz (30kHz / 30)
%% --- Session-specific Endpoint Targets ---
switch session
    case 'BL_RW_001_Session_1'
        EndPoint_pos = 32; EndPoint_neg = -26;
        search_folder = fullfile(base_folder, 'Renamed2');
        trial_mat_files = dir(fullfile(search_folder, '*_Cal.mat'));
        baseline_file_nums = [26];
        trial_indices = [20, 21];
    case 'BL_RW_002_Session_1'
        EndPoint_pos = 49.5857; EndPoint_neg = -34.4605;
        baseline_file_nums = [4, 27];
        trial_indices = [21];
    case 'BL_RW_003_Session_1'
        EndPoint_pos = 41; EndPoint_neg = -33;
        baseline_file_nums = [4, 11, 16];
        trial_indices = [6, 7, 8, 9, 12, 13, 14, 18]; %9 should have stim but doesn't
    otherwise
        EndPoint_pos = 30; EndPoint_neg = -30;
end
%% --- Load Metadata ---
T = readtable(metadata_csv);
all_trial_indices = sort([baseline_file_nums, trial_indices]);
%% --- Locate Trial Files ---
stim_files = dir(fullfile(cal_folder, '**', '*_Cal_stim.mat'));
nonstim_files = dir(fullfile(cal_folder, '**', '*_Cal.mat'));
all_files = [stim_files; nonstim_files];

file_map = containers.Map('KeyType', 'double', 'ValueType', 'char');
extract_br = @(name) str2double(regexp(name, 'STIM_\d+_(\d+)_Cal', 'tokens', 'once'));

for i = 1:numel(all_files)
    br = double(extract_br(all_files(i).name));
    if ~isnan(br)
        file_map(br) = fullfile(all_files(i).folder, all_files(i).name);
    end
end

all_br = sort(cell2mat(keys(file_map)));

%% --- Process Each Trial ---
tmp = load(raw_metrics_path, 'raw_metrics_all');
raw_metrics_all = tmp.raw_metrics_all;

fr_segments_all = cell(height(T), 1);
segment_fields_random = {'active_like_stim_pos_nan', 'active_like_stim_pos_0', ...
                         'active_like_stim_pos_100', 'active_like_stim_pos_200', ...
                         'active_like_stim_neg_nan', 'active_like_stim_neg_0', ...
                         'active_like_stim_neg_100', 'active_like_stim_neg_200'};

prev_intan_id = -1;
smoothed_fr_all = {};

for i = 1:height(T)
    br_id = T.BR_File(i);
    intan_id = T.Intan_File(i);

    if ~isKey(file_map, br_id)
        warning("Missing .mat for BR %d", br_id);
        continue;
    end

    % Load segments from mat
    mat_path = file_map(br_id);
    tmp = load(mat_path, 'Data');
    if ~isfield(tmp.Data, 'segments'), continue; end
    segments = tmp.Data.segments;

    % Load firing rate only if needed
    if intan_id ~= prev_intan_id
        intan_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));
        intan_dirs = intan_dirs([intan_dirs.isdir]);
        intan_names = {intan_dirs.name};

        if intan_id > numel(intan_names), warning("Bad intan ID"); continue; end
        fr_path = fullfile(intan_folder, intan_names{intan_id}, 'firing_rate_data.mat');
        if ~isfile(fr_path), warning("Missing FR: %s", fr_path); continue; end

        fr_data = load(fr_path, 'smoothed_fr_all');
        smoothed_fr_all = fr_data.smoothed_fr_all;
        prev_intan_id = intan_id;
    end

    % Segment fields
    segment_fields = {'active_like_stim_pos', 'active_like_stim_neg'};
    if ismember('Stim_Delay', T.Properties.VariableNames)
        delay_val = string(T.Stim_Delay(i));
        if strcmpi(delay_val, "Random")
            segment_fields = segment_fields_random;
        end
    end

    % Inject FR data into each segment field
    for f = 1:length(segment_fields)
        field = segment_fields{f};
        if ~isfield(segments, field) || ~isfield(raw_metrics_all{i}, field)
            continue;
        end

        onsets = segments.(field)(:, 1);
        n_seg = size(onsets, 1);

        ch_traces = cell(size(smoothed_fr_all));
        for ch = 1:length(smoothed_fr_all)
            fr = smoothed_fr_all{ch};
            traces = nan(n_seg, diff(window_ms)+1);
            for s = 1:n_seg
                idx = round(onsets(s)/30) + (window_ms(1):window_ms(2));
                if min(idx) > 0 && max(idx) <= length(fr)
                    traces(s, :) = fr(idx);
                end
            end
            ch_traces{ch} = traces;
        end

        % Now inject into raw_metrics
        raw_metrics_all{i}.(field).fr_traces = ch_traces;
    end
end

%% Save raw FR segments before merging
save(raw_metrics_path, ...
    'raw_metrics_all', 'T', 'segment_fields', 'EndPoint_pos', 'EndPoint_neg');
fprintf('Saved raw FR segments to: %s\n', raw_metrics_with_fr_path);

%% Merge
% Initialize merged baseline struct
merged_baseline = raw_metrics_all{baseline_file_nums(1)};
fields = fieldnames(merged_baseline);

for f = 2:length(baseline_file_nums)
    current = raw_metrics_all{baseline_file_nums(f)};
    for field = fields'
        key = field{1};
        if isfield(current, key)
            % Concatenate trial metrics and traces
            fnames = fieldnames(merged_baseline.(key));
            for subf = fnames'
                subkey = subf{1};
                if isfield(current.(key), subkey)
                    merged_baseline.(key).(subkey) = ...
                        [merged_baseline.(key).(subkey); current.(key).(subkey)];
                end
            end
        end
    end
end
% --- Calculate summaries for merged baseline ---
all_fields = fieldnames(merged_baseline);
ipsi_fields = all_fields(contains(all_fields, '_pos'));
contra_fields = all_fields(contains(all_fields, '_neg'));

summary_ipsi = calculate_mean_metrics(merged_baseline, ipsi_fields, 'ipsi');
summary_contra = calculate_mean_metrics(merged_baseline, contra_fields, 'contra');

% --- Merge into summary struct ---
merged_baseline_summary = merged_baseline;
for s = fieldnames(summary_ipsi)'
    merged_baseline_summary.(s{1}) = summary_ipsi.(s{1});
end
for s = fieldnames(summary_contra)'
    merged_baseline_summary.(s{1}) = summary_contra.(s{1});
end

% Save for later access (e.g., in plotting)
save(merged_baseline_with_fr_path, 'merged_baseline_summary');
%% Add baseline to random trial baselines
summary_struct = struct();

% Define mapping of baseline + condition segments
combine_map = struct( ...
    'active_like_stim_pos_nan', {{'active_like_stim_pos', 'active_like_stim_pos_nan'}}, ...
    'active_like_stim_neg_nan', {{'active_like_stim_neg', 'active_like_stim_neg_nan'}} ...
    );

% Identify random trial indices
is_random = strcmpi(string(T.Stim_Delay), 'Random');
random_indices = find(is_random);

% Loop through all trials
for i = all_trial_indices
    cond_metrics = raw_metrics_all{i};
    if isempty(cond_metrics), continue; end

    if ismember(i, baseline_file_nums)
        merged = cond_metrics;
    elseif ismember(i, random_indices)
        merged = combine_baseline_with_rand(merged_baseline_summary, cond_metrics, combine_map);
    else
        merged = cond_metrics;
    end

    % --- Identify all segment fields ---
    all_fields = fieldnames(merged);

    % Split fields by type
    ipsi_fields_main   = all_fields(contains(all_fields, '_pos') & ~contains(all_fields, 'catch'));
    contra_fields_main = all_fields(contains(all_fields, '_neg') & ~contains(all_fields, 'catch'));
    ipsi_fields_catch  = all_fields(contains(all_fields, 'catch_pos'));
    contra_fields_catch= all_fields(contains(all_fields, 'catch_neg'));

    % --- Calculate summaries ---
    summary_ipsi_main   = calculate_mean_metrics(merged, ipsi_fields_main, 'ipsi');
    summary_contra_main = calculate_mean_metrics(merged, contra_fields_main, 'contra');
    summary_ipsi_catch  = calculate_mean_metrics(merged, ipsi_fields_catch, 'ipsi');
    summary_contra_catch= calculate_mean_metrics(merged, contra_fields_catch, 'contra');

    % --- Combine all summaries ---
    merged_with_summary = merged;

    for s = fieldnames(summary_ipsi_main)'
        merged_with_summary.(s{1}) = summary_ipsi_main.(s{1});
    end
    for s = fieldnames(summary_contra_main)'
        merged_with_summary.(s{1}) = summary_contra_main.(s{1});
    end
    if ~isempty(fieldnames(summary_ipsi_catch))
        merged_with_summary.ipsi_catch_summary = summary_ipsi_catch;
    end
    if ~isempty(fieldnames(summary_contra_catch))
        merged_with_summary.contra_catch_summary = summary_contra_catch;
    end

    % --- Store result ---
    summary_struct(i).merged_with_summary = merged_with_summary;
    summary_struct(i).BR_File = T.BR_File(i);
end



%% Save summarized metrics
save(summary_with_fr_path, 'summary_struct', 'merged_baseline_summary', '-v7.3');
fprintf('Saved merged FR summary to: %s\n', summary_with_fr_path);
