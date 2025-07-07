clear; close all; clc;
addpath(genpath('functions'));

%% --- Setup Session and Paths ---
session = 'BL_RW_003_Session_1';
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);

intan_folder     = fullfile(base_folder, 'Intan');
cal_folder       = fullfile(base_folder, 'Calibrated');
metadata_csv     = fullfile(base_folder, 'Metadata', [session, '_metadata.csv']);
save_path        = fullfile(base_folder, 'Checkpoints', [session, '_fr_segments.mat']);

window_ms = [-800, 1200];
window_samples = round(window_ms * 30 / 1000);  % Convert ms to 1 kHz (30kHz / 30)

%% --- Load Metadata ---
T = readtable(metadata_csv);

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
fr_segments_all = cell(height(T), 1);
segment_fields_random = {'active_like_stim_pos_nan', 'active_like_stim_pos_0', ...
                         'active_like_stim_pos_100', 'active_like_stim_pos_200', ...
                         'active_like_stim_neg_nan', 'active_like_stim_neg_0', ...
                         'active_like_stim_neg_100', 'active_like_stim_neg_200'};

for i = 1:height(T)
    br_id = T.BR_File(i);
    if ~isKey(file_map, br_id), warning("Missing .mat for BR %d", br_id); continue; end

    mat_path = file_map(br_id);
    try
        % Load segments
        tmp = load(mat_path, 'Data');
        if ~isfield(tmp.Data, 'segments'), continue; end
        segments = tmp.Data.segments;

        % --- Locate Intan folder ---
        if ismember('Intan_Index', T.Properties.VariableNames) && ~isnan(T.Intan_Index(i))
            intan_index = T.Intan_Index(i);
        else
            match = regexp(mat_path, 'IntanFile_(\d+)', 'tokens');
            if ~isempty(match)
                intan_index = str2double(match{1}{1});
            else
                warning("Cannot find Intan index for BR %d", br_id); continue;
            end
        end

        intan_subdirs = dir(fullfile(intan_folder, sprintf('BL_closed_loop_STIM_%s_*', session(end))));
        intan_subdirs = intan_subdirs([intan_subdirs.isdir]);
        intan_subdirs = sort({intan_subdirs.name});

        if intan_index > length(intan_subdirs)
            warning("Intan index %d out of bounds for BR %d", intan_index, br_id); continue;
        end

        fr_path = fullfile(intan_folder, intan_subdirs{intan_index}, 'firing_rate_data.mat');
        if ~isfile(fr_path), warning("Missing FR data: %s", fr_path); continue; end

        fr_data = load(fr_path, 'smoothed_fr_all');
        smoothed_fr_all = fr_data.smoothed_fr_all;

        % Determine segment fields
        segment_fields = {'active_like_stim_pos', 'active_like_stim_neg'};
        if ismember('Stim_Delay', T.Properties.VariableNames)
            delay_val = T.Stim_Delay(i);
            if iscell(delay_val), delay_val = string(delay_val{1}); end
            if isnumeric(delay_val), delay_val = string(num2str(delay_val)); end
            if strcmpi(delay_val, "Random")
                segment_fields = segment_fields_random;
            end
        end

        % Extract aligned FR traces
        fr_traces = struct();
        for f = 1:length(segment_fields)
            field = segment_fields{f};
            if ~isfield(segments, field), continue; end
            onsets = segments.(field)(:, 1);  % sample idx in 30kHz
            n_seg = size(onsets, 1);

            ch_traces = cell(size(smoothed_fr_all));
            for ch = 1:length(smoothed_fr_all)
                fr = smoothed_fr_all{ch};
                traces = nan(n_seg, diff(window_ms)+1);
                for s = 1:n_seg
                    idx = round(onsets(s) / 30) + (window_ms(1):window_ms(2));  % convert to 1 kHz
                    if min(idx) > 0 && max(idx) <= length(fr)
                        traces(s, :) = fr(idx);
                    end
                end
                ch_traces{ch} = traces;
            end
            fr_traces.(field) = ch_traces;
        end

        fr_segments_all{i} = struct('BR_File', br_id, ...
                                    'fr_traces', fr_traces, ...
                                    'segment_fields', segment_fields);
    catch ME
        warning("Error processing BR %d: %s", br_id, ME.message);
    end
end

%% Save raw FR segments before merging
save_path = fullfile(base_folder, 'Checkpoints', [session, '_fr_segments_raw.mat']);
save(save_path, 'fr_segments_all', 'T', 'window_ms', 'window_samples');
fprintf('Saved raw FR segments to: %s\n', save_path);

%% --- Merging baselines and calculating mean/var ---
baseline_file_nums = [4, 11, 16];  % Modify as needed per session
valid_baselines = baseline_file_nums(~cellfun(@isempty, fr_segments_all(baseline_file_nums)));

merged_baseline_fr = fr_segments_all{valid_baselines(1)};
segment_fields_all = merged_baseline_fr.segment_fields;

for b = 2:length(valid_baselines)
    new_fr = fr_segments_all{valid_baselines(b)};
    for f = 1:length(segment_fields_all)
        field = segment_fields_all{f};
        if isfield(new_fr.fr_traces, field)
            for ch = 1:length(new_fr.fr_traces.(field))
                merged_baseline_fr.fr_traces.(field){ch} = ...
                    [merged_baseline_fr.fr_traces.(field){ch}; new_fr.fr_traces.(field){ch}];
            end
        end
    end
end

%% --- Add baseline to random trial baselines ---
combine_map = struct( ...
    'active_like_stim_pos_nan', {{'active_like_stim_pos', 'active_like_stim_pos_nan'}}, ...
    'active_like_stim_neg_nan', {{'active_like_stim_neg', 'active_like_stim_neg_nan'}} ...
);

summary_struct_fr = struct();
is_random = strcmpi(string(T.Stim_Delay), 'Random');
random_indices = find(is_random);

for i = 1:height(T)
    cond_fr = fr_segments_all{i};
    if isempty(cond_fr), continue; end

    if ismember(i, baseline_file_nums)
        merged = cond_fr;
    elseif ismember(i, random_indices)
        merged = cond_fr;
        for fieldname = fieldnames(combine_map)'
            cond_field = fieldname{1};
            baseline_fields = combine_map.(cond_field);

            if all(isfield(merged.fr_traces, baseline_fields))
                merged.fr_traces.(cond_field) = cell(size(merged.fr_traces.(baseline_fields{1})));
                for ch = 1:length(merged.fr_traces.(cond_field))
                    merged.fr_traces.(cond_field){ch} = ...
                        [merged_baseline_fr.fr_traces.(baseline_fields{1}){ch}; ...
                         merged.fr_traces.(baseline_fields{2}){ch}];
                end
            end
        end
    else
        merged = cond_fr;
    end

    summary_struct_fr(i).merged_fr = merged.fr_traces;
    summary_struct_fr(i).segment_fields = merged.segment_fields;
    summary_struct_fr(i).BR_File = T.BR_File(i);
end

%% Save merged FR data
save_path_summary = fullfile(base_folder, 'Checkpoints', [session, '_fr_segments_merged.mat']);
save(save_path_summary, 'summary_struct_fr', 'merged_baseline_fr', '-v7.3');
fprintf('Saved merged FR summary to: %s\n', save_path_summary);
