clear all; close all; clc;
addpath(genpath('functions'));
session = 'BL_RW_003_Session_1';

%% --- Setup Paths ---
base_folder = fullfile(['/Volumes/CullenLab_Server/Current Project Databases - NHP' ...
    '/2025 Cerebellum prosthesis/Bryan/Data'], session);
search_folder = fullfile(base_folder, 'Calibrated');
trial_mat_files = dir(fullfile(search_folder, 'IntanFile_*', '*_Cal.mat'));
save_figs = true;  % Set to false to skip saving .fig files
show_figs = true;
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
        EndPoint_pos = 41; EndPoint_neg = -33;
        segment_fields_random = {'active_like_stim_pos_nan', 'active_like_stim_pos_0', ...
            'active_like_stim_pos_100', 'active_like_stim_pos_200', ...
            'active_like_stim_neg_nan', 'active_like_stim_neg_0', ...
            'active_like_stim_neg_100', 'active_like_stim_neg_200'};
        baseline_file_nums = [4, 11, 16];
        trial_indices = [6, 7, 8, 9, 12, 13, 14, 18];
    otherwise
        EndPoint_pos = 30; EndPoint_neg = -30;
end
%% --- Load Metadata ---
T = readtable(fullfile(base_folder, [session, '_metadata.csv']));
%% --- Loop over each trial and analyze ---
raw_metrics_all = cell(height(T), 1);  % Save all raw metrics (from analyze_metrics)
all_trial_indices = sort([baseline_file_nums, trial_indices]);

for i = all_trial_indices
    fname = trial_mat_files(i).name;
    tokens = regexp(fname, 'STIM_\d+_(\d+)_Cal\.mat', 'tokens');
    if isempty(tokens), warning("Couldn't parse file: %s", fname); continue; end
    br_id = str2double(tokens{1}{1});
    row_idx = find(T.BR_File == br_id);
    if isempty(row_idx), warning("No metadata match: %d", br_id); continue; end

    metadata_row = T(row_idx, :);
    filename = fullfile(trial_mat_files(i).folder, fname);

    % Decide segment fields
    segment_fields = {'active_like_stim_pos', 'active_like_stim_neg'};
    if ismember('Stim_Delay', metadata_row.Properties.VariableNames)
        stim_delay_val = metadata_row.Stim_Delay;
        if iscell(stim_delay_val), stim_delay_val = string(stim_delay_val{1}); end
        if isnumeric(stim_delay_val), stim_delay_val = string(num2str(stim_delay_val)); end
        if strcmpi(stim_delay_val, "Random")
            segment_fields = segment_fields_random;
        end
    end

    % === Check for catch fields in Data.segments ===
    try
        tmp = load(filename, 'Data');
        if isfield(tmp.Data, 'segments')
            if isfield(tmp.Data.segments, 'catch_pos')
                segment_fields{end+1} = 'catch_pos';
            end
            if isfield(tmp.Data.segments, 'catch_neg')
                segment_fields{end+1} = 'catch_neg';
            end
        end

        raw_metrics = analyze_metrics(filename, EndPoint_pos, EndPoint_neg, ...
            metadata_row, segment_fields);
        raw_metrics_all{row_idx} = raw_metrics;
    catch ME
        warning("Error in trial %d: %s", br_id, ME.message);
    end
end


% Save all raw data
save(fullfile(base_folder, [session '_raw_metrics_all.mat']), ...
    'raw_metrics_all', 'T', 'segment_fields', 'EndPoint_pos', 'EndPoint_neg');
%% Merging baselines and calculating mean/var
load(fullfile(base_folder, [session '_raw_metrics_all.mat']));
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
save(fullfile(base_folder, [session '_merged_baseline.mat']), 'merged_baseline_summary');

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
save(fullfile(base_folder, [session '_summarized_metrics.mat']), 'summary_struct', 'merged_baseline_summary');
%% --- Compare Traces ---
save_dir = fullfile(base_folder, 'Figures');
if ~exist(save_dir, 'dir'), mkdir(save_dir); end
% Ensure save directories exist in the correct hierarchy
base_dirs = {'vsBaselineTraces', 'ComparisonTraces'};
sub_dirs = {'figFigs', 'pngFigs'};

for b = 1:length(base_dirs)
    for s = 1:length(sub_dirs)
        out_path = fullfile(save_dir, base_dirs{b}, sub_dirs{s});
        if ~exist(out_path, 'dir')
            mkdir(out_path);
        end
    end
end

load(fullfile(base_folder, [session '_summarized_metrics.mat']), 'summary_struct', 'merged_baseline_summary');

% Loop over condition trials
for i = 18
    cond_struct = summary_struct(i);
    if isempty(cond_struct.merged_with_summary), continue; end

    cond_data = cond_struct.merged_with_summary;
    cond_br = cond_struct.BR_File;
    meta_cond = T(T.BR_File == cond_br, :);

    % === Determine baseline source ===
    has_merged = isfield(cond_data, 'active_like_stim_pos_nan') || ...
        isfield(cond_data, 'active_like_stim_neg_nan');

    if has_merged
        base_data = cond_data;
        baseline_file_used = i;
    else
        base_data = merged_baseline_summary;
        baseline_file_used = NaN;
    end

    if iscell(meta_cond.Stim_Delay)
        raw_delay = meta_cond.Stim_Delay{1};
    else
        raw_delay = meta_cond.Stim_Delay;
    end

    % Special case for Random: run multiple fixed-delay comparisons
    if ischar(raw_delay) && strcmpi(raw_delay, 'Random')
        delays = [0, 100, 200];
        for delay_val = delays
            for polarity = {'pos', 'neg'}
                side_label = sprintf('active_like_stim_%s_%d', polarity{1}, delay_val);
                try
                    base_data = cond_data;
                    fig = plot_traces(base_data, cond_data, side_label, false, ...
                        meta_cond, true, show_figs);

                    % === Save ===
                    side_short = strrep(side_label, 'active_like_stim_', '');  % e.g., 'pos' or 'neg'
                    stim_str_for_file = sprintf('%dCh_%dHz_%duA_%dmsdelay_%s', ...
                        meta_cond.Channels, ...
                        meta_cond.Stim_Frequency_Hz, ...
                        meta_cond.Current_uA, ...
                        delay_val, ...
                        meta_cond.Movement_Trigger{1});
                    % Define base name without any extension

                    save_filename_base = sprintf('%s_Condition_%03d_%s_vsBaselineTraces', ...
                        stim_str_for_file, cond_br, side_short);
                    if save_figs
                        % Save .fig
                        fig_save_path = fullfile(save_dir, 'vsBaselineTraces', 'figFigs', [save_filename_base, '.fig']);
                        savefig(fig, fig_save_path);
                    end
                    % Save .png
                    png_save_path = fullfile(save_dir, 'vsBaselineTraces', 'pngFigs', [save_filename_base, '.png']);
                    print(fig, png_save_path, '-dpng', '-r300');

                    close(fig);
                    fprintf('Saved Random Delay Comparison: %s\n', save_filename_base);

                catch ME
                    warning('Plot failed for Random trial %d (%s, delay %d): %s', ...
                        i, polarity{1}, delay_val, ME.message);
                end
            end
        end
        for polarity = {'pos', 'neg'}
            side_label = sprintf('active_like_stim_%s', polarity{1});
            try
                base_data = cond_data;
                fig = plot_rand_condition_traces(base_data, cond_data, side_label, false, ...
                    meta_cond, true, show_figs);

                % === Save ===
                side_short = strrep(side_label, 'active_like_stim_', '');  % e.g., 'pos' or 'neg'
                trigger_clean = strrep(strtrim(meta_cond.Movement_Trigger{1}), ' ', '_');
                stim_str_for_file = sprintf('%dCh_%dHz_%duA_%s', ...
                    meta_cond.Channels, ...
                    meta_cond.Stim_Frequency_Hz, ...
                    meta_cond.Current_uA, ...
                    trigger_clean);


                % Define base name without any extension
                save_filename_base = sprintf('%s_Condition_%03d_%s_ComparisonAcrossDelay', ...
                    stim_str_for_file, cond_br, side_short);


                if save_figs
                    % Save .fig
                    fig_save_path = fullfile(save_dir, 'ComparisonTraces', 'figFigs', [save_filename_base, '.fig']);
                    savefig(fig, fig_save_path);
                end
                % Save .png
                png_save_path = fullfile(save_dir, 'ComparisonTraces', 'pngFigs', [save_filename_base, '.png']);
                print(fig, png_save_path, '-dpng', '-r300');

                close(fig);
                fprintf('Saved Random Delay Comparison: %s\n', save_filename_base);

            catch ME
                warning('Plot failed for Random trial %d (%s, delay %d): %s', ...
                    i, polarity{1}, ME.message);
            end
        end
        continue;  % skip default pos/neg loop
    end

    % Loop over sides
    for side = {'active_like_stim_pos', 'active_like_stim_neg'}
        side_label = side{1};
        try
            fig = plot_traces(base_data, cond_data, side_label, false, ...
                meta_cond, true, show_figs);
            % Convert delay to numeric if it's a cell
            if iscell(meta_cond.Stim_Delay)
                delay_value = str2double(meta_cond.Stim_Delay{1});
            else
                delay_value = meta_cond.Stim_Delay;
            end
            % === Save ===
            side_short = strrep(side_label, 'active_like_stim_', '');  % e.g., 'pos' or 'neg'
            trigger_clean = strrep(strtrim(meta_cond.Movement_Trigger{1}), ' ', '_');
            stim_str_for_file = sprintf('%dCh_%dHz_%duA_%dmsdelay_%s', ...
                meta_cond.Channels, ...
                meta_cond.Stim_Frequency_Hz, ...
                meta_cond.Current_uA, ...
                delay_value, ...
                trigger_clean);

            % Define base name without any extension
            save_filename_base = sprintf('%s_Condition_%03d_%s_vsBaselineTraces', ...
                stim_str_for_file, cond_br, side_short);
            if save_figs
                % Save .fig
                fig_save_path = fullfile(save_dir, 'vsBaselineTraces', 'figFigs', [save_filename_base, '.fig']);
                savefig(fig, fig_save_path);
            end
            % Save .png
            png_save_path = fullfile(save_dir, 'vsBaselineTraces', 'pngFigs', [save_filename_base, '.png']);
            print(fig, png_save_path, '-dpng', '-r300');

            close(fig);
            fprintf('Saved: %s\n', save_filename_base);
        catch ME
            warning('Plot failed for trial %d (%s): %s', i, side_label, ME.message);
        end
    end
end