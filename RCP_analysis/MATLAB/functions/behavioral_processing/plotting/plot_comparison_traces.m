function plot_comparison_traces(base_folder, session, summary_struct, T, baseline_file_num, Summary_all, offset, use_ci)
    % Setup
    save_dir = fullfile(base_folder, 'Figures', 'ComparisonTraces');
    if ~exist(save_dir, 'dir'), mkdir(save_dir); end

    % Gather trial file list
    subfolder = 'Calibrated';  % or customize per session
    file_pattern = fullfile('IntanFile_*', '*_Cal.mat');
    search_folder = fullfile(base_folder, subfolder);
    trial_mat_files = dir(fullfile(search_folder, file_pattern));
    sides = {'active_like_stim_pos', 'active_like_stim_neg'};

    for i = 1:length(summary_struct)
        if isempty(summary_struct(i)), continue; end

        condition_file_num = summary_struct(i).BR_File;

        % Determine baseline file: default or condition-specific
        has_merged_pos = isfield(summary_struct(i), 'merged_with_summary') && ...
                         isfield(summary_struct(i).merged_with_summary, 'active_like_stim_nan_pos');
        has_merged_neg = isfield(summary_struct(i), 'merged_with_summary') && ...
                         isfield(summary_struct(i).merged_with_summary, 'active_like_stim_nan_neg');
        use_custom_baseline = has_merged_pos || has_merged_neg;

        baseline_num = use_custom_baseline * condition_file_num + ~use_custom_baseline * baseline_file_num;

        % Load files and metadata
        condition_data = load_trial_data(trial_mat_files, condition_file_num);
        baseline_data  = load_trial_data(trial_mat_files, baseline_num);
        metadata_cond  = T(T.BR_File == condition_file_num, :);
        metadata_base  = T(T.BR_File == baseline_num, :);

        % Plot for both sides
        for s = 1:length(sides)
            side = sides{s};
            fig = plot_traces(baseline_data, condition_data, side, offset, ...
                              metadata_base, metadata_cond, save_dir, ...
                              baseline_num, condition_file_num, Summary_all, use_ci);

            % Save
            stim_str = sprintf('%dCh_%dHz_%duA_%dmsdelay_%s', ...
                metadata_cond.Channels, metadata_cond.Stim_Frequency_Hz, ...
                metadata_cond.Current_uA, metadata_cond.Stim_Delay, ...
                metadata_cond.Movement_Trigger{1});
            side_short = strrep(side, 'active_like_stim_', '');
            filename = sprintf('%s_Condition_%03d_%s_ComparisonTraces.png', stim_str, condition_file_num, side_short);
            print(fig, fullfile(save_dir, filename), '-dpng', '-r300');
            close(fig);
        end
    end
end

function Data = load_trial_data(trial_mat_files, file_num)
    match_str = sprintf('_%03d_Cal.mat$', file_num);
    match_idx = find(~cellfun(@isempty, regexp({trial_mat_files.name}, match_str)));
    if isempty(match_idx), error("File %d not found", file_num); end
    file_path = fullfile(trial_mat_files(match_idx).folder, trial_mat_files(match_idx).name);
    tmp = load(file_path, 'Data');
    Data = tmp.Data;
end
