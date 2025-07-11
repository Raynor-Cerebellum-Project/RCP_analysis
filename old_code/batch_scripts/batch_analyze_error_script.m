addpath(genpath('functions'));
session = 'BL_RW_001_Session_1';

%% --- Setup Paths ---
base_folder = fullfile(['/Volumes/CullenLab_Server/Current Project Databases - NHP' ...
    '/2025 Cerebellum prosthesis/Bryan/Data'], session);
% search_folder = fullfile(base_folder, 'Calibrated');
% trial_mat_files = dir(fullfile(search_folder, 'IntanFile_*', '*_Cal.mat'));
% 
search_folder = fullfile(base_folder, 'Renamed2');
trial_mat_files = dir(fullfile(search_folder, '*_Cal.mat'));
%% --- Session-specific Endpoint Targets ---
switch session
    case 'BL_RW_001_Session_1'
        EndPoint_pos = 32; EndPoint_neg = -26;
    case 'BL_RW_002_Session_1'
        EndPoint_pos = 49.5857; EndPoint_neg = -34.4605;
    otherwise
        EndPoint_pos = 30; EndPoint_neg = -30;
end
%% --- Load Metadata ---
T = readtable(fullfile(base_folder, [session, '_metadata.csv']));

%% --- Analyze Each Trial ---
[all_abs_err_mean, all_abs_err_std, all_var_mean, all_var_std] = deal(nan(height(T), 3));
endPointErr_all = cell(height(T), 1);     % Structs per trial
endPointVar_all = cell(height(T), 1);  % Structs per trial

opts = struct('endpoint_plot', false, ...
    'save_stacked', true, ...
    'offset', false);

for i = 1:length(trial_mat_files)
    fname = trial_mat_files(i).name;
    tokens = regexp(fname, 'STIM_\d+_(\d+)_Cal\.mat', 'tokens');
    if isempty(tokens), warning("Couldn't parse file: %s", fname); continue; end

    br_id = str2double(tokens{1}{1});
    row_idx = find(T.BR_File == br_id);
    if isempty(row_idx), warning("No metadata match: %d", br_id); continue; end

    isBaseline = startsWith(string(T.Movement_Trigger(row_idx)), "Baseline");
    metadata_row = T(row_idx, :);
    filename = fullfile(trial_mat_files(i).folder, fname);

    try
        [abs_err_mean, abs_err_std, var_mean, var_std, all_err, all_var] = extract_endpoint_error( ...
            filename, EndPoint_pos, EndPoint_neg, ...
            opts.endpoint_plot, opts.save_stacked, opts.offset, isBaseline, metadata_row);

        all_abs_err_mean(row_idx, :) = abs_err_mean;
        all_abs_err_std(row_idx, :)  = abs_err_std;
        all_var_mean(row_idx, :) = var_mean;
        all_var_std(row_idx, :)  = var_std;

        endPointErr_all{row_idx} = all_err;
        endPointVar_all{row_idx} = all_var;
    catch
        warning("Error processing %s", fname);
    end
end


%% --- Analyze Each Trial ---
[all_abs_err_mean, all_abs_err_std, all_var_mean, all_var_std] = deal(nan(height(T), 3));
endPointErr_all = cell(height(T), 1);     % Structs per trial
endPointVar_all = cell(height(T), 1);  % Structs per trial

opts = struct('endpoint_plot', false, ...
    'save_stacked', true, ...
    'offset', false);

for i = 1:length(trial_mat_files)
    fname = trial_mat_files(i).name;
    tokens = regexp(fname, 'STIM_\d+_(\d+)_Cal\.mat', 'tokens');
    if isempty(tokens), warning("Couldn't parse file: %s", fname); continue; end

    br_id = str2double(tokens{1}{1});
    row_idx = find(T.BR_File == br_id);
    if isempty(row_idx), warning("No metadata match: %d", br_id); continue; end

    isBaseline = startsWith(string(T.Movement_Trigger(row_idx)), "Baseline");
    metadata_row = T(row_idx, :);
    filename = fullfile(trial_mat_files(i).folder, fname);

    try
        [abs_err_mean, abs_err_std, var_mean, var_std, all_err, all_var] = extract_endpoint_error( ...
            filename, EndPoint_pos, EndPoint_neg, ...
            opts.endpoint_plot, opts.save_stacked, opts.offset, isBaseline, metadata_row);

        all_abs_err_mean(row_idx, :) = abs_err_mean;
        all_abs_err_std(row_idx, :)  = abs_err_std;
        all_var_mean(row_idx, :) = var_mean;
        all_var_std(row_idx, :)  = var_std;

        endPointErr_all{row_idx} = all_err;
        endPointVar_all{row_idx} = all_var;
    catch
        warning("Error processing %s", fname);
    end
end


%% --- Save Enriched Metadata ---
T.ErrMean_ipsi     = all_abs_err_mean(:,1);
T.ErrMean_contra   = all_abs_err_mean(:,2);
T.ErrMean_combined = all_abs_err_mean(:,3);
T.ErrStd_ipsi      = all_abs_err_std(:,1);
T.ErrStd_contra    = all_abs_err_std(:,2);
T.ErrStd_combined  = all_abs_err_std(:,3);
T.varMean_ipsi     = all_var_mean(:,1);
T.varMean_contra   = all_var_mean(:,2);
T.varMean_combined = all_var_mean(:,3);
T.varStd_ipsi      = all_var_std(:,1);
T.varStd_contra    = all_var_std(:,2);
T.varStd_combined  = all_var_std(:,3);

writetable(T, fullfile(base_folder, [session '_metadata_with_errors.csv']));
%% Read in the checkpoint file
T = readtable(fullfile(base_folder, [session, '_metadata_with_errors.csv']));

%% Save all relevant variables
save(fullfile(base_folder, [session, '_endpoints_checkpoint.mat']), ...
    'endPointErr_all', 'endPointVar_all', 'T');
%% Plotting options Session 1
% Default sorting order
sort_key_index = {'Trigger', 'Channels', 'Freq', 'Dur', 'Delay', 'Current', 'Depth'};
sort_key_order = {'ascend', 'descend', 'descend', 'descend', 'ascend', 'ascend', 'ascend'};

% Plot # of channels
exclude_trials = [8];
eval_condition = 'All';

% Channels
% exclude_trials = [8, 27, 28, 3, 4];
% eval_condition = 'Channels Num';

% Beginning vs end
% exclude_trials = [8];
% eval_condition = 'Beginning vs end';

% Groups of channels
% exclude_trials = [8, 27, 28, 3, 4, 21, 22, 5];
% eval_condition = 'Groups';

% % Sort only by trigger for now
% sort_key = sortrows(sort_key, ...
%     {'Trigger', 'Channels', 'Freq', 'Dur', 'Delay', 'Current', 'Depth'}, ...
%     {'ascend', 'descend', 'descend', 'descend', 'ascend', 'ascend', 'ascend'});

% Hertz
% exclude_trials = [12, 13, 4, 11, 19];
% eval_condition = 'Hertz';
%% Plotting options Session 2
% Plot # of channels
% exclude_trials = [12, 13, 4, 11, 19];
% eval_condition = 'All';

% Delay
% exclude_trials = [12, 13, 14, 15, 4, 11, 19];
% eval_condition = 'Delay';

close all;
%% Sorting
% Initialize all labels as empty strings
T.cond_label = strings(height(T), 1);

% Rows with MovementTrigger == "Beginning" or "End"
stim_idx = (T.Movement_Trigger == "Beginning") | (T.Movement_Trigger == "End");
nonstim_idx = ~stim_idx;

% Filter out excluded trials first
valid_idx = ~ismember(T.BR_File, exclude_trials);
T = T(valid_idx, :); % Filter main table

% Recompute stim_idx and nonstim_idx after filtering
stim_idx = (T.Movement_Trigger == "Beginning") | (T.Movement_Trigger == "End");
nonstim_idx = ~stim_idx;


% Determine which parameters vary across stim_idx trials
fields_to_check = {'Stim_Frequency_Hz', 'Stim_Delay', 'Current_uA', 'Depth_mm', 'Stim_Duration_ms'};
varying = false(size(fields_to_check));
for i = 1:numel(fields_to_check)
    vals = T{stim_idx, fields_to_check{i}};
    varying(i) = numel(unique(vals(~ismissing(vals)))) > 1;
end


% Initialize empty label array
T.cond_label(stim_idx) = "";

% Construct label for each row individually
for k = 1:sum(stim_idx)
    i = find(stim_idx);
    row = i(k);
    parts = [ ...
        string(T.Channels(row)) + "ch"
    ];

    if varying(1), parts(end+1) = string(T.Stim_Frequency_Hz(row)) + "Hz"; end
    if varying(2), parts(end+1) = string(T.Stim_Delay(row)) + "ms"; end
    if varying(3), parts(end+1) = string(T.Current_uA(row)) + "uA"; end
    if varying(4), parts(end+1) = string(T.Depth_mm(row)) + "mm"; end
    if varying(5), parts(end+1) = string(T.Stim_Duration_ms(row)) + "ms"; end

    % Lookup original index in full metadata
    original_idx = find(T.BR_File(row) == T.BR_File);  % assumes T is original table

    if ~isempty(original_idx) && original_idx <= length(endPointErr_all)
        val = endPointErr_all{original_idx};
        if ismatrix(val) && size(val, 2) == 2
            n_ipsi = sum(~isnan(val(:,1)));
            n_contra = sum(~isnan(val(:,2)));
        else
            n_ipsi = 0; n_contra = 0;
        end
    else
        n_ipsi = 0; n_contra = 0;
    end

    % Build final label with newline to split rows
    trial_count_str = sprintf('Ip: %d, Co: %d', n_ipsi, n_contra);
    cond_str = sprintf('(Cond.: %d)', T.BR_File(row));
    T.cond_label(row) = strjoin([strjoin(parts, '/'), trial_count_str, cond_str], '\n');
end

for row = find(nonstim_idx)'
    base_str = string(T.Movement_Trigger(row));
    
    % Use BR_File to find original index
    original_idx = find(T.BR_File(row) == T.BR_File);

    if ~isempty(original_idx) && original_idx <= length(endPointErr_all)
        val = endPointErr_all{original_idx};
        if ismatrix(val) && size(val, 2) == 2
            n_ipsi = sum(~isnan(val(:,1)));
            n_contra = sum(~isnan(val(:,2)));
        else
            n_ipsi = 0; n_contra = 0;
        end
    else
        n_ipsi = 0; n_contra = 0;
    end

    trial_count_str = sprintf('Ip: %d, Co: %d', n_ipsi, n_contra);
    cond_str = sprintf('Cond.: %d', T.BR_File(row));
    
    % Use \n for line break
    T.cond_label(row) = sprintf('%s\n%s, %s', base_str, trial_count_str, cond_str);
end


% Everything else, Baseline Active HoB, Beginning, then End
all_triggers = unique(string(T.Movement_Trigger));
special_triggers = ["Baseline Active HoB"; "Beginning"; "End"];
general_triggers = setdiff(all_triggers, special_triggers, 'stable');
trigger_order = [general_triggers; special_triggers];

% Order
T.TriggerCat = categorical(T.Movement_Trigger, trigger_order, 'Ordinal', true);

% Build sortable table
sort_key = table( ...
    T.TriggerCat, T.Stim_Frequency_Hz, T.Stim_Duration_ms, T.Channels, ...
    T.Stim_Delay, T.Current_uA, T.Depth_mm, T.cond_label, ...
    T.ErrMean_ipsi, T.ErrMean_contra, T.ErrMean_combined, ...
    T.ErrStd_ipsi, T.ErrStd_contra, T.ErrStd_combined, ...
    T.varMean_ipsi, T.varMean_contra, T.varMean_combined, ...
    T.varStd_ipsi, T.varStd_contra, T.varStd_combined, ...
    T.("BR_File"), ...
    'VariableNames', {'Trigger','Freq','Dur','Channels', ...
    'Delay', 'Current', 'Depth', 'Label', ...
                      'MeanIpsi','MeanContra', 'MeanCombined', ...
                      'StdIpsi','StdContra','StdCombined', ...
                      'VarIpsi','VarContra','VarCombined', ...
                      'VarStdIpsi','VarStdContra','VarStdCombined' ...
                      ,'TrialID'} ...
);

% Sort only by trigger for now
sort_key = sortrows(sort_key, sort_key_index, sort_key_order);

% Restrict baseline group (None/Movement/etc.) to Trial 26 only
is_baseline = ~ismember(string(sort_key.Trigger), ["Beginning", "End"]);
is_baseline_active_HoB = ismember(string(sort_key.Trigger), "Baseline Active HoB");

% Keep trial 26 if it's baseline, or any other non-baseline
keep_rows = ~is_baseline | (is_baseline & is_baseline_active_HoB);
sort_key = sort_key(keep_rows, :);
sort_key = sort_key(~ismember(sort_key.TrialID, exclude_trials), :);
T_filtered = T(ismember(T.BR_File, sort_key.TrialID), :);

%% Define title
% Extract unique values from the sorted trials
params = {
    'Freq',     'Hz';
    'Dur',      'ms';
    'Delay',    'ms';
    'Current',  'uA';
    'Depth',    'mm';
};

title_suffix = "";
title_components = {};
for j = 1:numel(fields_to_check)
    if ~varying(j)
        val = unique(T{stim_idx, fields_to_check{j}});
        if isnumeric(val)
            title_components{end+1} = sprintf('%s: %g', strrep(fields_to_check{j}, '_', ' '), val);
        else
            title_components{end+1} = sprintf('%s: %s', strrep(fields_to_check{j}, '_', ' '), string(val));
        end
    end
end

% Final title string
plot_title = strjoin(title_components, ', ');
% %% === Call plotting functions ===
% plot_endpoint_error(sort_key, T_filtered, eval_condition, plot_title, base_folder, session);
% plot_endpoint_variability(sort_key, T_filtered, eval_condition, plot_title, base_folder, session);
% % plot_combined_line(sort_key, T_filtered, eval_condition, plot_title, base_folder, session);
% 
% %% Scatter (jittered) + Mean ± Error Bars, signed errors
% plot_endpoint_error_scatter(sort_key, endPointErr_all, eval_condition, plot_title, base_folder, session);
% plot_endpoint_variability_scatter(sort_key, endPointVar_all, eval_condition, plot_title, base_folder, session);