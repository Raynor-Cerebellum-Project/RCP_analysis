clear all; close all; clc;
addpath(genpath(fullfile('..', 'functions')));

%% --- Setup Paths and Define session
session = 'Nike/20260304_NRR_RW020_Fastig';
[~, session_name] = fileparts(session);
% Get machine-specific root path
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);
relative_path = fullfile('Current Project Databases - NHP', ...
                         '2025 Cerebellum prosthesis', 'Bryan', 'Data', session_name);
%%
search_folder      = fullfile(base_folder, 'Calibrated');
intan_folder      = fullfile(base_folder, 'Intan');
metadata_csv_path  = fullfile(base_folder, 'Metadata', [session_name, '_metadata.csv']);
stim_files = dir(fullfile(search_folder, '**', '*_Cal_stim.mat'));
nonstim_files = dir(fullfile(search_folder, '**', '*_Cal.mat'));
intanfiles = dir(fullfile(intan_folder, '**', 'stim_data.mat'));

file_map = containers.Map('KeyType', 'double', 'ValueType', 'char');
% extract_br = @(name) str2double(regexp(name, 'STIM_\d+_(\d+)_Cal', 'tokens', 'once'));
extract_br = @(name) str2double(regexp(name, 'fastig_(\d+)_Cal', 'tokens', 'once'));

% First add stim files
for trial = 1:numel(stim_files)
    br = double(extract_br(stim_files(trial).name));
    if ~isnan(br)
        file_map(br) = fullfile(stim_files(trial).folder, stim_files(trial).name);
    end
end

% Add non-stim files only if not already included
for trial = 1:numel(nonstim_files)
    br = double(extract_br(nonstim_files(trial).name));
    if ~isnan(br)
        file_map(br) = fullfile(nonstim_files(trial).folder, nonstim_files(trial).name);
    end
end

intan_map = containers.Map('KeyType','double','ValueType','char');

for trial = 1:numel(intanfiles)
    fpath = fullfile(intanfiles(trial).folder, intanfiles(trial).name);
    % Try extracting BR number from folder
    tokens = regexp(intanfiles(trial).folder, 'fastig_(\d+)', 'tokens');
    if ~isempty(tokens)
        br = str2double(tokens{1}{1});
        intan_map(br) = fpath;
    end
end
% Sort Intan stim_data files by folder name
[~, intan_sort_idx] = sort({intanfiles.folder});
intanfiles = intanfiles(intan_sort_idx);

% Reconstruct sorted struct array
all_br = sort(cell2mat(keys(file_map)));
trial_mat_files = struct('name', {}, 'folder', {}, 'stim_path', {});

for trial = 1:numel(all_br)
    br = all_br(trial);
    fpath = file_map(br);
    [fldr, fname, ext] = fileparts(fpath);

    trial_mat_files(trial).folder = fldr;
    trial_mat_files(trial).name   = [fname, ext];

    if trial <= numel(intanfiles)
        trial_mat_files(trial).stim_path = fullfile(intanfiles(trial).folder, intanfiles(trial).name);
    else
        trial_mat_files(trial).stim_path = '';
    end
end

save_figs = false;  % Set to false to skip saving .fig files
show_figs = false;
trace_analysis_plot = false;
% --- Sort trial_mat_files by BR_File number extracted from filename ---
br_nums = zeros(length(trial_mat_files), 1);
for trial = 1:length(trial_mat_files)
    tokens = regexp(trial_mat_files(trial).name, 'fastig_(\d+)_Cal', 'tokens');
    if ~isempty(tokens)
        br_nums(trial) = str2double(tokens{1}{1});
    else
        br_nums(trial) = NaN;
    end
end

[~, sort_idx] = sort(br_nums);
trial_mat_files = trial_mat_files(sort_idx);
%% --- Session-specific Endpoint Targets ---
switch session
    case 'Nike/20260304_NRR_RW020_Fastig'
        EndPoint_pos = 30; EndPoint_neg = -30;
        baseline_file_nums = [1, 6];
        trial_indices = [2, 3, 4];
        at_rest_indices = [5];
        segment_fields_random = {'both', 'ipsi', 'contra', 'ipsi_0', 'contra_0', 'ipsi_100', 'contra_100', 'ipsi_200', 'contra_200'};
        segment_fields = {'both', 'ipsi' , 'contra'};
    otherwise
        EndPoint_pos = 30; EndPoint_neg = -30;
end
%% --- Load Metadata ---
T = readtable(metadata_csv_path);
%% --- Loop over each trial and analyze ---
for trial = 4 % trial_indices
    fname = trial_mat_files(trial).name;
    tokens = regexp(fname, 'fastig_(\d+)_Cal', 'tokens');
    if isempty(tokens)
        warning("Couldn't parse file: %s", fname);
        continue;
    end

    br_id = str2double(tokens{1}{1});
    row_idx = find(T.BR_File == br_id);
    if isempty(row_idx)
        warning("No metadata match: %d", br_id);
        continue;
    end

    metadata_row = T(row_idx, :);
    cal_path = fullfile(trial_mat_files(trial).folder, fname);
    stim_path = trial_mat_files(trial).stim_path;

    % Load calibrated file
    cal_tmp = load(cal_path, 'Data');
    Data = cal_tmp.Data;

    clear cal_tmp;

    % Load stim_data.mat if available
    stim_ext = [];
    if ~isempty(stim_path) && isfile(stim_path)
        stim_tmp = load(stim_path, 'Stim_data');

        if isfield(stim_tmp, 'Stim_data') && ~isempty(stim_tmp.Stim_data)
            stim_data = stim_tmp.Stim_data;

            % stim_data expected to be (n_channels x n_samples)
            stim_ext = extract_stim_triggers_and_blocks(stim_data);

            active_channels = stim_ext.active_channels;
            trigger_pairs = stim_ext.trigger_pairs;
            block_bounds_samples = stim_ext.block_bounds_samples;
            pulse_sizes = stim_ext.pulse_sizes;

            fprintf('BR file %d: found %d pulses across %d blocks\n', ...
                br_id, size(trigger_pairs,1), size(block_bounds_samples,1));
        else
            warning('Stim_data field missing or empty for BR file %d', br_id);
        end
    else
        warning('No stim_data.mat found for BR file %d', br_id);
    end

    % Decide segment fields
    if ismember('Stim_Delay', metadata_row.Properties.VariableNames)
        stim_delay_val = metadata_row.Stim_Delay;

        if iscell(stim_delay_val)
            stim_delay_val = string(stim_delay_val{1});
        end
        if isnumeric(stim_delay_val)
            stim_delay_val = string(num2str(stim_delay_val));
        end

        if strcmpi(stim_delay_val, "Random")

            % Example: detect yaw velocity threshold crossings from Data
            if isfield(Data, 'yaw_vel') && ~isempty(stim_ext)
                yaw_vel = Data.yaw_vel;
                pos_cross = find(yaw_vel(1:end-1) < 85  & yaw_vel(2:end) >= 85) + 1;
                neg_cross = find(yaw_vel(1:end-1) > -85 & yaw_vel(2:end) <= -85) + 1;
                
                onset_intan = block_bounds_samples(:, 1);
                onset_idx = round((onset_intan - Data.Intan_idx(1)) / 30) + 1;
                onset_idx = onset_idx(onset_idx >= 1 & onset_idx <= length(Data.yaw_vel));
                
                max_delay = 300;
                all_cross = [pos_cross(:); neg_cross(:)];
                
                % helper: find first threshold crossing in segment
                get_trig = @(seg) min(all_cross(all_cross >= seg(1) & all_cross <= seg(2)));
                % helper: find stim after trig within max_delay
                get_stim = @(trig) min(onset_idx(onset_idx >= trig & onset_idx <= trig + max_delay));
                
                % classify ipsi
                ipsi_cond = NaN(size(Data.segments.ipsi, 1), 1);
                for i = 1:size(Data.segments.ipsi, 1)
                    trig = get_trig(Data.segments.ipsi(i,:));
                    if ~isempty(trig) && ~isnan(trig)
                        stim = get_stim(trig);
                        if ~isempty(stim)
                            d = stim - trig;
                            if d < 75;        ipsi_cond(i) = 0;
                            elseif d < 175;   ipsi_cond(i) = 100;
                            elseif d < 275;   ipsi_cond(i) = 200;
                            end
                        end
                    end
                end
                
                % classify contra
                contra_cond = NaN(size(Data.segments.contra, 1), 1);
                for i = 1:size(Data.segments.contra, 1)
                    trig = get_trig(Data.segments.contra(i,:));
                    if ~isempty(trig) && ~isnan(trig)
                        stim = get_stim(trig);
                        if ~isempty(stim)
                            d = stim - trig;
                            if d < 75;        contra_cond(i) = 0;
                            elseif d < 175;   contra_cond(i) = 100;
                            elseif d < 275;   contra_cond(i) = 200;
                            end
                        end
                    end
                end
                % ipsi - build Nx4
                ipsi_full = NaN(size(Data.segments.ipsi, 1), 4);
                ipsi_full(:, 1:2) = Data.segments.ipsi;
                for i = 1:size(Data.segments.ipsi, 1)
                    trig = get_trig(Data.segments.ipsi(i,:));
                    if ~isempty(trig) && ~isnan(trig)
                        ipsi_full(i, 3) = trig;
                        stim = get_stim(trig);
                        if ~isempty(stim)
                            ipsi_full(i, 4) = stim;
                        end
                    end
                end
                
                Data.segments.ipsi_0   = ipsi_full(ipsi_cond == 0,   :);
                Data.segments.ipsi_100 = ipsi_full(ipsi_cond == 100, :);
                Data.segments.ipsi_200 = ipsi_full(ipsi_cond == 200, :);
                Data.segments.ipsi_nan = ipsi_full(isnan(ipsi_cond), :);
                
                % contra - build Nx4
                contra_full = NaN(size(Data.segments.contra, 1), 4);
                contra_full(:, 1:2) = Data.segments.contra;
                for i = 1:size(Data.segments.contra, 1)
                    trig = get_trig(Data.segments.contra(i,:));
                    if ~isempty(trig) && ~isnan(trig)
                        contra_full(i, 3) = trig;
                        stim = get_stim(trig);
                        if ~isempty(stim)
                            contra_full(i, 4) = stim;
                        end
                    end
                end
                
                Data.segments.contra_0   = contra_full(contra_cond == 0,   :);
                Data.segments.contra_100 = contra_full(contra_cond == 100, :);
                Data.segments.contra_200 = contra_full(contra_cond == 200, :);
                Data.segments.contra_nan = contra_full(isnan(contra_cond), :);
            else
                warning('No yaw_vel or stim data found for BR file %d', br_id);
            end
            segment_fields = segment_fields_random;
        end
    end
    save(cal_path, 'Data', '-v7.3');
    fprintf('Saved updated Data to %s\n', cal_path);
end