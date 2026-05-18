% APPEND_STIM_TO_NEURAL
% For each _Cal.mat in Calibrated/IntanFile_N/, loads the corresponding
% stim_data.mat, finds the most active Stim_data channel, and writes it
% into Data.Neural(3, :) mapped to Blackrock 1 kHz time.
%
% Usage: set session_root then run.

session_root = '/Volumes/data/Current Project Databases - NHP/2025 Cerebellum prosthesis/Nike/20260318_NRR_RW025_fastig';

calibrated_root = fullfile(session_root, 'Calibrated');
intan_root      = fullfile(session_root, 'Intan');

% Get sorted list of Intan folders (lexicographic = chronological for timestamp names)
intan_folders = dir(intan_root);
intan_folders = intan_folders([intan_folders.isdir] & ~startsWith({intan_folders.name}, '.'));
[~, sort_order] = sort({intan_folders.name});
intan_folders   = intan_folders(sort_order);

% Get IntanFile_N subfolders sorted by integer index (not lexicographic)
cal_subfolders = dir(calibrated_root);
cal_subfolders = cal_subfolders([cal_subfolders.isdir] & ~startsWith({cal_subfolders.name}, '.'));

% Extract integer indices and sort numerically
cal_indices = nan(length(cal_subfolders), 1);
for k = 1:length(cal_subfolders)
    tok = regexp(cal_subfolders(k).name, 'IntanFile_(\d+)', 'tokens');
    if ~isempty(tok)
        cal_indices(k) = str2double(tok{1}{1});
    end
end
valid = ~isnan(cal_indices);
cal_subfolders = cal_subfolders(valid);
cal_indices    = cal_indices(valid);
[~, sort_order] = sort(cal_indices);
cal_subfolders  = cal_subfolders(sort_order);
cal_indices     = cal_indices(sort_order);

for fi = 1:length(cal_subfolders)
    intan_idx  = cal_indices(fi);
    cal_folder = fullfile(calibrated_root, cal_subfolders(fi).name);

    if intan_idx > length(intan_folders)
        warning('IntanFile_%d exceeds number of Intan folders (%d), skipping', intan_idx, length(intan_folders));
        continue
    end

    stim_mat_path = fullfile(intan_root, intan_folders(intan_idx).name, 'stim_data.mat');
    if ~exist(stim_mat_path, 'file')
        fprintf('No stim_data.mat for IntanFile_%d, skipping\n', intan_idx);
        continue
    end

    cal_files = dir(fullfile(cal_folder, '*_Cal.mat'));
    cal_files = cal_files(~startsWith({cal_files.name}, '.'));
    if isempty(cal_files)
        fprintf('No _Cal.mat in %s, skipping\n', cal_folder);
        continue
    end

    % Load stim_data once per Intan folder
    fprintf('Loading stim_data: %s\n', stim_mat_path);
    S = load(stim_mat_path);
    if ~isfield(S, 'Stim_data')
        warning('stim_data.mat has no Stim_data field, skipping');
        continue
    end

    % Find most active channel (most nonzero samples)
    n_nonzero      = sum(S.Stim_data ~= 0, 2);  % 128 x 1
    [~, active_ch] = max(n_nonzero);
    stim_signal    = S.Stim_data(active_ch, :);  % 1 x N_intan at 30 kHz
    fprintf('  Active channel: %d (%d nonzero samples)\n', active_ch, n_nonzero(active_ch));

    for ci = 1:length(cal_files)
        cal_path = fullfile(cal_files(ci).folder, cal_files(ci).name);
        fprintf('  Processing: %s\n', cal_files(ci).name);

        load(cal_path, 'Data');

        if ~isfield(Data, 'Intan_idx')
            warning('Data has no Intan_idx in %s, skipping', cal_files(ci).name);
            continue
        end
        if ~isfield(Data, 'Neural')
            warning('Data has no Neural field in %s, skipping', cal_files(ci).name);
            continue
        end

        N_bk = size(Data.Neural, 1);  % samples x channels

        % Assert Neural has at least 3 columns
        if size(Data.Neural, 2) < 3
            warning('Data.Neural has only %d cols in %s — expected at least 3, skipping', ...
                size(Data.Neural, 2), cal_files(ci).name);
            continue
        end

        % Check Neural row count matches Intan_idx length; clamp if off (clock-ratio rounding)
        N_idx = size(Data.Intan_idx, 2);
        if N_idx ~= N_bk
            warning('Intan_idx has %d cols but Neural has %d rows in %s — delta=%d, clamping', ...
                N_idx, N_bk, cal_files(ci).name, N_bk - N_idx);
        end
        N_use = min(N_idx, N_bk);

        intan_map = double(Data.Intan_idx(1, 1:N_use));  % 1 x N_use

        % Map each Blackrock sample to its Intan sample; tail beyond N_use stays zero
        stim_col = zeros(N_bk, 1);
        for bk = 1:N_use
            intan_sample = max(1, min(intan_map(bk), length(stim_signal)));
            stim_col(bk) = stim_signal(intan_sample);
        end

        Data.Neural(:, 3) = stim_col;

        % save back to cal_path
        save(cal_path, 'Data', '-v7.3');
        fprintf('    Done — stim mapped to Neural(3,:), saved\n');
    end
end

fprintf('All done.\n');