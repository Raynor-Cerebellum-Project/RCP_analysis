function [abs_err_mean, abs_err_std, var_mean, var_std, all_err, all_var] = extract_endpoint_error(filename, EndPoint_pos, EndPoint_neg, do_plot, save_stacked, offset, isBaseline, metadata_row)
% extract_endpoint_error: Computes endpoint position error and movement variability.
%
% INPUTS:
%   - filename: path to .mat file containing Data structure
%   - EndPoint_pos: ground truth endpoint for ipsilateral movements
%   - EndPoint_neg: ground truth endpoint for contralateral movements
%   - do_plot: whether to plot each segment (default false)
%    
% OUTPUTS:
%   - Err_mean: [1x2] mean absolute error for pos and neg sides
%   - Err_std:  [1x2] std of absolute error for pos and neg sides
%   - var_mean: [1x2] mean of post-endpoint velocity std for pos/neg
%   - var_std:  [1x2] std of velocity std across trials
%   - EndPoint: struct of endpoint positions per trial
%   - EndPointErr: struct of endpoint errors per trial
    if nargin < 4
        do_plot = false;
    end
    if nargin < 5, save_stacked = true; end

    if nargin < 6, offset = false; end

    if nargin < 7, isBaseline = false; end
    
    % -----------------------------
    % Load Data and Identify Sides
    % -----------------------------
    % 
    if nargin < 8, row_idx = []; end
% % Debug
%     EndPoint_pos = 32; EndPoint_neg = -26;
%     filename = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_001_Session_1/Renamed2/BL_closed_loop_STIM_001_003_Cal.mat';
%     % Read metadata table
%     [base_folder, session_name, ~] = fileparts(fileparts(fileparts(filename)));
%     T_path = fullfile(base_folder, [session_name, '_metadata_with_errors.csv']);
% 
%     if exist(T_path, 'file') && ~isempty(row_idx)
%         T = readtable(T_path);
%         metadata_row = T(row_idx, :);
%     else
%         metadata_row = table();  % fallback
%     end

% Function
    load(filename); % loads variable `Data`
    if isfield(Data, 'segments') && ~isempty(fieldnames(Data.segments))
        segment_fields = fieldnames(Data.segments);

        % Filter to segment types
        Sides = segment_fields(contains(segment_fields, 'active_like'));

        % Prioritize 'stim' sides, fallback to plain
        pos_idx = find(strcmp(Sides, 'active_like_stim_pos'), 1);
        neg_idx = find(strcmp(Sides, 'active_like_stim_neg'), 1);

        if ~isempty(pos_idx) && ~isempty(neg_idx)
            Sides = {'active_like_stim_pos', 'active_like_stim_neg'};
        else
            % pos_idx = find(strcmp(Sides, 'active_like_pos'), 1);
            % neg_idx = find(strcmp(Sides, 'active_like_neg'), 1);
            % 
            % if ~isempty(pos_idx) && ~isempty(neg_idx)
            %     Sides = {'active_like_pos', 'active_like_neg'};
            % else
            %     warning(['Expected pos/neg side segments not found in ' filename]);
            %     Sides = {};
            % end
            Sides = {};
        end
    else
        Sides = {};
    end

    % -----------------------------
    % Handle Missing Segment Cases
    % -----------------------------
    if isempty(Sides)
        abs_err_mean = [NaN NaN NaN];
        abs_err_std = [NaN NaN NaN];
        var_mean = [NaN NaN NaN];
        var_std = [NaN NaN NaN];
        all_err      = nan(1, 2);  % or zeros(0,2) if you want an empty matrix
        all_var      = nan(1, 2);
        return;
    end

    % -----------------------------
    % Initialize & Preprocess Data
    % -----------------------------
    EndPoint = struct();
    EndPointErr = struct();
    EndPointVar = struct();

    accel = diff(Data.headYawVel) / 0.001; % Acceleration (deg/s²)
    t = linspace(-800, 1200, 2001);        % Time vector in ms (2s window)

    % -----------------------------
    % Loop Over Each Stimulation Side
    % -----------------------------
    for j = 1:length(Sides)
        side = Sides{j};
        Segment = Data.segments.(side);

        % Align each segment to stimulation onset
        for i = 1:size(Segment, 1)
            abs_loc = Segment(i,1);
            Data.segments3.(side)(i,1) = abs_loc - 800;
            Data.segments3.(side)(i,2) = abs_loc + 1200;
        end

        % Loop over aligned trials
        for i = 1:size(Data.segments3.(side),1)
            idx1 = Data.segments3.(side)(i,1);
            idx2 = Data.segments3.(side)(i,2);
            seg_len = idx2 - idx1 + 1;

            if idx2 > length(Data.headYawVel)
                continue
            end

            % Extract velocity, acceleration, and position segments
            vel_seg = Data.headYawVel(idx1:idx2);
            accel_seg = accel(idx1:idx2);
            pos_seg = Data.headYawPos(idx1:idx2);
            
            % -----------------------------
            % Compute Endpoint from Acceleration Zero-Crossing
            % -----------------------------
            if seg_len >= 1300
                accel_slice = accel_seg(1050:1300);   % 250–500 ms post-stim
                pos_slice   = pos_seg(1050:1300);
                t_slice     = t(1050:1300);

                % Find first zero-crossing in accel
                zc_idx = find(diff(sign(accel_slice)) ~= 0, 1);
                if ~isempty(zc_idx)
                    zero_cross_pos = pos_slice(zc_idx);
                    zero_cross_time = t_slice(zc_idx);

                    % Store endpoint
                    EndPoint.(side)(i) = zero_cross_pos;
                    % Compute endpoint variability: velocity std after zero-crossing
                    win_idx = 800 + zc_idx; % Index in full vel_seg (t=0 is index 800)
                    if win_idx+250 <= length(vel_seg)
                        EndPointVar.(side)(i) = std(vel_seg(win_idx:win_idx+250));
                    else
                        EndPointVar.(side)(i) = NaN;
                    end

                    % -----------------------------
                    % Optional Plotting
                    % -----------------------------
                    if do_plot
                        subplot(1,2,1); hold on;
                        plot(t, vel_seg, 'm'); plot(t, accel_seg, 'k');
                        plot(t_slice(zc_idx), accel_slice(zc_idx), 'ro');

                        subplot(1,2,2); hold on;
                        plot(t, pos_seg, 'b');
                        plot(t_slice(zc_idx), zero_cross_pos, 'ro');
                        sgtitle(['File: ' filename ' - ' strrep(side, '_', ' ')])
                    end
                end
            end
        end
    end
    % === Compute signed endpoint error and variability matrix ===
    side_pos = Sides{1};  % ipsi
    side_neg = Sides{2};  % contra
    
    EndPointErr.(side_pos) = EndPoint.(side_pos) - EndPoint_pos;
    EndPointErr.(side_neg) = -1*(EndPoint.(side_neg) - EndPoint_neg);
    
    n_pos = length(EndPointErr.(side_pos));
    n_neg = length(EndPointErr.(side_neg));
    max_trials = max(n_pos, n_neg);
    
    % Initialize NaN-padded matrices
    all_err = nan(max_trials, 2);  % [n_trials x 2]: col1 = ipsi, col2 = contra
    all_var = nan(max_trials, 2);
    
    % Assign values
    all_err(1:n_pos, 1) = EndPointErr.(side_pos);
    all_err(1:n_neg, 2) = EndPointErr.(side_neg);
    all_var(1:n_pos, 1) = EndPointVar.(side_pos);
    all_var(1:n_neg, 2) = EndPointVar.(side_neg);
    
    % === Compute mean and std of absolute error and variability ===
    abs_err_mean = nan(1, 3);
    abs_err_std  = nan(1, 3);
    var_mean     = nan(1, 3);
    var_std      = nan(1, 3);
    
    for j = 1:2
        abs_err_mean(j) = mean(abs(all_err(:,j)), 'omitnan');
        abs_err_std(j)  = std(abs(all_err(:,j)), 'omitnan');
        var_mean(j)     = mean(all_var(:,j), 'omitnan');
        var_std(j)      = std(all_var(:,j), 'omitnan');
    end
    
    % Combined stats (3rd column)
    abs_err_mean(3) = mean(abs(all_err), 'all', 'omitnan');
    abs_err_std(3)  = std(abs(all_err), 0, 'all', 'omitnan');
    var_mean(3)     = mean(all_var, 'all', 'omitnan');
    var_std(3)      = std(all_var, 0, 'all', 'omitnan');

    % Save stacked plots if requested
    if save_stacked
        for j = 1:length(Sides)
            side = Sides{j};
            if isfield(Data, 'segments3') && isfield(Data.segments3, side)
                fig = plot_stacked_velocity_position(Data, side, offset, metadata_row);

                % Parse base filename and condition number (e.g., 027)
                [~, base_name, ~] = fileparts(filename);
                condition_num = regexp(base_name, 'STIM_\d+_(\d+)', 'tokens', 'once');
                if isempty(condition_num)
                    condition_str = 'Unknown';
                else
                    condition_str = condition_num{1};
                end

                % Determine side label
                if contains(side, 'pos')
                    side_str = 'Ipsi';
                    file_side = 'pos';
                elseif contains(side, 'neg')
                    side_str = 'Contra';
                    file_side = 'neg';
                else
                    side_str = 'Unknown';
                    file_side = 'unk';
                end
                % Extract values
                % freq = metadata_row.Stim_Frequency_Hz;
                % current = metadata_row.Current_uA;
                % delay = metadata_row.Stim_Delay;
                % depth = metadata_row.Depth_mm;
                channels = metadata_row.Channels;
                duration = metadata_row.Stim_Duration_ms;
                trigger = metadata_row.Movement_Trigger;
               
                % Title and filename logic
                if isBaseline
                    sgtitle(sprintf('BASELINE: Condition %s, %s, Stacked Traces\n%s', ...
                        condition_str, side_str), ...
                        'FontWeight', 'bold', 'Color', [0.2 0.6 0.2]);
                    save_filename = sprintf('BASELINE_Condition_%s_%s_StackedTraces.png', ...
                        condition_str, file_side);
                else
                    trigger_str = metadata_row.Movement_Trigger{1};  % get string from cell
                    meta_str = sprintf('Freq: %gHz, Current: %gµA, Delay: %gms, Depth: %gmm', ...
                                   channels, duration, trigger_str);
                    sgtitle(sprintf('Condition %s, %s, Stacked Traces\n%s', ...
                        condition_str, side_str, meta_str), ...
                        'FontWeight', 'bold');
                    save_filename = sprintf('Condition_%s_%s_StackedTraces.png', ...
                        condition_str, file_side);
                end

                % Save figure
                session_dir = fileparts(fileparts(fileparts(filename)));
                save_dir = fullfile(session_dir, 'Figures', 'StackedTraces');
                if ~exist(save_dir, 'dir'); mkdir(save_dir); end

                save_name = fullfile(save_dir, save_filename);
                print(fig, save_name, '-dpng', '-r300');
                close(fig);
            end
        end
    end
end
