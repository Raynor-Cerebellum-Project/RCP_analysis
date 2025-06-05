function [SummaryMetrics] = analysis_metrics(filename, EndPoint_pos, EndPoint_neg, do_plot, isBaseline, metadata_row)
% extract_endpoint_error: Computes endpoint position error and movement variability.
%
% INPUTS:
%   - filename: path to .mat file containing Data structure
%   - EndPoint_pos: ground truth endpoint for ipsilateral movements
%   - EndPoint_neg: ground truth endpoint for contralateral movements
%   - do_plot: whether to plot each segment (default false)
%    
% OUTPUTS:
    if nargin < 4
        do_plot = false;
    end
    
% Debug
    % EndPoint_pos = 32; EndPoint_neg = -26;
    % filename = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_002_Session_1/Calibrated/IntanFile_19/BL_closed_loop_STIM_002_022_Cal.mat';
    % T_path = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_002_Session_1/BL_RW_002_Session_1_metadata_with_errors.csv';
    % 
    % if exist(T_path, 'file')
    %     T = readtable(T_path);
    %     metadata_row = T(22, :);
    % else
    %     metadata_row = table();  % fallback
    % end

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
            Sides = {};
        end
    else
        Sides = {};
    end

    % -----------------------------
    % Handle Missing Segment Cases
    % -----------------------------
    if isempty(Sides)
        SummaryMetrics = struct();
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

    % === High-pass filter entire headYawVel signal to avoid filter warm-up artifacts ===
    fs = 1000;        % Sampling frequency (Hz)
    cutoff = 2;       % High-pass cutoff frequency (Hz)
    order = 4;        % Filter order
    [b, a] = butter(order, cutoff / (fs/2), 'high');
    Data.headYawVel_filtered = filtfilt(b, a, Data.headYawVel);

    % -----------------------------
    % Loop Over Each Stimulation Side
    % -----------------------------
    for j = 1:length(Sides)
        side = Sides{j};
        Segment = Data.segments.(side);

        trigger_type = '';
        if nargin >= 6 && isfield(metadata_row, 'Movement_Trigger')
            trigger_type = metadata_row.Movement_Trigger{1};  % e.g., 'Start' or 'End'
        end
        
        for i = 1:size(Segment, 1)
            if ismember('Stim_Delay', metadata_row.Properties.VariableNames) && metadata_row.Stim_Delay > 0
                disp(metadata_row.Stim_Delay);
                % If delay is positive, align to start + delay
                abs_loc = Segment(i,1) + metadata_row.Stim_Delay;
            elseif strcmpi(trigger_type, 'End')
                abs_loc = Segment(i,2);  % Align to end
            else
                abs_loc = Segment(i,1);  % Default: align to start
            end
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
            if isnan(idx1) || isnan(idx2) || idx1 <= 0 || idx2 <= 0 || idx1 > idx2 || idx2 > length(Data.headYawVel)
                continue;  % Skip this trial due to invalid indices
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
        
        % -----------------------------
        % Additional Metrics Computation
        % -----------------------------
        vel_trials = nan(size(Segment,1), length(t));
        oscillation_counts = nan(size(Segment,1), 1);
        fft_power = nan(size(Segment,1), 1);
        for i = 1:size(Segment, 1)
            idx1 = Data.segments3.(side)(i,1);
            idx2 = Data.segments3.(side)(i,2);
            if idx2 > length(Data.headYawVel), continue; end
        
            if isnan(idx1) || isnan(idx2) || idx1 <= 0 || idx2 <= 0 || idx1 > idx2 || idx2 > length(Data.headYawVel)
                continue;  % Skip this trial due to invalid indices
            end
            vel_seg = Data.headYawVel(idx1:idx2);
            vel_trials(i,:) = vel_seg;
        
            % --- Max speed & Average speed
            max_speed(i, :) = max(abs(vel_seg(800:1300)));  % 0-500 ms post stim
            avg_speed(i, :) = mean(abs(vel_seg(800:1300)));
        
            % --- Endpoint oscillations (zero-crossings in post-peak)
            osc_count = sum(diff(sign(vel_seg(1050:end))) ~= 0);
            oscillation_counts(i) = osc_count;
        
            % --- Variability over 500ms
            var_500ms(i, :) = std(vel_seg(800:1300));
        
            % --- Frequency content > 2Hz
            % Parameters
            % --- Frequency content > 2Hz using pre-filtered velocity
            signal_filt = Data.headYawVel_filtered(idx1:idx2);
            fft_power(i) = mean(signal_filt(800:1300).^2);
        end
        
        % Save per-side
        SummaryMetrics.(side).max_speed     = max_speed;
        SummaryMetrics.(side).avg_speed     = avg_speed;
        SummaryMetrics.(side).oscillations  = oscillation_counts;
        SummaryMetrics.(side).var_500ms     = var_500ms;
        SummaryMetrics.(side).fft_power     = fft_power;

    end
    % === Compute signed endpoint error and variability matrix ===
    side_pos = Sides{1};  % ipsi
    side_neg = Sides{2};  % contra
    
    EndPointErr.(side_pos) = EndPoint.(side_pos) - EndPoint_pos;
    EndPointErr.(side_neg) = -1*(EndPoint.(side_neg) - EndPoint_neg);
    
    n_pos = length(EndPointErr.(side_pos));
    n_neg = length(EndPointErr.(side_neg));
    max_trials = max(n_pos, n_neg);
    
    % === Initialize NaN-padded matrices for error and variability ===
    all_err = nan(max_trials, 2);  % [n_trials x 2]: col1 = ipsi, col2 = contra
    all_var = nan(max_trials, 2);
    
    % Assign values to columns: col1 = ipsi (pos), col2 = contra (neg)
    all_err(1:n_pos, 1) = EndPointErr.(side_pos);
    all_err(1:n_neg, 2) = EndPointErr.(side_neg);
    all_var(1:n_pos, 1) = EndPointVar.(side_pos);
    all_var(1:n_neg, 2) = EndPointVar.(side_neg);

    side_fields = {side_pos, side_neg};  % {'active_like_stim_pos', 'active_like_stim_neg'}
    
    for j = 1:2
        side = side_fields{j};
    
        % Assign errors and variability vectors
        SummaryMetrics.(side).all_err = all_err(:, j);
        SummaryMetrics.(side).all_var = all_var(:, j);
    end
    
    % === Add summary fields for ipsi and contra ===
    summary_fields = {'all_err', 'all_var', 'max_speed', 'avg_speed', ...
                      'oscillations', 'var_500ms', 'fft_power'};
    summary_sides = {side_pos, side_neg};
    summary_names = {'ipsi_summary', 'contra_summary'};

    for j = 1:2
        side = summary_sides{j};
        summary_struct = struct();
        % Assign number of valid trials
        if j == 1
            summary_struct.n_trials = n_pos;
        else
            summary_struct.n_trials = n_neg;
        end

        for k = 1:length(summary_fields)
            field = summary_fields{k};
            if isfield(SummaryMetrics.(side), field)
                data = SummaryMetrics.(side).(field);
                summary_struct.([field '_mean']) = mean(data, 'omitnan');
                summary_struct.([field '_var'])  = var(data, 'omitnan');
            end
            if strcmp(field, 'all_err')
                summary_struct.([field '_abs_mean']) = mean(abs(data), 'omitnan');
                summary_struct.([field '_abs_var'])  = var(abs(data), 'omitnan');
            end
        end

        SummaryMetrics.(summary_names{j}) = summary_struct;
    end

end