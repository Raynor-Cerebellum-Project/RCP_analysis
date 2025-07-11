function fig = plot_stacked_velocity_position(Data, side_label, offset, metadata_row)
    % plot_stacked_velocity_position: Plots stacked velocity and position traces
    % with shaded stimulation window for a given side label.
    %
    % Inputs:
    %   - Data: Struct containing fields .headYawVel, .headYawPos, .segments3
    %   - side_label: string, e.g., 'active_like_stim_pos'
    if nargin < 3
        offset = true; % default to true if not specified
    end
    
    trace_spacing = 20 * offset; % 0 if no offset

    Segment = Data.segments3.(side_label);
    n_trials = size(Segment, 1);
    cmap = winter(n_trials); % trial color map
    t = linspace(-800, 1200, 2001);
    vel_traces = nan(n_trials, 2001);  % Preallocate
    pos_traces = nan(n_trials, 2001);

    fig = figure('Visible','on');
    %% === Subplot 1: Velocity ===
    subplot(2,1,1);
    hold on;
    title('Velocity Traces');
    ylabel('Velocity (deg/s)');
    box off; set(gca, 'TickDir', 'out');
    
    max_offset_vel = -inf;
    min_offset_vel = inf;
    legend_entries = {};
    
    for i = 1:n_trials
        idx1 = Segment(i,1); idx2 = Segment(i,2);
        if idx1 < 1 || idx2 > length(Data.headYawVel), continue; end
    
        vel_trace = Data.headYawVel(idx1:idx2);
        vel_trace = vel_trace - mean(vel_trace(1:10));
        stacked_vel = vel_trace + i * trace_spacing;
        vel_traces(i, :) = vel_trace;  % Save unstacked trace
    
        plot(t, stacked_vel, 'Color', cmap(i,:), 'LineWidth', 1.5);
        legend_entries{end+1} = ['Trial ' num2str(i)];
    
        max_offset_vel = max(max_offset_vel, max(stacked_vel));
        min_offset_vel = min(min_offset_vel, min(stacked_vel));
    end
    
    ylim([min_offset_vel-10, max_offset_vel+10]);
    xlim([-800 1200]);
    h_fill1 = fill([0 100 100 0], ...
                   [min_offset_vel-10 min_offset_vel-10 max_offset_vel+10 max_offset_vel+10], ...
                   [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    uistack(h_fill1, 'bottom');
    %% === Subplot 2: Position ===
    subplot(2,1,2);
    hold on;
    title('Position Traces');
    xlabel('Time (ms)');
    ylabel('Position (deg)');
    box off; set(gca, 'TickDir', 'out');
    
    max_offset_pos = -inf;
    min_offset_pos = inf;
    
    for i = 1:n_trials
        idx1 = Segment(i,1); idx2 = Segment(i,2);
        if idx1 < 1 || idx2 > length(Data.headYawPos), continue; end
    
        pos_trace = Data.headYawPos(idx1:idx2);
        pos_trace = pos_trace - mean(pos_trace(1:10));
        stacked_pos = pos_trace + i * trace_spacing;
        pos_traces(i, :) = pos_trace;  % Save unstacked trace
    
        plot(t, stacked_pos, 'Color', cmap(i,:), 'LineWidth', 1.5);
    
        max_offset_pos = max(max_offset_pos, max(stacked_pos));
        min_offset_pos = min(min_offset_pos, min(stacked_pos));
    end
    
    ylim([min_offset_pos-10, max_offset_pos+10]);
    xlim([-800 1200]);
    h_fill2 = fill([0 100 100 0], ...
                   [min_offset_pos-10 min_offset_pos-10 max_offset_pos+10 max_offset_pos+10], ...
                   [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    uistack(h_fill2, 'bottom');
    % === Save velocity and position traces ===
    trace_struct = struct();
    trace_struct.time = t;
    trace_struct.velocity = vel_traces;
    trace_struct.position = pos_traces;
    trace_struct.side = side_label;
    
    session_dir = fileparts(fileparts(fileparts(Data.filename)));  % Or pass in filename
    trace_save_dir = fullfile(session_dir, 'Traces');
    if ~exist(trace_save_dir, 'dir'); mkdir(trace_save_dir); end
    
    trace_file = fullfile(trace_save_dir, ['Traces_' side_label '.mat']);
    save(trace_file, '-struct', 'trace_struct');
end
