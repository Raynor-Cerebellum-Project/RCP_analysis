%% Load data
% load('BL_closed_loop_STIM_001_026_Cal.mat') % 19 and baseline: 26
load('BL_closed_loop_STIM_001_026_Cal.mat')

%%
Sides = {'active_like_stim_pos', 'active_like_stim_neg'};
EndPoint = struct(); % Initialize result structure
EndPointErr = struct(); % Initialize result structure
EndPoint_pos = 32;
EndPoint_neg = -26;

for j = 1:length(Sides)
    side = Sides{j};
    
    Segment = Data.segments.(side);
    accel = diff(Data.headYawVel)/0.001; % Compute acceleration

        % Segment 2: Align based on peak velocity
         for i = 1:length(Segment) % Align to peak velocity
    
            vel = (Data.headYawVel(Segment(i,1) - 200:Segment(i,2)+200)); %**Stim var 2 for pitch (extract pm 200 of the window)
            vel = vel - mean(vel(1:10));
            [pk loc] = max(abs(vel));
            abs_loc = loc+Segment(i,1) - 200
            Data.segments2.(side)(i,1)=abs_loc-500; %-500 ***modify this window for motor command condition to not align with peak
            Data.segments2.(side)(i,2)=abs_loc+700; %500
    
         end
         
         % Segment 3: Align to stimulation onset
         for i = 1:length(Segment) % align to stimulation onset

            vel = (Data.headYawVel(Segment(i,1):Segment(i,2))); %**Stim var 2 for pitch
%             vel = vel - mean(vel(1:10));
            abs_loc = Segment(i,1)
            Data.segments3.(side)(i,1)=abs_loc-800; %-500 ***modify this window for motor command condition to not align with peak
            Data.segments3.(side)(i,2)=abs_loc+1200; %500

         end

    t = linspace(-800,1200,2001);
    figure;
    subplot(1,2,1);

    % Plots
    for i = 1:length(Data.segments3.(side))
        side_clean = strrep(side, '_', ' ');
        plot(t,Data.headYawVel(Data.segments3.(side)(i,1):Data.segments3.(side)(i,2)), 'm', 'linewidth', 2);
%         hold on
%         plot(t,accel(Data.segments3.(side)(i,1):Data.segments3.(side)(i,2)), 'k', 'linewidth', 2);
        title(['Head Velocity Segments - ' side_clean]);
        xlabel('Time (ms)');
        ylabel('Head Velocity (deg/s)');
        hold on
        box off; set(gca, 'TickDir', 'out');
    end

    subplot(1,2,2);
    for i = 1:length(Data.segments3.(side))
        side_clean = strrep(side, '_', ' ');
        plot(t,Data.headYawPos(Data.segments3.(side)(i,1):Data.segments3.(side)(i,2)), 'b', 'linewidth', 2);
        title(['Head Position Segments - ' side_clean]);
        xlabel('Time (ms)');
        ylabel('Head Position (deg/s)');
        hold on
        box off; set(gca, 'TickDir', 'out');
    end

figure; 
subplot(1,2,1); % Top subplot: velocity & acceleration
hold on;
title(['Head Velocity Segments - ' strrep(side,'_',' ')]);
xlabel('Time (ms)');
ylabel('Velocity / Acceleration');
box off;

subplot(1,2,2); % Bottom subplot: position
hold on;
title(['Head Position Segments - ' strrep(side,'_',' ')]);
xlabel('Time (ms)');
ylabel('Position (deg)');
box off; set(gca, 'TickDir', 'out');

% Identify Endpoint from Acceleration Zero-Crossing
    for i = 1:size(Data.segments3.(side),1)
        idx1 = Data.segments3.(side)(i,1);
        idx2 = Data.segments3.(side)(i,2);
        seg_len = idx2 - idx1 + 1;
    
    %     t_seg = t(idx1:idx2);
        t_seg = t;
        vel_seg = Data.headYawVel(idx1:idx2);
        accel_seg = accel(idx1:idx2);
        pos_seg = Data.headYawPos(idx1:idx2);
    
        % Plot velocity and acceleration
        subplot(1,2,1);
        plot(t_seg, vel_seg, 'm', 'linewidth', 2);
        plot(t_seg, accel_seg, 'k', 'linewidth', 2);
    
        % Plot position trace
        subplot(1,2,2);
        plot(t_seg, pos_seg, 'b', 'linewidth', 2);
    
        % Compute Endpoint Errors
        % Check segment is long enough
        if seg_len >= 1300
            accel_slice = accel_seg(1050:1300);
            t_slice = t_seg(1050:1300);
            pos_slice = pos_seg(1050:1300);
    
            % Find first zero-crossing
            zc_idx = find(diff(sign(accel_slice)) ~= 0, 1);
            if ~isempty(zc_idx)
                zero_cross_time = t_slice(zc_idx);
                zero_cross_value = accel_slice(zc_idx);
                zero_cross_pos = pos_slice(zc_idx);
    
                % Plot red dot on acceleration plot
                subplot(1,2,1);
                plot(zero_cross_time, zero_cross_value, 'ro', 'MarkerSize', 6, 'LineWidth', 2);
    
                % Plot red dot on position plot
                subplot(1,2,2);
                plot(zero_cross_time, zero_cross_pos, 'ro', 'MarkerSize', 6, 'LineWidth', 2);
    
                % Save position value at zero-crossing
                field_name = ['Segment' num2str(i)];
                EndPoint.(side)(i) = zero_cross_pos;
                EndPointVar.(side)(i) = std(vel_seg(zero_cross_time+800:zero_cross_time+1300));
            end
        end
    end
end

%% Stacked Velocity and Position Traces with Shaded Stim Window
Sides = {'active_like_stim_pos', 'active_like_stim_neg'};
trace_spacing = 20; % vertical space between trials

for j = 1:length(Sides)
    side = Sides{j};
    Segment = Data.segments3.(side);
    n_trials = size(Segment, 1);
    cmap = winter(n_trials); % trial color map
    t = linspace(-800, 1200, 2001);

    figure;

    %% === Subplot 1: Velocity ===
    subplot(2,1,1);
    hold on;
    title(['Stacked Head Velocity Traces - ' strrep(side, '_', ' ')]);
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

        plot(t, stacked_vel, 'Color', cmap(i,:), 'LineWidth', 1.5);
        legend_entries{end+1} = ['Trial ' num2str(i)];

        max_offset_vel = max(max_offset_vel, max(stacked_vel));
        min_offset_vel = min(min_offset_vel, min(stacked_vel));
    end

    % Set limits before plotting fill
    ylim([min_offset_vel-10, max_offset_vel+10]);
    xlim([-800 1200]);

    % Add shaded stim region
    h_fill1 = fill([0 100 100 0], ...
                   [min_offset_vel-10 min_offset_vel-10 max_offset_vel+10 max_offset_vel+10], ...
                   [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    uistack(h_fill1, 'bottom');

    legend(legend_entries, 'Location', 'eastoutside');

    %% === Subplot 2: Position ===
    subplot(2,1,2);
    hold on;
    title(['Stacked Head Position Traces - ' strrep(side, '_', ' ')]);
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

        plot(t, stacked_pos, 'Color', cmap(i,:), 'LineWidth', 1.5);

        max_offset_pos = max(max_offset_pos, max(stacked_pos));
        min_offset_pos = min(min_offset_pos, min(stacked_pos));
    end

    % Set limits before plotting fill
    ylim([min_offset_pos-10, max_offset_pos+10]);
    xlim([-800 1200]);

    % Add shaded stim region
    h_fill2 = fill([0 100 100 0], ...
                   [min_offset_pos-10 min_offset_pos-10 max_offset_pos+10 max_offset_pos+10], ...
                   [0.8 0.1 0.1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    uistack(h_fill2, 'bottom');
end

%%

side = Sides{1} %pos side
EndPointErr.(side) = EndPoint.(side)-EndPoint_pos;

Err_mean(1,1)=mean(abs(EndPointErr.(side))); % for bar graph
Err_std(1,1)=std(abs(EndPointErr.(side)));
var_mean(1,1) = mean(EndPointVar.(side));
var_std(1,1) = std(EndPointVar.(side));


side = Sides{2} %neg side
EndPointErr.(side) = EndPoint.(side)-EndPoint_neg;

Err_mean(1,2)=mean(abs(EndPointErr.(side))); % for bar graph
Err_std(1,2)=std(abs(EndPointErr.(side)));
var_mean(1,2) = mean(EndPointVar.(side));
var_std(1,2) = std(EndPointVar.(side));

%% Bar Plot of Absolute Errors
figure;
x = categorical({'ipsi', 'contra'}); %for 5 directions
x = reordercats(x,{'ipsi', 'contra'}); %for 5 directions
bar(x, Err_mean);
hold on;

errorbar(Err_mean, Err_std,'.k')
xlabel('Stimulation Side');
ylabel('Absolute Endpoint Error (deg)');
title('Endpoint Error by Stimulation Side');
box off; set(gca, 'TickDir', 'out');

%% Bar graph of signed erros 
side = Sides{1} %pos side
EndPointErr.(side) = EndPoint.(side)-EndPoint_pos; %positive is overshoot

Err_mean(1,1)=mean(EndPointErr.(side)); % for bar graph
Err_std(1,1)=std(EndPointErr.(side));


side = Sides{2} %neg side
EndPointErr.(side) = (EndPoint.(side)-EndPoint_neg)*-1; %positive is overshoot

Err_mean(1,2)=mean(EndPointErr.(side)); % for bar graph
Err_std(1,2)=std(EndPointErr.(side));

%% Bar Plot of signed Errors
figure;
x = categorical({'ipsi', 'contra'}); %for 5 directions
x = reordercats(x,{'ipsi', 'contra'}); %for 5 directions
bar(x, Err_mean);
hold on;

errorbar(Err_mean, Err_std,'.k')
xlabel('Stimulation Side');
ylabel('Signed Endpoint Error (deg)');
title('Endpoint Error by Stimulation Side');
box off; set(gca, 'TickDir', 'out');
ylim([-20 8])

%% Bar Plot of End point variability
figure;
x = categorical({'ipsi', 'contra'}); %for 5 directions
x = reordercats(x,{'ipsi', 'contra'}); %for 5 directions
bar(x, var_mean);
hold on;

errorbar(var_mean, var_std,'.k')
xlabel('Stimulation Side');
ylabel('Endpoint Variability (deg/s)');
title('Endpoint Variability by Stimulation Side');
box off; set(gca, 'TickDir', 'out');
ylim([0 30])
%%
%If you want to plot a segment labelled "raw_example"
% figure;
% plot(Data.headYawVel(Data.segments.raw_example(1,1):Data.segments.raw_example(1,2)), 'k', 'linewidth', 2);
% hold on
% plot(Data.stim_trig(Data.segments.raw_example(1,1):Data.segments.raw_example(1,2))/100, 'k', 'linewidth', 2);
% box off

%% Comparison between baseline and stim endpoint errors