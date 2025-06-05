%% create neural file list of intended tirals
% datafolder = 'E:\neuraldata\Daphne_003_mat\Seperate_cells\mid_bot_003_no2021\Kilosort4\Daphne_003_mat_CELL_81_kilo_79_good\';
% neuralFiles = dir(fullfile(datafolder, '*_neural.mat'));
% neuralFiles = struct2cell(neuralFiles);
% neuralFiles = neuralFiles(1,:);
[datafolder, neuralFiles] = readfolder("", "*_neural.mat");
trial_number =[10:17, 19, 32:38, 40, 43:47, 49];
file_indices = [];
num_list = [];

for i = 1:length(neuralFiles)
    % Extract the number from the filename
    filename = neuralFiles{i};
    fileidx = split(filename, ["-","_","."]);
    fileNumber = str2double(fileidx(1));
    % Check if the file number is in the selected ranges
    if ~isempty(trial_number)
        if any(fileNumber == trial_number)
            file_indices = [file_indices, i]; % Add the index to the list
            num_list = [num_list, fileNumber];
        end
    else 
        file_indices = [file_indices, i];
        num_list = [num_list, fileNumber];
    end
end
%% read data file with neural data
%[10, 40]   8	1--8	9--16	100	200
%[11, 42]   8	1--8	9--16	100	400 no 42 intan
%[12, 43]   8	1--8	9--16	50	400
%[13, 44]   8	1--8	9--16	50	200
%[14, 45]   16	1--16	17--32	100	200
% 
% trial_num = [12, 43];
% duration = 50;
% channel_num = 8;
% freq = 400;
% 
% trial_num = [10, 45];
% duration = 100;
% channel_num = 16;
% freq = 200;


% trial_num = [10, 40];
% duration = 100;
% channel_num = 8;
% freq = 200;

trial_num = [14, 45];
duration = 100;
channel_num = 16;
freq = 200;
%% preprocess and post_process on the data 
p.prebuffer = 100; %prepulse length ms
p.postbuffer = 150; %postpulse length ms
% p.num_pulse_threshold = 5; %An BR file should have >5 pulses
p.fc1 = 125;
p.fc2 = 75;
p.fs = 1000; 
p.flag = 1;
p.threshs = [45, 0.3, 0.6, 0.3, 0.3];
session_name = "Daphne-session-3";
Processed_Data = cell(2, 1);
% Removed_Data = cell(size(Filelist, 1), 1);
for i = 1:2
    file_path = fullfile(datafolder, neuralFiles{file_indices(num_list == trial_num(i))});
    [~, Processed_Data{i}, ~] = pipeline_neural(file_path, p, session_name); 
end
%%
Refined_Data = post_process_neural(Processed_Data);
%%

%%
label = {['current steering trial #' num2str(trial_num(1))], ['non current steering trial #' num2str(trial_num(2))]};
coloring = {'b', 'r'};
fig = figure;
tiledlayout(4,2)
title_txt = sprintf("stim with %i channels at %i Hz in %i ms",channel_num, freq, duration);
for i = 1:2
nexttile;
segs = Refined_Data{i};
plot(segs.timeframe, segs.ipsi_ehp_avg, 'DisplayName', label{i});
hold on
x_plot = [segs.timeframe, fliplr(segs.timeframe)]; 
y1_plot = [segs.CI_ipsi_ehp_lower, fliplr(segs.CI_ipsi_ehp_upper)];
fill(x_plot, y1_plot, 1,'FaceColor', coloring{i},'FaceAlpha',0.3, 'EdgeColor','none', 'DisplayName', '95% CI');%fill the confidence interval with color
xline(0, '--r', 'DisplayName','Stimulus onset');
xline(duration, 'k--', 'DisplayName',num2str(duration))
hold off
box off
axis([-50 150 -15 5]);
title(sprintf("Average Eye horizontal Position stim with %i channels at %i Hz in %i ms",channel_num, freq, duration),'Fontsize',12);
legend
xlabel("time (ms)")
ylabel("Eye Horizontal Position (deg)")
    
end


for i = 1:2
nexttile;
segs = Refined_Data{i};
plot(segs.timeframe, 1000*segs.ipsi_ehv_avg, 'DisplayName', label{i});
hold on  
x_plot = [segs.timeframe, fliplr(segs.timeframe)]; 
y3_plot = [segs.CI_ipsi_ehv_lower, fliplr(segs.CI_ipsi_ehv_upper)];
fill(x_plot, 1000*y3_plot, 1, 'FaceColor', coloring{i},'FaceAlpha',0.3, 'EdgeColor','none', 'DisplayName', '95% CI');%fill the confidence interval with color
xline(0, '--r', 'DisplayName','Stimulus onset');
xline(duration, 'k--', 'DisplayName',num2str(duration))
hold off
box off
axis([-50 150 -250 200]);
title(sprintf("Average Eye horizontal Velocity stim with %i channels at %i Hz in %i ms",channel_num, freq, duration),'Fontsize',12);
legend
xlabel("time (ms)")
ylabel("Eye Horizontal Velocity (deg/s)")   
end
 

nexttile;
n = 1;
segs = Refined_Data{n};
ua= segs.ua;
C = bone(length(ua)*2);
C=flip(C(1:length(ua),:));
timeframe = segs.timeframe;
hold on
for row = 1:length(ua)
    ua_seg = ua{row}; % Extract the binary array
    x = find(ua_seg == 1);    % Get indices of 1s
    yStart = row - 0.45;            % Start position of vertical line
    yEnd = row + 0.45;              % End position of vertical line
    
                % Plot vertical lines at each '1' position
    for k = 1:length(x)
        plot([timeframe(x(k)) timeframe(x(k))], [yStart yEnd], 'Color', C(row,:), 'LineWidth', 2);
    end
end
        
% add shaded region that denotes stim duration

    x = [0, duration, duration, 0];
    y = [0, 0, length(ua)+1, length(ua)+1];
    patch(x, y, 'yellow', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

            % Formatting
xlim([-50 150])
ylim([0 length(ua)+1]);
xlabel('time (ms)');
ylabel('trials');
yticks(1:length(ua));
title(sprintf('Raster Plot %s %s', label{n}, title_txt), 'FontSize', 16);
grid off;
box off;
hold off;
ax = gca;
ax.FontSize = 16; 

nexttile;
n = 2;
segs = Refined_Data{n};
ua= segs.ua;
C = bone(length(ua)*2);
C=flip(C(1:length(ua),:));
timeframe = segs.timeframe;
hold on
for row = 1:length(ua)
    ua_seg = ua{row}; % Extract the binary array
    x = find(ua_seg == 1);    % Get indices of 1s
    yStart = row - 0.45;            % Start position of vertical line
    yEnd = row + 0.45;              % End position of vertical line
    
                % Plot vertical lines at each '1' position
    for k = 1:length(x)
        plot([timeframe(x(k)) timeframe(x(k))], [yStart yEnd], 'Color', C(row,:), 'LineWidth', 2);
    end
end
        
% add shaded region that denotes stim duration

    x = [0, duration, duration, 0];
    y = [0, 0, length(ua)+1, length(ua)+1];
    patch(x, y, 'yellow', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

            % Formatting
xlim([-50 150])
ylim([0 length(ua)+1]);
xlabel('time (ms)');
ylabel('trials');
yticks(1:length(ua));
title(sprintf('Raster Plot %s %s', label{n}, title_txt), 'FontSize', 16);
grid off;
box off;
hold off;
ax = gca;
ax.FontSize = 16; 

for i = 1:2
nexttile;
segs = Refined_Data{i};
x = segs.timeframe;
y = segs.fr_avg;
x_fill = [x, fliplr(x)];
y_fill = [y, zeros(size(y))];

fill(x_fill, y_fill, coloring{i}, 'FaceAlpha', 0.5, 'EdgeColor', 'k', 'DisplayName',label{i});
% plot(segs.timeframe, segs.fr_avg, 'DisplayName', label{i});
hold on  
% x_plot = [segs.timeframe, fliplr(segs.timeframe)]; 
% y3_plot = [segs.CI_fr_lower, fliplr(segs.CI_fr_upper)];
% fill(x_plot, y3_plot, 1, 'FaceColor', coloring{i},'FaceAlpha',0.3, 'EdgeColor','none', 'DisplayName', '95% CI');%fill the confidence interval with color
xline(0, '--r', 'DisplayName','Stimulus onset');
xline(duration, 'k--', 'DisplayName',num2str(duration))
hold off
box off
axis([-50 150 0 200]);
legend
title(sprintf("Average Firing Rate for one Cell stim with %i channels at %i Hz in %i ms",channel_num, freq, duration),'Fontsize',12);
xlabel("time (ms)")
ylabel("Unit Firing Rate")   


end

%%