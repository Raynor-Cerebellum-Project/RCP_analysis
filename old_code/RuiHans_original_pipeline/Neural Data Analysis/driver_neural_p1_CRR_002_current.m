%% create neural file list of intended tirals

[datafolder, neuralFiles] = readfolder("", "*_neural.mat");
% trial_number =[10:17, 19, 32:38, 40, 43:47, 49];
file_indices = [];
num_list = [];
trial_number = [];
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
experiment_setting = readtable("D:\Experiment Summary\Experiment_Setting_Summary.xlsx", "Sheet","CRR_NXPL_STIM_002");
durs = experiment_setting.Duration_ms_;
freqs = experiment_setting.Frequency_Hz_;
ids = experiment_setting.BR;
locs = experiment_setting.region_of_stim;
curs = experiment_setting.Current;
[locs_numeric, uniq_locs] = grp2idx(locs(1:24));
[durs_numeric, uniq_durs] = grp2idx(durs(1:24));
[freqs_numeric, uniq_freqs] = grp2idx(freqs(1:24));
[curs_numeric, uniq_curs] = grp2idx(curs(1:24));
locs_numeric(isnan(locs_numeric))=0;
durs_numeric(isnan(durs_numeric))=0;
freqs_numeric(isnan(freqs_numeric))=0;
curs_numeric(isnan(curs_numeric))=0;
%%
A_numeric = locs_numeric;
A_uniq = uniq_locs;
B_numeric = durs_numeric;
B_uniq = uniq_durs;
C_numeric = freqs_numeric;
C_uniq = uniq_freqs;
D_numeric = curs_numeric;
D_uniq = uniq_curs;

%% Varying region of stim
% Comp1 = findComp(A_numeric, B_numeric, C_numeric, D_numeric, ids, A_uniq, B_uniq, C_uniq, D_uniq );
% trial_nums = Comp1.trial_nums;
% durations = Comp1.B;
% currents = Comp1.D;
% frequencies = Comp1.C;
% regions = Comp1.A;

%% Varying currents
Comp1 = findComp(D_numeric, B_numeric, C_numeric, A_numeric, ids, D_uniq, B_uniq, C_uniq, A_uniq );
trial_nums = Comp1.trial_nums;
durations = Comp1.B;
currents = Comp1.A;
frequencies = Comp1.C;
regions = Comp1.D;

%% Varying frequency
% Comp1 = findComp(C_numeric, B_numeric, D_numeric, A_numeric, ids, C_uniq, B_uniq, D_uniq, A_uniq );
% trial_nums = Comp1.trial_nums;
% durations = Comp1.B;
% currents = Comp1.C;
% frequencies = Comp1.A;
% regions = Comp1.D;

%% Varying duration
% Comp1 = findComp(B_numeric, A_numeric, C_numeric, D_numeric, ids, B_uniq, A_uniq, C_uniq, D_uniq );
% trial_nums = Comp1.trial_nums;
% durations = Comp1.A;
% currents = Comp1.D;
% frequencies = Comp1.C;
% regions = Comp1.B;

%% preprocess and post_process on the data 
p.prebuffer = 100; %prepulse length ms
p.postbuffer = 150; %postpulse length ms
% p.num_pulse_threshold = 5; %An BR file should have >5 pulses
p.fc1 = 125;
p.fc2 = 75;
p.fs = 1000; 
p.flag = 1;
p.threshs = [45, 0.3, 0.9, 0.5, 0.5];
session_name = "Caesar-session-2";
Processed_Data = cell(2, 1);
% Removed_Data = cell(size(Filelist, 1), 1);
%%
temp = 1:length(trial_nums);
n = 7;
trial_num = trial_nums(temp(n), :);
duration = durations{temp(n)};
freq = frequencies{temp(n)};
region = regions{temp(n)};
cur = currents(temp(n),:);

for i = 1:2
    file_path = fullfile(datafolder, neuralFiles{file_indices(num_list == trial_num(i))});
    [~, Processed_Data{i}, ~] = pipeline_neural(file_path, p, session_name); 
end

Refined_Data = post_process_neural(Processed_Data);



% %% varying current
label = {['trial # ' num2str(trial_num(1)) ' stim with ' cur{1} ' \muA'], ['trial # ' num2str(trial_num(2)) ' stim with ' cur{2} ' \muA']};
title_txt = sprintf("at %s Hz in %s ms at " + region, freq, duration);
coloring = {'b', 'r'};
fig = figure;
tiledlayout(4,2)

for i = 1:2
nexttile;
segs = Refined_Data{i};
plot(segs.timeframe, segs.ipsi_ehp_avg, 'DisplayName', label{i});
hold on
x_plot = [segs.timeframe, fliplr(segs.timeframe)]; 
y1_plot = [segs.CI_ipsi_ehp_lower, fliplr(segs.CI_ipsi_ehp_upper)];
fill(x_plot, y1_plot, 1,'FaceColor', coloring{i},'FaceAlpha',0.3, 'EdgeColor','none', 'DisplayName', '95% CI');%fill the confidence interval with color
xline(0, '--r', 'DisplayName','Stimulus onset');
xline(str2double(duration), 'k--', 'DisplayName',[duration ' ms'])
hold off
box off
axis([-50 segs.timeframe(end) -50 10]);
title(sprintf("Average Eye horizontal Position with stim %s",  title_txt),'Fontsize',12);
legend
xlabel("time (ms)")
ylabel("Eye Horizontal Position (deg)")
ax = gca;
ax.FontSize = 16; 
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
xline(str2double(duration), 'k--', 'DisplayName',[duration ' ms'])
hold off
box off
axis([-50 segs.timeframe(end) -800 600]);
title(sprintf("Average Eye horizontal Velocity with stim %s", title_txt),'Fontsize',12);
legend
xlabel("time (ms)")
ylabel("Eye Horizontal Velocity (deg/s)")   
ax = gca;
ax.FontSize = 16; 
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
if ~isempty(duration)
    x = [0, str2double(duration), str2double(duration), 0];
    y = [0, 0, length(ua)+1, length(ua)+1];
    patch(x, y, 'yellow', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end
            % Formatting
xlim([-50 segs.timeframe(end)])
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
if ~isempty(duration)
    x = [0, str2double(duration), str2double(duration), 0];
    y = [0, 0, length(ua)+1, length(ua)+1];
    patch(x, y, 'yellow', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end
            % Formatting
xlim([-50 segs.timeframe(end)])
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
xline(str2double(duration), 'k--', 'DisplayName',[duration ' ms'])
hold off
box off
axis([-50 segs.timeframe(end) 0 150]);
legend
title(sprintf("Average Firing Rate for one Cell stim %s", title_txt ),'Fontsize',12);
xlabel("time (ms)")
ylabel("Unit Firing Rate")   
ax = gca;
ax.FontSize = 16; 
end



%% plot channel maps
file_path = fullfile(datafolder, neuralFiles{file_indices(num_list == trial_num(1))});
load(file_path);
stim_channels1 = Data.stim_channels;
file_path = fullfile(datafolder, neuralFiles{file_indices(num_list == trial_num(2))});
load(file_path);
stim_channels2 = Data.stim_channels;
channels = Data.Neural_channels;
chanMap = 'ImecPrimateStimRec128_042421.mat';
load(chanMap, 'xcoords', 'ycoords');


figure
scatter(xcoords, ycoords,'LineWidth',3, 'DisplayName', 'Neuropixel Probe Channel Map')
hold on
scatter(xcoords(stim_channels1), ycoords(stim_channels1), 'red', 'filled','LineWidth',3,'DisplayName', ['Stim Channels ' 'at ' region ])
% scatter(xcoords(stim_channels2), ycoords(stim_channels2), 'yellow', 'filled','LineWidth',3,'DisplayName', ['Stim Channels ' 'at ' region{2} ])
scatter(xcoords(channels), ycoords(channels), 'green', 'filled','LineWidth',3,'DisplayName', 'Cluster Sites')
hold off
legend
title('Neuropixel Probe Channel Map')
xlabel('xcoords')
ylabel('ycoords')
axis([(min(xcoords)-40*range(xcoords)) (max(xcoords)+40*range(xcoords)) 0 (max(ycoords))]);
ax = gca;
ax.FontSize = 16; 

%%
function allComps = findComp(A_numeric, B_numeric, C_numeric, D_numeric, ids, A_uniq, B_uniq, C_uniq, D_uniq )
groups = findgroups(B_numeric, C_numeric, D_numeric);
grp_idx = find_groups_appearing_twice(groups);

allComps.trial_nums = ids(grp_idx(:,2:end));
allComps.B = B_uniq(B_numeric(grp_idx(:,2)));
allComps.C = C_uniq(C_numeric(grp_idx(:,2)));
allComps.D = D_uniq(D_numeric(grp_idx(:,2)));
allComps.A = A_uniq(A_numeric(grp_idx(:,2:end)));

end
%%
function result = find_groups_appearing_twice(lst)
    % Count occurrences of each number
    unique_vals = unique(lst);
    counts = histcounts(lst,'BinMethod','integers');
    % Find numbers that appear exactly twice
    numbers_twice = unique_vals(counts == 2);
    
    % Initialize result matrix
    result = zeros(length(numbers_twice), 3);
    
    % Find indices and store in result
    for i = 1:length(numbers_twice)
        num = numbers_twice(i);
        indices = find(lst == num);
        result(i, :) = [num, indices(1), indices(2)];
    end
end