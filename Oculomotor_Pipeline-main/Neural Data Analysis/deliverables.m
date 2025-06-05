clear all

trigger_file_path = uigetdir(pwd, "select folder for session trigger files");
% 
% file_path= uigetdir(pwd, "select folder for kilosort4 results files");

[bin_file, location] = uigetfile('*.bin', "select .bin file that store the artifact removed neural data");

%%
[neural_file, location_neural] = uigetfile('*_neural.mat', 'select the file that store the unit activity');
file_indices = [];
% trial_number =[10:17, 19, 32:38, 40, 43:45];
% trial_number = [10:17, 40, 43:45];
% trigger_file_path = 'D:\Oculomotor Research\Current_non-currtent\Neural data analysis\bin_test\mid_bot_all_session_trigger\';
% trial_number = [4, 8, 14, 20];
trial_number = [1, 7, 13, 19];
file_num_list = [1, 7, 13, 19];

segment_marks = zeros(1, length(file_num_list)+1);
for i = 2:length(file_num_list)+1
    trigger_file_name = ['session_trigger_' num2str(file_num_list(i-1)) '.mat'];
    session_trigger = fullfile(trigger_file_path, trigger_file_name);
    trigger = load(session_trigger);
    segment_marks(i) = length(trigger.session_trigger);
end
segment_marks = cumsum(segment_marks);

%%
fileidx = split(neural_file, ["-","_","."]);
fileNumber = str2double(fileidx(1));
load([location_neural, neural_file]);
trial_num = find(file_num_list == fileNumber);
chan = Data.cluster_sites(1);
segment = 1+ segment_marks(trial_num): segment_marks(trial_num+1);
artifact_removed = ReadBin([location bin_file],128,chan,segment);
% preprocessed_filtered = ReadBin([file_path '\temp_wh.dat'], 128, chan, segment);

%% preprocess compared
[neural_file, location_neural] = uigetfile('*_neural.mat', 'select the file that store the unit activity');
fileidx = split(neural_file, ["-","_","."]);
fileNumber = str2double(fileidx(1));
load([location_neural, neural_file]);
%%
set(groot,'defaultLineLineWidth',1.0)
segments_data = struct2cell(Data.segments);
samples = segments_data{1};
buffer = 10000;
n = 10;
sample  = samples(n, 2)*30-buffer/2: samples(n, 2)*30+buffer;
if size(Data.Neural, 2)>=128
    chan_neural = Data.Neural_channels(1);
else
    chan_neural = 1;
end
figure 
subplot(3,2,1)
plot(sample /30000, Data.Neural(sample, chan_neural), 'DisplayName','Target Channel')
hold on
% plot(sample /30000,artifact_removed(sample), 'DisplayName','artifact removed')
% plot(sample /30000,preprocessed_filtered(sample), 'DisplayName','whitened (kilsoort4)')
xline(samples(n, 1)*30/30000, 'DisplayName','stimulation onset','LineWidth',2.0, 'LineStyle','--','Color','r')
xline(samples(n, 2)*30/30000, 'DisplayName','stimulation end','LineWidth',2.0, 'LineStyle','--','Color','r')

box off
hold off
legend('FontSize',20)
xlabel('time(s)')
ylabel('Whitened Neural Recording', 'FontSize',16)
set(gca,'ytick',[])
title(['Spike Sorting result Caesar Session 2 experiment # ' num2str(fileNumber) ' stim trial # ' num2str(n)], 'FontSize',30)

subplot(3,2,3)
plot(sample /30000,Data.spktimes_ua(sample), 'DisplayName','Unit Activity','LineWidth',2.0)
xline(samples(n, 1)*30/30000, 'DisplayName','stimulation onset','LineWidth',2.0, 'LineStyle','--','Color','r')
xline(samples(n, 2)*30/30000, 'DisplayName','stimulation end','LineWidth',2.0, 'LineStyle','--','Color','r')
set(gca,'ytick',[])
ylabel('unit activity', 'FontSize',16)
box off

subplot(3,2,5)
sample_lo_rate = samples(n, 2) -round(buffer/2/30): samples(n, 2) + round(buffer/30);
set(gca,'ytick',[])
plot(sample_lo_rate/1000, Data.ehp_left_3d(sample_lo_rate),'LineWidth',2.0)
xline(samples(n, 1)/1000, 'DisplayName','stimulation onset','LineWidth',2.0, 'LineStyle','--','Color','r')
xline(samples(n, 2)/1000, 'DisplayName','stimulation end','LineWidth',2.0, 'LineStyle','--','Color','r')
box off
% ylim([-40 20])
xlabel('time (s)', 'FontSize',16)
ylabel('EHP (deg)', 'FontSize',16)


%% rastor plot

%% channel map 
stim_channels = Data.stim_channels;
channels = Data.Neural_channels(1:end-2);
chanMap = 'ImecPrimateStimRec128_042421.mat';
load(chanMap, 'xcoords', 'ycoords');

figure
scatter(xcoords, ycoords,'LineWidth',2, 'DisplayName', 'Neuropixel Probe Channel Map')
hold on
scatter(xcoords(stim_channels), ycoords(stim_channels), 'red', 'filled','LineWidth',3,'DisplayName', 'Stim Channels ')
scatter(xcoords(channels), ycoords(channels), 'green', 'filled','LineWidth',3,'DisplayName', 'Cluster Sites')
hold off
box off
axis off
legend
title('Neuropixel Probe Channel Map')
xlabel('xcoords')
ylabel('ycoords')
axis([(min(xcoords)-40*range(xcoords)) (max(xcoords)+40*range(xcoords)) 0 (max(ycoords))]);
ax = gca;
ax.FontSize = 20; 
%% 

