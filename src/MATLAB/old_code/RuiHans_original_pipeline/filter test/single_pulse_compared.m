Data_Raw = load("D:\neuraldata\Caesar_002\seperate_cells\Kilosort4\Project 1 - Occulomotor Kinematics_CELL_7_kilo_11_good\019-16channels-50ms-400hz-80uA_Neural.mat");
Data_ERASER = load("D:\neuraldata\CRR_002_seg1_ERASER\seperate_cells\Kilosort4\seperate_cells_CELL_107_kilo_186_good\019-16channels-50ms-400hz-80uA_Neural.mat");

Data_ERAASR = load("D:\neuraldata\CRR_002_ERAASR_rev4\seperate_cells\Kilosort4\seperate_cells_CELL_107_kilo_147_good\019-16channels-50ms-400hz-80uA_Neural.mat");
%%
Z = ZoomPlot([Data_ERASER.Data.Neural(:, end-1), Data_ERAASR.Data.Neural(:, end-1), Data_Raw.Data.Neural(:, 131)*500])
%%
samples = Data_ERAASR.Data.segments;
samples = struct2cell(samples);
samples = samples{1};
%%

n =12;
buffer = 5;
sample = (samples(n, 1)+buffer*3)*30:(samples(n, 2)+buffer)*30;
figure 
tiledlayout(5,1)

nexttile
plot(sample /30000, Data_ERASER.Data.Neural(sample, end-1), 'DisplayName','ERASER')
hold on
plot(sample/30000, Data_Raw.Data.Neural(sample, 131)*500, 'LineStyle','--','DisplayName','Stimulation')
% plot(sample /30000,artifact_removed(sample), 'DisplayName','artifact removed')
% plot(sample /30000,preprocessed_filtered(sample), 'DisplayName','whitened (kilsoort4)')
% xline(samples(n, 1)/1000, 'DisplayName','stimulation onset','LineWidth',2.0, 'LineStyle','--','Color','r')
xline((samples(n, 2)-5)/1000, 'DisplayName','stimulation end','LineWidth',2.0, 'LineStyle','--','Color','r')
ylim([-500 500])
box off
hold off
legend('FontSize',16)
% xlabel('time(s)')
ylabel('Highpassed (300 Hz) Neural Recording', 'FontSize',16)

title(['Spike Sorting result Caesar Session 2 experiment # 19'  ' pulse train # ' num2str(n)], 'FontSize',30)

nexttile;
plot(sample /30000,Data_ERASER.Data.spktimes_ua(sample), 'DisplayName','Unit Activity','LineWidth',2.0)
% xline(samples(n, 1)/1000, 'DisplayName','stimulation onset','LineWidth',2.0, 'LineStyle','--','Color','r')
xline((samples(n, 2)-5)/1000, 'DisplayName','stimulation end','LineWidth',2.0, 'LineStyle','--','Color','r')
set(gca,'ytick',[])
ylabel('unit activity', 'FontSize',16)
box off


nexttile
plot(sample /30000, Data_ERAASR.Data.Neural(sample, end-1), 'DisplayName','ERAASR')
hold on
plot(sample/30000, Data_Raw.Data.Neural(sample, 131)*500, 'LineStyle','--','DisplayName','Stimulation')
% plot(sample /30000,artifact_removed(sample), 'DisplayName','artifact removed')
% plot(sample /30000,preprocessed_filtered(sample), 'DisplayName','whitened (kilsoort4)')
% xline(samples(n, 1)/1000, 'DisplayName','stimulation onset','LineWidth',2.0, 'LineStyle','--','Color','r')
xline((samples(n, 2)-5)/1000, 'DisplayName','stimulation end','LineWidth',2.0, 'LineStyle','--','Color','r')
ylim([-500 500])
box off
hold off
legend('FontSize',16)
% xlabel('time(s)')
ylabel('Highpassed (300 Hz) Neural Recording', 'FontSize',16)


nexttile;
plot(sample /30000,Data_ERAASR.Data.spktimes_ua(sample), 'DisplayName','Unit Activity','LineWidth',2.0)
% xline(samples(n, 1)/1000, 'DisplayName','stimulation onset','LineWidth',2.0, 'LineStyle','--','Color','r')
xline((samples(n, 2)-5)/1000, 'DisplayName','stimulation end','LineWidth',2.0, 'LineStyle','--','Color','r')
set(gca,'ytick',[])
ylabel('unit activity', 'FontSize',16)
box off

nexttile;
sample_lo_rate = (samples(n, 1)+buffer*3):(samples(n, 2)+buffer);
plot(sample_lo_rate/1000, Data_Raw.Data.ehp_left_3d(sample_lo_rate),'LineWidth',2.0)
% xline(samples(n, 1)/1000, 'DisplayName','stimulation onset','LineWidth',2.0, 'LineStyle','--','Color','r')
xline((samples(n, 2)-5)/1000, 'DisplayName','stimulation end','LineWidth',2.0, 'LineStyle','--','Color','r')
box off
% ylim([-40 20])
xlabel('time (s)', 'FontSize',16)
ylabel('EHP (deg)', 'FontSize',16)

%% channel map 
stim_channels = Data_ERASER.Data.stim_channels;
channels = Data_ERASER.Data.Neural_channels(1:end-2);
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
