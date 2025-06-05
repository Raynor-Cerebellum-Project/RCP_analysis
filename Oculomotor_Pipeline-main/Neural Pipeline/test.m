
load("\\10.16.59.34\cullenlab_server\Current Project Databases - NHP\2021 Abducens Stimulation (Neuropixel)\Data\Project 1 - Occulomotor Kinematics\Caesar_Session_2 - Copy\Renamed\014-17channels-50ms-200hz-85uA_Neural.mat",'Data'); %load data structure from a .mat file
Data_back = Data;
%%
neuropixel_index = [    18, 19, 20, 21, 22, 23, 24, 25, ...
   26, 27, 29, 17, 2,  32, 1,  30, ...
    31, 39, 3,  36, 38, 28, 35, 37, ...
    4,  34, 16, 33, 15, 14, 13, 12, ...
    11, 10, 9,  8,  7,  6,  5,  63, ...
    59, 56, 64, 58, 55, 40, 57, 54, ...                                                                                               
    41, 60, 53, 43, 61, 52, 44, 62, ...
    51, 42, 47, 50, 45, 48, 49, 46, ...
    65, 96, 69, 66, 95, 68, 67, 94, ...
    70, 83, 93, 72, 84, 92, 71, 85, ...
    91, 73, 88, 90, 81, 87, 89, 82, ...
    86, 108, 107, 106,105,104,103,102,...
    101,100,99, 98, 80, 97, 79, 109,...
    76, 78, 117,75, 77, 110,74, 114,...
    115,112,113,111,128,116,118,119,...
    120,121,122,123,124,125,126,127];
load("E:\neuraldata\Caesar_002\seperate_cells\Kilosort4\Project 1 - Occulomotor Kinematics_CELL_78_kilo_138_good\007-16channels-50ms-400hz-80uA_Neural.mat")

channels = Data.Neural_channels;
mat_chan = neuropixel_index(channels);
Data.Neural_channels = [1:128]';
save('test.mat', 'Data')

%%

seg = [5.8*1000:5.95*1000];
ua = Data.ua(seg);
channel = 84;
neural_seg = 5.8*30000:5.95*30000;
neuraldat = Data.Neural(neural_seg, channel);
EP = Data.ehp_left_3d(seg);

figure 
tiledlayout(3,1)
nexttile;
plot(seg/1000, EP)
box off
xlabel('time (s)')
ylabel('EP (deg)')

nexttile;
plot(neural_seg/30000, neuraldat)
box off
xlabel('time (s)')

nexttile;
plot(seg/1000, ua)
box off
xlabel('time (s)')
