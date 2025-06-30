clear all

%%

trigger_file_path = uigetdir(pwd, "select folder for session trigger files");

file_path= uigetdir(pwd, "select folder for kilosort4 results files");
%%
[bin_file, location] = uigetfile('*.bin', "select .bin file that store the artifact removed neural data");

file_indices = [];
% trial_number =[10:17, 19, 32:38, 40, 43:45];
% trial_number = [10:17, 40, 43:45];
% trigger_file_path = 'D:\Oculomotor Research\Current_non-currtent\Neural data analysis\bin_test\mid_bot_all_session_trigger\';
% trial_number = [4, 8, 14, 20];
trial_number = [1,7,13,19];
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

chan = 78;
chan_npxl = find(neuropixel_index==chan);
chan2 =80;
chan_npxl2 = find(neuropixel_index==chan2);
chan3 = 77;
chan_npxl3 = find(neuropixel_index==chan3);
sample = 1+ segment_marks(2): segment_marks(3);
artifact_removed = ReadBin([location bin_file],128,chan,sample);
raw_neural = ReadBin("D:\filter_test\seg1_no_filter\all_files_no_filter.bin", 128, chan, sample);
artifact_removed2 = ReadBin([location bin_file],128,chan,sample);
artifact_removed3 = ReadBin([location bin_file],128,chan,sample);
preprocessed_filtered = ReadBin([file_path '\temp_wh.dat'], 128, chan, sample);

preprocessed_raw = ReadBin("D:\filter_test\seg1_no_filter\kilosort4\temp_wh.dat", 128, chan, sample);
set(groot,'defaultLineLineWidth',5.0)
%%
artifact_removed_reg = ReadBin("D:\filter_test\seg1_25104\all_files_seg1_25104.bin", 128, chan, sample);
artifact_removed_reg2 = ReadBin("D:\filter_test\seg1_25104\all_files_seg1_25104.bin", 128, chan, sample);
artifact_removed_reg3 = ReadBin("D:\filter_test\seg1_25104\all_files_seg1_25104.bin", 128, chan, sample);
preprocessed_reg = ReadBin("D:\filter_test\seg1_25104\kilosort4\temp_wh.dat", 128, chan, sample);
% Z = ZoomPlot([artifact_removed_reg, artifact_removed_reg3, preprocessed_reg ]);
 Z = ZoomPlot([artifact_removed, artifact_removed3, preprocessed_filtered ]);
legend('target channel', 'neighbor channel', 'whitened')
%%
preprocessed_reg = ReadBin("D:\filter_test\seg1_25104\kilosort4\temp_wh.dat", 128, chan, sample);

Z = ZoomPlot([preprocessed_filtered, preprocessed_reg, preprocessed_raw]);
legend('whitened (sliding mean template)', 'Whitened (regular mean template)', 'whitened (raw)')

%%
artifact_removed_reg = ReadBin("D:\filter_test\seg1_25104\all_files_seg1_25104.bin", 128, chan, sample);
Z = ZoomPlot([artifact_removed, artifact_removed_reg, raw_neural ]);
legend('target channel (sliding mean template)', 'target channel (regular mean template)',  'target channel (raw)')


%%
set(groot,'defaultLineLineWidth',1.0)