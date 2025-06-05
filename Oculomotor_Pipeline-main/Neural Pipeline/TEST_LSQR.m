clear all
close all
clc
%% reading file
% file_name = '001-16channels-50ms-200hz-80uA';
% file_name = '001-16channels-50ms-200hz-80uA';
% file_directory = uigetdir('\\10.16.59.34\cullenlab_server\Current Project Databases - NHP\2021 Abducens Stimulation (Neuropixel)\Data\Project 1 - Occulomotor Kinematics\Caesar_Session_2 - Copy\Renamed\', 'select the file that contains the raw _neural.mat data');
[rawDatafile, file_directory] = uigetfile(fullfile("D:\neuraldata\Caesar_002\bin_files_ERAASR", '*.bin'), 'select the .bin file that contains the raw data for ERAASR');
[stimDatafile, file_directory2] = uigetfile(fullfile(file_directory, '*.bin'), 'select the .bin file that contains the raw stim data for ERAASR');
trigger_file_path = uigetdir(file_directory, "select folder for session trigger files");
[~, trigger_files] = readfolder(trigger_file_path, 'session_trigger_*');
neural_directory = uigetdir(file_directory, 'select the file that contains the raw _neural.mat data');
outputfolder = uigetdir(file_directory, 'select output folder');

if ~exist([outputfolder '\filtered'], 'dir')
    mkdir(fullfile(outputfolder, 'filtered'))
end

fileNumberlist = [];
for i = 1:length(trigger_files)
        % Extract the number from the filename
        filename = trigger_files{i};
        fileidx = split(filename, ["_",".", "-"]);
        fileNumber = str2double(fileidx(3));
        fileNumberlist = [fileNumberlist fileNumber];
end
% trial_number =[10:17, 19, 32:38, 40, 43:45];
% [~,~,sorted_idx] = intersect(trial_number, fileNumberlist);
[~, sorted_idx] = sort(fileNumberlist);
trigger_files = trigger_files(sorted_idx);



file_num_list = [];
segment_marks = zeros(1, length(trigger_files)+1);
for i = 2:length(trigger_files)+1
    trigger_file_name = trigger_files{i-1};
    fileidx = split(trigger_file_name, ["-","_","."]);
    fileNumber = str2double(fileidx(3));
    file_num_list = [file_num_list, fileNumber];
    session_trigger = fullfile(trigger_file_path, trigger_file_name);
    trigger = load(session_trigger);
    segment_marks(i) = length(trigger.session_trigger);
end
segment_marks = cumsum(segment_marks);

%%
file_index = 7;
sample = segment_marks(file_index)+1:segment_marks(file_index+1);
% file_directory = '\\10.16.59.34\cullenlab_server\Current Project Databases - NHP\2021 Abducens Stimulation (Neuropixel)\Data\Project 1 - Occulomotor Kinematics\Caesar_Session_2 - Copy\Renamed\';
rawData = ReadBin([file_directory rawDatafile],128,[1:128], sample);
file_name = file_names{file_index};
% file_directory = '\\10.16.59.34\cullenlab_server\Current Project Databases - NHP\2021 Abducens Stimulation (Neuropixel)\Data\Project 1 - Occulomotor Kinematics\Caesar_Session_2 - Copy\Renamed\';

load([neural_directory,'\', file_name, '_Neural.mat']); % matlab file
% rawData = Data.Neural(:, 1:128);
stimData = ReadBin([file_directory2 stimDatafile],1,1, sample);
% stimData = Data.Neural(:, 131);
% rawData = ERASER.ReadBin(dataFileDir , 128, [1:128], [1:30*Data.N]);
% rawData = Data.Neural(:, 1:128);
% TRIGDAT =Data.Neural(:, 131);
%%
TRIGDAT =stimData;
% STIM_CHANS = find(any(stim_data~=0, 2));
% TRIGDAT = stim_data(STIM_CHANS(1),:)';
trigs1 = find(diff(TRIGDAT) < 0); 
trigs2 = find(diff(TRIGDAT) > 0);
if length(trigs1) > length(trigs2)
    trigs  = trigs1;
else
    trigs = trigs2;
end
trigs = trigs(1:2:end);
period = trigs(2) - trigs(1);

NSTIM = length(trigs);

segments_aligned = [];
time_diffs = diff(trigs);
repeat_gap_threshold = period*2;
repeat_boundaries = [0; find(time_diffs > repeat_gap_threshold); numel(trigs)];
num_repeats = numel(repeat_boundaries) - 1;
num_pulse = NSTIM/num_repeats;


for i = 1:NSTIM
    segment = (1 + trigs(i) ):(period+ trigs(i)); 
    segments_aligned = [segments_aligned; segment];  
end

fs = 30000; % samplig rate at 30kHz
fc = 300; % highpass at 300 Hz
f = num_pulse*20;  % frequency of stim wave
cutoff = 2*f;  % cutoff frequency (just above fundamental)
[b, a] = butter(4, 10/ (30000 / 2) , 'high');  % 4th-order Butterworth filter
sample_chans = 1:128;
sample_trials = 1:num_repeats;
prebuffer =60;
postbuffer =400;
raw_signal_segs = zeros(length(sample_trials), prebuffer+num_pulse*period+postbuffer, length(sample_chans));
stim_segs = zeros(length(sample_trials), prebuffer+num_pulse*period+postbuffer, length(sample_chans));
for i = 1:length(sample_trials)
    sample_trial = sample_trials(i);
    for j = 1:length(sample_chans)
        sample_chan = sample_chans(j);
        sample_pulses = (1+(sample_trial-1)*num_pulse:sample_trial*num_pulse);
        train_seg = reshape(segments_aligned(sample_pulses, :)', 1, []);
        prebuffer_seg = -prebuffer+train_seg(1):train_seg(1)-1;
        postbuffer_seg = train_seg(end)+1:postbuffer+train_seg(end);
        segment = [prebuffer_seg, train_seg, postbuffer_seg];
        raw_signal_segs(i, :, j) = filtfilt(b, a, rawData(segment, sample_chan));
        % raw_signal_segs(i, :, j) =rawData(segment, sample_chan);
        stim_segs(i, :, j) =  TRIGDAT(segment,:);
    end
end