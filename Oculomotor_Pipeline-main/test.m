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
%% powerspectrum
% Fs = 30000;
% [b, a] = butter(4, 1000/ (30000 / 2) , 'high');  % 4th-order Butterworth filter
% 
% n1 = 13;
% n2 = 107;
% b = 400;
% x_test1 = rawData(b+1 :b+length(raw_signal_segs(i,:,j)));
% x_test2 = raw_signal_segs(i, :, j);
% t = ((1:length(x_test1))/Fs)';
% xTable1 = timetable(seconds(t),x_test1');
% xTable2 = timetable(seconds(t),x_test2');
% xTable3 =  timetable(seconds(t),a_est_all');
% [pxx1,f1] = pspectrum(xTable1);
% [pxx2,f2] = pspectrum(xTable2);
% [pxx3,f3] = pspectrum(xTable3);
% plot(f1,pow2db(pxx1), 'DisplayName','raw')
% hold on
% plot(f2,pow2db(pxx2), 'DisplayName','filtered')
% plot(f3,pow2db(pxx3), 'DisplayName','artifact')
% grid on
% hold off
% xlabel('Frequency (Hz)')
% ylabel('Power Spectrum (dB)')
% 
% legend

%%
filterOrder =16;       % Number of lags to include (x(t), x(t-1), ...)
windowSize =12;       % Number of time points in each window
n = filterOrder+windowSize;
i = 12;
j = 107;
dataCleaned = raw_signal_segs;
% [b, a] = butter(4, 600/ (30000 / 2) , 'high');        
        y_true_all = raw_signal_segs(i, :,j);
        a_est_all = raw_signal_segs(i, :, j);
        % y_test = filtfilt(b, a, raw_signal_segs(i, :,j));
        for l = 1:num_pulse
            segment = ((prebuffer -n+1):(prebuffer+period))+(l-1)*period;
            x = stim_segs(i,segment,j);
            % x = resample(x, 10, 1);

           %  x(x<0) = x(x<0)/3;
            y_true = raw_signal_segs(i, segment,j);
            %y = y_test(segment);
            y = y_true;
            % y = resample(y, 10, 1);
            % Assume x and y are column vectors of the same length
            N = length(x);
            

            a_est = zeros(N, 1);   % Output estimate of a(t)

            for t = windowSize + filterOrder - 1 : N
                % Construct y_local from the current window
                y_local = y(t - windowSize + 1 : t);  % [windowSize x 1]

                % Initialize X_local
                X_local = zeros(windowSize, filterOrder);  % [windowSize x filterOrder]
    
                for k = 1:filterOrder
                    % Fill each column with shifted x values
                    
                    X_local(:, k) = x(t - windowSize - k + 2 : t - k + 1);
                end
                % X_local = [X_local, ones(windowSize, 1)];
                % Solve least squares: theta = (X'X)^-1 X'y
                theta = pinv(X_local) * y_local';  % [filterOrder x 1]
                
                % Use most recent x values to estimate a(t)
                % x_recent = [x(t:-1:t - filterOrder + 1),  1];  % [filterOrder x 1]
                x_recent = x(t:-1:t - filterOrder + 1);
                a_est(t) = x_recent(:)' * theta;        % scalar

            end
            nonzero_mask = a_est~=0;
            onsets = find(diff([0; nonzero_mask]) == 1, 1);
            offsets = find(diff([ nonzero_mask; 0]) == -1, 1, 'last');
%             zero_idx = find(a_est~=0);  % or logical mask where it's zero
%             nonzero_idx = find(a_est==0);
            if ~isempty(offsets)
               
                % a_est(30:onsets) = linspace(0, a_est(onsets), length(30:onsets));
                
                % a_est(offsets+1:end) = linspace(y(offsets+1), 0, length(a_est(offsets+1:end)));
            end
            y_fit = y_true(offsets+1:end);
            x_fit = 1:length(y_fit);
            p = polyfit(x_fit, y_fit, 6);
            template = polyval(p, x_fit);
            % template(1) = y_fit(1);
            y_true(offsets+1:end) = y_true(offsets+1:end) - template;
            % y_true(zero_idx) =0.15*interp1(nonzero_idx, y_true(nonzero_idx), zero_idx, 'pchip');
            y_true = y_true - a_est';
            y_true_all(segment(n+1:end))  = y_true(n+1:end) -linspace(y_true(n+1), y_true(end), length(segment(n+1:end)));
            a_est_all(segment(n+1:end)) = - y_true_all(segment(n+1:end)) + y(n+1:end);
        end
        
        y_true_all(1:prebuffer)  = raw_signal_segs(i, 1:prebuffer, j) - linspace(y_true_all(1), y_true_all(prebuffer), length(1:prebuffer));
        y_true_all(segment(end)+1:end) = y_true_all(segment(end)+1:end) - linspace(y_true_all(segment(end)+1), 0, length(y_true_all(segment(end)+1:end)));
        

        % [b, a] = butter(4, 1000/ (30000 / 2) , 'low');
% y_test = filtfilt(b, a, raw_signal_segs(n1, :,n2));
[b, a] = butter(4, 250/ (30000 / 2) , 'high');
figure
subplot(3,1,1); 
plot( raw_signal_segs(i, :, j), 'DisplayName','raw','LineWidth',2);
hold on
plot(  a_est_all(:), 'DisplayName','estimated artifact');
title('raw data');
box off
legend
subplot(3,1,2); 
plot( filtfilt(b, a, y_true_all(:)));
hold on
% plot(filtfilt(b, a, a_est_all(:)))
title('Recovered y_{true}(t)');
box off
subplot(3,1,3); 
plot(  a_est_all(:));
% hold on
% plot(y_test)

title('artifact');
box off
%%
ZoomPlot([ ...
    dataCleaned(i, :, j)', raw_signal_segs(i, :, j)', stim_segs(i, :, j)'*500])

%%
figure;
subplot(3,1,1); 
plot(y, 'DisplayName','Raw Data'); 
hold on
plot([0, 0, 0, x(1:end-3)]*200, 'DisplayName','Stim Signal')
legend
box off
subplot(3,1,2); 
plot(y, 'LineWidth',3.0, 'DisplayName','raw signal')
hold on
plot(a_est, 'LineWidth',1.0, 'DisplayName','estimated artifact'); 

% [b, a] = butter(4, 250/ (30000 / 2) , 'high');  % 4th-order Butterworth filter
% title('Estimated a(t)');
% legend
% box off
%%
% [b, a] = butter(4, 1000/ (30000 / 2) , 'low');
% y_test = filtfilt(b, a, raw_signal_segs(n1, :,n2));
[b, a] = butter(4, 250/ (30000 / 2) , 'high');
figure
subplot(3,1,1); 
plot( raw_signal_segs(i, :, j), 'DisplayName','raw');
hold on
plot(  a_est_all(:), 'DisplayName','estimated artifact');
title('raw data');
box off
legend
subplot(3,1,2); 
plot( filtfilt(b, a, y_true_all(:)));
hold on
% plot(filtfilt(b, a, a_est_all(:)))
title('Recovered y_{true}(t)');
box off
subplot(3,1,3); 
plot(  a_est_all(:));

title('artifact');
box off
%%

ZoomPlot( y_true_all')
