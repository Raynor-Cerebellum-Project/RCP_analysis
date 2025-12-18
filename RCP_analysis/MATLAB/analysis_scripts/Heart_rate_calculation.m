% Read the VOG CSV file into a numeric matrix
% VOG_data = readmatrix('NRR_RW011_027.csv');
% VOG_data = readmatrix('NRR_RW006_003_2025_1029_145324.csv'); %baseline reaches
% load('NRR_RW006_003_Cal.mat') %baseline reaches

VOG_data = readmatrix('NRR_RW006_009_2025_1029_152200.csv'); %32ch stim
load('NRR_RW006_008_Cal.mat') %32ch stim

% Extract each column as a separate vector
TimeStamp = VOG_data(:, 1); 
FrameNumber = VOG_data(:, 2); 
HPos_pix = VOG_data(:, 4); 
VPos_pix = VOG_data(:, 5); 
HPos_deg = VOG_data(:, 6); 
VPos_deg = VOG_data(:, 7); 
area = VOG_data(:, 8); 

idx_extr = find(diff(area)>4*std(diff(area)));

expanded_idx_extr = [];
% Loop through original indices and add one on either side
for i = 1:length(idx_extr)
    current_idx = idx_extr(i);
    
    % Add current index and its neighboring indices
    % Check boundaries to avoid going out of bounds
    if current_idx > 1
        expanded_idx_extr = [expanded_idx_extr, current_idx - 1]; % Add index before
    end
    
    expanded_idx_extr = [expanded_idx_extr, current_idx]; % Add current index
    
    if current_idx < length(area)
        expanded_idx_extr = [expanded_idx_extr, current_idx + 1]; % Add index after
    end
end

% Remove duplicate indices and sort
expanded_idx_extr = unique(expanded_idx_extr);

area(expanded_idx_extr)=nan;
figure;plot(area)

med_pupil = nanmedian(area)
MAD_pupil = nanmedian(abs(area - med_pupil))


%%

% Assuming Data.heart_rate contains the plethysmography signal
% signal = Data.heart_rate;
signal = hr_sig;
% Parameters for peak detection
fs = 1000; % Sampling frequency (Hz), adjust based on your data
threshold = 0.1; % Adjust threshold for peak detection as needed
min_peak_distance = round(0.3 * fs); % Minimum distance between peaks (60 bpm)

% Find peaks in the signal
[peaks, locs] = findpeaks(signal, 'MinPeakHeight', threshold, 'MinPeakDistance', min_peak_distance);

% Calculate the time intervals between the peaks
if length(locs) >= 2
    intervals = diff(locs) / fs; % Time (in seconds) between peaks
else
    error('Not enough peaks detected to calculate heart rate.');
end

% Calculate instantaneous heart rate in beats per minute (BPM)
inst_heart_rate = 60 ./ intervals;

% Apply a moving average filter to smooth the instantaneous heart rate
window_size = 6; % You can set this to 3 or 5 depending on your preference
smoothed_heart_rate = movmean(inst_heart_rate, window_size);

% Prepare time vector for interpolation
time_inst_hr = locs(2:end)/fs; % Time vector corresponding to instantaneous HR
time_smoothed_hr = locs(2:end)/fs; % Time vector for smoothed HR

% Interpolation to match the length of the original signal
original_time = (0:length(signal)-1) / fs; % Time vector for the original signal
interpolated_smoothed_hr = interp1(time_smoothed_hr, smoothed_heart_rate, original_time, 'linear', 'extrap');

% To visualize the results
figure;
plot(original_time, signal);
hold on;
plot(locs/fs, peaks, 'ro'); % Red circles for detected peaks
xlabel('Time (s)');
ylabel('Signal Amplitude');
title('Plethysmography Signal with Detected Peaks');

% Display interpolated smoothed heart rate over original time
figure;
plot(original_time, interpolated_smoothed_hr, 'b-', 'DisplayName', 'Interpolated Smoothed HR'); % Interpolated Smoothed HR
xlabel('Time (s)');
ylabel('Heart Rate (BPM)');
title('Interpolated Smoothed Heart Rate');
legend;

mean(interpolated_smoothed_hr)
std(interpolated_smoothed_hr)

%% PLOT bar graph of heart rate and pupil diameter for different baselin/stim conditions
HR_mean(1,1)=162;
HR_std(1,1)=7.4;
HR_mean(1,2)=152; %2
HR_std(1,2)=12.2;
HR_mean(1,3)=158; %3
HR_std(1,3)=8.7;
HR_mean(1,4)=157; %4
HR_std(1,4)=12.9;

figure;
x = categorical({'Baseline', '16ch 400Hz', '64ch 400Hz', '64ch 130Hz'});
x = reordercats(x,{'Baseline', '16ch 400Hz', '64ch 400Hz', '64ch 130Hz'});
bar(x,HR_mean)
hold on
errorbar(HR_mean,HR_std,'.k')
ylim([0 200])


pup_mean(1,1) = (5332/5332)*4.8; %equivalent to 5mm
pup_std(1,1) = (425.5/5332)*4.8;
pup_mean(1,2) = (5405/5332)*4.8; %2
pup_std(1,2) = (480/5332)*4.8;
pup_mean(1,3) = (5711/5332)*4.8; %3
pup_std(1,3) = (498/5332)*4.8;
pup_mean(1,4) = (5528/5332)*4.8; %4
pup_std(1,4) = (368/5332)*4.8;

figure;
x = categorical({'Baseline', '16ch 400Hz', '64ch 400Hz', '64ch 130Hz'});
x = reordercats(x,{'Baseline', '16ch 400Hz', '64ch 400Hz', '64ch 130Hz'});
bar(x,pup_mean)
hold on
errorbar(pup_mean,pup_std,'.k')
ylim([0 7])

%%
% Example Data
fs = 1000; % Sampling frequency in Hz
% stim_ms = [1000, 5000, 10000]; % Stimulation times in milliseconds
stim_times = stim_ms / 1000; % Convert milliseconds to seconds
duration_before_stim = 4; % Duration before the stimulus (in seconds)
duration_after_stim = 8; % Duration after the stimulus (in seconds)

% Create an example heart rate time series (replace this with your actual HR data)
t = 0:1/fs:15; % Time vector (15 seconds)
% heart_rate = 60 + 5 * sin(2 * pi * 0.1 * t) + randn(size(t)); % Example heart rate signal with noise
heart_rate = interpolated_smoothed_hr;
% Convert the heart rate to samples (in the context of your time series)
stim_samples = round(stim_times * fs); % Convert stimulus times to sample indices

% Initialize a matrix to hold the segments
segments = [];

% Loop through each stimulus time
for i = 1:length(stim_samples)
    stim_idx = stim_samples(i);
    
    % Define the segment around the stimulus
    start_idx = stim_idx - duration_before_stim * fs; % Start index
    end_idx = stim_idx + duration_after_stim * fs;   % End index
    
    % Check for bounds and extract segment
    if start_idx > 0 && end_idx <= length(heart_rate)
        segments(:, i) = heart_rate(start_idx:end_idx); % Store the segment
    end
end

% Remove any empty columns from segments
segments(:, all(isnan(segments))) = []; 

% Compute the mean across stimuli
mean_trace = nanmean(segments, 2);

% Create time vector for the segments
segment_time_vector = linspace(-duration_before_stim, duration_after_stim, size(segments, 1));

% Plot individual traces and the mean trace
figure;
hold on;
% Plot individual traces
for i = 1:size(segments, 2)
    plot(segment_time_vector, segments(:, i), 'Color', [0.7 0.7 0.7]); % Light grey for individual traces
end

% Plot mean trace
plot(segment_time_vector, mean_trace, 'b-', 'LineWidth', 2, 'DisplayName', 'Mean Trace');

% Customization of the plot
xlabel('Time (s)');
ylabel('Heart Rate (BPM)');
title('Stimulus-Triggered Average Heart Rate');
legend('Individual traces', 'Mean Trace');
grid off;
hold off;
ylim([80 220])
fig = gcf;
print(fig, 'stim_evoked_hr_change_32ch_130_BR25.emf', '-depsc','-painters'); %force it to vector format

%%
% Plot trigger-averaged eye movement
fs = 1000; % Sampling frequency in Hz
% stim_ms = [1000, 5000, 10000]; % Stimulation times in milliseconds
duration_before_stim = .05; % Duration before the stimulus (in seconds)
duration_after_stim = .08; % Duration after the stimulus (in seconds)

% Create an example heart rate time series (replace this with your actual HR data)
t = 0:1/fs:15; % Time vector (15 seconds)
% heart_rate = 60 + 5 * sin(2 * pi * 0.1 * t) + randn(size(t)); % Example heart rate signal with noise
h_VOG = vog_sig;
h_VOG(h_VOG>-3e6)=nan;
h_VOG(h_VOG<-4.99e6)=nan;
% Convert the heart rate to samples (in the context of your time series)
stim_samples = round(stim_ms/10); % Convert stimulus times to sample indices

% Initialize a matrix to hold the segments
segments = [];

% Loop through each stimulus time
for i = 1:length(stim_samples)
    stim_idx = stim_samples(i);
    
    % Define the segment around the stimulus
    start_idx = stim_idx - duration_before_stim * fs; % Start index
    end_idx = stim_idx + duration_after_stim * fs;   % End index
    
    % Check for bounds and extract segment
%     if start_idx > 0 && end_idx <= length(h_VOG)
        segments(:, i) = h_VOG(start_idx:end_idx)-mean(h_VOG(start_idx:start_idx+100)); % Store the segment
%     end
end

segments = segments/-1.2668e4;
% Remove any empty columns from segments
segments(:, all(isnan(segments))) = []; 

% Compute the mean across stimuli
mean_trace = nanmean(segments, 2);

% Create time vector for the segments
segment_time_vector = linspace(-duration_before_stim, duration_after_stim, size(segments, 1))*1000;

% Plot individual traces and the mean trace
figure;
hold on;
% Plot individual traces
for i = 1:size(segments, 2)
    plot(segment_time_vector, segments(:, i), 'Color', [0.7 0.7 0.7]); % Light grey for individual traces
end

% Plot mean trace
plot(segment_time_vector, mean_trace, 'b-', 'LineWidth', 2, 'DisplayName', 'Mean Trace');

% Customization of the plot
xlabel('Time (ms)');
ylabel('Horizontal eye position (deg)');
title('Stimulus-Triggered eye movement');
legend('Individual traces', 'Mean Trace');
grid off;
hold off;
ylim([-4 4])
fig = gcf;
print(fig, 'stim_evoked_hr_change_32ch_130_BR25.emf', '-depsc','-painters'); %force it to vector format

