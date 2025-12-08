
load('NRR_RW011_025_Cal.mat')
%%

% Assuming Data.heart_rate contains the plethysmography signal
signal = Data.heart_rate;

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
window_size = 5; % You can set this to 3 or 5 depending on your preference
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

%%
load('aligned__NRR_RW011_251203_155431__Intan_025__BR_025.mat')