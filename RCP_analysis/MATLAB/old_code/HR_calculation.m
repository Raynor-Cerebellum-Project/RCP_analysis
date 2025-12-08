% Assuming Data.heart_rate contains the plethysmography signal
signal = Data.heart_rate;

% Parameters for peak detection
fs = 1000; % Sampling frequency (Hz), adjust based on your data
threshold = 0.1; % Adjust threshold for peak detection as needed
min_peak_distance = round(0.2 * fs); % Minimum distance between peaks (60 bpm)

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

% To visualize, you can plot the result
time = (0:length(signal)-1) / fs;

% Plotting the plethysmography signal and the detected peaks
figure;
plot(time, signal);
hold on;
plot(locs/fs, peaks, 'ro'); % Red circles for detected peaks
xlabel('Time (s)');
ylabel('Signal Amplitude');
title('Plethysmography Signal with Detected Peaks');

% Display instantaneous heart rate over time
figure;
plot(locs(2:end)/fs, inst_heart_rate, 'g'); % Instantaneous HR
xlabel('Time (s)');
ylabel('Heart Rate (BPM)');
title('Instantaneous Heart Rate');