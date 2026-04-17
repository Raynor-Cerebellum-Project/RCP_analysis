% % File name (make sure it's in your current folder or add full path)
% filename = 'NRR_RW022_009_Cam-1DLC_Resnet50_Nike_reach_buttonNov6shuffle7_snapshot_140.csv';
% 
% % Read the CSV file
% data = readmatrix(filename);
% 
% % Extract columns K and L (11 and 12), rows 4 to end
% wrist_x = data(4:end, 11);
% wrist_y = data(4:end, 12);
% 
% % Create time vector (sampling rate = 100 Hz)
% Fs = 100; % Hz
% t = (0:length(wrist_x)-1) / Fs;
% 
% % Plot
% figure;
% plot(t, wrist_x, 'b', 'DisplayName', 'Wrist X');
% hold on;
% plot(t, wrist_y, 'r', 'DisplayName', 'Wrist Y');
% xlabel('Time (s)');
% ylabel('Position');
% title('Wrist Position Over Time');
% legend;
% grid on;

%% ---- Load data ----
filename = 'NRR_RW022_009_Cam-1DLC_Resnet50_Nike_reach_buttonNov6shuffle7_snapshot_140.csv';

data = readmatrix(filename);

% Extract wrist (columns K=11, L=12), rows 4:end
wrist_x = data(4:end, 11);
wrist_y = data(4:end, 12);

%% ---- Time vector ----
Fs = 100; % sampling rate (Hz)
t = (0:length(wrist_x)-1)/Fs;

%% ---- Select 20–30 seconds ----
idx = t >= 21 & t <= 28;

x = wrist_x(idx);
y = wrist_y(idx);
t_seg = t(idx);

%% ---- Low-pass filter to reduce marker jitter ----
% Fs: sampling rate (Hz)
% x, y: your wrist data for 20–30 s segment

Fc = 10;   % cutoff frequency (Hz)
[b,a] = butter(4, Fc/(Fs/2), 'low');  % 4th-order low-pass Butterworth

% Apply zero-phase filtering
x = filtfilt(b, a, x);
y = filtfilt(b, a, y);

% Handle missing values (important for smooth video)
x = fillmissing(x,'linear');
y = fillmissing(y,'linear');

%% ---- Interpolate to video frame rate ----
videoFPS = 50;

t_uniform = 20:1/videoFPS:30;  % exact frame timestamps

x_interp = interp1(t_seg, x, t_uniform);
y_interp = interp1(t_seg, y, t_uniform);

%% ---- Video setup ----
outputFile = fullfile('D:\DLC_v2\NRR_RW022', 'wrist_trajectory_21_28s_fast.mp4');

v = VideoWriter(outputFile, 'MPEG-4');
v.FrameRate = videoFPS;
v.Quality = 100; % improves smoothness
open(v);

%% ---- Create offscreen figure (critical for smoothness) ----
%% ---- Minimalist figure for video ----
%% ---- Minimalist figure with fixed axes ----
fig = figure('Visible','off', 'Position',[100 100 560 420], 'Color','w');

ax = axes(fig);
hold(ax,'on');
axis(ax,'off');       % remove axes ticks and labels
axis(ax,'equal');     % preserve aspect ratio
axis(ax,'manual');    % prevent MATLAB from auto-scaling

% ---- Fix axis limits based on your data ----
xPadding = 0.05*(max(x_interp)-min(x_interp));
yPadding = 0.05*(max(y_interp)-min(y_interp));
xlim(ax, [min(x_interp)-xPadding, max(x_interp)+xPadding]);
ylim(ax, [min(y_interp)-yPadding, max(y_interp)+yPadding]);

% Optional: flip Y if needed
set(ax, 'YDir', 'reverse');

% ---- Plot objects ----
purple = [0.5 0 0.5];
trail = plot(ax, nan, nan, '-', 'Color', purple, 'LineWidth', 1.5);
point = plot(ax, nan, nan, 'o', 'MarkerEdgeColor', purple, 'MarkerFaceColor', purple);
%% ---- Generate video frames (NO drawnow = smooth) ----
for i = 1:length(t_uniform)

    % Update trajectory
    set(trail, 'XData', x_interp(1:i), 'YData', y_interp(1:i));
    
    % Current position
    set(point, 'XData', x_interp(i), 'YData', y_interp(i));
    
    % Time display
%     title(ax, sprintf('Wrist trajectory (t = %.2f s)', t_uniform(i)));
    
    % Capture frame
    frame = getframe(fig);
    writeVideo(v, frame);
end

%% ---- Cleanup ----
close(v);
close(fig);

fprintf('Video saved to:\n%s\n', outputFile);