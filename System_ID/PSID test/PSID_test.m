clear;clc;close all;
%% Load data
% Data.Intan_idx: the index values of the behavioural signals that correspond to intan samples
% Data.bsv: body in space velocity (yaw)
% Data.hsv: head in space velocity (yaw) 
% Data.segments - contains the indexes of the start and end of a movement in each row
addpath(genpath('C:\Users\bryan\Documents\workspace\Raynor\PSID test\'));
i = 4;
base_path = 'Z:\Current Project Databases - NHP\2025 Cerebellum prosthesis\Bryan\CRR_NPXL_011_Session_1\Seperate cells\Kilosort_all_cells\CRR_NPXL_011_Session_1\';
condition_files = {
    'BuH_yaw_active_like',
    'BuH_yaw_sine',
    'HoB_yaw_active',
    'HoB_yaw_active_like',
    'HoB_yaw_active_multiple_perturbations',
    'HoB_yaw_sine',
    'WB_yaw_active_like',
    'WB_yaw_active_like_multilevel',
    'WB_yaw_sine'
};
file_name = [condition_files{i} '.mat'];
file_path = fullfile(base_path, file_name);
load(file_path);  % Loads the 'Data' struct

condition_name = condition_files{i};  % Remove ".mat"
disp(['Running analysis on: ' condition_name]);

cluster_numbers = [213 205 219 45 67 214 212 217 142 151 158 165];

Intan_idx = Data.Intan_idx;
behavior_hz = 1000;
bsv = Data.bsv;
hsv = Data.hsv;
segments = Data.segments;
t_vec = linspace(0,length(bsv)/behavior_hz,length(bsv));

spike_rates_mat = [];
rate_matrix = [];

for i = 1:length(cluster_numbers)
    num = cluster_numbers(i);
    field_name = ['fr_' num2str(num)];
    data = Data.(field_name);  % Access Data.ua_number
    spike_rates_mat = [spike_rates_mat; data'];  % Concatenate as rows
end

% Find columns that are all zeros
dead_neurons = all(spike_rates_mat == 0, 2);

% Remove dead channels
spike_rates_mat_clean = spike_rates_mat(~dead_neurons, :);

% spike_times = readNPY('spike_times.npy');
% load('D:\CRR_NPXL_011_Session_1\Total (Neural)\SPIKES.mat');
%% Analysis
figure;
colororder([0, 0.4470, 0.7410; 0, 0, 0]);
ax1 = subplot(3, 1, 1);
hold on;
yyaxis left;
plot(t_vec, bsv, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1.5);
plot(t_vec, hsv+50, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5, 'LineStyle', '-');
xlim([0 20.3])
legend('Body in space velocity', 'Head in space velocity (Offset by 50)');
ylabel('Rotation (deg/s)');
title('Head and body in space velocity');
yyaxis right;
plot(t_vec, spike_rates_mat_clean(1,:), 'DisplayName', ['Neuron 1'], 'Color', 'k');
yl = ylim;
ylim([yl(1), yl(2) + 0.25 * range(yl)]);  % Add vertical padding
ylabel('Voltage (mV)');
set(gca, 'TickDir', 'out');

ax2 = subplot(3, 1, 2); hold on;
yyaxis left;
plot(t_vec, bsv, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1.5);
plot(t_vec, hsv+50, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5, 'LineStyle', '-');
xlim([0 20.3])
legend('Body in space velocity', 'Head in space velocity (Offset by 50)');
ylabel('Rotation (deg/s)');
title('Head and body in space velocity');
yyaxis right;
plot(t_vec, spike_rates_mat_clean(6,:), 'DisplayName', ['Neuron 7'], 'Color', 'k');
yl = ylim;
ylim([yl(1), yl(2) + 0.25 * range(yl)]);  % Add vertical padding
ylabel('Voltage (mV)');
set(gca, 'TickDir', 'out');

ax2 = subplot(3, 1, 3); hold on;
yyaxis left;
plot(t_vec, bsv, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1.5);
plot(t_vec, hsv+50, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5, 'LineStyle', '-');
xlim([0 20.3])
legend('Body in space velocity', 'Head in space velocity (Offset by 50)');
ylabel('Rotation (deg/s)');
title('Head and body in space velocity');
yyaxis right;
plot(t_vec, spike_rates_mat_clean(7,:), 'DisplayName', ['Neuron 8'], 'Color', 'k');
yl = ylim;
ylim([yl(1), yl(2) + 0.25 * range(yl)]);  % Add vertical padding
ylabel('Voltage (mV)');
set(gca, 'TickDir', 'out');

% for i = 1:length(cluster_numbers)
% end
% yyaxis right;
% ylabel('Voltage (mV)');
% xlabel('Time (s)');
% xlim([0 20.3])

% Save
filename = ['WB_yaw_sine_three_neurons.pdf'];
print(gcf, filename, '-dpdf', '-bestfit');
%% PSID
nx = 10;
n1 = 2;
i = 12;
idSys = cell(1,10);

for nx = 6:10
    idSys{nx} = PSID(spike_rates_mat_clean', hsv, nx, n1, i);
end
%% Plot spike rates
fs = 1000;                         % Sampling rate in Hz
n_timepoints = size(spike_rates_mat_clean, 2);      % Number of samples
t = (0:n_timepoints-1) / fs;       % Time vector in seconds

figure;
hold on;

% Plot each row with vertical offset
offset = 200;  % adjust based on your data scale
for i = 1:size(spike_rates_mat_clean, 1)
    plot(t, spike_rates_mat(i,:) + (i-1)*offset, 'DisplayName', ['Neuron ' num2str(i)]);
end

xlabel('Time (s)');
ylabel('Firing rate (offset for clarity)');
title('Neural Signals over Time');
legend;
set(gca, 'TickDir', 'out', 'ytick',[])

% Save
filename = ['WB_yaw_sine_spike_rates.pdf'];
print(gcf, filename, '-dpdf', '-bestfit');
%% Plot predictions
nx_size_chosen = 8;
n_ahead = 100;

% Data
Y = spike_rates_mat_clean';   % [T × m], neural activity (used for inference)
Z = hsv';               % [T × 1], behavior (target to predict)

% Get system matrices
A = idSys{1, nx_size_chosen}.A;
C_y = idSys{1, nx_size_chosen}.Cy;   % C matrix for input (neural activity)
K = idSys{1, nx_size_chosen}.K;
C_z = idSys{1, nx_size_chosen}.Cz;  % Optional: behavior output matrix, if PSID returns it

[T, m] = size(Y);
valid_T = T - n_ahead;

% Initialize prediction matrices
Z_preds = nan(valid_T, n_ahead);
Z_true = nan(valid_T, n_ahead);

for k = 1:valid_T
    y_k = Y(k, :)';
    x_t = pinv(C_y) * y_k;
    for i = 1:n_ahead
        x_t = A * x_t;
        Z_preds(k, i) = C_z * x_t;
    end
    Z_true(k, :) = Z(k+1:k+n_ahead)';
end


% --- Compute average values across all timepoints ---
mean_pred = mean(Z_preds, 1, 'omitnan');
mean_true = mean(Z_true, 1, 'omitnan');

% --- Compute relative MSE for each step ---
rel_mse = sum((Z_preds - Z_true).^2, 1, 'omitnan') ./ sum(Z_true.^2, 1, 'omitnan');

% --- Compute correlation across all steps ---
corrs = nan(1, n_ahead);
for i = 1:n_ahead
    valid_idx = ~isnan(Z_preds(:, i)) & ~isnan(Z_true(:, i));
    if sum(valid_idx) > 2
        corrs(i) = corr(Z_preds(valid_idx, i), Z_true(valid_idx, i));
    end
end

% % --- Plot 1: Mean predicted vs actual behavior at each prediction step ---
% figure; hold on;
% plot(1:n_ahead, mean_true, 'b', 'LineWidth', 1.5);
% plot(1:n_ahead, mean_pred, 'r--', 'LineWidth', 1.5);
% legend('Mean Actual', 'Mean Predicted');
% xlabel('Prediction Step Ahead');
% ylabel('Behavior');
% title('Mean Behavior Across Time (Per Step Ahead)');
% grid on;



% --- Start combined figure ---
figure('Position', [100 100 1200 800]);

% Subplot 1: Relative MSE
subplot(2, 2, 1); hold on;
plot(1:n_ahead, rel_mse, 'k', 'LineWidth', 1.5);
scatter(1:n_ahead, rel_mse, 'kx', 'LineWidth', 1.5);
xlabel('Prediction Step Ahead (ms)');
ylabel('Relative MSE');
title('Relative MSE vs. Prediction Horizon');
set(gca, 'TickDir', 'out');

% Subplot 2: Correlation vs. Prediction Step
subplot(2, 2, 2); hold on;
plot(1:n_ahead, corrs, 'LineWidth', 1.5);
scatter(1:n_ahead, corrs, 40, 'x', 'MarkerEdgeColor', [0 0.4470 0.7410], 'LineWidth', 1.5);
xlabel('Prediction Step Ahead (ms)');
ylabel('Correlation Coefficient (r)');
title('Prediction Accuracy vs. Horizon');
ylim([min(corrs)-0.01 1]);
set(gca, 'TickDir', 'out');

% Subplot 3: 10 ms ahead prediction
subplot(2, 2, 3); hold on;
step = 10;
t_vec = (1:valid_T) + step;
plot(t_vec/1000, Z_true(:, step), 'b', 'LineWidth', 1.5);
plot(t_vec/1000, Z_preds(:, step), 'r--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Behavior');
legend('Actual', ['Predicted (' num2str(step) ' ms)'], 'Location', 'northeast');
title(['Prediction at ' num2str(step) ' ms']);
xlim([0 20]);
set(gca, 'TickDir', 'out');

% Subplot 4: 30 ms ahead prediction
subplot(2, 2, 4); hold on;
step = 100;
t_vec = (1:valid_T) + step;
plot(t_vec/1000, Z_true(:, step), 'b', 'LineWidth', 1.5);
plot(t_vec/1000, Z_preds(:, step), 'r--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Behavior');
legend('Actual', ['Predicted (' num2str(step) ' ms)'], 'Location', 'northeast');
title(['Prediction at ' num2str(step) ' ms']);
xlim([0 20]);
set(gca, 'TickDir', 'out');

% --- Save and close figure ---
filename = [condition_name '_figure.pdf'];
print(gcf, filename, '-dpdf', '-bestfit');

%%
for cond_i = 1:length(condition_files)
    condition_name = condition_files{cond_i};
    file_path = fullfile(base_path, [condition_name '.mat']);
    disp(['Processing: ' condition_name]);

    % Load data
    load(file_path);
    Intan_idx = Data.Intan_idx;
    bsv = Data.bsv;
    hsv = Data.hsv;
    segments = Data.segments;

    % Neural data: stack across clusters
    spike_rates_mat = [];
    for i = 1:length(cluster_numbers)
        num = cluster_numbers(i);
        field_name = ['fr_' num2str(num)];
        data = Data.(field_name);
        spike_rates_mat = [spike_rates_mat; data'];
    end
    % Clean zero rows
    dead_neurons = all(spike_rates_mat == 0, 2);
    spike_rates_mat_clean = spike_rates_mat(~dead_neurons, :);

    % PSID
    idSys = PSID(spike_rates_mat_clean', hsv, nx_size_chosen, n1, 12);
    A = idSys.A;
    C_y = idSys.Cy;
    K = idSys.K;
    C_z = idSys.Cz;

    % Prediction
    Y = spike_rates_mat_clean';  % [T × m]
    Z = hsv';  % [T × 1]
    [T, ~] = size(Y);
    valid_T = T - n_ahead;
    Z_preds = nan(valid_T, n_ahead);
    Z_true = nan(valid_T, n_ahead);

    for k = 1:valid_T
        y_k = Y(k, :)';
        x_t = pinv(C_y) * y_k;
        for i = 1:n_ahead
            x_t = A * x_t;
            Z_preds(k, i) = C_z * x_t;
        end
        Z_true(k, :) = Z(k+1:k+n_ahead)';
    end

    % --- Compute overall RMSE for this condition ---
    errors = Z_preds - Z_true;
    rmse_by_condition(cond_i) = sqrt(nanmean(errors(:).^2));
end

figure;
bar(rmse_by_condition, 'FaceColor', [0.2 0.6 0.8]);
xticks(1:length(condition_files));
xticklabels(strrep(condition_files, '_', '\_'));
xtickangle(45);
ylabel('RMSE (rad/s)');
title('Prediction Error (RMSE) per Condition');
box off; set(gca, 'TickDir', 'out');

% Optional save
print('Condition_RMSE_Summary.pdf', '-dpdf', '-bestfit');
