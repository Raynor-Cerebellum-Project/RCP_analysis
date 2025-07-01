function batch_PSID_intracondition()
% Add PSID utilities
addpath(genpath('/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Analysis Codes/System_ID/PSID test'));
base_path = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/CRR_NPXL_011_Session_1/Seperate cells/Kilosort_all_cells/CRR_NPXL_011_Session_1/';
save_path = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/CRR_NPXL_011_Session_1/Figures';
save_dir = fullfile(save_path, 'withinCondition_test');
if ~exist(save_dir, 'dir'); mkdir(save_dir); end
%%
% Conditions
condition_files = {
    'BuH_yaw_active_like', 'BuH_yaw_sine', ...
    'HoB_yaw_active', 'HoB_yaw_active_like', 'HoB_yaw_active_multiple_perturbations', ...
    'HoB_yaw_sine', 'WB_yaw_active_like', 'WB_yaw_active_like_multilevel', 'WB_yaw_sine'
    };

cluster_numbers = [213 205 219 45 67 214 212 217 142 151 158 165];
n_ahead = 100;
nx = 8; n1 = 2;
bsv_steps = [10, 100];
hsv_steps = [10, 100];

for cond_i = 1:length(condition_files)
    condition_name = condition_files{cond_i};
    disp(['Processing: ' condition_name]);
    file_path = fullfile(base_path, [condition_name '.mat']);
    load(file_path);  % loads Data

    % === Neural data ===
    spike_rates_mat = [];
    for i = 1:length(cluster_numbers)
        field_name = ['fr_' num2str(cluster_numbers(i))];
        if isfield(Data, field_name)
            spike_rates_mat = [spike_rates_mat; Data.(field_name)'];
        end
    end
    dead_neurons = all(spike_rates_mat == 0, 2);
    spike_rates = spike_rates_mat(~dead_neurons, :)';

    % === Behavioral signals ===
    bsv = Data.bsv(:);
    hsv = Data.hsv(:);

    % === Train/test split ===
    T = size(spike_rates, 1);
    T_train = floor(0.8 * T);
    Y_train = spike_rates(1:T_train, :);
    Z_train_bsv = bsv(1:T_train);
    Z_train_hsv = hsv(1:T_train);
    Y_test = spike_rates(T_train+1:end, :);
    Z_test_bsv = bsv(T_train+1:end);
    Z_test_hsv = hsv(T_train+1:end);

    % === Train PSID ===
    id_bsv = PSID(Y_train, Z_train_bsv, nx, n1, 12);
    id_hsv = PSID(Y_train, Z_train_hsv, nx, n1, 12);

    % === Predict ===
    [Zpb, Ztb, mse_bsv, r_bsv] = predictPSID(id_bsv.A, id_bsv.Cy, id_bsv.Cz, Y_test, Z_test_bsv, n_ahead);
    [Zph, Zth, mse_hsv, r_hsv] = predictPSID(id_hsv.A, id_hsv.Cy, id_hsv.Cz, Y_test, Z_test_hsv, n_ahead);

    fig = figure('Visible', 'off', ...
        'Units', 'inches', ...
        'Position', [0 0 8.5 15], ... % standard portrait layout
        'PaperPositionMode', 'auto');

    t = tiledlayout(6, 2, ...
        'TileSpacing', 'tight', ...
        'Padding', 'tight');

    font_size = 9;
    lw = 2;

    % bsv
    nexttile(t, 1); plot(1:n_ahead, mse_bsv, 'k-x', 'LineWidth', lw); title('bsv Rel. MSE');
    xlabel('Step Ahead (ms)'); ylabel('Relative MSE'); set(gca,'TickDir','out');

    nexttile(t, 2); plot(1:n_ahead, r_bsv, 'b-x', 'LineWidth', lw); title('bsv Corr');
    xlabel('Step Ahead (ms)'); ylabel('Correlation'); set(gca,'TickDir','out');

    bsv_tiles = [3, 5];

    for idx = 1:length(bsv_steps)
        s = bsv_steps(idx);
        if s > n_ahead
            continue;  % skip if step exceeds prediction range
        end
        nexttile(t, bsv_tiles(idx), [1 2]);
        t_ax = (1:size(Zpb,1)) + s;
        plot(t_ax/1000, Ztb(:,s), 'b', 'LineWidth', lw); hold on;
        plot(t_ax/1000, Zpb(:,s), 'r--', 'LineWidth', lw);
        title(sprintf('bsv @%dms', s));
        xlabel('Time (s)'); ylabel('Body velocity (deg/s)');
    end

    % hsv
    nexttile(t, 7); plot(1:n_ahead, mse_hsv, 'k-x', 'LineWidth', lw); title('hsv Rel. MSE');
    xlabel('Step Ahead (ms)'); ylabel('Relative MSE'); set(gca,'TickDir','out');

    nexttile(t, 8); plot(1:n_ahead, r_hsv, 'b-x', 'LineWidth', lw); title('hsv Corr');
    xlabel('Step Ahead (ms)'); ylabel('Correlation'); set(gca,'TickDir','out');

    hsv_tiles = [9, 11];
    for idx = 1:length(hsv_steps)
        s = hsv_steps(idx);
        if s > n_ahead
            continue;  % skip if step exceeds prediction range
        end
        nexttile(t, hsv_tiles(idx), [1 2]);
        t_ax = (1:size(Zph,1)) + s;
        plot(t_ax/1000, Zth(:,s), 'b', 'LineWidth', lw); hold on;
        plot(t_ax/1000, Zph(:,s), 'r--', 'LineWidth', lw);
        title(sprintf('hsv @%dms', s));
        xlabel('Time (s)'); ylabel('Head velocity (deg/s)');
    end


    % Add legend directly to last plot (e.g., hsv @30ms)
    legend({'True', 'Predicted'}, ...
        'Orientation', 'horizontal', ...
        'Location', 'southoutside', ...
        'Box', 'off', ...
        'FontSize', font_size);


    sgtitle(sprintf('%s - head and body velocity predictions\nTrain: %.1f seconds | Test: %.1f seconds (fs = 1kHz)', ...
        strrep(condition_name, '_', ' '), T_train/1000, (T - T_train)/1000), ...
        'FontSize', font_size + 2);
    set(findall(fig, '-property', 'FontSize'), 'FontSize', font_size);
    % Save figure in PDF, SVG, and JPG to separate folders
    pdf_dir = fullfile(save_dir, 'pdfFigs');
    svg_dir = fullfile(save_dir, 'svgFigs');
    jpg_dir = fullfile(save_dir, 'jpgFigs');
    if ~exist(pdf_dir, 'dir'); mkdir(pdf_dir); end
    if ~exist(svg_dir, 'dir'); mkdir(svg_dir); end
    if ~exist(jpg_dir, 'dir'); mkdir(jpg_dir); end

    fname_pdf = fullfile(pdf_dir, [condition_name '_prediction']);
    fname_svg = fullfile(svg_dir, [condition_name '_prediction']);
    fname_jpg = fullfile(jpg_dir, [condition_name '_prediction']);

    print([fname_pdf '.pdf'], '-dpdf', '-bestfit');
    print([fname_svg '.svg'], '-dsvg');
    print([fname_jpg '.jpg'], '-djpeg', '-r300');  % 300 dpi for good quality

    close(fig);
end
end

% === Predict Function ===
function [Z_pred, Z_true, rel_mse, corrs] = predictPSID(A, C_y, C_z, Y, Z, n_ahead)
T_test = size(Y, 1);
valid_T = T_test - n_ahead;
Z_pred = nan(valid_T, n_ahead);
Z_true = nan(valid_T, n_ahead);
for k = 1:valid_T
    y_k = Y(k,:)';
    x_t = pinv(C_y) * y_k;
    for i = 1:n_ahead
        x_t = A * x_t;
        Z_pred(k, i) = C_z * x_t;
    end
    Z_true(k, :) = Z(k+1:k+n_ahead)';
end
rel_mse = sum((Z_pred - Z_true).^2, 1) ./ sum(Z_true.^2, 1);
corrs = arrayfun(@(i) corr(Z_pred(:,i), Z_true(:,i), 'rows','complete'), 1:n_ahead);
end