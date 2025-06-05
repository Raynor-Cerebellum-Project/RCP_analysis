function plot_psid_results(Z_preds, Z_true, rel_mse, corrs, condition_name, n_ahead, behavior_label, base_path)

    close all;
    figure('Position', [100 100 1200 800]);

    if strcmp(behavior_label, 'hsv')
        y_label_str = 'Head Rotation Velocity (deg/s)';
    elseif strcmp(behavior_label, 'bsv')
        y_label_str = 'Body Rotation Velocity (deg/s)';
    else
        y_label_str = 'Behavior';
    end

    % Add title
    sgtitle([strrep(condition_name, '_', ' ') ' - ' behavior_label], 'FontSize', 16, 'FontWeight', 'bold');

    % Subplot 1: Relative MSE
    subplot(2, 2, 1); hold on;
    plot(1:n_ahead, rel_mse, 'k', 'LineWidth', 1.5);
    scatter(1:n_ahead, rel_mse, 'kx', 'LineWidth', 1.5);
    xlabel('Prediction Step Ahead (ms)');
    ylabel('Relative MSE');
    title(['Relative MSE']);
    set(gca, 'TickDir', 'out');

    % Subplot 2: Correlation
    subplot(2, 2, 2); hold on;
    plot(1:n_ahead, corrs, 'LineWidth', 1.5);
    scatter(1:n_ahead, corrs, 40, 'x', 'MarkerEdgeColor', [0 0.4470 0.7410], 'LineWidth', 1.5);
    xlabel('Prediction Step Ahead (ms)');
    ylabel('Correlation Coefficient (r)');
    title(['Prediction Accuracy']);
    ylim([min(corrs)-0.01 1]);
    set(gca, 'TickDir', 'out');

    % Subplot 3: 10 ms ahead
    subplot(2, 2, 3); hold on;
    step = 10;
    valid_T = size(Z_preds, 1);
    t_vec = ((1:valid_T) + step) / 1000;  % Convert to seconds
    plot(t_vec, Z_true(:, step), 'b', 'LineWidth', 1.5);
    plot(t_vec, Z_preds(:, step), 'r--', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel(y_label_str);
    yl = ylim;
    ylim([yl(1), yl(2) + 0.25 * range(yl)]);  % Add vertical padding
    legend('Actual', ['Predicted (' num2str(step) ' ms)'], 'Location', 'northeast');
    title(['Prediction at ' num2str(step) ' ms']);
    xlim([0 20]);
    set(gca, 'TickDir', 'out');
    
    % Subplot 4: 30 ms ahead
    subplot(2, 2, 4); hold on;
    step = n_ahead;
    t_vec = ((1:valid_T) + step) / 1000;  % Convert to seconds
    plot(t_vec, Z_true(:, step), 'b', 'LineWidth', 1.5);
    plot(t_vec, Z_preds(:, step), 'r--', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel(y_label_str);
    yl = ylim;
    ylim([yl(1), yl(2) + 0.25 * range(yl)]);  % Add vertical padding
    legend('Actual', ['Predicted (' num2str(step) ' ms)'], 'Location', 'northeast');
    title(['Prediction at ' num2str(step) ' ms']);
    xlim([0 20]);
    set(gca, 'TickDir', 'out');

    % Save
    base_dir = base_path;  % Get current working directory
    output_dir = fullfile(base_dir, 'PSID_Figures');
    
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    filename = fullfile(output_dir, [condition_name '_' behavior_label '_plots.pdf']);
    disp(output_dir);
    print(gcf, filename, '-dpdf', '-bestfit');
    close(gcf);
end