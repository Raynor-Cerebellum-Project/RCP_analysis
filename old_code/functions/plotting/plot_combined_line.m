function plot_combined_line(sort_key, T_filtered, eval_condition, plot_title, base_folder, session)
    %% Plot line plot: Combined Error and Variability
    [~, match_idx] = ismember(sort_key.TrialID, T_filtered.BR_File);
    T_filtered_sorted = T_filtered(match_idx, :);
    y_err = T_filtered_sorted.ErrMean_combined;
    y_var = T_filtered_sorted.varMean_combined;

    x = categorical(sort_key.Label, sort_key.Label, 'Ordinal', true);
    x_vals = 1:length(x);

    figure('Position', [100 100 1000 400]);
    yyaxis left
    plot(x_vals, y_err, '-o', 'LineWidth', 2, 'Color', [0.2 0.2 0.8]);
    ylabel('Endpoint Error (deg)');

    yyaxis right
    plot(x_vals, y_var, '-s', 'LineWidth', 2, 'Color', [0.85 0.33 0.1]);
    ylabel('Endpoint Variability (deg)');

    xticks(x_vals);
    xticklabels(cellstr(x));
    xtickangle(45);
    xlabel('Stimulation Condition');
    xlim([0.2, length(x) + 0.5]);

    title(sprintf('Endpoint Errors and Variability: %s\n%s', eval_condition, plot_title), 'Units', 'normalized', 'Position', [0.5, 1.00, 0], 'FontWeight', 'normal', 'Interpreter', 'none');
    box off; set(gca, 'TickDir', 'out');

    draw_section_lines(sort_key, x_vals);

    filename = fullfile(base_folder, 'Figures', [eval_condition, '_', session, '_', 'Combined_Line_Graph_Comparison.jpg']);
    print(gcf, filename, '-djpeg', '-r300');
end