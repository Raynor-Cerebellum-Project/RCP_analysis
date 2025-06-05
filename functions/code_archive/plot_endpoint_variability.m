function plot_endpoint_variability(sort_key, T_filtered, eval_condition, plot_title, base_folder, session)
    %% Plot bar graph: Endpoint Variability (Ipsi/Contra)
    x = categorical(sort_key.Label, sort_key.Label, 'Ordinal', true);
    bar_data = [sort_key.VarIpsi, sort_key.VarContra];
    bar_errs = [sort_key.VarStdIpsi, sort_key.VarStdContra];

    figure('Position', [100 100 1200 500]);
    b = bar(x, bar_data, 'grouped'); hold on;

    ngroups = size(bar_data, 1);
    nbars   = size(bar_data, 2);
    xvals   = nan(nbars, ngroups);
    for i = 1:nbars
        xvals(i,:) = b(i).XEndPoints;
    end
    for i = 1:nbars
        errorbar(xvals(i,:), bar_data(:,i), bar_errs(:,i), '.k', 'LineWidth', 1.5);
    end

    ylabel('Absolute Endpoint Variability (deg)');
    xlabel('Stimulation Condition');
    legend(b, {'Ipsi', 'Contra'});
    sgtitle(sprintf('Endpoint Variability: %s\n%s', eval_condition, plot_title), 'FontWeight', 'bold', 'FontSize', 14);
    xtickangle(45); box off; set(gca, 'TickDir', 'out');

    draw_section_lines(sort_key, b);

    filename = fullfile(base_folder, 'Figures', [eval_condition, '_', session, '_', 'End_point_Variability_Comparison.jpg']);
    print(gcf, filename, '-djpeg', '-r300');
end