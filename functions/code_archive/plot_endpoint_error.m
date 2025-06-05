function plot_endpoint_error(sort_key, T_filtered, eval_condition, plot_title, base_folder, session)
    %% Plot bar graph: Absolute Endpoint Error (Ipsi/Contra)
    x = categorical(sort_key.Label, sort_key.Label, 'Ordinal', true);
    bar_data = [sort_key.MeanIpsi, sort_key.MeanContra];
    bar_errs = [sort_key.StdIpsi, sort_key.StdContra];

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

    ylabel('Absolute Endpoint Error (deg)');
    xlabel('Stimulation Condition');
    legend(b, {'Ipsi', 'Contra'});
    sgtitle(sprintf('Endpoint Errors: %s\n%s', eval_condition, plot_title), 'FontWeight', 'bold', 'FontSize', 14);
    xtickangle(45); box off; set(gca, 'TickDir', 'out');

    draw_section_lines(sort_key, b);

    filename = fullfile(base_folder, 'Figures', [eval_condition, '_', session, '_', 'Mean_Error_Comparison.jpg']);
    print(gcf, filename, '-djpeg', '-r300');
end

