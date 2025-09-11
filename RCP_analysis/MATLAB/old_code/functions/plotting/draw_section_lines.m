function draw_section_lines(sort_key, x_handle)
    %% Helper function to draw group boundaries and section labels
    trigger_str = string(sort_key.Trigger);

    begin_end_idx = find(trigger_str == "Beginning" | trigger_str == "End");
    other_idx     = find(~(trigger_str == "Beginning" | trigger_str == "End"));

    last_other     = max(other_idx);
    last_beginning = max(find(trigger_str == "Beginning"));

    % Determine x positions
    if isa(x_handle, 'matlab.graphics.chart.primitive.Bar') && isprop(x_handle(1), 'XEndPoints')
        xpos = x_handle(1).XEndPoints;
    elseif isnumeric(x_handle)
        xpos = x_handle;
    else
        error('Unsupported x_handle type');
    end

    % Draw vertical lines
    if ~isempty(last_other) && last_other < length(xpos)
        xline(xpos(last_other) + 0.5, '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
    if ~isempty(last_beginning) && last_beginning < length(xpos)
        xline(xpos(last_beginning) + 0.5, '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end

    % Expand ylim to make room for text
    yl = ylim;
    ylim([yl(1), yl(2) * 1.15]);
    yl = ylim;
    label_y = yl(2) * 0.98;

    % Add section labels
    if ~isempty(last_other)
        group1_idx = 1:last_other;
        text(mean(xpos(group1_idx)), label_y, 'Baselines', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
    end
    if ~isempty(last_other) && ~isempty(last_beginning)
        group2_idx = (last_other+1):last_beginning;
        text(mean(xpos(group2_idx)), label_y, 'Beginning', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
    end
    if ~isempty(last_beginning)
        group3_idx = (last_beginning+1):length(xpos);
        text(mean(xpos(group3_idx)), label_y, 'End', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
    end
end