function shade_stim_blocks(repeat_boundaries, trigs_beg_sec, trigs_end_sec, t_center, zoom_win, y_limits)
    for r = 1:length(repeat_boundaries) - 1
        idx_start = repeat_boundaries(r) + 1;
        idx_end   = repeat_boundaries(r+1);

        t1 = trigs_beg_sec(idx_start);
        t2 = trigs_end_sec(idx_end);

        % Skip if block is outside of current time window
        if t2 < (t_center - zoom_win) || t1 > (t_center + zoom_win)
            continue;
        end

        x1 = max(t1, t_center - zoom_win);
        x2 = min(t2, t_center + zoom_win);
        patch([x1 x2 x2 x1], [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
              [0.8 0.9 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);  % light blue
    end
end
