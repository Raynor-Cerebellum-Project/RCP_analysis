function STDplot(t, Y, color)
    t = reshape(t, 1, []);
    
    % Check dimension alignment
    if size(Y,2) == length(t)
        % OK
    elseif size(Y,1) == length(t)
        Y = Y';  % transpose so rows = trials
    else
        error('Time vector t must match one dimension of Y');
    end

    % Handle multiple trials
    if size(Y,1) > 1
        mu = mean(Y, 'omitnan');
        S  = std(Y, 'omitnan') ./ sqrt(sum(~isnan(Y), 1));  % correct N for SEM

        % Handle all-NaN cases gracefully
        if all(isnan(mu))
            warning('All values are NaN — skipping STDplot.');
            return
        end
        
        mu = reshape(mu, 1, []);
        S  = reshape(S, 1, []);
        
        M = mu + S;
        m = mu - S;
        fill([t, fliplr(t)], [M, fliplr(m)], color, ...
            'FaceAlpha', 0.2, 'EdgeColor', color, 'EdgeAlpha', 0, 'LineStyle', '-');
        hold on;
        plot(t, mu, 'Color', color, 'LineWidth', 1.5);
    
    elseif size(Y,1) == 1
        plot(t, Y, 'Color', color, 'LineWidth', 1.5);
    end
end
